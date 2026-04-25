"""
Shared agentic rollout loop for MetroCrowdManager.

Both `inference.py` and `train_grpo.py` import from here. Keeping the
loop in one place guarantees that the `turn_history` shape consumed by
the environment's reward functions stays consistent between training
and evaluation.

The loop accepts a `Stepper` — anything with ``reset(**kwargs)`` and
``step(action)`` methods returning OpenEnv-compatible observations. In
training we pass the in-process environment directly; in inference we
pass the WebSocket-backed `MetrocrowdmanagerEnv` client.

A `model_generate_fn(messages) -> str` is supplied by the caller so the
same rollout works against an OpenAI-compatible HTTP model, an Unsloth
in-process model, or a vLLM server.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:
    from MetroCrowdManager.models import SubmitResponseAction
except ImportError:  # pragma: no cover
    from models import SubmitResponseAction

from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)


# ---------------------------------------------------------------------------
# System prompts (one per task)
# ---------------------------------------------------------------------------

_TOOL_FORMAT_BLOCK = """\
You may invoke MCP tools by emitting one or more blocks of the form:

    <tool_call>{"name": "<tool_name>", "arguments": { ... }}</tool_call>

Tool results will be returned in the next user message inside
<tool_result> tags. When you are ready to give your FINAL answer, reply
with plain text containing NO <tool_call> blocks.

TOOL ARGUMENT RULES (apply to every <tool_call>):
- "arguments" must be a JSON object whose keys EXACTLY match the parameter
  names listed for each tool below.
- Do NOT rename, alias, or invent keys (e.g. `origin_station`,
  `destination_station`, `to`, `from`, `dest` are all invalid). Wrong key
  names cause the call to fail validation.
- Use the exact types shown (str / int / float). Pass {} for tools that
  take no arguments.
- Emit one <tool_call> at a time and wait for its <tool_result> before
  issuing the next call.
"""

_TICKET_BOOKING_TOOLS = """\
Available tools (call ONLY with the parameter names shown):

  * validate_destination(destination: str) — verify a station name; returns
    {"valid": bool, "normalized": str|null, "destination": str}.
    Example: <tool_call>{"name": "validate_destination", "arguments": {"destination": "greenfield"}}</tool_call>

  * get_ticket_cost(source: str, destination: str, passenger_count: int) —
    quote the fare. Returns {"cost": number|null, "currency": "INR",
    "found": bool, "source": str, "destination": str, "passenger_count": int}.
    Example: <tool_call>{"name": "get_ticket_cost", "arguments": {"source": "South Gate", "destination": "Greenfield", "passenger_count": 2}}</tool_call>

  * initiate_payment(amount: float, passenger_count: int) — start a payment;
    returns {"payment_id": str, "status": "pending", "amount": float,
    "passenger_count": int}.
    Example: <tool_call>{"name": "initiate_payment", "arguments": {"amount": 60.0, "passenger_count": 2}}</tool_call>

  * check_payment_status(payment_id: str) — poll a payment; resolves to
    success/failed after a few polls. Returns {"payment_id": str,
    "status": "pending"|"success"|"failed"|"unknown"}.
    Example: <tool_call>{"name": "check_payment_status", "arguments": {"payment_id": "PAY-abc123"}}</tool_call>

  * list_valid_stations() — full station list and source. No arguments.
    Returns {"stations": [str...], "source_station": str}.
    Example: <tool_call>{"name": "list_valid_stations", "arguments": {}}</tool_call>

  * get_current_time() — station-system time as HH:MM. No arguments.
    Returns {"time": "HH:MM"}.
    Example: <tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>
"""

_TICKET_ISSUANCE_TOOLS = """\
Available tools (call ONLY with the parameter names shown):

  * get_platform_for_destination(destination: str) — platform that serves
    `destination`. Returns {"platform": int|null, "found": bool,
    "destination": str}.
    Example: <tool_call>{"name": "get_platform_for_destination", "arguments": {"destination": "Greenfield"}}</tool_call>

  * get_platform_crowd(platform: int) — per-zone platform crowd percentages
    (10 zones, A-J). Returns {"platform": int, "found": bool,
    "zones": [int×10], "zone_labels": ["A".."J"]}.
    Example: <tool_call>{"name": "get_platform_crowd", "arguments": {"platform": 3}}</tool_call>

  * get_train_crowd_occupation(platform: int) — per-coach occupancy of the
    train at `platform` (10 coaches, A-J). Returns {"platform": int,
    "found": bool, "coaches": [int×10], "coach_labels": ["A".."J"]}.
    Example: <tool_call>{"name": "get_train_crowd_occupation", "arguments": {"platform": 3}}</tool_call>

  * get_ideal_zone(platform: int) — single best zone (A-J) for one
    passenger. Returns {"platform": int, "found": bool,
    "zone": "A".."J"|null, "zone_index": int, "reasoning": str}.
    Example: <tool_call>{"name": "get_ideal_zone", "arguments": {"platform": 3}}</tool_call>

  * get_current_time() — station-system time as HH:MM. No arguments.
    Returns {"time": "HH:MM"}.
    Example: <tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>
"""

_CROWD_ANNOUNCEMENT_TOOLS = """\
Available tools (call ONLY with the parameter names shown):

  * get_platform_crowd(platform: int) — per-zone platform crowd percentages
    (10 zones, A-J). Returns {"platform": int, "found": bool,
    "zones": [int×10], "zone_labels": ["A".."J"]}.
    Example: <tool_call>{"name": "get_platform_crowd", "arguments": {"platform": 3}}</tool_call>

  * get_train_crowd_occupation(platform: int) — per-coach occupancy of the
    train at `platform`. Returns {"platform": int, "found": bool,
    "coaches": [int×10], "coach_labels": ["A".."J"]}.
    Example: <tool_call>{"name": "get_train_crowd_occupation", "arguments": {"platform": 3}}</tool_call>

  * get_ideal_distribution(platform: int) — full 10-zone ideal
    redistribution. Returns {"platform": int, "found": bool,
    "distribution": [int×10], "zone_labels": ["A".."J"]}.
    Example: <tool_call>{"name": "get_ideal_distribution", "arguments": {"platform": 3}}</tool_call>
"""

SYSTEM_PROMPTS: Dict[str, str] = {
    "ticket_booking": (
        "You are a polite metro station ticket agent. Your job is to talk with "
        "passengers and help them buy tickets through the available MCP tools.\n\n"
        "Workflow rules:\n"
        "  1. Read the passenger's latest message carefully and extract any details they\n"
        "     already provided.\n"
        "  2. If the destination is already stated, do NOT ask for it again; validate it via\n"
        "     `validate_destination` and ask only for missing information such as passenger\n"
        "     count.\n"
        "  3. Look up the price with `get_ticket_cost` (use the normalized destination).\n"
        "  4. Only then call `initiate_payment`.\n"
        "  5. Poll `check_payment_status` (with the same payment_id) a few times until it\n"
        "     resolves to `success` or `failed`. Do not poll more than necessary.\n"
        "  6. Communicate the outcome politely. If payment failed, offer to retry.\n\n"
        "Stay polite throughout. Do NOT initiate payment before you know both the\n"
        "destination AND the passenger count.\n\n"
        + _TOOL_FORMAT_BLOCK
        + "\n"
        + _TICKET_BOOKING_TOOLS
    ),
    "ticket_issuance": (
        "You issue metro tickets. Use the MCP tools to gather the data you need,\n"
        "then return a single JSON object with exactly these keys:\n"
        '  {"time": "HH:MM", "from": "<station>", "to": "<station>",\n'
        '   "price": <number>, "platform": <int>, "ideal_zone": "<A-J>"}\n\n'
        "Use values returned by tools — do not invent prices or zones.\n\n"
        + _TOOL_FORMAT_BLOCK
        + "\n"
        + _TICKET_ISSUANCE_TOOLS
    ),
    "crowd_announcement": (
        "You are an automated crowd controller. For each train arrival, use the MCP\n"
        "tools below (with the announced platform) to gather crowd data, then produce\n"
        "a polite redirection announcement in this exact structured format:\n\n"
        '  Announcement: "<your announcement>"\n'
        "  Recommended Platform Distribution: [<10 ints>]\n"
        "  Platform Zone Color Codes: [<10 hex>]\n"
        "  Train Coach Color Codes: [<10 hex>]\n\n"
        "Color reference: #008000 (≤40), #FFFF00 (40–60), #FF8C00 (60–80), #FF0000 (>80).\n\n"
        + _TOOL_FORMAT_BLOCK
        + "\n"
        + _CROWD_ANNOUNCEMENT_TOOLS
    ),
}


# ---------------------------------------------------------------------------
# Tool-call parsing (ported from demoMCP.py)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool call payloads from a model's text. Malformed blocks are
    ignored (they will fail `format_reward`)."""
    calls: List[Dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(text or ""):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "name" not in payload:
            continue
        calls.append(
            {
                "name": payload["name"],
                "arguments": payload.get("arguments") or {},
            }
        )
    return calls


def has_tool_calls(text: str) -> bool:
    return bool(_TOOL_CALL_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Rollout types
# ---------------------------------------------------------------------------

GenerateFn = Callable[[List[Dict[str, str]]], Awaitable[str]]


@dataclass
class RolloutResult:
    task_name: str
    reward: float
    reward_breakdown: Dict[str, float]
    turn_history: List[dict]
    final_text: str
    truncated: bool
    error: Optional[str] = None
    per_step_rewards: List[float] = field(default_factory=list)
    per_step_breakdowns: List[Dict[str, float]] = field(default_factory=list)


def replay_completion_sync(
    env: Any,
    task_name: str,
    completion_text: str,
    *,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Replay a single static completion against a sync env.

    This is shared by local/docker and in-process training so the reward
    replay and debug traces stay identical across both paths.
    """
    obs = _unwrap(env.reset(task=task_name, seed=seed))
    initial_observation = {
        "prompt_text": getattr(obs, "prompt_text", "") or "",
        "passenger_message": getattr(obs, "passenger_message", "") or "",
        "scenario_summary": dict(getattr(obs, "scenario_summary", {}) or {}),
        "current_step": getattr(obs, "current_step", 0),
        "max_steps": getattr(obs, "max_steps", 1),
    }

    debug_context: Dict[str, Any] = {}
    if task_name == "crowd_announcement":
        debug_context["crowd_snapshot"] = _fetch_crowd_snapshot(
            env, initial_observation["scenario_summary"]
        )

    turn_history: List[dict] = []
    text = completion_text or ""
    idx = 0
    while True:
        chunk_end = text.find("</tool_call>", idx)
        if chunk_end < 0:
            break
        chunk = text[idx:chunk_end + len("</tool_call>")]
        calls = parse_tool_calls(chunk)
        turn_record = {"text": chunk, "tool_calls": []}
        for call in calls:
            result_obs = env.step(
                CallToolAction(
                    tool_name=call["name"], arguments=call["arguments"]
                )
            )
            result_obs = _unwrap(result_obs)
            data, error = _extract_call_result(result_obs)
            turn_record["tool_calls"].append(
                {
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "result": data,
                    "error": error,
                }
            )
        turn_history.append(turn_record)
        idx = chunk_end + len("</tool_call>")

    final_text = text[idx:].strip()
    turn_history.append({"text": final_text, "tool_calls": []})
    step_result = env.step(
        SubmitResponseAction(
            content=final_text,
            metadata={"turn_history": turn_history},
        )
    )
    obs = _unwrap(step_result)
    reward = _reward_value(step_result, obs)
    breakdown = dict(getattr(obs, "reward_breakdown", {}) or {})
    return {
        "reward": reward,
        "breakdown": breakdown,
        "turn_history": turn_history,
        "final_text": final_text,
        "initial_observation": initial_observation,
        "debug_context": debug_context,
    }


def format_training_debug_log(
    *,
    step: int,
    sample_idx: int,
    task_name: str,
    replay_result: Dict[str, Any],
) -> str:
    """Build a task-specific multiline debug trace for training stdout."""
    lines = [f"[debug][step={step}][sample={sample_idx}][task={task_name}]"]
    initial = replay_result.get("initial_observation", {}) or {}
    debug_context = replay_result.get("debug_context", {}) or {}
    turn_history = replay_result.get("turn_history", []) or []
    final_text = replay_result.get("final_text", "") or ""

    if task_name == "ticket_booking":
        passenger = _compact_text(initial.get("passenger_message", ""))
        if passenger:
            lines.append(f"passenger: {passenger}")
        for turn in turn_history:
            text = _compact_text(turn.get("text", ""))
            if text:
                lines.append(f"ai_assistant: {text}")

    elif task_name == "crowd_announcement":
        snapshot = debug_context.get("crowd_snapshot", {}) or {}
        platform = snapshot.get("platform")
        if platform is not None:
            lines.append(f"current_platform: {platform}")
        platform_crowd = snapshot.get("platform_crowd")
        if platform_crowd:
            lines.append(
                f"platform_crowd: {json.dumps(platform_crowd, sort_keys=True)}"
            )
        coach_crowd = snapshot.get("coach_crowd")
        if coach_crowd:
            lines.append(
                f"coach_crowd: {json.dumps(coach_crowd, sort_keys=True)}"
            )
        ideal_distribution = snapshot.get("ideal_distribution")
        if ideal_distribution:
            lines.append(
                "ideal_distribution: "
                f"{json.dumps(ideal_distribution, sort_keys=True)}"
            )
        announcement = _compact_text(final_text)
        if announcement:
            lines.append(f"ai_assistant_announcement: {announcement}")

    elif task_name == "ticket_issuance":
        parsed = _try_parse_json(final_text)
        if parsed is not None:
            lines.append("ticket_details_json:")
            lines.append(json.dumps(parsed, indent=2, sort_keys=True))
        else:
            raw = _compact_text(final_text)
            lines.append(f"ticket_details_raw: {raw}")

    lines.append(f"reward: {float(replay_result.get('reward', 0.0)):.3f}")
    breakdown = replay_result.get("breakdown", {}) or {}
    lines.append(f"breakdown: {json.dumps(breakdown, sort_keys=True)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async rollout (works against the WebSocket client)
# ---------------------------------------------------------------------------


async def agentic_episode_async(
    env: Any,
    task_name: str,
    generate: GenerateFn,
    *,
    seed: Optional[int] = None,
    max_turns: int = 10,
    max_steps: Optional[int] = None,
) -> RolloutResult:
    """Run one episode against the (async) client/env.

    Works for single-step tasks (ticket_booking, ticket_issuance) and the
    multi-step crowd_announcement task — when the env reports
    ``done=False`` after a SubmitResponseAction we restart the inner loop
    with a fresh prompt for the next train arrival.
    """
    sys_prompt = SYSTEM_PROMPTS[task_name]
    obs = await _maybe_async(env.reset, task=task_name, seed=seed)
    obs = _unwrap(obs)

    per_step_rewards: List[float] = []
    per_step_breakdowns: List[Dict[str, float]] = []
    final_text = ""
    truncated = False
    turn_history_all: List[dict] = []

    step_idx = 0
    while True:
        prompt = obs.prompt_text or _fallback_prompt(task_name)
        user_lines = [prompt]
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n".join(user_lines)},
        ]
        turn_history: List[dict] = []
        submitted = False

        for turn in range(max_turns):
            text = await generate(messages)
            tool_calls = parse_tool_calls(text)

            if tool_calls:
                turn_record = {"text": text, "tool_calls": []}
                messages.append({"role": "assistant", "content": text})
                tool_results_blocks: List[str] = []
                for tc in tool_calls:
                    result_obs = await _maybe_async(
                        env.step,
                        CallToolAction(
                            tool_name=tc["name"], arguments=tc["arguments"]
                        ),
                    )
                    result_obs = _unwrap(result_obs)
                    data, error = _extract_call_result(result_obs)
                    turn_record["tool_calls"].append(
                        {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                            "result": data,
                            "error": error,
                        }
                    )
                    tool_results_blocks.append(
                        f"<tool_result name=\"{tc['name']}\">{json.dumps(data)[:512]}</tool_result>"
                    )
                turn_history.append(turn_record)
                messages.append(
                    {"role": "user", "content": "\n".join(tool_results_blocks)}
                )
                continue

            # No tool calls: treat as final submission for this step.
            turn_history.append({"text": text, "tool_calls": []})
            submission = SubmitResponseAction(
                content=text,
                metadata={"turn_history": turn_history},
            )
            obs = await _maybe_async(env.step, submission)
            step_result = obs
            obs = _unwrap(obs)
            final_text = text
            per_step_rewards.append(_reward_value(step_result, obs))
            per_step_breakdowns.append(dict(obs.reward_breakdown or {}))
            turn_history_all.extend(turn_history)
            submitted = True
            break

        if not submitted:
            # Max turns hit — force-submit whatever the model said last.
            forced_text = turn_history[-1]["text"] if turn_history else ""
            obs = await _maybe_async(
                env.step,
                SubmitResponseAction(
                    content=forced_text,
                    metadata={"turn_history": turn_history},
                ),
            )
            step_result = obs
            obs = _unwrap(obs)
            final_text = forced_text
            per_step_rewards.append(_reward_value(step_result, obs))
            per_step_breakdowns.append(dict(obs.reward_breakdown or {}))
            turn_history_all.extend(turn_history)
            truncated = True

        step_idx += 1
        if obs.done:
            break
        if max_steps is not None and step_idx >= max_steps:
            break

    avg_reward = (
        sum(per_step_rewards) / len(per_step_rewards) if per_step_rewards else 0.0
    )
    aggregated_breakdown = _average_breakdowns(per_step_breakdowns)

    return RolloutResult(
        task_name=task_name,
        reward=avg_reward,
        reward_breakdown=aggregated_breakdown,
        turn_history=turn_history_all,
        final_text=final_text,
        truncated=truncated,
        per_step_rewards=per_step_rewards,
        per_step_breakdowns=per_step_breakdowns,
    )


# ---------------------------------------------------------------------------
# Sync wrapper (for in-process training without an event loop)
# ---------------------------------------------------------------------------


def agentic_episode_sync(
    env: Any,
    task_name: str,
    generate_sync: Callable[[List[Dict[str, str]]], str],
    *,
    seed: Optional[int] = None,
    max_turns: int = 10,
    max_steps: Optional[int] = None,
) -> RolloutResult:
    """Sync sibling of `agentic_episode_async`. Calls `env.step` and
    `env.reset` directly (no awaiting). Works with the in-process
    `MetrocrowdmanagerEnvironment`."""
    sys_prompt = SYSTEM_PROMPTS[task_name]
    obs = env.reset(task=task_name, seed=seed)

    per_step_rewards: List[float] = []
    per_step_breakdowns: List[Dict[str, float]] = []
    final_text = ""
    truncated = False
    turn_history_all: List[dict] = []

    step_idx = 0
    while True:
        prompt = obs.prompt_text or _fallback_prompt(task_name)
        user_lines = [prompt]
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n".join(user_lines)},
        ]
        turn_history: List[dict] = []
        submitted = False

        for turn in range(max_turns):
            text = generate_sync(messages)
            tool_calls = parse_tool_calls(text)

            if tool_calls:
                turn_record = {"text": text, "tool_calls": []}
                messages.append({"role": "assistant", "content": text})
                tool_results_blocks: List[str] = []
                for tc in tool_calls:
                    result_obs = env.step(
                        CallToolAction(
                            tool_name=tc["name"], arguments=tc["arguments"]
                        )
                    )
                    data, error = _extract_call_result(result_obs)
                    turn_record["tool_calls"].append(
                        {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                            "result": data,
                            "error": error,
                        }
                    )
                    tool_results_blocks.append(
                        f"<tool_result name=\"{tc['name']}\">{json.dumps(data)[:512]}</tool_result>"
                    )
                turn_history.append(turn_record)
                messages.append(
                    {"role": "user", "content": "\n".join(tool_results_blocks)}
                )
                continue

            turn_history.append({"text": text, "tool_calls": []})
            obs = env.step(
                SubmitResponseAction(
                    content=text,
                    metadata={"turn_history": turn_history},
                )
            )
            step_result = obs
            obs = _unwrap(obs)
            final_text = text
            per_step_rewards.append(_reward_value(step_result, obs))
            per_step_breakdowns.append(dict(obs.reward_breakdown or {}))
            turn_history_all.extend(turn_history)
            submitted = True
            break

        if not submitted:
            forced_text = turn_history[-1]["text"] if turn_history else ""
            obs = env.step(
                SubmitResponseAction(
                    content=forced_text,
                    metadata={"turn_history": turn_history},
                )
            )
            step_result = obs
            obs = _unwrap(obs)
            final_text = forced_text
            per_step_rewards.append(_reward_value(step_result, obs))
            per_step_breakdowns.append(dict(obs.reward_breakdown or {}))
            turn_history_all.extend(turn_history)
            truncated = True

        step_idx += 1
        if obs.done:
            break
        if max_steps is not None and step_idx >= max_steps:
            break

    avg_reward = (
        sum(per_step_rewards) / len(per_step_rewards) if per_step_rewards else 0.0
    )
    aggregated_breakdown = _average_breakdowns(per_step_breakdowns)

    return RolloutResult(
        task_name=task_name,
        reward=avg_reward,
        reward_breakdown=aggregated_breakdown,
        turn_history=turn_history_all,
        final_text=final_text,
        truncated=truncated,
        per_step_rewards=per_step_rewards,
        per_step_breakdowns=per_step_breakdowns,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _maybe_async(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    out = fn(*args, **kwargs)
    if hasattr(out, "__await__"):
        return await out
    return out


def _unwrap(maybe_step_result: Any) -> Any:
    """The async client returns StepResult(.observation, ...); the sync env
    returns observations directly. Unwrap so the loop sees a uniform shape."""
    obs = getattr(maybe_step_result, "observation", maybe_step_result)
    return obs


def _reward_value(step_result: Any, obs: Any) -> float:
    reward = getattr(obs, "reward", None)
    if reward is None:
        reward = getattr(step_result, "reward", None)
    return float(reward or 0.0)


def _extract_call_result(obs: Any) -> tuple[Any, Optional[str]]:
    """Pull the data dict (and any error string) out of a CallToolObservation."""
    if isinstance(obs, CallToolObservation):
        if obs.error is not None:
            err = getattr(obs.error, "message", str(obs.error))
            return None, err
        result = obs.result
        if result is None:
            return None, "no result"
        # FastMCP CallToolResult exposes .data and .structured_content
        data = getattr(result, "data", None)
        if data is None:
            data = getattr(result, "structured_content", None)
        if isinstance(data, dict) and "result" in data and len(data) == 1:
            data = data["result"]
        return data, None
    # remote case: obs.result may already be a dict
    raw = getattr(obs, "result", None)
    if isinstance(raw, dict):
        if "data" in raw:
            return raw["data"], raw.get("error")
        return raw, None
    return None, None


def _fetch_crowd_snapshot(env: Any, scenario_summary: Dict[str, Any]) -> Dict[str, Any]:
    platform = scenario_summary.get("current_platform")
    if platform is None:
        return {}

    platform_data, _ = _call_tool_sync(
        env, "get_platform_crowd", {"platform": platform}
    )
    coach_data, _ = _call_tool_sync(
        env, "get_train_crowd_occupation", {"platform": platform}
    )
    ideal_data, _ = _call_tool_sync(
        env, "get_ideal_distribution", {"platform": platform}
    )

    return {
        "platform": platform,
        "platform_crowd": _labelled_values(
            (platform_data or {}).get("zone_labels", []),
            (platform_data or {}).get("zones", []),
        ),
        "coach_crowd": _labelled_values(
            (coach_data or {}).get("coach_labels", []),
            (coach_data or {}).get("coaches", []),
        ),
        "ideal_distribution": _labelled_values(
            (ideal_data or {}).get("zone_labels", []),
            (ideal_data or {}).get("distribution", []),
        ),
    }


def _call_tool_sync(
    env: Any, tool_name: str, arguments: Dict[str, Any]
) -> tuple[Any, Optional[str]]:
    result_obs = env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
    result_obs = _unwrap(result_obs)
    return _extract_call_result(result_obs)


def _fallback_prompt(task_name: str) -> str:
    return f"Continue the {task_name} task using the available tools."


def _average_breakdowns(breakdowns: List[Dict[str, float]]) -> Dict[str, float]:
    if not breakdowns:
        return {}
    keys = breakdowns[0].keys()
    return {
        k: sum(b.get(k, 0.0) for b in breakdowns) / len(breakdowns) for k in keys
    }


def _compact_text(text: str) -> str:
    return " ".join((text or "").split())


def _labelled_values(labels: List[str], values: List[Any]) -> Dict[str, Any]:
    if not labels or not values:
        return {}
    return {str(label): value for label, value in zip(labels, values)}


def _try_parse_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
