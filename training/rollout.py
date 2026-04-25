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
"""

SYSTEM_PROMPTS: Dict[str, str] = {
    "ticket_booking": (
        "You are a polite metro station ticket agent. Your job is to talk with "
        "passengers and help them buy tickets through the available MCP tools.\n\n"
        "Workflow rules:\n"
        "  1. Ask the passenger what their destination is and how many people are travelling.\n"
        "  2. Validate the destination via `validate_destination` before quoting any fare.\n"
        "  3. Look up the price with `get_ticket_cost` (use the normalized destination).\n"
        "  4. Only then call `initiate_payment`.\n"
        "  5. Poll `check_payment_status` (with the same payment_id) a few times until it\n"
        "     resolves to `success` or `failed`. Do not poll more than necessary.\n"
        "  6. Communicate the outcome politely. If payment failed, offer to retry.\n\n"
        "Stay polite throughout. Do NOT initiate payment before you know both the\n"
        "destination AND the passenger count.\n\n" + _TOOL_FORMAT_BLOCK
    ),
    "ticket_issuance": (
        "You issue metro tickets. Use the MCP tools to gather the data you need:\n"
        "  * `get_platform_for_destination` — which platform serves the destination\n"
        "  * `get_platform_crowd` and `get_train_crowd_occupation` — current crowd state\n"
        "  * `get_ideal_zone` — best zone on that platform for a single passenger\n"
        "  * `get_current_time` — for the ticket timestamp\n\n"
        "Your FINAL message must be a single JSON object with exactly these keys:\n"
        '  {"time": "HH:MM", "from": "<station>", "to": "<station>",\n'
        '   "price": <number>, "platform": <int>, "ideal_zone": "<A-J>"}\n\n'
        "Use values returned by tools — do not invent prices or zones.\n\n"
        + _TOOL_FORMAT_BLOCK
    ),
    "crowd_announcement": (
        "You are an automated crowd controller. For each train arrival, use the MCP\n"
        "tools `get_platform_crowd`, `get_train_crowd_occupation`, and\n"
        "`get_ideal_distribution` (with the announced platform), then produce a\n"
        "polite redirection announcement in this exact structured format:\n\n"
        '  Announcement: "<your announcement>"\n'
        "  Recommended Platform Distribution: [<10 ints>]\n"
        "  Platform Zone Color Codes: [<10 hex>]\n"
        "  Train Coach Color Codes: [<10 hex>]\n\n"
        "Color reference: #008000 (≤40), #FFFF00 (40–60), #FF8C00 (60–80), #FF0000 (>80).\n"
        + _TOOL_FORMAT_BLOCK
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
        passenger = getattr(obs, "passenger_message", "") or ""
        user_lines = [prompt]
        if passenger:
            user_lines.append(f"Passenger: \"{passenger}\"")
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
            obs = _unwrap(obs)
            final_text = text
            per_step_rewards.append(float(obs.reward or 0.0))
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
            obs = _unwrap(obs)
            final_text = forced_text
            per_step_rewards.append(float(obs.reward or 0.0))
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
        passenger = getattr(obs, "passenger_message", "") or ""
        user_lines = [prompt]
        if passenger:
            user_lines.append(f"Passenger: \"{passenger}\"")
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
            final_text = text
            per_step_rewards.append(float(obs.reward or 0.0))
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
            final_text = forced_text
            per_step_rewards.append(float(obs.reward or 0.0))
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


def _fallback_prompt(task_name: str) -> str:
    return f"Continue the {task_name} task using the available tools."


def _average_breakdowns(breakdowns: List[Dict[str, float]]) -> Dict[str, float]:
    if not breakdowns:
        return {}
    keys = breakdowns[0].keys()
    return {
        k: sum(b.get(k, 0.0) for b in breakdowns) / len(breakdowns) for k in keys
    }
