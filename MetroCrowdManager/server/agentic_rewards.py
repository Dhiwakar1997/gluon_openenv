"""
Agentic reward functions for the MCP refactor.

These complement the 11 existing text/crowd rewards in `rewards.py` and
score the orchestration dimension: did the agent call the right tools in
the right order, did it use real tool outputs, did it avoid spamming, and
did it respect task-specific discipline rules?

All functions return a float in [0.0, 1.0]. They are deterministic and
cheap — no LLM calls, no network.

A `turn_history` entry has the shape:

    {
        "text": "<model's raw output>",
        "tool_calls": [                       # may be empty
            {
                "name": "<tool_name>",
                "arguments": { ... },
                "result": { ... } | None,     # parsed JSON of the tool's return
                "error": str | None,
            },
            ...
        ],
    }

The rollout loop in `training/rollout.py` is the source of truth for this
shape. The environment's `_step_impl` passes its own copy in when it
computes rewards at submit time.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

try:
    from .scenarios import Scenario, compute_ticket_cost
except ImportError:  # pragma: no cover
    from scenarios import Scenario, compute_ticket_cost


# ---------------------------------------------------------------------------
# Expected tool sequences per task
# ---------------------------------------------------------------------------

# Ordered "checkpoints" — each inner list is a group of alternatives, and
# each group must appear at least once, in order. This allows natural
# flexibility (e.g. the agent may call `list_valid_stations` before
# `validate_destination`) while still checking the critical backbone.
EXPECTED_SEQUENCE: Dict[str, List[List[str]]] = {
    "ticket_booking": [
        ["validate_destination"],
        ["get_ticket_cost"],
        ["initiate_payment"],
        ["check_payment_status"],
    ],
    "ticket_issuance": [
        ["get_platform_for_destination"],
        ["get_platform_crowd", "get_train_crowd_occupation"],
        ["get_platform_crowd", "get_train_crowd_occupation"],
        ["get_ideal_zone"],
        ["get_current_time"],
    ],
    "crowd_announcement": [
        ["get_platform_crowd", "get_train_crowd_occupation"],
        ["get_platform_crowd", "get_train_crowd_occupation"],
        ["get_ideal_distribution"],
    ],
}

EXPECTED_MIN_CALLS: Dict[str, int] = {
    "ticket_booking": 5,
    "ticket_issuance": 5,
    "crowd_announcement": 3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_tool_calls(turn_history: List[dict]) -> List[dict]:
    calls = []
    for turn in turn_history:
        calls.extend(turn.get("tool_calls", []) or [])
    return calls


def count_valid_tool_calls(turn_history: List[dict]) -> int:
    """Number of parsed tool calls with a non-empty `name`. Used by the
    env to decide whether the agent did *any* real tool use this episode."""
    return sum(1 for c in _all_tool_calls(turn_history) if c.get("name"))


def _final_text(turn_history: List[dict], fallback: str = "") -> str:
    for turn in reversed(turn_history):
        text = turn.get("text") or ""
        if text.strip():
            return text
    return fallback


_REDUNDANT_DESTINATION_PATTERNS = (
    r"\bwhat(?:'s| is)? your destination\b",
    r"\bwhere (?:are you going|would you like to go)\b",
    r"\bwhich station\b",
    r"\btell me where\b",
    r"\b(?:tell me|confirm) (?:your )?destination\b",
)


def _asked_for_destination_again(turn_history: List[dict]) -> bool:
    if not turn_history:
        return False
    first_text = (turn_history[0].get("text") or "").lower()
    if not first_text.strip():
        return False
    return any(re.search(pattern, first_text) for pattern in _REDUNDANT_DESTINATION_PATTERNS)


def has_malformed_tool_call(turn_history: List[dict]) -> bool:
    """Detect raw tool-call markup that never formed a valid parsed call."""
    for turn in turn_history:
        text = turn.get("text", "") or ""
        if "<tool_call" not in text:
            continue
        matches = _TOOL_CALL_RE.findall(text)
        if "<tool_call>" in text and len(matches) == 0:
            return True
        for block in matches:
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                return True
            if not isinstance(payload, dict) or "name" not in payload:
                return True
    return False


# ---------------------------------------------------------------------------
# 1. tool_sequence_reward
# ---------------------------------------------------------------------------


def tool_sequence_reward(turn_history: List[dict], task_name: str) -> float:
    """Fraction of the expected ordered checkpoints the agent hit in order.

    Each checkpoint is a group of acceptable tool names (alternatives).
    We scan the agent's call log left-to-right, consuming checkpoints as
    their first accepted tool is encountered.
    """
    expected = EXPECTED_SEQUENCE.get(task_name, [])
    if not expected:
        return 1.0
    calls = _all_tool_calls(turn_history)
    if not calls:
        return 0.0

    idx = 0
    for call in calls:
        if idx >= len(expected):
            break
        if call.get("name") in expected[idx]:
            idx += 1
    return idx / len(expected)


# ---------------------------------------------------------------------------
# 2. tool_fidelity_reward
# ---------------------------------------------------------------------------


def tool_fidelity_reward(turn_history: List[dict], task_name: Optional[str] = None) -> float:
    """Reward using real tool outputs in downstream tool call arguments.

    Specifically checks for each "upstream → downstream" link below:

      * validate_destination.normalized → get_ticket_cost.destination
      * validate_destination.normalized → get_platform_for_destination.destination
      * get_ticket_cost.cost → initiate_payment.amount
      * initiate_payment.payment_id → check_payment_status.payment_id
      * get_platform_for_destination.platform → get_platform_crowd.platform
      * get_platform_for_destination.platform → get_train_crowd_occupation.platform
      * get_platform_for_destination.platform → get_ideal_zone.platform
      * get_platform_for_destination.platform → get_ideal_distribution.platform

    Returns the fraction of applicable links the agent got right.
    A link is "applicable" when both upstream and downstream tools were
    called.
    """
    calls = _all_tool_calls(turn_history)

    links = [
        ("validate_destination", "normalized", "get_ticket_cost", "destination"),
        ("validate_destination", "normalized", "get_platform_for_destination", "destination"),
        ("get_ticket_cost", "cost", "initiate_payment", "amount"),
        ("initiate_payment", "payment_id", "check_payment_status", "payment_id"),
        ("get_platform_for_destination", "platform", "get_platform_crowd", "platform"),
        ("get_platform_for_destination", "platform", "get_train_crowd_occupation", "platform"),
        ("get_platform_for_destination", "platform", "get_ideal_zone", "platform"),
        ("get_platform_for_destination", "platform", "get_ideal_distribution", "platform"),
    ]

    applicable = 0
    satisfied = 0
    for up_tool, up_field, down_tool, down_arg in links:
        upstream_values = [
            _result_field(c, up_field)
            for c in calls
            if c.get("name") == up_tool
        ]
        upstream_values = [v for v in upstream_values if v is not None]
        downstream_args = [
            c.get("arguments", {}).get(down_arg)
            for c in calls
            if c.get("name") == down_tool
        ]
        downstream_args = [v for v in downstream_args if v is not None]
        if not upstream_values or not downstream_args:
            continue
        applicable += 1
        if any(_fuzzy_equal(u, d) for u in upstream_values for d in downstream_args):
            satisfied += 1

    if applicable == 0:
        if has_malformed_tool_call(turn_history):
            return 0.0
        # No applicable upstream→downstream link means the agent didn't
        # invoke a chained pair. For tasks where the EXPECTED sequence
        # demands tool use, returning 0.5 hands free reward to a model
        # that submitted nothing — collapse to 0.0 instead.
        if task_name and EXPECTED_SEQUENCE.get(task_name):
            calls = _all_tool_calls(turn_history)
            if not calls:
                return 0.0
        return 0.5
    return satisfied / applicable


# ---------------------------------------------------------------------------
# 3. tool_economy_reward
# ---------------------------------------------------------------------------


def tool_economy_reward(turn_history: List[dict], task_name: str) -> float:
    """Penalise runaway tool calls.

    Reward = min(1, expected / actual), clipped to [0, 1]. Below the
    expected-minimum count we return 1.0 (the agent hasn't padded).

    A special cap is applied to `check_payment_status`: at most 8 polls
    allowed before economy degrades. Spamming that tool is the obvious
    hack against this reward.
    """
    calls = _all_tool_calls(turn_history)
    if not calls:
        if has_malformed_tool_call(turn_history):
            return 0.0
        # If the task has an expected minimum tool count, zero calls is a
        # failure mode, not "polite economy". Returning 1.0 here let
        # ticket_issuance models earn 0.10 for emitting an empty answer.
        if EXPECTED_MIN_CALLS.get(task_name, 0) > 0:
            return 0.0
        return 1.0
    actual = len(calls)
    expected = EXPECTED_MIN_CALLS.get(task_name, actual)

    poll_count = sum(1 for c in calls if c.get("name") == "check_payment_status")
    poll_penalty = 0.0
    if poll_count > 8:
        poll_penalty = min(0.5, (poll_count - 8) * 0.05)

    if actual <= expected:
        return max(0.0, 1.0 - poll_penalty)
    return max(0.0, (expected / actual) - poll_penalty)


# ---------------------------------------------------------------------------
# 4. format_reward
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def format_reward(turn_history: List[dict]) -> float:
    """Fraction of agent turns that were well-formed.

    A turn is well-formed if:
      * It contains only well-parsed `<tool_call>...</tool_call>` blocks
        (JSON with `name` and `arguments`), OR
      * It is a final submission (non-empty text with NO `<tool_call>` block).

    Malformed tool-call tags (open but not closed, non-JSON body) fail.
    """
    if not turn_history:
        return 0.0
    good = 0
    for turn in turn_history:
        text = turn.get("text", "") or ""
        declared = turn.get("tool_calls") or []
        matches = _TOOL_CALL_RE.findall(text)
        # Count matches between declared and parseable
        if declared:
            all_valid = True
            for block in matches:
                try:
                    payload = json.loads(block)
                    if not isinstance(payload, dict) or "name" not in payload:
                        all_valid = False
                        break
                except json.JSONDecodeError:
                    all_valid = False
                    break
            # Also: if the text has an unclosed <tool_call> without a match,
            # penalize.
            if "<tool_call>" in text and len(matches) == 0:
                all_valid = False
            if all_valid:
                good += 1
        else:
            # Final submission turn. Should have no stray <tool_call> tags.
            if "<tool_call>" not in text and text.strip():
                good += 1
    return good / len(turn_history)


# ---------------------------------------------------------------------------
# 5. info_sufficiency_reward (Task 1 only)
# ---------------------------------------------------------------------------


def info_sufficiency_reward(turn_history: List[dict], scenario: Scenario) -> float:
    """Reward agent for collecting destination and passenger count BEFORE
    initiating payment. Splits into three checks worth 1/3 each:

      * validate_destination was called and returned `valid=True` before
        any call to `initiate_payment`.
      * get_ticket_cost was called with `passenger_count > 0` before
        `initiate_payment`.
      * `initiate_payment` happened (and only after the above).

    If initiate_payment is never called, returns the pre-payment score so
    the agent still gets credit for asking the right questions.
    """
    calls = _all_tool_calls(turn_history)
    had_valid_dest = False
    had_cost_lookup = False
    had_payment = False
    premature_payment = False

    for call in calls:
        name = call.get("name")
        if name == "validate_destination":
            res = call.get("result") or {}
            if res.get("valid") is True:
                had_valid_dest = True
        elif name == "get_ticket_cost":
            args = call.get("arguments", {}) or {}
            pc = args.get("passenger_count")
            try:
                if pc is not None and int(pc) > 0:
                    had_cost_lookup = True
            except (TypeError, ValueError):
                pass
        elif name == "initiate_payment":
            if not (had_valid_dest and had_cost_lookup):
                premature_payment = True
            had_payment = True

    score = 0.0
    if had_valid_dest:
        score += 1.0 / 3.0
    if had_cost_lookup:
        score += 1.0 / 3.0
    if had_payment and not premature_payment:
        score += 1.0 / 3.0
    if premature_payment:
        score = max(0.0, score - 0.5)
    if scenario.passenger_goal is not None and _asked_for_destination_again(turn_history):
        score = max(0.0, score - 0.25)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 6. payment_discipline_reward (Task 1 only)
# ---------------------------------------------------------------------------


_RETRY_MARKERS = (
    "retry",
    "try again",
    "reattempt",
    "another attempt",
    "alternate",
    "alternative payment",
    "different method",
)

_FAILURE_COMM_MARKERS = (
    "payment failed",
    "payment did not",
    "could not process",
    "unable to process",
    "payment was unsuccessful",
)

_SUCCESS_COMM_MARKERS = (
    "payment successful",
    "payment was successful",
    "booking confirmed",
    "ticket has been booked",
    "successfully paid",
    "payment complete",
)


def payment_discipline_reward(turn_history: List[dict], scenario: Scenario) -> float:
    """Composite: (a) final outcome communicated clearly, (b) polling cadence
    reasonable (not 0 polls, not > 8).
    """
    calls = _all_tool_calls(turn_history)
    poll_calls = [c for c in calls if c.get("name") == "check_payment_status"]
    payment_calls = [c for c in calls if c.get("name") == "initiate_payment"]

    if not payment_calls:
        return 0.0

    final_status: Optional[str] = None
    for call in reversed(poll_calls):
        result = call.get("result") or {}
        status = result.get("status")
        if status in {"success", "failed"}:
            final_status = status
            break

    final_text = _final_text(turn_history).lower()
    comm_score = 0.0
    if final_status == "success":
        comm_score = 1.0 if any(m in final_text for m in _SUCCESS_COMM_MARKERS) else 0.3
    elif final_status == "failed":
        has_fail = any(m in final_text for m in _FAILURE_COMM_MARKERS)
        has_retry = any(m in final_text for m in _RETRY_MARKERS)
        comm_score = (0.6 if has_fail else 0.0) + (0.4 if has_retry else 0.0)
    else:
        comm_score = 0.2  # outcome never reached

    polls = len(poll_calls)
    if polls == 0:
        cadence_score = 0.0
    elif polls <= 6:
        cadence_score = 1.0
    elif polls <= 10:
        cadence_score = 0.6
    else:
        cadence_score = max(0.0, 1.0 - (polls - 10) * 0.1)

    return 0.5 * comm_score + 0.5 * cadence_score


# ---------------------------------------------------------------------------
# 7. task_success_reward (Task 1 only)
# ---------------------------------------------------------------------------


_COUNT_QUESTION_PATTERNS = (
    r"\bhow many\b",
    r"\bhow many (?:people|passengers|travellers|travelers|tickets)\b",
    r"\bnumber of passengers\b",
    r"\bpassenger count\b",
    r"\bgroup size\b",
)

_FARE_MARKERS = (
    "fare",
    "cost",
    "price",
    "total",
    "inr",
    "rs.",
    "rs ",
    "rupees",
    "ticket will be",
)

_BOOKING_MARKERS = (
    "ticket",
    "booking",
    "booked",
    "payment",
    "confirmed",
    "confirmation",
)


def task_success_reward(turn_history: List[dict], scenario: Scenario) -> float:
    """End-to-end success score for a ticket-booking trajectory."""
    goal = scenario.passenger_goal
    if goal is None:
        return 0.0

    calls = _all_tool_calls(turn_history)
    expected_cost = compute_ticket_cost(
        scenario,
        scenario.source_station,
        goal.destination,
        goal.passenger_count,
    )

    valid_destination = any(
        call.get("name") == "validate_destination"
        and (call.get("result") or {}).get("valid") is True
        and _fuzzy_equal((call.get("result") or {}).get("normalized"), goal.destination)
        for call in calls
    )

    correct_cost_lookup = any(
        call.get("name") == "get_ticket_cost"
        and _fuzzy_equal((call.get("arguments") or {}).get("source"), scenario.source_station)
        and _fuzzy_equal((call.get("arguments") or {}).get("destination"), goal.destination)
        and _num((call.get("arguments") or {}).get("passenger_count"), default=-1)
        == goal.passenger_count
        and abs(_num((call.get("result") or {}).get("cost"), default=-1) - expected_cost) < 0.5
        for call in calls
    )

    payment_call = next(
        (
            call
            for call in calls
            if call.get("name") == "initiate_payment"
            and abs(_num((call.get("arguments") or {}).get("amount"), default=-1) - expected_cost)
            < 0.5
            and _num((call.get("arguments") or {}).get("passenger_count"), default=-1)
            == goal.passenger_count
        ),
        None,
    )

    payment_id = None
    if payment_call is not None:
        payment_id = (payment_call.get("result") or {}).get("payment_id")

    terminal_status: Optional[str] = None
    if payment_id is not None:
        for call in reversed(calls):
            if call.get("name") != "check_payment_status":
                continue
            if not _fuzzy_equal((call.get("arguments") or {}).get("payment_id"), payment_id):
                continue
            status = (call.get("result") or {}).get("status")
            if status in {"success", "failed"}:
                terminal_status = status
                break

    final_text = _final_text(turn_history).lower()
    mentions_destination = goal.destination.lower() in final_text
    mentions_booking = any(marker in final_text for marker in _BOOKING_MARKERS)
    mentions_amount = (
        "inr" in final_text
        or "rupees" in final_text
        or str(int(round(expected_cost))) in final_text
    )

    final_outcome = 0.0
    if terminal_status == "success":
        if any(marker in final_text for marker in _SUCCESS_COMM_MARKERS):
            final_outcome = 0.6
            if mentions_booking:
                final_outcome += 0.2
            if mentions_destination or mentions_amount:
                final_outcome += 0.2
    elif terminal_status == "failed":
        has_failure = any(marker in final_text for marker in _FAILURE_COMM_MARKERS)
        has_retry = any(marker in final_text for marker in _RETRY_MARKERS)
        final_outcome = (0.5 if has_failure else 0.0) + (0.5 if has_retry else 0.0)

    score = 0.0
    if valid_destination:
        score += 0.2
    if correct_cost_lookup:
        score += 0.2
    if payment_call is not None:
        score += 0.2
    if terminal_status in {"success", "failed"}:
        score += 0.2
    score += 0.2 * final_outcome
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 8. conversation_quality_reward (Task 1 only)
# ---------------------------------------------------------------------------


def conversation_quality_reward(turn_history: List[dict], scenario: Scenario) -> float:
    """Score whether the dialogue was useful, relevant, and orderly."""
    if has_malformed_tool_call(turn_history) and not _all_tool_calls(turn_history):
        return 0.0
    texts = [
        _strip_tool_blocks(turn.get("text", ""))
        for turn in turn_history
        if (turn.get("text") or "").strip()
    ]
    if not texts:
        return 0.0

    lower_texts = [text.lower() for text in texts]
    calls = _all_tool_calls(turn_history)

    count_question_turns = [
        idx
        for idx, text in enumerate(lower_texts)
        if any(re.search(pattern, text) for pattern in _COUNT_QUESTION_PATTERNS)
    ]
    redundant_destination = any(
        any(re.search(pattern, text) for pattern in _REDUNDANT_DESTINATION_PATTERNS)
        for text in lower_texts
    )
    fare_explained = any(
        any(marker in text for marker in _FARE_MARKERS) for text in lower_texts
    )
    final_text = _final_text(turn_history).lower()
    final_useful = any(marker in final_text for marker in _BOOKING_MARKERS)

    had_valid_destination = any(
        call.get("name") == "validate_destination"
        and (call.get("result") or {}).get("valid") is True
        for call in calls
    )
    payment_idx = next(
        (idx for idx, call in enumerate(calls) if call.get("name") == "initiate_payment"),
        None,
    )
    count_before_payment = False
    if payment_idx is None:
        count_before_payment = bool(count_question_turns)
    else:
        seen_tool_calls = 0
        for turn in turn_history:
            text = _strip_tool_blocks(turn.get("text", "")).lower()
            if any(re.search(pattern, text) for pattern in _COUNT_QUESTION_PATTERNS):
                count_before_payment = True
                break
            seen_tool_calls += len(turn.get("tool_calls", []) or [])
            if seen_tool_calls > payment_idx:
                break

    normalized_texts = [_normalize_turn_text(text) for text in texts]
    repeated_turns = len(normalized_texts) - len({text for text in normalized_texts if text})

    score = 0.0
    if had_valid_destination and not redundant_destination:
        score += 0.25
    if count_before_payment:
        score += 0.25
    if fare_explained:
        score += 0.20
    if repeated_turns == 0 and len(count_question_turns) <= 1 and not redundant_destination:
        score += 0.15
    if final_useful:
        score += 0.15

    if payment_idx is not None and not count_before_payment:
        score = max(0.0, score - 0.40)
    if len(texts) > 8:
        score = max(0.0, score - min(0.30, (len(texts) - 8) * 0.10))

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 9. turn_efficiency_reward (Task 1 only)
# ---------------------------------------------------------------------------


def turn_efficiency_reward(turn_history: List[dict], max_turns: int = 8) -> float:
    """Reward concise ticket-booking episodes and penalize looping."""
    if has_malformed_tool_call(turn_history) and not _all_tool_calls(turn_history):
        return 0.0
    assistant_turns = sum(
        1 for turn in turn_history if (turn.get("text") or "").strip() or turn.get("tool_calls")
    )
    if assistant_turns <= 0:
        return 0.0

    if assistant_turns <= max_turns - 2:
        score = 1.0
    elif assistant_turns <= max_turns:
        score = 0.8
    elif assistant_turns <= max_turns + 2:
        score = 0.5
    elif assistant_turns <= max_turns + 4:
        score = 0.2
    else:
        score = 0.0

    normalized = [
        _normalize_turn_text(_strip_tool_blocks(turn.get("text", "")))
        for turn in turn_history
        if (turn.get("text") or "").strip()
    ]
    repeated = len(normalized) - len({text for text in normalized if text})
    score -= min(0.4, repeated * 0.15)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 10. ticket_schema_validity (Task 2 only)
# ---------------------------------------------------------------------------


_REQUIRED_TICKET_FIELDS = ("time", "from", "to", "price", "platform", "ideal_zone")


def ticket_schema_validity(final_text: str, scenario: Scenario) -> float:
    """1.0 if the submission contains a valid JSON ticket with all fields
    AND values match the per-episode ground truth. 0.5 if schema valid but
    values wrong. 0.0 if the JSON can't be parsed.
    """
    parsed = _extract_json(final_text)
    if parsed is None or not isinstance(parsed, dict):
        return 0.0
    missing = [f for f in _REQUIRED_TICKET_FIELDS if f not in parsed]
    if missing:
        return 0.0

    goal = scenario.passenger_goal
    if goal is None:
        return 0.5

    score = 0.0
    # time
    if str(parsed.get("time")).strip() == scenario.current_time:
        score += 1.0 / 6.0
    # from / to
    if _norm(parsed.get("from")) == _norm(scenario.source_station):
        score += 1.0 / 6.0
    dest = goal.destination
    if _norm(parsed.get("to")) == _norm(dest):
        score += 1.0 / 6.0
    # platform
    expected_platform = scenario.platform_map.get(dest)
    if expected_platform is not None and _num(parsed.get("platform")) == expected_platform:
        score += 1.0 / 6.0
    # price
    expected_cost = compute_ticket_cost(
        scenario, scenario.source_station, dest, goal.passenger_count
    )
    if expected_cost > 0 and abs(_num(parsed.get("price"), default=-1) - expected_cost) < 0.5:
        score += 1.0 / 6.0
    # ideal_zone — accept any single-letter zone A-J as structurally valid
    zone = str(parsed.get("ideal_zone", "")).strip().upper()
    if len(zone) == 1 and "A" <= zone <= "J":
        score += 1.0 / 6.0

    if score == 0.0:
        return 0.5  # schema valid, no matching values
    return max(0.5, min(1.0, score))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _result_field(call: dict, field: str) -> Any:
    result = call.get("result")
    if result is None:
        return None
    if isinstance(result, dict):
        return result.get(field)
    return None


def _fuzzy_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-3
    try:
        sa, sb = str(a).strip().casefold(), str(b).strip().casefold()
    except Exception:
        return False
    return sa == sb


def _strip_tool_blocks(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text or "").strip()


def _normalize_turn_text(text: str) -> str:
    text = _strip_tool_blocks(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # fallback: first balanced-looking JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
