"""
MCP tool implementations for the MetroCrowdManager environment.

Tools are defined here as pure functions that take the episode's `Scenario`
(and optional args) and return a dict. The environment class binds thin
wrappers around these onto its `FastMCP` server so that the tools can be
listed and invoked via `ListToolsAction` / `CallToolAction`.

All tools return simulated data. None of them make network calls. Tool
results are small (≤ ~256 chars when stringified) so they don't blow up
the agent's context during multi-turn rollouts.

Catalog (12 tools):

Shared (Tasks 2, 3):
  1. get_platform_for_destination(destination)
  2. get_platform_crowd(platform)
  3. get_train_crowd_occupation(platform)
  4. get_current_time()

Task 1 (ticket_booking):
  5. validate_destination(destination)
  6. get_ticket_cost(source, destination, passenger_count)
  7. initiate_payment(amount, passenger_count)
  8. check_payment_status(payment_id)
  11. list_valid_stations()

Task 2 (ticket_issuance):
  9. get_ideal_zone(platform)

Task 3 (crowd_announcement):
  10. get_ideal_distribution(platform)
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional

try:
    from .rewards import _compute_ideal
    from .scenarios import (
        Scenario,
        ZONE_LABELS,
        compute_ticket_cost,
        new_payment_id,
    )
except ImportError:  # pragma: no cover — support direct module-style imports
    from rewards import _compute_ideal
    from scenarios import (
        Scenario,
        ZONE_LABELS,
        compute_ticket_cost,
        new_payment_id,
    )


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------


def get_platform_for_destination(scenario: Scenario, destination: str) -> Dict:
    """Return the platform number that serves trains to `destination`.

    Matches case-insensitively and with a small edit-distance tolerance so
    the agent isn't punished for trivial casing differences.
    """
    match = _best_match(destination, scenario.station_list)
    if match is None:
        return {"platform": None, "found": False, "destination": destination}
    return {
        "platform": scenario.platform_map[match],
        "found": True,
        "destination": match,
    }


def get_platform_crowd(scenario: Scenario, platform: int) -> Dict:
    """Return per-zone platform crowd percentages (10 zones, A-J)."""
    if platform not in scenario.platform_crowd:
        return {"platform": platform, "found": False, "zones": []}
    zones = scenario.platform_crowd[platform]
    return {
        "platform": platform,
        "found": True,
        "zones": list(zones),
        "zone_labels": list(ZONE_LABELS),
    }


def get_train_crowd_occupation(scenario: Scenario, platform: int) -> Dict:
    """Return per-coach occupancy percentages for the train at `platform`."""
    if platform not in scenario.train_crowd:
        return {"platform": platform, "found": False, "coaches": []}
    coaches = scenario.train_crowd[platform]
    return {
        "platform": platform,
        "found": True,
        "coaches": list(coaches),
        "coach_labels": list(ZONE_LABELS),
    }


def get_current_time(scenario: Scenario) -> Dict:
    return {"time": scenario.current_time}


# ---------------------------------------------------------------------------
# Task 1 — ticket_booking
# ---------------------------------------------------------------------------


def validate_destination(scenario: Scenario, destination: str) -> Dict:
    """Check whether `destination` is a known station. Returns the canonical
    form under `normalized` when matched."""
    match = _best_match(destination, scenario.station_list)
    if match is None or match == scenario.source_station:
        return {"valid": False, "normalized": None, "destination": destination}
    return {"valid": True, "normalized": match, "destination": destination}


def get_ticket_cost(
    scenario: Scenario,
    source: str,
    destination: str,
    passenger_count: int,
) -> Dict:
    """Simulated fare lookup. Cost scales with hops and passenger count."""
    src = _best_match(source, scenario.station_list) or source
    dst = _best_match(destination, scenario.station_list) or destination
    if src not in scenario.platform_map or dst not in scenario.platform_map:
        return {"cost": None, "currency": "INR", "found": False}
    cost = compute_ticket_cost(scenario, src, dst, int(passenger_count))
    return {
        "cost": cost,
        "currency": "INR",
        "found": True,
        "source": src,
        "destination": dst,
        "passenger_count": int(passenger_count),
    }


def initiate_payment(
    scenario: Scenario, amount: float, passenger_count: int
) -> Dict:
    """Kick off a payment. Returns a payment_id the agent must poll with
    `check_payment_status`. The payment will complete after 2-5 polls, with
    a ~12% probability of failure (pre-decided at scenario creation).
    """
    state = scenario.payment_state
    state.payment_id = new_payment_id()
    state.amount = float(amount)
    state.passenger_count = int(passenger_count)
    state.status = "pending"
    return {
        "payment_id": state.payment_id,
        "status": "pending",
        "amount": state.amount,
        "passenger_count": state.passenger_count,
    }


def check_payment_status(scenario: Scenario, payment_id: str) -> Dict:
    """Poll the payment's status. Each call decrements the internal tick
    counter; once it hits zero the status resolves to `success` or `failed`.
    """
    state = scenario.payment_state
    if state.payment_id is None or payment_id != state.payment_id:
        return {
            "payment_id": payment_id,
            "status": "unknown",
            "error": "No payment with that ID",
        }
    if state.status in {"success", "failed"}:
        return {"payment_id": state.payment_id, "status": state.status}
    state.ticks_remaining -= 1
    if state.ticks_remaining <= 0:
        state.status = "failed" if state.will_fail else "success"
    return {"payment_id": state.payment_id, "status": state.status}


def list_valid_stations(scenario: Scenario) -> Dict:
    """Return the full list of stations served on the current network."""
    return {
        "stations": list(scenario.station_list),
        "source_station": scenario.source_station,
    }


# ---------------------------------------------------------------------------
# Task 2 — ticket_issuance
# ---------------------------------------------------------------------------


def get_ideal_zone(scenario: Scenario, platform: int) -> Dict:
    """Recommend a single ideal platform zone for a single passenger boarding
    the train at `platform`. Picks the zone that the `_compute_ideal` algorithm
    would load most heavily (i.e. the one with the most free coach capacity)."""
    if platform not in scenario.platform_crowd or platform not in scenario.train_crowd:
        return {"platform": platform, "found": False, "zone": None}
    train = scenario.train_crowd[platform]
    plat = scenario.platform_crowd[platform]
    ideal = _compute_ideal(train, plat, len(train))
    # The "best" zone for a single passenger is the one that benefits most
    # from additional load — i.e. the highest value in the ideal distribution.
    best_idx = max(range(len(ideal)), key=lambda i: ideal[i])
    return {
        "platform": platform,
        "found": True,
        "zone": ZONE_LABELS[best_idx],
        "zone_index": best_idx,
        "reasoning": (
            f"Zone {ZONE_LABELS[best_idx]} has the most remaining train capacity "
            f"({100 - train[best_idx]}%) and should absorb single passengers."
        ),
    }


# ---------------------------------------------------------------------------
# Task 3 — crowd_announcement
# ---------------------------------------------------------------------------


def get_ideal_distribution(scenario: Scenario, platform: int) -> Dict:
    """Recommend the full 10-zone ideal distribution for `platform`."""
    if platform not in scenario.platform_crowd or platform not in scenario.train_crowd:
        return {"platform": platform, "found": False, "distribution": []}
    train = scenario.train_crowd[platform]
    plat = scenario.platform_crowd[platform]
    ideal = _compute_ideal(train, plat, len(train))
    return {
        "platform": platform,
        "found": True,
        "distribution": list(ideal),
        "zone_labels": list(ZONE_LABELS),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_match(query: str, options: List[str]) -> Optional[str]:
    """Case-insensitive fuzzy match against a list of station names."""
    if not query:
        return None
    q = query.strip().casefold()
    for opt in options:
        if opt.casefold() == q:
            return opt
    close = difflib.get_close_matches(q, [o.casefold() for o in options], n=1, cutoff=0.85)
    if not close:
        return None
    for opt in options:
        if opt.casefold() == close[0]:
            return opt
    return None


TOOL_NAMES = [
    "get_platform_for_destination",
    "get_platform_crowd",
    "get_train_crowd_occupation",
    "get_current_time",
    "validate_destination",
    "get_ticket_cost",
    "initiate_payment",
    "check_payment_status",
    "list_valid_stations",
    "get_ideal_zone",
    "get_ideal_distribution",
]
