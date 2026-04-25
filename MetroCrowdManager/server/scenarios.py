"""
Per-episode scenario generators for the MetroCrowdManager MCP environment.

A Scenario is the immutable (per-episode) ground truth that MCP tools query:
the set of valid destination stations, the platform-to-station map, crowd
state for each platform, the passenger's goal (Task 1), payment ticks, and
the current wall-clock time.

Tools never compute or mutate the scenario — they only read from it (or
mutate small per-episode state like `payment_state`). All random draws are
made inside the scenario factory so that a seeded reset is reproducible.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


STATION_POOL = [
    "Central Station",
    "Riverside",
    "University",
    "Market Square",
    "Tech Park",
    "Lakeside",
    "Airport Terminal",
    "Old Town",
    "Harbor View",
    "Greenfield",
    "North Bridge",
    "South Gate",
    "Hillcrest",
    "Eastwood",
    "Westbrook",
    "Banyan Grove",
]

NUM_COACHES = 10
ZONE_LABELS = [chr(ord("A") + i) for i in range(NUM_COACHES)]

BASE_FARE = 20.0
PER_HOP_FARE = 8.0


@dataclass
class PaymentState:
    payment_id: Optional[str] = None
    amount: float = 0.0
    passenger_count: int = 0
    ticks_remaining: int = 0
    will_fail: bool = False
    status: str = "none"


@dataclass
class PassengerGoal:
    source: str
    destination: str
    passenger_count: int


@dataclass
class Scenario:
    task_name: str
    station_list: List[str]
    source_station: str
    platform_map: Dict[str, int]
    platform_crowd: Dict[int, List[int]]
    train_crowd: Dict[int, List[int]]
    current_time: str
    passenger_goal: Optional[PassengerGoal] = None
    payment_state: PaymentState = field(default_factory=PaymentState)
    train_arrivals: List[dict] = field(default_factory=list)
    current_arrival_idx: int = 0

    @property
    def valid_destinations(self) -> List[str]:
        return [s for s in self.station_list if s != self.source_station]

    def platform_for(self, destination: str) -> Optional[int]:
        return self.platform_map.get(destination)

    def current_platform(self) -> int:
        """For tasks that operate on the 'current' train's platform."""
        if self.task_name == "crowd_announcement" and self.train_arrivals:
            return self.train_arrivals[self.current_arrival_idx]["platform"]
        if self.passenger_goal:
            return self.platform_map[self.passenger_goal.destination]
        return next(iter(self.platform_crowd))


def _pick_stations(rng: random.Random, k: int = 8) -> List[str]:
    return rng.sample(STATION_POOL, k)


def _generate_crowd(
    rng: random.Random,
    pattern: str,
    num_coaches: int = NUM_COACHES,
) -> List[int]:
    half = num_coaches // 2
    if pattern == "uniform_low":
        return [rng.randint(20, 45) for _ in range(num_coaches)]
    if pattern == "uniform_high":
        return [rng.randint(70, 95) for _ in range(num_coaches)]
    if pattern == "front_heavy":
        return [rng.randint(70, 95) for _ in range(half)] + [
            rng.randint(20, 45) for _ in range(num_coaches - half)
        ]
    if pattern == "rear_heavy":
        return [rng.randint(20, 45) for _ in range(half)] + [
            rng.randint(70, 95) for _ in range(num_coaches - half)
        ]
    if pattern == "one_outlier_empty":
        crowd = [rng.randint(55, 85) for _ in range(num_coaches)]
        crowd[rng.randint(0, num_coaches - 1)] = rng.randint(15, 30)
        return crowd
    if pattern == "one_outlier_packed":
        crowd = [rng.randint(30, 55) for _ in range(num_coaches)]
        crowd[rng.randint(0, num_coaches - 1)] = rng.randint(85, 95)
        return crowd
    return [rng.randint(20, 95) for _ in range(num_coaches)]


def _pick_pattern(rng: random.Random, bias_hard: bool = False) -> str:
    patterns = [
        "uniform_low",
        "uniform_high",
        "front_heavy",
        "rear_heavy",
        "one_outlier_empty",
        "one_outlier_packed",
        "random",
    ]
    if bias_hard:
        weights = [5, 25, 20, 20, 5, 5, 20]
    else:
        weights = [15, 15, 15, 15, 10, 10, 20]
    return rng.choices(patterns, weights=weights, k=1)[0]


def _random_time(rng: random.Random) -> str:
    hour = rng.randint(6, 23)
    minute = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    return f"{hour:02d}:{minute:02d}"


def build_scenario(task_name: str, seed: Optional[int] = None) -> Scenario:
    """Build a fresh per-episode scenario for the given task."""
    rng = random.Random(seed)
    stations = _pick_stations(rng, k=8)
    source = stations[0]
    platform_map = {st: i + 1 for i, st in enumerate(stations)}
    current_time = _random_time(rng)

    platform_crowd: Dict[int, List[int]] = {}
    train_crowd: Dict[int, List[int]] = {}
    for st, platform in platform_map.items():
        platform_crowd[platform] = _generate_crowd(rng, _pick_pattern(rng))
        train_crowd[platform] = _generate_crowd(rng, _pick_pattern(rng))

    passenger_goal = None
    if task_name == "ticket_booking":
        dest = rng.choice([s for s in stations if s != source])
        passenger_count = rng.choices([1, 2, 3, 4], weights=[55, 25, 12, 8], k=1)[0]
        passenger_goal = PassengerGoal(
            source=source, destination=dest, passenger_count=passenger_count
        )

    if task_name == "ticket_booking":
        payment_state = PaymentState(
            ticks_remaining=rng.randint(2, 5),
            will_fail=rng.random() < 0.12,
        )
    else:
        payment_state = PaymentState()

    train_arrivals: List[dict] = []
    if task_name == "crowd_announcement":
        num_arrivals = rng.randint(3, 4)
        for i in range(num_arrivals):
            platform = rng.choice(list(platform_map.values()))
            train_arrivals.append(
                {
                    "platform": platform,
                    "train_crowd": _generate_crowd(rng, _pick_pattern(rng, bias_hard=True)),
                    "platform_crowd": _generate_crowd(rng, _pick_pattern(rng, bias_hard=True)),
                }
            )

    return Scenario(
        task_name=task_name,
        station_list=stations,
        source_station=source,
        platform_map=platform_map,
        platform_crowd=platform_crowd,
        train_crowd=train_crowd,
        current_time=current_time,
        passenger_goal=passenger_goal,
        payment_state=payment_state,
        train_arrivals=train_arrivals,
        current_arrival_idx=0,
    )


def compute_ticket_cost(
    scenario: Scenario, source: str, destination: str, passenger_count: int
) -> float:
    """Simulated ticket cost.

    Cost scales with station-index distance in the per-episode map.
    Lives here (not in tools.py) so reward functions can recompute the
    ground-truth price without going through the MCP client.
    """
    if source not in scenario.platform_map or destination not in scenario.platform_map:
        return 0.0
    hops = abs(scenario.platform_map[source] - scenario.platform_map[destination])
    base = BASE_FARE + hops * PER_HOP_FARE
    return round(base * max(1, passenger_count), 2)


def new_payment_id() -> str:
    return f"PAY-{uuid.uuid4().hex[:8].upper()}"
