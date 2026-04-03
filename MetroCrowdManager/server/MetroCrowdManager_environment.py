# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MetroCrowdManager Environment Implementation.

A stateful metro station crowd management environment where an AI agent
produces redirection announcements evaluated across 10 reward dimensions.
Supports 3 tasks with progressive difficulty:
  - crowd_assessment (easy):  single-step color code mapping
  - redirection (medium):     single-step full announcement
  - multi_train (hard):       8-step evolving crowd management
"""

import random
import re
from typing import Any, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MetrocrowdmanagerAction, MetrocrowdmanagerObservation
except ImportError:
    from models import MetrocrowdmanagerAction, MetrocrowdmanagerObservation

try:
    from .rewards import (
        compute_clarity,
        compute_color_grading,
        compute_conservation_accuracy,
        compute_distribution_accuracy,
        compute_factual_accuracy,
        compute_feasibility_accuracy,
        compute_language_consistency,
        compute_noop_detection,
        compute_platform_mention,
        compute_politeness,
        compute_sequential_direction,
    )
except ImportError:
    from rewards import (
        compute_clarity,
        compute_color_grading,
        compute_conservation_accuracy,
        compute_distribution_accuracy,
        compute_factual_accuracy,
        compute_feasibility_accuracy,
        compute_language_consistency,
        compute_noop_detection,
        compute_platform_mention,
        compute_politeness,
        compute_sequential_direction,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATION_NAMES = [
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
]

NUM_COACHES = 10


def _coach_labels(num_coaches: int) -> List[str]:
    """Generate coach labels A, B, C, ... dynamically."""
    return [chr(ord("A") + i) for i in range(num_coaches)]

TASK_CONFIG = {
    "crowd_assessment": {"max_steps": 1},
    "redirection": {"max_steps": 1},
    "multi_train": {"max_steps": 8},
}


# ---------------------------------------------------------------------------
# Crowd generation helpers
# ---------------------------------------------------------------------------

def generate_train_crowd(
    num_coaches: int = NUM_COACHES,
    pattern_override: Optional[str] = None,
    bias_hard: bool = False,
) -> List[int]:
    """Generate train coach occupancy percentages using variety patterns."""
    if pattern_override:
        pattern = pattern_override
    elif bias_hard:
        # Hard task: 70% chance of tough patterns
        pattern = random.choices(
            ["uniform_low", "uniform_high", "front_heavy", "rear_heavy",
             "one_outlier_empty", "one_outlier_packed", "random"],
            weights=[5, 25, 20, 20, 5, 5, 20],
            k=1,
        )[0]
    else:
        pattern = random.choices(
            ["uniform_low", "uniform_high", "front_heavy", "rear_heavy",
             "one_outlier_empty", "one_outlier_packed", "random"],
            weights=[15, 15, 15, 15, 10, 10, 20],
            k=1,
        )[0]

    half = num_coaches // 2

    if pattern == "uniform_low":
        return [random.randint(20, 45) for _ in range(num_coaches)]
    elif pattern == "uniform_high":
        return [random.randint(70, 95) for _ in range(num_coaches)]
    elif pattern == "front_heavy":
        return [random.randint(70, 95) for _ in range(half)] + [
            random.randint(20, 45) for _ in range(num_coaches - half)
        ]
    elif pattern == "rear_heavy":
        return [random.randint(20, 45) for _ in range(half)] + [
            random.randint(70, 95) for _ in range(num_coaches - half)
        ]
    elif pattern == "one_outlier_empty":
        crowd = [random.randint(55, 85) for _ in range(num_coaches)]
        crowd[random.randint(0, num_coaches - 1)] = random.randint(15, 30)
        return crowd
    elif pattern == "one_outlier_packed":
        crowd = [random.randint(30, 55) for _ in range(num_coaches)]
        crowd[random.randint(0, num_coaches - 1)] = random.randint(85, 95)
        return crowd
    else:  # "random"
        return [random.randint(20, 95) for _ in range(num_coaches)]


def _add_new_passengers(
    platform_crowd: List[float],
    num_coaches: int,
    bias_hard: bool = False,
) -> List[float]:
    """Add new passengers arriving at the platform between trains."""
    if bias_hard:
        growth = random.choices(
            ["rush", "offpeak", "event", "normal"],
            weights=[50, 10, 20, 20],
            k=1,
        )[0]
    else:
        growth = random.choices(
            ["rush", "offpeak", "event", "normal"],
            weights=[20, 20, 10, 50],
            k=1,
        )[0]

    event_zones = set()
    if growth == "event":
        event_zones = set(random.sample(range(num_coaches), min(2, num_coaches)))

    result = list(platform_crowd)
    for i in range(num_coaches):
        if growth == "rush":
            new = random.randint(10, 60)
        elif growth == "offpeak":
            new = random.randint(2, 10)
        elif growth == "event" and i in event_zones:
            new = random.randint(40, 60)
        elif growth == "event":
            new = random.randint(5, 15)
        else:  # normal
            new = random.randint(5, 30)

        # Entrance bias: zones A & B get slightly more arrivals
        if i < 2:
            new = int(new * random.uniform(1.1, 1.5))

        result[i] = min(100.0, result[i] + new)

    return result


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MetrocrowdmanagerEnvironment(Environment):
    """
    Metro station crowd management RL environment.

    The agent receives train coach occupancy and platform zone crowd
    percentages, then produces structured redirection responses evaluated
    across 10 reward dimensions.

    Tasks:
        crowd_assessment (easy):  Map crowd percentages to hex color codes.
        redirection (medium):     Full announcement with redistribution + colors.
        multi_train (hard):       8-step evolving crowd management.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name = "crowd_assessment"
        self._max_steps = 1
        self._num_coaches = NUM_COACHES
        self._train_crowd: List[int] = []
        self._platform_crowd: List[float] = []
        self._station_name = ""
        self._platform_number: int = 1
        self._total_reward = 0.0
        self._step_rewards: List[float] = []
        self._crowd_history: List[dict] = []

    # ---- reset -----------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MetrocrowdmanagerObservation:
        """Reset the environment and start a new episode.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional custom episode identifier.
            **kwargs: task (str) — one of "crowd_assessment", "redirection",
                      "multi_train". Defaults to "crowd_assessment".
        """
        task = kwargs.get("task", "crowd_assessment")
        if task not in TASK_CONFIG:
            task = "crowd_assessment"
        self._task_name = task
        self._num_coaches = kwargs.get("num_coaches", NUM_COACHES)
        nc = self._num_coaches

        if seed is not None:
            random.seed(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._max_steps = TASK_CONFIG[task]["max_steps"]
        self._station_name = random.choice(STATION_NAMES)
        self._platform_number = random.randint(1, 8)
        self._total_reward = 0.0
        self._step_rewards = []

        # --- Initialise platform crowd ---
        if task == "crowd_assessment":
            self._platform_crowd = [float(random.randint(20, 50)) for _ in range(nc)]
        elif task == "redirection":
            if random.random() < 0.15:
                # Balanced scenario — low variance
                base = random.randint(40, 60)
                self._platform_crowd = [
                    float(max(0, min(100, base + random.randint(-8, 8))))
                    for _ in range(nc)
                ]
            else:
                self._platform_crowd = [float(random.randint(15, 85)) for _ in range(nc)]
        else:  # multi_train — start empty, add initial wave
            self._platform_crowd = [0.0] * nc
            for i in range(nc):
                base = float(random.randint(5, 25))
                if i < 2:
                    base = min(100.0, base * random.uniform(1.1, 1.5))
                self._platform_crowd[i] = base

        # --- Generate first train ---
        if task == "crowd_assessment":
            self._train_crowd = generate_train_crowd(num_coaches=nc, pattern_override="random")
        elif task == "multi_train":
            self._train_crowd = generate_train_crowd(num_coaches=nc, bias_hard=True)
        else:
            self._train_crowd = generate_train_crowd(num_coaches=nc)

        self._crowd_history = [
            {
                "step": 0,
                "train_crowd": list(self._train_crowd),
                "platform_crowd": self._rounded_platform(),
            }
        ]

        return MetrocrowdmanagerObservation(
            platform_number=self._platform_number,
            num_coaches=nc,
            train_crowd=list(self._train_crowd),
            platform_crowd=self._rounded_platform(),
            prompt_text=self._build_prompt(),
            current_step=1,
            max_steps=self._max_steps,
            station_name=self._station_name,
            task_name=self._task_name,
            done=False,
            reward=0.0,
        )

    # ---- step ------------------------------------------------------------

    def step(
        self,
        action: MetrocrowdmanagerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MetrocrowdmanagerObservation:
        """Process the agent's response and advance the environment state."""
        self._state.step_count += 1
        response_text = action.response_text
        platform_int = self._rounded_platform()
        nc = self._num_coaches

        # --- Compute all 11 rewards ---
        rewards = {
            "politeness": compute_politeness(response_text, self._train_crowd, platform_int, nc),
            "distribution_accuracy": compute_distribution_accuracy(response_text, self._train_crowd, platform_int, nc),
            "conservation_accuracy": compute_conservation_accuracy(response_text, self._train_crowd, platform_int, nc),
            "feasibility_accuracy": compute_feasibility_accuracy(response_text, self._train_crowd, platform_int, nc),
            "color_grading": compute_color_grading(response_text, self._train_crowd, platform_int, nc),
            "language_consistency": compute_language_consistency(response_text, self._train_crowd, platform_int, nc),
            "noop_detection": compute_noop_detection(response_text, self._train_crowd, platform_int, nc),
            "clarity": compute_clarity(response_text, self._train_crowd, platform_int, nc),
            "sequential_direction": compute_sequential_direction(response_text, self._train_crowd, platform_int, nc),
            "factual_accuracy": compute_factual_accuracy(response_text, self._train_crowd, platform_int, nc),
            "platform_mention": compute_platform_mention(response_text, self._platform_number),
        }

        # --- Task-specific weighting ---
        if self._task_name == "crowd_assessment":
            total_reward = (
                0.50 * rewards["color_grading"]
                + 0.25 * rewards["language_consistency"]
                + 0.25 * rewards["clarity"]
            )
        else:
            total_reward = (
                0.30 * rewards["distribution_accuracy"]
                + 0.10 * rewards["conservation_accuracy"]
                + 0.10 * rewards["feasibility_accuracy"]
                + 0.10 * rewards["color_grading"]
                + 0.05 * rewards["politeness"]
                + 0.10 * rewards["factual_accuracy"]
                + 0.05 * rewards["noop_detection"]
                + 0.05 * rewards["clarity"]
                + 0.05 * rewards["sequential_direction"]
                + 0.05 * rewards["language_consistency"]
                + 0.05 * rewards["platform_mention"]
            )

        self._total_reward += total_reward
        self._step_rewards.append(total_reward)

        # --- Done? ---
        done = self._state.step_count >= self._max_steps

        # --- Evolve crowd for multi_train (if not final step) ---
        if self._task_name == "multi_train" and not done:
            self._evolve_crowd(response_text, total_reward)

        self._crowd_history.append(
            {
                "step": self._state.step_count,
                "train_crowd": list(self._train_crowd),
                "platform_crowd": self._rounded_platform(),
                "rewards": rewards,
                "total_reward": total_reward,
            }
        )

        return MetrocrowdmanagerObservation(
            platform_number=self._platform_number,
            num_coaches=nc,
            train_crowd=list(self._train_crowd),
            platform_crowd=self._rounded_platform(),
            prompt_text=self._build_prompt() if not done else "",
            current_step=self._state.step_count + (0 if done else 1),
            max_steps=self._max_steps,
            station_name=self._station_name,
            task_name=self._task_name,
            done=done,
            reward=total_reward,
            metadata={
                "rewards": rewards,
                "total_reward": total_reward,
                "step": self._state.step_count,
            },
        )

    # ---- state -----------------------------------------------------------

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_name=self._task_name,
            max_steps=self._max_steps,
            train_crowd=list(self._train_crowd),
            platform_crowd=self._rounded_platform(),
            station_name=self._station_name,
            platform_number=self._platform_number,
            total_reward=self._total_reward,
            step_rewards=list(self._step_rewards),
            crowd_history=self._crowd_history,
            done=self._state.step_count >= self._max_steps,
        )

    # ---- internal --------------------------------------------------------

    def _rounded_platform(self) -> List[int]:
        return [max(0, min(100, round(p))) for p in self._platform_crowd]

    def _evolve_crowd(self, response_text: str, step_reward: float) -> None:
        """Evolve the platform crowd state for multi_train task."""
        nc = self._num_coaches

        # 1. Simulate redirection effect based on agent's recommendation
        proposed = self._parse_proposed_distribution(response_text)
        if proposed is not None and len(proposed) == nc:
            # Better responses → higher compliance (0.3 to 0.7)
            compliance = 0.3 + 0.4 * step_reward
            for i in range(nc):
                self._platform_crowd[i] = (
                    (1 - compliance) * self._platform_crowd[i]
                    + compliance * proposed[i]
                )

        # 2. Simulate train departure — passengers board
        for i in range(nc):
            boarding_rate = (100 - self._train_crowd[i]) / 100.0 * 0.5
            self._platform_crowd[i] = max(0.0, self._platform_crowd[i] * (1 - boarding_rate))

        # 3. Generate next train
        self._train_crowd = generate_train_crowd(num_coaches=nc, bias_hard=True)

        # 4. Add new passengers arriving at the platform
        self._platform_crowd = _add_new_passengers(
            self._platform_crowd, nc, bias_hard=True,
        )

        # 5. With 15% chance, force a balanced state (no-op test)
        if random.random() < 0.15:
            avg = sum(self._platform_crowd) / nc
            self._platform_crowd = [
                max(0.0, min(100.0, avg + random.uniform(-5, 5)))
                for _ in range(nc)
            ]

    @staticmethod
    def _parse_proposed_distribution(response_text: str) -> Optional[List[float]]:
        pattern = r"Recommended Platform Distribution\s*:\s*\[([^\]]+)\]"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if not match:
            return None
        try:
            return [float(x.strip().rstrip("%")) for x in match.group(1).split(",")]
        except (ValueError, IndexError):
            return None

    def _build_prompt(self) -> str:
        """Build the prompt text for the current step and task."""
        tc = self._train_crowd
        pc = self._rounded_platform()
        nc = self._num_coaches
        labels = _coach_labels(nc)
        last = labels[-1]

        coach_str = ", ".join(f"Coach {labels[i]}: {tc[i]}%" for i in range(nc))
        zone_str = ", ".join(f"Zone {labels[i]}: {pc[i]}%" for i in range(nc))

        if self._task_name == "multi_train":
            header = (
                f"Upcoming train arriving at {self._station_name} station, Platform {self._platform_number}. "
                f"[Step {self._state.step_count + 1}/{self._max_steps}]"
            )
        else:
            header = f"Train arriving at {self._station_name} station, Platform {self._platform_number}."

        lines = [
            header,
            f"Coach occupancy: {coach_str}",
            f"Platform crowd at each coach zone: {zone_str}",
            "",
        ]

        if self._task_name == "crowd_assessment":
            lines += [
                "Provide the color codes for each platform zone and train coach based on their crowd levels.",
                "",
                "Respond in the following structured format:",
                "",
                f"Platform Zone Color Codes: [<hex color for Zone A>, <hex color for Zone B>, ..., <hex color for Zone {last}>]",
                "",
                f"Train Coach Color Codes: [<hex color for Coach A>, <hex color for Coach B>, ..., <hex color for Coach {last}>]",
                "",
                "Color code reference: #008000 (Green, <=40%), #FFFF00 (Yellow, 40-60%), #FF8C00 (Orange, 60-80%), #FF0000 (Red, >80%)",
            ]
        else:
            lines += [
                "Your announcement must begin by addressing passengers on the correct platform number.",
                "",
                "Respond in the following structured format:",
                "",
                'Announcement: "<your crowd redirection announcement>"',
                "",
                f"Recommended Platform Distribution: [<target % for Zone A>, <target % for Zone B>, ..., <target % for Zone {last}>]",
                "",
                f"Platform Zone Color Codes: [<hex color for Zone A>, <hex color for Zone B>, ..., <hex color for Zone {last}>]",
                "",
                f"Train Coach Color Codes: [<hex color for Coach A>, <hex color for Coach B>, ..., <hex color for Coach {last}>]",
                "",
                "Color code reference: #008000 (Green, <=40%), #FFFF00 (Yellow, 40-60%), #FF8C00 (Orange, 60-80%), #FF0000 (Red, >80%)",
            ]

        return "\n".join(lines)
