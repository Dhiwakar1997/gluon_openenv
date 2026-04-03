# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MetroCrowdManager Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MetrocrowdmanagerAction, MetrocrowdmanagerObservation


class MetrocrowdmanagerEnv(
    EnvClient[MetrocrowdmanagerAction, MetrocrowdmanagerObservation, State]
):
    """
    Client for the MetroCrowdManager Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Example:
        >>> async with MetrocrowdmanagerEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(task="crowd_assessment")
        ...     print(result.observation.prompt_text)
        ...
        ...     result = await client.step(
        ...         MetrocrowdmanagerAction(response_text="Platform Zone Color Codes: [...]")
        ...     )
        ...     print(result.reward)
    """

    def _step_payload(self, action: MetrocrowdmanagerAction) -> Dict[str, Any]:
        """Convert action to JSON payload for step message."""
        return {
            "response_text": action.response_text,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MetrocrowdmanagerObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = MetrocrowdmanagerObservation(
            num_coaches=obs_data.get("num_coaches", 6),
            train_crowd=obs_data.get("train_crowd", []),
            platform_crowd=obs_data.get("platform_crowd", []),
            prompt_text=obs_data.get("prompt_text", ""),
            current_step=obs_data.get("current_step", 0),
            max_steps=obs_data.get("max_steps", 1),
            station_name=obs_data.get("station_name", ""),
            task_name=obs_data.get("task_name", "crowd_assessment"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name"),
            max_steps=payload.get("max_steps"),
            train_crowd=payload.get("train_crowd"),
            platform_crowd=payload.get("platform_crowd"),
            station_name=payload.get("station_name"),
            total_reward=payload.get("total_reward"),
            step_rewards=payload.get("step_rewards"),
            done=payload.get("done"),
        )
