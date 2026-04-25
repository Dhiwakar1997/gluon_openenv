"""
Data models for the MetroCrowdManager MCP environment.

MetroCrowdManager subclasses `MCPEnvironment`. Three action types flow
through `step()`:

* `ListToolsAction` — ask the server which tools are available. Routed
  automatically by the MCP base class. Not defined here.
* `CallToolAction` — invoke one of the registered MCP tools with
  arguments. Also routed automatically. Not defined here.
* `SubmitResponseAction` — the agent's final answer for the episode
  (Task 1, 2) or the current arrival step (Task 3). This is the only
  action handled by our `_step_impl()`, and rewards are computed here.

Observations have two shapes:

* `CallToolObservation` (from MCP base class) — returned for every tool
  call. Carries the tool's return value in `.result.data`.
* `MetrocrowdmanagerObservation` — returned from `reset()` and from
  `_step_impl()` after a `SubmitResponseAction`. Carries the prompt for
  the next turn, reward breakdown, and episode metadata.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import ConfigDict, Field


class SubmitResponseAction(Action):
    """Submit the agent's final text response for the current episode step.

    After a `SubmitResponseAction`:
      * In `ticket_booking` and `ticket_issuance` the episode ends.
      * In `crowd_announcement` the step advances to the next train
        arrival until all arrivals are consumed.

    `extra="allow"` so payloads survive the HTTP roundtrip when the
    serializer falls back to validating against the env's registered
    action class — MCP fields (``tool_name``, ``arguments``, ``tools``)
    are tolerated and unused.
    """

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    type: str = Field(
        default="submit_response",
        description="Action type discriminator",
    )
    content: str = Field(
        default="",
        description="Final text the agent wants the environment to evaluate",
    )


# Kept as a re-export so existing imports (`from models import
# MetrocrowdmanagerAction`) don't break during migration. `SubmitResponseAction`
# is the canonical name.
MetrocrowdmanagerAction = SubmitResponseAction


class MetrocrowdmanagerObservation(Observation):
    """Observation emitted on reset and after a SubmitResponseAction.

    Tool-call observations use MCP's `CallToolObservation` and do not
    flow through this type.
    """

    task_name: str = Field(
        default="ticket_booking",
        description="Active task: ticket_booking | ticket_issuance | crowd_announcement",
    )
    prompt_text: str = Field(
        default="",
        description="Human-readable prompt for the agent's next turn",
    )
    current_step: int = Field(default=0, description="1-indexed step number")
    max_steps: int = Field(default=1, description="Total steps in this episode")
    passenger_message: str = Field(
        default="",
        description="Scripted passenger's latest utterance (Task 1 only, '' otherwise)",
    )
    scenario_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Public episode metadata (source station, valid destinations, "
            "current platform, arrival index). Excludes ground-truth values "
            "the agent should discover via tools."
        ),
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-reward-function scores for the most recent SubmitResponseAction. "
            "Empty on reset."
        ),
    )
