"""MetroCrowdManager async client for the agentic MCP environment."""

from typing import Any, Dict, Union

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
    ListToolsObservation,
    Tool,
    ToolError,
)
from openenv.core.env_server.types import Action, Observation, State

try:
    from .models import MetrocrowdmanagerObservation, SubmitResponseAction
except ImportError:
    from models import MetrocrowdmanagerObservation, SubmitResponseAction


ClientAction = Union[
    SubmitResponseAction,
    CallToolAction,
    ListToolsAction,
]
ClientObservation = Union[
    MetrocrowdmanagerObservation,
    CallToolObservation,
    ListToolsObservation,
]


class MetrocrowdmanagerEnv(
    EnvClient[ClientAction, ClientObservation, State]
):
    """Client for the MetroCrowdManager MCP environment.

    Supports three action shapes:

      * ``SubmitResponseAction(content=...)`` — submit final answer for the
        current step. Episode terminates for ``ticket_booking`` and
        ``ticket_issuance``; advances to next train arrival for
        ``crowd_announcement``.
      * ``CallToolAction(tool_name=..., arguments=...)`` — invoke an MCP
        tool. Returns a ``CallToolObservation``.
      * ``ListToolsAction()`` — discover available tools.

    Example::

        async with MetrocrowdmanagerEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset(task="ticket_booking")
            tools = await env.step(ListToolsAction())
            r = await env.step(CallToolAction(
                tool_name="validate_destination",
                arguments={"destination": "Tech Park"},
            ))
            done = await env.step(SubmitResponseAction(
                content='{"time": "10:15", "from": "Riverside", ...}',
                metadata={"turn_history": [...]},
            ))
    """

    def _step_payload(self, action: ClientAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ClientObservation]:
        obs_data = payload.get("observation") or payload
        obs_type = obs_data.get("type") or _infer_observation_type(obs_data)
        observation: Observation
        if obs_type == "list_tools":
            tools = [
                t if isinstance(t, Tool) else Tool(**t)
                for t in obs_data.get("tools", [])
            ]
            observation = ListToolsObservation(
                tools=tools,
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
        elif obs_type == "call_tool":
            err_data = obs_data.get("error")
            error = (
                ToolError(**err_data)
                if isinstance(err_data, dict)
                else err_data
            )
            observation = CallToolObservation(
                tool_name=obs_data.get("tool_name", ""),
                result=obs_data.get("result"),
                error=error,
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
        else:
            observation = MetrocrowdmanagerObservation(
                task_name=obs_data.get("task_name", "ticket_booking"),
                prompt_text=obs_data.get("prompt_text", ""),
                current_step=obs_data.get("current_step", 0),
                max_steps=obs_data.get("max_steps", 1),
                passenger_message=obs_data.get("passenger_message", ""),
                scenario_summary=obs_data.get("scenario_summary", {}),
                reward_breakdown=obs_data.get("reward_breakdown", {}),
                done=obs_data.get("done", False),
                reward=obs_data.get("reward"),
                metadata=obs_data.get("metadata", {}),
            )
        return StepResult(
            observation=observation,
            reward=payload.get("reward") if "reward" in payload else observation.reward,
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(**{k: v for k, v in payload.items() if k != "type"})


def _infer_observation_type(obs_data: Dict[str, Any]) -> str:
    if "tools" in obs_data:
        return "list_tools"
    if "tool_name" in obs_data:
        return "call_tool"
    return "submission"
