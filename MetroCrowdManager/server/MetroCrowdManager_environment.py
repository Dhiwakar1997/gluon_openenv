"""
MetroCrowdManager — agentic MCP environment.

Subclasses `MCPEnvironment`. Three tasks, each driven by tool calls:

    ticket_booking      — converse with a (scripted) passenger, validate
                          destination, quote a fare, run a payment loop.
    ticket_issuance     — fetch platform + crowd info, compute an ideal
                          zone, emit a structured JSON ticket.
    crowd_announcement  — across 3-4 train arrivals, fetch crowd info and
                          emit a redirection announcement per arrival.

Tool calls flow through the inherited MCP routing. The agent's final
text submission flows through `_step_impl` as a `SubmitResponseAction`,
which is where rewards are computed.

This module is intentionally light on per-turn logging — the rollout
loop in `training/rollout.py` is the source of truth for `turn_history`,
which it passes to `_step_impl` via `SubmitResponseAction.metadata`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv.core.env_server.types import Action, Observation, State

try:
    from ..models import MetrocrowdmanagerObservation, SubmitResponseAction
except ImportError:
    from models import MetrocrowdmanagerObservation, SubmitResponseAction

try:
    from . import tools as tools_mod
    from .agentic_rewards import (
        conversation_quality_reward,
        count_valid_tool_calls,
        format_reward,
        has_malformed_tool_call,
        payment_discipline_reward,
        task_success_reward,
        ticket_schema_validity,
        tool_economy_reward,
        tool_fidelity_reward,
        tool_sequence_reward,
        turn_efficiency_reward,
    )
    from .passenger_sim import PassengerSim
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
    from .scenarios import Scenario, build_scenario
except ImportError:  # pragma: no cover
    import tools as tools_mod
    from agentic_rewards import (
        conversation_quality_reward,
        count_valid_tool_calls,
        format_reward,
        has_malformed_tool_call,
        payment_discipline_reward,
        task_success_reward,
        ticket_schema_validity,
        tool_economy_reward,
        tool_fidelity_reward,
        tool_sequence_reward,
        turn_efficiency_reward,
    )
    from passenger_sim import PassengerSim
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
    from scenarios import Scenario, build_scenario


VALID_TASKS = ("ticket_booking", "ticket_issuance", "crowd_announcement")
DEFAULT_TASK = "ticket_booking"


class MetrocrowdmanagerEnvironment(MCPEnvironment):
    """Agentic metro environment exposing 11 simulated MCP tools."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        # Build the FastMCP server first; MCPEnvironment.__init__ wires it up.
        self._mcp = FastMCP("MetroCrowdManager")
        self._scenario: Optional[Scenario] = None
        self._passenger: Optional[PassengerSim] = None
        self._task_name: str = DEFAULT_TASK
        self._max_steps: int = 1
        self._current_step: int = 0
        self._episode_id: str = str(uuid4())
        self._step_rewards: List[float] = []
        self._reward_breakdowns: List[Dict[str, float]] = []

        self._register_tools()
        super().__init__(self._mcp)

    # ------------------------------------------------------------------ tools

    def _register_tools(self) -> None:
        """Register tool wrappers that close over `self` and read live state.

        Each wrapper guards against being invoked before `reset()` has
        populated `self._scenario` and rejects calls in tasks where the tool
        isn't applicable (so the agent gets fast feedback rather than silently
        useful data from the wrong context).
        """
        env = self

        def _require_scenario() -> Scenario:
            if env._scenario is None:
                raise RuntimeError("Environment has not been reset; call reset() first.")
            return env._scenario

        @self._mcp.tool()
        def get_platform_for_destination(destination: str) -> dict:
            """Look up the platform number for a destination station."""
            return tools_mod.get_platform_for_destination(_require_scenario(), destination)

        @self._mcp.tool()
        def get_platform_crowd(platform: int) -> dict:
            """Get per-zone platform crowd percentages (10 zones, A-J)."""
            return tools_mod.get_platform_crowd(_require_scenario(), int(platform))

        @self._mcp.tool()
        def get_train_crowd_occupation(platform: int) -> dict:
            """Get per-coach occupancy percentages for the train at a platform."""
            return tools_mod.get_train_crowd_occupation(_require_scenario(), int(platform))

        @self._mcp.tool()
        def get_current_time() -> dict:
            """Return the current station-system time as HH:MM."""
            return tools_mod.get_current_time(_require_scenario())

        @self._mcp.tool()
        def validate_destination(destination: str) -> dict:
            """Check whether a destination is a valid station on this network."""
            return tools_mod.validate_destination(_require_scenario(), destination)

        @self._mcp.tool()
        def get_ticket_cost(
            source: str, destination: str, passenger_count: int
        ) -> dict:
            """Quote the ticket fare for source -> destination given a count."""
            return tools_mod.get_ticket_cost(
                _require_scenario(), source, destination, int(passenger_count)
            )

        @self._mcp.tool()
        def initiate_payment(amount: float, passenger_count: int) -> dict:
            """Start a payment for the given amount; returns a payment_id to poll."""
            return tools_mod.initiate_payment(
                _require_scenario(), float(amount), int(passenger_count)
            )

        @self._mcp.tool()
        def check_payment_status(payment_id: str) -> dict:
            """Poll a payment_id; resolves to success/failed after a few polls."""
            return tools_mod.check_payment_status(_require_scenario(), payment_id)

        @self._mcp.tool()
        def list_valid_stations() -> dict:
            """List every station currently served on the network."""
            return tools_mod.list_valid_stations(_require_scenario())

        @self._mcp.tool()
        def get_ideal_zone(platform: int) -> dict:
            """Recommend a single ideal platform zone for a single passenger."""
            return tools_mod.get_ideal_zone(_require_scenario(), int(platform))

        @self._mcp.tool()
        def get_ideal_distribution(platform: int) -> dict:
            """Recommend the full 10-zone ideal crowd distribution."""
            return tools_mod.get_ideal_distribution(_require_scenario(), int(platform))

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MetrocrowdmanagerObservation:
        task = kwargs.get("task") or kwargs.get("task_name") or DEFAULT_TASK
        if task not in VALID_TASKS:
            task = DEFAULT_TASK
        self._task_name = task
        self._scenario = build_scenario(task, seed=seed)
        self._episode_id = episode_id or str(uuid4())
        self._current_step = 0
        self._step_rewards = []
        self._reward_breakdowns = []

        if task == "ticket_booking":
            assert self._scenario.passenger_goal is not None
            self._passenger = PassengerSim(
                goal=self._scenario.passenger_goal,
                rng_seed=seed if seed is not None else 0,
            )
            self._max_steps = 1
        elif task == "ticket_issuance":
            self._passenger = None
            self._max_steps = 1
        else:  # crowd_announcement
            self._passenger = None
            self._max_steps = max(1, len(self._scenario.train_arrivals))

        return self._build_observation(done=False, reward_breakdown={})

    # ------------------------------------------------------------------ step

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Route MCP types via discriminator BEFORE delegating to the parent.

        The HTTP serializer registers ``SubmitResponseAction`` as
        ``action_cls`` (with ``extra="allow"``), so MCP payloads come
        through the wire validated as a SubmitResponseAction with extra
        ``tool_name`` / ``arguments`` / ``tools`` fields. We rebuild the
        proper MCP action from those fields and hand it back to the
        parent's ``step()``.
        """
        coerced = self._coerce_to_mcp_if_possible(action)
        return super().step(coerced, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        coerced = self._coerce_to_mcp_if_possible(action)
        return await super().step_async(coerced, timeout_s=timeout_s, **kwargs)

    def _coerce_to_mcp_if_possible(self, action: Action) -> Action:
        if isinstance(action, (CallToolAction, ListToolsAction)):
            return action
        action_type = getattr(action, "type", None)
        # extras may be in __pydantic_extra__ (Pydantic v2) or attributes
        extras = getattr(action, "__pydantic_extra__", None) or {}

        def _f(name: str, default: Any = None) -> Any:
            return getattr(action, name, extras.get(name, default))

        if action_type == "list_tools":
            return ListToolsAction()
        if action_type == "call_tool":
            return CallToolAction(
                tool_name=_f("tool_name", ""),
                arguments=_f("arguments", {}) or {},
            )
        return action

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle SubmitResponseAction. MCP actions were already routed above."""
        if self._scenario is None:
            raise RuntimeError("Environment has not been reset; call reset() first.")

        content, turn_history = self._extract_submission(action)

        if self._task_name == "crowd_announcement":
            return self._handle_announcement_step(content, turn_history)
        return self._handle_terminal_submission(content, turn_history)

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> State:
        sc = self._scenario
        return State(
            episode_id=self._episode_id,
            step_count=self._current_step,
            task_name=self._task_name,
            max_steps=self._max_steps,
            done=self._current_step >= self._max_steps,
            station_list=list(sc.station_list) if sc else [],
            source_station=sc.source_station if sc else "",
            current_time=sc.current_time if sc else "",
            passenger_state=self._passenger.snapshot() if self._passenger else None,
            step_rewards=list(self._step_rewards),
        )

    # ------------------------------------------------------------------ helpers

    def _extract_submission(self, action: Action) -> tuple[str, List[dict]]:
        """Pull `content` and `turn_history` from a SubmitResponseAction or
        a generic Action carrying the same fields."""
        content = ""
        turn_history: List[dict] = []
        if isinstance(action, SubmitResponseAction):
            content = action.content
            turn_history = action.metadata.get("turn_history", []) or []
        else:
            data = getattr(action, "metadata", {}) or {}
            content = data.get("content", "") or getattr(action, "content", "") or ""
            turn_history = data.get("turn_history", []) or []
        return content, turn_history

    # --- terminal submission (Tasks 1, 2) -----------------------------

    def _handle_terminal_submission(
        self, content: str, turn_history: List[dict]
    ) -> MetrocrowdmanagerObservation:
        breakdown = self._compute_rewards(content, turn_history)
        total = sum(breakdown.values())
        self._current_step += 1
        self._step_rewards.append(total)
        self._reward_breakdowns.append(breakdown)
        return self._build_observation(
            done=True, reward_breakdown=breakdown, last_reward=total
        )

    # --- per-arrival submission (Task 3) ------------------------------

    def _handle_announcement_step(
        self, content: str, turn_history: List[dict]
    ) -> MetrocrowdmanagerObservation:
        breakdown = self._compute_rewards(content, turn_history)
        total = sum(breakdown.values())
        self._current_step += 1
        self._step_rewards.append(total)
        self._reward_breakdowns.append(breakdown)

        if self._scenario is not None:
            self._scenario.current_arrival_idx = min(
                self._current_step, len(self._scenario.train_arrivals) - 1
            )

        done = self._current_step >= self._max_steps
        return self._build_observation(
            done=done, reward_breakdown=breakdown, last_reward=total
        )

    # --- reward composition -------------------------------------------

    def _compute_rewards(self, content: str, turn_history: List[dict]) -> Dict[str, float]:
        task = self._task_name
        if task == "ticket_booking":
            return self._reward_ticket_booking(content, turn_history)
        if task == "ticket_issuance":
            return self._reward_ticket_issuance(content, turn_history)
        return self._reward_crowd_announcement(content, turn_history)

    def _reward_ticket_booking(
        self, content: str, turn_history: List[dict]
    ) -> Dict[str, float]:
        sc = self._scenario
        assert sc is not None

        seq = tool_sequence_reward(turn_history, "ticket_booking")
        fid = tool_fidelity_reward(turn_history, "ticket_booking")
        eco = tool_economy_reward(turn_history, "ticket_booking")
        fmt = format_reward(turn_history)
        task_success = task_success_reward(turn_history, sc)
        convo = conversation_quality_reward(turn_history, sc)
        efficiency = turn_efficiency_reward(turn_history)
        pay = payment_discipline_reward(turn_history, sc)
        malformed = has_malformed_tool_call(turn_history)
        final_has_tool_markup = "<tool_call" in (content or "")
        if malformed or final_has_tool_markup:
            polite = 0.0
            clarity = 0.0
        else:
            polite = compute_politeness(content, [], [], 10)
            clarity = compute_clarity(content, [], [], 10)

        return {
            "task_success":        0.30 * task_success,
            "tool_sequence":       0.10 * seq,
            "tool_fidelity":       0.10 * fid,
            "tool_economy":        0.03 * eco,
            "format":              0.02 * fmt,
            "conversation_quality": 0.20 * convo,
            "turn_efficiency":     0.10 * efficiency,
            "payment_discipline":  0.10 * pay,
            "politeness":          0.03 * polite,
            "clarity":             0.02 * clarity,
        }

    def _reward_ticket_issuance(
        self, content: str, turn_history: List[dict]
    ) -> Dict[str, float]:
        sc = self._scenario
        assert sc is not None

        seq = tool_sequence_reward(turn_history, "ticket_issuance")
        fid = tool_fidelity_reward(turn_history, "ticket_issuance")
        eco = tool_economy_reward(turn_history, "ticket_issuance")
        fmt = format_reward(turn_history)
        ticket = ticket_schema_validity(content, sc)
        lang = compute_language_consistency(content, [], [], 10)

        # Heavy penalty: ticket_issuance is a tool-grounded task. If the
        # agent submitted without making a single valid tool call, no
        # field in the JSON could have come from real env data — zero
        # every component except a small format/language signal so the
        # model can't farm easy reward by emitting empty or hallucinated
        # answers.
        if count_valid_tool_calls(turn_history) == 0:
            return {
                "tool_sequence":          0.0,
                "tool_fidelity":          0.0,
                "tool_economy":           0.0,
                "format":                 0.10 * fmt,
                "ticket_schema_validity": 0.0,
                "language_consistency":   0.0,
            }

        return {
            "tool_sequence":          0.25 * seq,
            "tool_fidelity":          0.20 * fid,
            "tool_economy":           0.10 * eco,
            "format":                 0.10 * fmt,
            "ticket_schema_validity": 0.30 * ticket,
            "language_consistency":   0.05 * lang,
        }

    def _reward_crowd_announcement(
        self, content: str, turn_history: List[dict]
    ) -> Dict[str, float]:
        sc = self._scenario
        assert sc is not None
        arrival = (
            sc.train_arrivals[self._current_step]
            if self._current_step < len(sc.train_arrivals)
            else sc.train_arrivals[-1]
        )
        train = arrival["train_crowd"]
        plat = arrival["platform_crowd"]
        platform_num = arrival["platform"]
        nc = len(train)

        seq = tool_sequence_reward(turn_history, "crowd_announcement")
        fid = tool_fidelity_reward(turn_history, "crowd_announcement")
        eco = tool_economy_reward(turn_history, "crowd_announcement")
        fmt = format_reward(turn_history)
        dist = compute_distribution_accuracy(content, train, plat, nc)
        cons = compute_conservation_accuracy(content, train, plat, nc)
        feas = compute_feasibility_accuracy(content, train, plat, nc)
        color = compute_color_grading(content, train, plat, nc)
        fact = compute_factual_accuracy(content, train, plat, nc)
        platf = compute_platform_mention(content, platform_num)
        noop = compute_noop_detection(content, train, plat, nc)
        seqd = compute_sequential_direction(content, train, plat, nc)
        clarity = compute_clarity(content, train, plat, nc)

        return {
            "tool_sequence":         0.15 * seq,
            "tool_fidelity":         0.10 * fid,
            "tool_economy":          0.05 * eco,
            "format":                0.05 * fmt,
            "distribution_accuracy": 0.20 * dist,
            "conservation_accuracy": 0.10 * cons,
            "feasibility_accuracy":  0.05 * feas,
            "color_grading":         0.05 * color,
            "factual_accuracy":      0.05 * fact,
            "platform_mention":      0.05 * platf,
            "noop_detection":        0.05 * noop,
            "sequential_direction":  0.05 * seqd,
            "clarity":               0.05 * clarity,
        }

    # --- observation builder ------------------------------------------

    def _build_observation(
        self,
        done: bool,
        reward_breakdown: Dict[str, float],
        last_reward: float = 0.0,
    ) -> MetrocrowdmanagerObservation:
        prompt = "" if done else self._build_prompt()
        passenger_message = ""
        if self._passenger and not done:
            passenger_message = self._passenger.last_utterance
        scenario_summary = self._build_scenario_summary()

        return MetrocrowdmanagerObservation(
            task_name=self._task_name,
            prompt_text=prompt,
            current_step=min(self._current_step + 1, self._max_steps),
            max_steps=self._max_steps,
            passenger_message=passenger_message,
            scenario_summary=scenario_summary,
            reward_breakdown=dict(reward_breakdown),
            done=done,
            reward=float(last_reward) if reward_breakdown else 0.0,
            metadata={
                "episode_id": self._episode_id,
                "step": self._current_step,
            },
        )

    def _build_scenario_summary(self) -> Dict[str, Any]:
        sc = self._scenario
        if sc is None:
            return {}
        summary: Dict[str, Any] = {
            "task": sc.task_name,
            "source_station": sc.source_station,
            "valid_destinations": sc.valid_destinations,
        }
        if sc.task_name == "crowd_announcement":
            arrival = sc.train_arrivals[
                min(self._current_step, len(sc.train_arrivals) - 1)
            ]
            summary["current_arrival_idx"] = self._current_step
            summary["total_arrivals"] = len(sc.train_arrivals)
            summary["current_platform"] = arrival["platform"]
        return summary

    # --- prompt builders ----------------------------------------------

    def _build_prompt(self) -> str:
        if self._task_name == "ticket_booking":
            return self._prompt_ticket_booking()
        if self._task_name == "ticket_issuance":
            return self._prompt_ticket_issuance()
        return self._prompt_crowd_announcement()

    def _prompt_ticket_booking(self) -> str:
        sc = self._scenario
        passenger = self._passenger
        utterance = passenger.opening_line() if passenger else ""
        return "\n".join(
            [
                f"You are a metro station ticket agent at {sc.source_station}.",
                "A passenger has approached you. Your job is to:",
                "  1. Read the passenger's request carefully and extract any details already provided.",
                "  2. If the destination is already stated, do not ask for it again.",
                "  3. Confirm the destination is valid and ask only for missing details.",
                "  4. Quote the fare clearly.",
                "  5. Initiate the payment and poll until it resolves.",
                "  6. Communicate the outcome and any next step to the passenger.",
                "",
                "Use the available MCP tools to validate destinations, look up fares,",
                "and run the payment loop. Respond politely and only initiate payment",
                "AFTER you know the destination and passenger count.",
                "",
                f"Passenger: \"{utterance}\"",
            ]
        )

    def _prompt_ticket_issuance(self) -> str:
        sc = self._scenario
        goal = sc.passenger_goal
        # ticket_issuance scenarios don't carry a passenger_goal; pick a destination
        # deterministically from the station list.
        dest = (
            goal.destination
            if goal is not None
            else next(
                (s for s in sc.station_list if s != sc.source_station),
                sc.station_list[-1],
            )
        )
        return "\n".join(
            [
                f"You are issuing a metro ticket at {sc.source_station}.",
                f"Destination: {dest}",
                "",
                "Use the MCP tools to look up the platform, current platform crowd,",
                "current train coach occupancy, and the ideal platform zone for this",
                "passenger. Then return a single JSON object with these fields:",
                "",
                '  {"time": "HH:MM", "from": "<station>", "to": "<station>",',
                '   "price": <number>, "platform": <int>, "ideal_zone": "<A-J>"}',
                "",
                "Use only data returned by tools — do not guess. Respond with the JSON",
                "as your final message (no `<tool_call>` blocks).",
            ]
        )

    def _prompt_crowd_announcement(self) -> str:
        sc = self._scenario
        idx = min(self._current_step, len(sc.train_arrivals) - 1)
        arrival = sc.train_arrivals[idx]
        platform = arrival["platform"]
        return "\n".join(
            [
                f"Train arrival {idx + 1} of {len(sc.train_arrivals)} at {sc.source_station}.",
                f"Approaching platform: {platform}",
                "",
                "Use the MCP tools to fetch this platform's current crowd, the train's",
                "current coach occupancy, and the ideal redistribution. Then produce a",
                "polite redirection announcement in the existing structured format:",
                "",
                '  Announcement: "<your announcement>"',
                "  Recommended Platform Distribution: [<10 ints>]",
                "  Platform Zone Color Codes: [<10 hex>]",
                "  Train Coach Color Codes: [<10 hex>]",
                "",
                "Color reference: #008000 ≤40, #FFFF00 40-60, #FF8C00 60-80, #FF0000 >80.",
                f"Mention platform {platform} in your announcement.",
            ]
        )

    # --- close --------------------------------------------------------

    def close(self) -> None:
        super().close()
        self._scenario = None
        self._passenger = None
