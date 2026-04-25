"""
Scripted passenger state machine for Task 1 (ticket_booking).

The passenger follows a deterministic state progression:
    awaiting_destination
        └─ agent asks → passenger reveals destination
    awaiting_count
        └─ agent asks about passenger count → passenger reveals count
    awaiting_cost_confirmation
        └─ agent quotes the fare → passenger confirms
    awaiting_payment_result
        └─ agent reports payment outcome → passenger acknowledges
    done

At each state the passenger returns one of several phrasing variants,
sampled per-episode. Keeping this scripted (not LLM-driven) guarantees
deterministic rewards and keeps the rollout loop cheap on T4.

State transitions are triggered by keyword heuristics over the agent's
most recent turn. This is intentionally loose — we want the agent to
learn tool discipline, not to memorise specific prompts.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .scenarios import PassengerGoal


STATES = (
    "awaiting_destination",
    "awaiting_count",
    "awaiting_cost_confirmation",
    "awaiting_payment_result",
    "done",
)


DESTINATION_TEMPLATES = [
    "Hello, I need a ticket to {dest}, please.",
    "Hi, could you book me a ride to {dest}?",
    "I'd like to travel to {dest}.",
    "One ticket to {dest}, please.",
    "Can you help me get to {dest}?",
    "I need to go to {dest}.",
]

COUNT_TEMPLATES = [
    "There are {n} of us travelling.",
    "It's a group of {n}.",
    "{n} passengers, please.",
    "We are {n} people.",
    "Just me — {n} ticket.",
    "Make that {n} tickets.",
]

COST_CONFIRM_TEMPLATES = [
    "Sounds good, please proceed.",
    "Yes, that works for me.",
    "Okay, go ahead and charge it.",
    "That's fine, please start the payment.",
    "Alright, please process the payment.",
]

PAYMENT_ACK_SUCCESS = [
    "Great, thank you!",
    "Perfect, thanks for your help.",
    "Thanks, that's wonderful.",
    "Thank you very much.",
]

PAYMENT_ACK_FAILURE = [
    "Oh no — can we try again?",
    "That's unfortunate, please try once more.",
    "Is there another way to pay?",
    "Hmm, can you retry the payment?",
]


_DEST_KEYWORDS = (
    "where",
    "destination",
    "going",
    "where to",
    "which station",
    "travel to",
    "which stop",
)

_COUNT_KEYWORDS = (
    "how many",
    "passenger count",
    "number of passengers",
    "how many people",
    "group size",
    "how many travellers",
    "how many travelers",
)

_COST_CONFIRM_KEYWORDS = (
    "total is",
    "fare is",
    "cost is",
    "that'll be",
    "that will be",
    "price is",
    "the total",
    "confirm",
    "shall i proceed",
    "should i proceed",
    "proceed with payment",
    "initiate the payment",
)

_PAYMENT_RESULT_KEYWORDS = (
    "payment successful",
    "payment failed",
    "payment did not",
    "payment was successful",
    "payment has failed",
    "successfully paid",
    "unable to process",
    "could not process",
)


@dataclass
class PassengerSim:
    goal: PassengerGoal
    state: str = "awaiting_destination"
    last_utterance: str = ""
    rng_seed: int = 0
    turns_in_state: int = 0
    template_choice: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = random.Random(self.rng_seed)
        self.template_choice = {
            "dest": rng.choice(DESTINATION_TEMPLATES),
            "count": rng.choice(COUNT_TEMPLATES),
            "cost_confirm": rng.choice(COST_CONFIRM_TEMPLATES),
            "ack_success": rng.choice(PAYMENT_ACK_SUCCESS),
            "ack_failure": rng.choice(PAYMENT_ACK_FAILURE),
        }
        self.last_utterance = self._speak_current()

    def _speak_current(self) -> str:
        if self.state == "awaiting_destination":
            return self.template_choice["dest"].format(dest=self.goal.destination)
        if self.state == "awaiting_count":
            return self.template_choice["count"].format(n=self.goal.passenger_count)
        if self.state == "awaiting_cost_confirmation":
            return self.template_choice["cost_confirm"]
        if self.state == "awaiting_payment_result":
            return ""
        return ""

    def opening_line(self) -> str:
        """First utterance the agent sees at the start of the episode."""
        return self.template_choice["dest"].format(dest=self.goal.destination)

    def advance(self, agent_text: str, payment_outcome: Optional[str] = None) -> str:
        """Advance the state machine given the agent's latest turn.

        payment_outcome is only consulted in the awaiting_payment_result
        state; it should be "success" or "failed" once known.
        """
        text = (agent_text or "").lower()
        self.turns_in_state += 1

        if self.state == "awaiting_destination":
            if any(kw in text for kw in _DEST_KEYWORDS) or self.turns_in_state > 1:
                self.state = "awaiting_count"
                self.turns_in_state = 0
                self.last_utterance = self._speak_current()
                return self.last_utterance
            self.last_utterance = ""
            return ""

        if self.state == "awaiting_count":
            if any(kw in text for kw in _COUNT_KEYWORDS) or self.turns_in_state > 1:
                self.state = "awaiting_cost_confirmation"
                self.turns_in_state = 0
                self.last_utterance = self._speak_current()
                return self.last_utterance
            self.last_utterance = ""
            return ""

        if self.state == "awaiting_cost_confirmation":
            if any(kw in text for kw in _COST_CONFIRM_KEYWORDS) or re.search(
                r"(inr|rs\.?|₹)\s*\d", text
            ):
                self.state = "awaiting_payment_result"
                self.turns_in_state = 0
                self.last_utterance = self._speak_current()
                return self.last_utterance
            self.last_utterance = ""
            return ""

        if self.state == "awaiting_payment_result":
            if payment_outcome == "success" or any(
                phrase in text
                for phrase in (
                    "payment successful",
                    "payment was successful",
                    "successfully paid",
                    "ticket has been booked",
                    "booking confirmed",
                )
            ):
                self.state = "done"
                self.last_utterance = self.template_choice["ack_success"]
                return self.last_utterance
            if payment_outcome == "failed" or any(
                phrase in text
                for phrase in (
                    "payment failed",
                    "payment did not",
                    "unable to process",
                    "could not process",
                )
            ):
                self.state = "done"
                self.last_utterance = self.template_choice["ack_failure"]
                return self.last_utterance
            self.last_utterance = ""
            return ""

        self.last_utterance = ""
        return ""

    def is_done(self) -> bool:
        return self.state == "done"

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "last_utterance": self.last_utterance,
            "goal": {
                "source": self.goal.source,
                "destination": self.goal.destination,
                "passenger_count": self.goal.passenger_count,
            },
        }
