# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the MetroCrowdManager Environment.

The MetroCrowdManager environment simulates metro station crowd management.
An agent receives train coach occupancy and platform zone crowd data,
then produces redirection announcements with color-coded crowd indicators.
"""

from typing import List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MetrocrowdmanagerAction(Action):
    """Action for the MetroCrowdManager environment.

    The agent produces a structured text response containing:
    - An announcement with crowd redirection instructions
    - Recommended platform distribution percentages
    - Platform zone color codes (hex)
    - Train coach color codes (hex)
    """

    response_text: str = Field(
        ..., description="Agent's structured response text with announcement and color codes"
    )


class MetrocrowdmanagerObservation(Observation):
    """Observation from the MetroCrowdManager environment.

    Provides the current crowd state at a metro station including
    train coach occupancy and platform zone crowd percentages.
    """

    num_coaches: int = Field(default=6, description="Number of train coaches / platform zones")
    train_crowd: List[int] = Field(
        default_factory=list,
        description="Current train coach occupancy percentages (0-100 per coach)",
    )
    platform_crowd: List[int] = Field(
        default_factory=list,
        description="Current platform zone crowd percentages (0-100 per zone)",
    )
    prompt_text: str = Field(default="", description="Human-readable scenario prompt with format instructions")
    current_step: int = Field(default=0, description="Current step number (1-indexed for display)")
    max_steps: int = Field(default=1, description="Total steps in this episode")
    station_name: str = Field(default="", description="Name of the metro station")
    task_name: str = Field(default="crowd_assessment", description="Active task name")
