---
title: MetroCrowdManager Environment Server
emoji: 🚇
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# MetroCrowdManager Environment

A metro station crowd management RL environment built on the OpenEnv framework.
An AI agent receives real-time train coach occupancy and platform zone crowd
percentages, then produces polite, structured redirection announcements with
color-coded crowd indicators.

## Motivation

In real metro systems, passengers cluster near platform entrances and have no
visibility into which coaches or zones are less crowded. This environment trains
and evaluates agents that can produce clear, mathematically sound, polite
crowd-redirection instructions — a genuine real-world task with immediate
practical value.

## Quick Start

```python
from MetroCrowdManager import MetrocrowdmanagerAction, MetrocrowdmanagerEnv

async with MetrocrowdmanagerEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="redirection")
    print(result.observation.prompt_text)

    result = await env.step(MetrocrowdmanagerAction(
        response_text='Announcement: "Dear passengers, please ..."\n'
                       'Recommended Platform Distribution: [50, 55, 60, 45, 40, 50]\n'
                       'Platform Zone Color Codes: [#FFFF00, #FFFF00, #FF8C00, #FFFF00, #008000, #FFFF00]\n'
                       'Train Coach Color Codes: [#FF8C00, #FF0000, #FF0000, #FF8C00, #FFFF00, #FF8C00]'
    ))
    print(f"Reward: {result.reward}")
```

## Action Space

**MetrocrowdmanagerAction** — a single field:

| Field | Type | Description |
|-------|------|-------------|
| `response_text` | `str` | Structured text containing announcement, recommended distribution, and color codes |

### Structured Output Format

For the full format (medium & hard tasks):

```
Announcement: "<crowd redirection announcement>"

Recommended Platform Distribution: [<target % for Zone A>, ..., <target % for Zone F>]

Platform Zone Color Codes: [<hex for Zone A>, ..., <hex for Zone F>]

Train Coach Color Codes: [<hex for Coach A>, ..., <hex for Coach F>]
```

For the easy task (crowd_assessment), only the two color code lines are required.

### Color Code Reference

| Hex Code | Color | Crowd Level |
|----------|-------|-------------|
| `#008000` | Green | Comfortable (<=40%) |
| `#FFFF00` | Yellow | Moderate (40-60%) |
| `#FF8C00` | Orange | Crowded (60-80%) |
| `#FF0000` | Red | Severely overcrowded (>80%) |

## Observation Space

**MetrocrowdmanagerObservation**:

| Field | Type | Description |
|-------|------|-------------|
| `num_coaches` | `int` | Number of coaches/zones (always 6) |
| `train_crowd` | `List[int]` | Coach occupancy percentages (0-100) |
| `platform_crowd` | `List[int]` | Platform zone crowd percentages (0-100) |
| `prompt_text` | `str` | Human-readable prompt with format instructions |
| `current_step` | `int` | Current step (1-indexed) |
| `max_steps` | `int` | Total steps in the episode |
| `station_name` | `str` | Name of the station |
| `task_name` | `str` | Active task name |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward from the last action |

## Tasks

### Task 1: Crowd Assessment (Easy)

**Objective**: Map crowd percentages to correct hex color codes.

- **Steps**: 1
- **Tests**: Basic perception — read percentages, apply thresholds, output hex codes
- **Scoring**: `0.50 * color_grading + 0.25 * language_consistency + 0.25 * clarity`

### Task 2: Redirection Announcement (Medium)

**Objective**: Produce a complete redirection response with polite announcement,
recommended platform distribution, and color codes.

- **Steps**: 1
- **Tests**: Math reasoning, language generation, structured formatting, color grading, sequential ordering, no-op detection
- **Scoring**: Equal weight across all 7 reward dimensions
- **Edge case**: 15% chance of balanced scenario requiring no-op response

### Task 3: Multi-Train Crowd Management (Hard)

**Objective**: Manage crowd across 8 consecutive train arrivals as platform crowd
evolves with crisis surges, event crowds, and varying train patterns.

- **Steps**: 8
- **Tests**: All medium-task abilities plus temporal adaptation, crisis handling, and consistency
- **Scoring**: All 7 dimensions scored per step, later steps weighted higher
- **Edge case**: 15% chance of balanced steps amidst chaos

## Reward Dimensions

All 7 rewards are rule-based heuristics returning `[0.0, 1.0]`:

| Reward | What it measures |
|--------|-----------------|
| `politeness` | Polite language markers, absence of aggressive language |
| `math_accuracy` | Proposed redistribution vs capacity-weighted ideal (MAE) |
| `color_grading` | Correct hex colors for platform zones + train coaches |
| `language_consistency` | English-only output (no non-Latin characters or foreign words) |
| `noop_detection` | Correctly identifies balanced scenarios (no unnecessary movement) |
| `clarity` | Short sentences, no jargon, structured formatting |
| `sequential_direction` | Coach-by-coach ordering (not simultaneous directives) |

## Setup & Usage

### Run Locally

```bash
cd MetroCrowdManager
uv sync
uv run server
```

Server starts at `http://localhost:8000`.

### Docker

```bash
# From MetroCrowdManager directory
docker build -t metrocrowdmanager:latest .
docker run -p 8000:8000 metrocrowdmanager:latest
```

### Run Inference

```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
cd MetroCrowdManager
python inference.py
```

### Deploy to Hugging Face Spaces

```bash
openenv push
```

## Baseline Scores

*Scores will be populated after running inference with the baseline model.*

| Task | Model | Avg Reward |
|------|-------|------------|
| crowd_assessment | Qwen2.5-72B-Instruct | TBD |
| redirection | Qwen2.5-72B-Instruct | TBD |
| multi_train | Qwen2.5-72B-Instruct | TBD |

## Project Structure

```
MetroCrowdManager/
├── __init__.py                 # Module exports
├── models.py                   # Action and Observation Pydantic models
├── client.py                   # MetrocrowdmanagerEnv WebSocket client
├── inference.py                # Baseline inference script (mandatory)
├── openenv.yaml                # OpenEnv manifest with task definitions
├── pyproject.toml              # Project metadata and dependencies
├── Dockerfile                  # Container image definition
├── README.md                   # This file
└── server/
    ├── __init__.py             # Server module exports
    ├── MetroCrowdManager_environment.py  # Core environment logic
    ├── rewards.py              # 7 rule-based reward functions
    └── app.py                  # FastAPI application
```
