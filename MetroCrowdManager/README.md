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
  - mcp
  - agentic
  - tool-use
  - reinforcement-learning
  - metro
  - hackathon
---

# MetroCrowdManager — Agentic MCP Environment for OpenEnv

<div align="center">

<a href="./blog.md">
  <img src="https://img.shields.io/badge/%F0%9F%93%9D%20%20READ%20THE%20BLOG-The%20story%20behind%20MetroCrowdManager-FFD21E?style=for-the-badge&labelColor=FF9D00" alt="Read the blog" height="44" />
</a>

### 👉 [**Read `blog.md` — the full story behind this project**](./blog.md) 👈

*How a missed train at Hauz Khas became a Scaler × Hugging Face hackathon submission — with the tools, rewards, and architecture explained end-to-end.*

</div>

---

[![GitHub repo](https://img.shields.io/badge/GitHub-Dhiwakar1997%2Fgluon__openenv-181717?logo=github)](https://github.com/Dhiwakar1997/gluon_openenv)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dhiwakar1997/gluon_openenv/blob/main/notebooks/train_on_hf_jobs.ipynb)
[![Trackio dashboard](https://img.shields.io/badge/Trackio-DhiwakarDev%2Fmcm--trackio-FF6F00?logo=huggingface)](https://huggingface.co/spaces/DhiwakarDev/mcm-trackio)

A multi-task **agentic** environment where an LLM plays the role of a metro
station assistant. The agent orchestrates **11 simulated MCP tools** (FastMCP
under the hood) to converse with passengers, look up platform / crowd state,
process payments, and issue announcements. Built on
[**OpenEnv**](https://github.com/meta-pytorch/OpenEnv) on top of `MCPEnvironment`.

> Hackathon submission for the OpenEnv RL Hackathon. The env is fully
> tool-use focused: every reward signal is grounded in real tool outputs, not
> in hallucinated text.

---

## What this env trains

A single LLM is asked to handle **three different metro-station tasks**, all
through the same MCP tool surface. The agent doesn't get the answers — it has
to query the environment for them.

| Task | Difficulty | Steps | What the agent must do |
|------|------------|-------|------------------------|
| `ticket_booking` | hard | 1 episode (multi-turn) | Converse with a scripted passenger, validate destination, quote fare, run the payment loop, communicate the outcome |
| `ticket_issuance` | medium | 1 | Use tools to gather platform + crowd intel, then emit a structured JSON ticket with the ideal boarding zone |
| `crowd_announcement` | hard | 3–4 (one per train arrival) | Across consecutive train arrivals, fetch crowd state via tools and produce a redirection announcement per arrival |

All three tasks share the same simulated metro network (stations, platforms,
crowd levels, train occupancies) but exercise different reward dimensions.

---

## MCP Tool Catalog

Eleven tools exposed via FastMCP. Tool calls flow through the inherited MCP
routing in [`server/MetroCrowdManager_environment.py`](server/MetroCrowdManager_environment.py)
and are implemented in [`server/tools.py`](server/tools.py).

| # | Tool | Used by | Purpose |
|---|------|---------|---------|
| 1 | `list_valid_stations()` | ticket_booking | Discover the network's stations |
| 2 | `validate_destination(destination)` | ticket_booking | Confirm a destination is real (fuzzy match) |
| 3 | `get_ticket_cost(source, destination, passenger_count)` | ticket_booking | Quote the fare in INR |
| 4 | `initiate_payment(amount, passenger_count)` | ticket_booking | Start a payment, returns a `payment_id` to poll |
| 5 | `check_payment_status(payment_id)` | ticket_booking | Poll until `success` / `failed` (~12% failure rate) |
| 6 | `get_platform_for_destination(destination)` | issuance / announcement | Look up which platform serves a destination |
| 7 | `get_platform_crowd(platform)` | issuance / announcement | Per-zone platform crowd % (10 zones, A–J) |
| 8 | `get_train_crowd_occupation(platform)` | issuance / announcement | Per-coach occupancy % for the train at a platform |
| 9 | `get_current_time()` | issuance | Current station-system time (`HH:MM`) |
| 10 | `get_ideal_zone(platform)` | ticket_issuance | Recommend a single ideal boarding zone |
| 11 | `get_ideal_distribution(platform)` | crowd_announcement | Recommend the full 10-zone target distribution |

---

## Reward Design

Rewards are grounded in **what the agent did** (tool calls + their results),
not just in surface text. Per-task reward weights live in
[`server/MetroCrowdManager_environment.py`](server/MetroCrowdManager_environment.py)
and the rule-based functions live in
[`server/agentic_rewards.py`](server/agentic_rewards.py) (orchestration) and
[`server/rewards.py`](server/rewards.py) (text/crowd accuracy).

### Orchestration rewards (used by all three tasks)

| Reward | What it measures |
|---|---|
| `tool_sequence` | Fraction of expected ordered checkpoints the agent hit in order |
| `tool_fidelity` | Did the agent actually pipe upstream tool outputs into downstream tool args? (e.g. `get_ticket_cost.cost` → `initiate_payment.amount`) |
| `tool_economy` | Penalises runaway tool spam, especially `check_payment_status` polling > 8 times |
| `format` | Fraction of agent turns that are well-formed `<tool_call>...</tool_call>` JSON or clean final answers |

### Task-specific rewards

**`ticket_booking`** (10 dims): `task_success` (30%), `tool_sequence` (10%),
`tool_fidelity` (10%), `tool_economy` (3%), `format` (2%),
`conversation_quality` (20%), `turn_efficiency` (10%), `payment_discipline`
(10%), `politeness` (3%), `clarity` (2%).

**`ticket_issuance`** (6 dims, hard floor when zero tools called):
`tool_sequence` (25%), `tool_fidelity` (20%), `tool_economy` (10%), `format`
(10%), `ticket_schema_validity` (30%), `language_consistency` (5%).

**`crowd_announcement`** (13 dims): `tool_sequence` (15%), `tool_fidelity`
(10%), `tool_economy` (5%), `format` (5%), `distribution_accuracy` (20%),
`conservation_accuracy` (10%), `feasibility_accuracy` (5%), `color_grading`
(5%), `factual_accuracy` (5%), `platform_mention` (5%), `noop_detection`
(5%), `sequential_direction` (5%), `clarity` (5%).

### Anti-gaming guards

- **Zero-tool-call floor** in `ticket_issuance`: every component except a
  small format signal collapses to 0.0 if the agent submitted without making
  a single valid tool call. No reward for hallucinated answers.
- **Malformed `<tool_call>` markup** zeroes politeness + clarity in
  `ticket_booking` so the agent can't game text rewards by leaking tool
  syntax.
- **Premature payment** (calling `initiate_payment` before validating
  destination + collecting passenger count) costs 0.5 from
  `info_sufficiency` and hard-caps `conversation_quality`.

---

## Quick Start

### Run the environment locally

```bash
cd MetroCrowdManager
uv sync
uv run server     # FastAPI server on http://localhost:8000
```

Or via Docker:

```bash
docker build -t metrocrowdmanager:latest .
docker run -p 8000:8000 metrocrowdmanager:latest
```

### Use it from Python (agentic loop)

```python
import asyncio
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from MetroCrowdManager import MetrocrowdmanagerEnv, SubmitResponseAction


async def main():
    async with MetrocrowdmanagerEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset(task="ticket_issuance")
        print(obs.observation.prompt_text)

        tools = await env.step(ListToolsAction())

        platform = await env.step(CallToolAction(
            tool_name="get_platform_for_destination",
            arguments={"destination": obs.observation.scenario_summary["destination"]},
        ))
        plat_num = platform.observation.result["data"]["platform"]

        crowd = await env.step(CallToolAction(
            tool_name="get_platform_crowd",
            arguments={"platform": plat_num},
        ))

        result = await env.step(SubmitResponseAction(
            content='{"time": "10:15", "from": "Riverside", "to": "Tech Park", '
                    '"platform": 3, "price": 80, "ideal_zone": "C"}',
            metadata={"turn_history": [...]},
        ))
        print(result.observation.reward_breakdown)


asyncio.run(main())
```

The `metadata={"turn_history": [...]}` field is how the rollout loop hands
the env the full record of `<tool_call>` outputs — that's what the
orchestration rewards score against. See
[`training/rollout.py`](https://github.com/Dhiwakar1997/gluon_openenv/blob/main/training/rollout.py)
for the canonical agentic loop.

---

## Train an agent on this env

GRPO with TRL on Hugging Face Jobs A100. The Colab notebook is the official
re-runnable entry point — it submits the job to HF Jobs, tails logs, embeds
the live trackio dashboard, and renders loss/reward plots inline.

- **Colab**: [`notebooks/train_on_hf_jobs.ipynb`](https://colab.research.google.com/github/Dhiwakar1997/gluon_openenv/blob/main/notebooks/train_on_hf_jobs.ipynb)
- **Training script**: [`training/hf_jobs_train_grpo.py`](https://github.com/Dhiwakar1997/gluon_openenv/blob/main/training/hf_jobs_train_grpo.py)
- **Live metrics**: [Trackio dashboard](https://huggingface.co/spaces/DhiwakarDev/mcm-trackio)

### What the training curves tell us

We ran GRPO over **`google/gemma-3-27b-it`** across all three tasks in parallel,
streaming per-step metrics into Trackio. Two snapshots from a representative
run — one orange line, one purple line, and one blue line per chart, each
corresponding to a different task — show how the policy improved.

<div align="center">

<img src="./images/train_reward.png" alt="Mean reward across training steps for all three tasks" width="85%" />

<sub><i>📈 <b>Reward / mean across runs.</b> Higher is better. Each line is a different task variant — the orange (booking) and purple (announcement) curves are both trending upward, while the blue line (issuance) starts higher because of how its reward weights collapse onto schema validity.</i></sub>

</div>

The reward chart is the headline result. Even on a short HF Jobs run with
**`google/gemma-3-27b-it`** as the base policy, **every task's mean reward
is climbing**. The orange
booking curve crosses **0.6 → 0.64** in just five steps, meaning the agent
went from "occasionally calling the right tool" to "running the full
booking → payment → confirmation chain end-to-end" inside a single training
window. The purple announcement curve climbs from `~0.19` to `~0.24` —
slower in absolute terms, but that task's reward stack has 9 dimensions all
multiplying to a single score, so even tenths of a point reflect real
behaviour shifts (correct hex color codes, proper sequential phrasing,
factually accurate crowd descriptions).

<div align="center">

<img src="./images/train_loss.png" alt="GRPO training loss across steps" width="85%" />

<sub><i>📉 <b>Train / loss across runs.</b> GRPO loss is policy-relative — values near zero mean the model is producing trajectories the reward says are "as good as expected", and values below zero mean it's exceeding its own baseline. The early dive (down to <code>-0.08</code>) is the model rapidly discovering the tool-use pattern; subsequent steps stabilise around zero as the policy converges.</i></sub>

</div>

The loss curve tells the *behavioural* half of the story. The big drop on
**step 1** is the moment the model figures out that wrapping its tool intent
in `<tool_call>` JSON unlocks reward — a lightbulb moment that's visible as
a single near-vertical line. After step 2 the loss settles into a tight band
around zero, which is exactly what you want from GRPO: the policy and the
reference agree, the variance has dropped, and any further reward gains
come from *refining* tool use rather than *discovering* it.

> 💡 **Takeaway:** with rewards grounded in real tool outputs (not LLM-as-judge),
> a Gemma-class open model on a short HF Jobs run is enough to teach genuine
> agentic behaviour — destination validation, payment polling, crowd-aware
> announcements — that transfers directly to a real metro deployment because
> the tool surface is identical.

---

## Run baseline inference

```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="google/gemma-3-27b-it"
cd MetroCrowdManager
python inference.py
```

`inference.py` runs the full agentic loop against a remote API model, prints
per-step reward + the full reward breakdown, and emits the standardised
`[START] / [STEP] / [END]` STDOUT contract used by the OpenEnv evaluator.

---

## Project Structure

```
MetroCrowdManager/
├── __init__.py                 # Module exports
├── models.py                   # SubmitResponseAction + Observation
├── client.py                   # MetrocrowdmanagerEnv async client
├── inference.py                # Baseline agentic-loop inference script
├── openenv.yaml                # OpenEnv manifest with task + tool spec
├── pyproject.toml              # Project metadata + dependencies
├── Dockerfile                  # Container image (Spaces-compatible)
├── README.md                   # This file (also rendered on the HF Space)
└── server/
    ├── app.py                  # FastAPI app entry
    ├── tools.py                # 11 MCP tool implementations
    ├── scenarios.py            # Per-episode scenario builder + RNG
    ├── passenger_sim.py        # Scripted passenger for ticket_booking
    ├── rewards.py              # 11 text/crowd-accuracy reward functions
    ├── agentic_rewards.py      # 10 orchestration reward functions
    └── MetroCrowdManager_environment.py  # MCPEnvironment subclass
```

---

## Links

- **GitHub repo**: <https://github.com/Dhiwakar1997/gluon_openenv>
- **Colab notebook**: [`notebooks/train_on_hf_jobs.ipynb`](https://colab.research.google.com/github/Dhiwakar1997/gluon_openenv/blob/main/notebooks/train_on_hf_jobs.ipynb)
- **Trackio dashboard**: <https://huggingface.co/spaces/DhiwakarDev/mcm-trackio>
- **Mini-blog write-up**: _coming soon_ — will be linked in the root [README.md](https://github.com/Dhiwakar1997/gluon_openenv/blob/main/README.md)

---

## Authors

Giridaran D, Dhiwakar Nagarajan
