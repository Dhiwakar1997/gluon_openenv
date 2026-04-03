# MetroPulse RL Environment — Detailed Implementation Plan

## Context

The existing MetroPulse environment at `/Users/dhiwakar/Documents/Gluon/MetroPulse/` is currently a simple echo environment scaffolded from the OpenEnv template. It needs to be transformed into a real RL evaluation environment for **metro station crowd management**.

**Problem**: Passengers near coach A don't know if coach E's platform zone is less crowded. A model trained via GRPO should take train + platform crowd percentages as input, then output polite, mathematically sound, coach-by-coach redirection instructions with color gradings. This applies to any metro system — the environment uses generic station names and is not tied to any specific city or line.

**Training approach**: unsloth + GRPO pattern (matching `unsloth_2048.ipynb` reference), not the multi-turn wordle rollout pattern. MetroPulse is multi-step but single-action-per-step: each step presents a new crowd snapshot (new train arriving, platform crowd evolving), the model responds with redirection instructions, and rewards are computed per-step across 7 dimensions.

**Hackathon alignment**: This plan follows the Gluon OpenEnv Hackathon Round 1 requirements — full OpenEnv spec compliance (`step()`/`reset()`/`state()`), 3 tasks with agent graders (easy/medium/hard), `openenv.yaml`, `inference.py` with mandatory stdout format, Dockerfile, and HF Spaces deployment.

---

## Architecture Overview

### Why Multi-Step Stateful (not Single-Turn)

The original plan was single-turn (one crowd state in, one response out, episode done). The updated design is **multi-step stateful** to better simulate real metro operations:

- A station serves multiple trains in sequence. The platform crowd **accumulates** between trains and is affected by the agent's redirection quality.
- `reset()` initializes an **empty platform** (all zones at 0%) and generates the first arriving train's coach occupancy.
- Each `step()`:
  1. Evaluates the agent's response against the current crowd state (7 reward dimensions)
  2. Simulates the effect of the response on platform crowd (good redirections reduce crowding imbalance)
  3. Generates a **new random `train_crowd`** (next train arriving with different occupancy)
  4. Adds **random new passengers** to the platform (people arriving at the station between trains)
  5. Returns the new observation with updated crowd state
- Episode runs for a fixed number of steps (configurable per task difficulty), then `done=True`.

This makes the environment genuinely stateful — the agent must adapt to evolving conditions, not just respond to isolated snapshots.

### Why unsloth Pattern (not TRL+vLLM rollout)

The wordle-grpo pattern uses `rollout_func`, `rollout_once`, `generate_rollout_completions`, and vLLM colocate mode — all designed for complex multi-turn environments with branching. Since MetroPulse steps are independent evaluations (each step's response is scored on its own merits, not on conversation history), the simpler unsloth pattern is appropriate:
- Reward functions receive `completions` list and return scores
- No custom rollout function needed
- GRPOTrainer handles generation internally

---

## Deliverables

### Core Environment (modify 4 files, create 2 new files)

| File | Status | Purpose |
|------|--------|---------|
| `MetroPulse/models.py` | **Modify** | Observation/Action Pydantic models |
| `MetroPulse/server/rewards.py` | **Create** | 7 standalone reward functions |
| `MetroPulse/server/MetroPulse_environment.py` | **Modify** | Stateful step/reset/state logic |
| `MetroPulse/client.py` | **Modify** | Updated payloads/parsing |
| `MetroPulse/server/__init__.py` | **Modify** | Export rewards module |
| `MetroPulse/openenv.yaml` | **Create** | OpenEnv metadata spec file |

### Hackathon Submission Files (create 3 new files)

| File | Status | Purpose |
|------|--------|---------|
| `inference.py` | **Create** | Baseline inference script (root directory, mandatory) |
| `Dockerfile` | **Create** | Containerized execution |
| `README.md` | **Create** | Environment docs per hackathon spec |

### Training Notebook (1 new file)

| File | Status | Purpose |
|------|--------|---------|
| `MetroPulse/metropulse_train.ipynb` | **Create** | GRPO training notebook (unsloth pattern) |

---

## State Space Design

### Episode State (internal, tracked across steps)

```python
class EpisodeState:
    current_step: int              # 0-indexed step counter
    max_steps: int                 # Determined by task difficulty
    train_crowd: List[int]         # Current train's coach occupancy [6 values, 0-95%]
    platform_crowd: List[float]    # Accumulated platform crowd [6 values, 0-100%]
    total_reward: float            # Sum of step rewards so far
    step_rewards: List[float]      # Per-step reward history
    station_name: str              # Current station (fixed per episode)
    crowd_history: List[dict]      # History of crowd states for state()
```

### Observation (what the model receives each step)

```python
class MetroPulseObservation(BaseModel):
    num_coaches: int = 6
    train_crowd: List[int]         # Current train's coach occupancy percentages
    platform_crowd: List[int]      # Current platform zone crowd percentages (rounded)
    prompt_text: str               # Human-readable scenario with format instructions
    current_step: int              # Which step we're on (1-indexed for display)
    max_steps: int                 # Total steps in this episode
    station_name: str              # Station name for context
```

### Action (what the model produces each step)

```python
class MetroPulseAction(BaseModel):
    response_text: str  # Structured text with the following format:
```

The model must output a **structured response** with three clearly labeled sections:

```
Announcement: "<Generated crowd redirection announcement>"

Recommended Platform Distribution: [<target % for Zone A>, ..., <target % for Zone F>]

Platform Zone Color Codes: [<hex color for Zone A>, ..., <hex color for Zone F>]

Train Coach Color Codes: [<hex color for Coach A>, ..., <hex color for Coach F>]
```

**Hex color mapping:**
- `#008000` (Green) — comfortable, <=40% effective crowd
- `#FFFF00` (Yellow) — moderate, 40-60% effective crowd
- `#FF8C00` (Orange) — crowded, 60-80% effective crowd
- `#FF0000` (Red) — severely overcrowded, >80% effective crowd

---

## Episode Dynamics (Stateful Step Logic)

### `reset()` — Start a New Episode

1. Pick a random station from a pool of generic metro station names (Central Station, Riverside, University, Market Square, Tech Park, Lakeside, Airport Terminal, Old Town, Harbor View, Greenfield, etc.)
2. Initialize **platform crowd to all zeros** — the platform is empty at the start of the episode
3. Generate the **first arriving train's coach occupancy**: `train_crowd[i]` randomly in `[20, 95]` for each of 6 coaches. Apply variety patterns (see Crowd Variety Simulation below).
4. Add **initial passenger wave** to the platform — small random values `[5, 25]` per zone to represent the first batch of waiting passengers
5. Set `current_step = 0`, `max_steps` based on task difficulty
6. Return initial observation with the crowd state and prompt

### `step(action)` — Process Agent Response & Advance State

1. **Evaluate** the agent's `response_text` against current `train_crowd` and `platform_crowd` using all 7 reward functions → compute step reward
2. **Simulate redirection effect** on platform crowd:
   - Parse the agent's `Recommended Platform Distribution` from the response
   - If parsed successfully, blend current platform crowd toward the recommendation:
     ```python
     # Agent's recommendation quality affects how much the crowd actually follows
     compliance_factor = 0.3 + 0.4 * step_reward  # Better responses → more compliance (0.3 to 0.7)
     for i in range(num_coaches):
         platform_crowd[i] = (1 - compliance_factor) * platform_crowd[i] + compliance_factor * proposed[i]
     ```
   - If not parsed, platform crowd stays as-is (people don't move without clear directions)
3. **Simulate train departure**: Some passengers board the train, reducing platform crowd:
   ```python
   # Passengers board based on available train capacity
   for i in range(num_coaches):
       boarding_rate = (100 - train_crowd[i]) / 100.0 * 0.5  # Up to 50% of zone boards
       platform_crowd[i] = max(0, platform_crowd[i] * (1 - boarding_rate))
   ```
4. **Generate new train**: Create fresh `train_crowd[i]` randomly for the next arriving train (see Crowd Variety Simulation)
5. **Add new passengers** to the platform (people arriving between trains):
   ```python
   for i in range(num_coaches):
       new_arrivals = random.randint(5, 30)  # Random passengers per zone
       # Bias: more arrivals near station entrance zones (A, B tend to be near entrances)
       if i < 2:
           new_arrivals = int(new_arrivals * random.uniform(1.1, 1.5))
       platform_crowd[i] = min(100, platform_crowd[i] + new_arrivals)
   ```
6. **Increment step counter**. If `current_step >= max_steps`, set `done = True`
7. Return new observation, step reward, done flag, and info dict with per-dimension reward breakdown

### `state()` — Return Current Episode State

Returns the full internal state for debugging/evaluation:

```python
def state():
    return {
        "current_step": self.current_step,
        "max_steps": self.max_steps,
        "train_crowd": self.train_crowd,
        "platform_crowd": self.platform_crowd,
        "station_name": self.station_name,
        "total_reward": self.total_reward,
        "step_rewards": self.step_rewards,
        "crowd_history": self.crowd_history,
        "done": self.current_step >= self.max_steps
    }
```

### Crowd Variety Simulation

To ensure the agent encounters diverse crowd patterns during training and evaluation, train crowd generation uses **variety patterns** selected randomly:

```python
def generate_train_crowd(num_coaches=6):
    pattern = random.choice([
        "uniform_low",       # 15% chance — all coaches 20-45% (nearly empty train)
        "uniform_high",      # 15% chance — all coaches 70-95% (packed train)
        "front_heavy",       # 15% chance — coaches A,B,C crowded, D,E,F empty
        "rear_heavy",        # 15% chance — coaches D,E,F crowded, A,B,C empty
        "one_outlier_empty",  # 10% chance — one coach very empty, rest moderate-high
        "one_outlier_packed", # 10% chance — one coach very full, rest moderate-low
        "random",            # 20% chance — fully random across full range [20, 95]
    ])

    if pattern == "uniform_low":
        return [random.randint(20, 45) for _ in range(num_coaches)]
    elif pattern == "uniform_high":
        return [random.randint(70, 95) for _ in range(num_coaches)]
    elif pattern == "front_heavy":
        return [random.randint(70, 95) for _ in range(3)] + [random.randint(20, 45) for _ in range(3)]
    elif pattern == "rear_heavy":
        return [random.randint(20, 45) for _ in range(3)] + [random.randint(70, 95) for _ in range(3)]
    elif pattern == "one_outlier_empty":
        crowd = [random.randint(55, 85) for _ in range(num_coaches)]
        crowd[random.randint(0, 5)] = random.randint(15, 30)
        return crowd
    elif pattern == "one_outlier_packed":
        crowd = [random.randint(30, 55) for _ in range(num_coaches)]
        crowd[random.randint(0, 5)] = random.randint(85, 95)
        return crowd
    else:  # random
        return [random.randint(20, 95) for _ in range(num_coaches)]
```

**Platform crowd growth patterns** (applied during new passenger addition in step):
- **Rush hour surge** (20% chance): `new_arrivals` multiplied by 2x — simulates peak hours
- **Off-peak trickle** (20% chance): `new_arrivals` reduced to `[2, 10]` — quiet periods
- **Event crowd** (10% chance): One or two zones get massive influx `[40, 60]` — simulates crowd from a connecting line or event
- **Normal** (50% chance): Standard `[5, 30]` per zone

With **15% probability per episode**, generate a **balanced** step where `std(platform_crowd) < 10` — this tests whether the model correctly identifies that no redistribution is needed.

### Prompt Text Generation

Each step generates a fresh prompt with current crowd state:

```
Upcoming train arriving at {station_name} station. [Step {current_step}/{max_steps}]
Coach occupancy: Coach A: {train_crowd[0]}%, Coach B: {train_crowd[1]}%, Coach C: {train_crowd[2]}%, Coach D: {train_crowd[3]}%, Coach E: {train_crowd[4]}%, Coach F: {train_crowd[5]}%
Platform crowd at each coach zone: Zone A: {platform_crowd[0]}%, Zone B: {platform_crowd[1]}%, Zone C: {platform_crowd[2]}%, Zone D: {platform_crowd[3]}%, Zone E: {platform_crowd[4]}%, Zone F: {platform_crowd[5]}%

Respond in the following structured format:

Announcement: "<your crowd redirection announcement>"

Recommended Platform Distribution: [<target % for Zone A>, <target % for Zone B>, ..., <target % for Zone F>]

Platform Zone Color Codes: [<hex color for Zone A>, <hex color for Zone B>, ..., <hex color for Zone F>]

Train Coach Color Codes: [<hex color for Coach A>, <hex color for Coach B>, ..., <hex color for Coach F>]

Color code reference: #008000 (Green, <=40%), #FFFF00 (Yellow, 40-60%), #FF8C00 (Orange, 60-80%), #FF0000 (Red, >80%)
```

---

## Three Tasks with Agent Graders

The hackathon requires minimum 3 tasks with difficulty progression (easy -> medium -> hard), each with a programmatic grader scoring 0.0-1.0. Each task tests a **progressively broader set of agent abilities**, not just the same process under harder conditions.

### Task 1: Crowd Assessment (Easy)

**Objective**: Given a single crowd snapshot, produce correct **color codes** for all platform zones and train coaches.

**What it tests**: Basic perception — can the agent read percentage values and map them to the correct hex color thresholds?

**Configuration**:
- `max_steps = 1` (single step)
- Platform crowd starts with moderate values `[20, 50]` per zone
- Train crowd uses `"random"` pattern only
- **Simplified action format** — agent only needs to output the two color code lists:
  ```
  Platform Zone Color Codes: [<hex>, <hex>, ..., <hex>]
  Train Coach Color Codes: [<hex>, <hex>, ..., <hex>]
  ```

**Active reward dimensions** (3 of 7):
- `color_grading` (weight 0.5) — primary objective
- `language_consistency` (weight 0.25) — must respond in English
- `clarity` (weight 0.25) — output should be clean and well-formatted

**Grader**:

```python
def grade_crowd_assessment(env, agent):
    obs = env.reset(task="crowd_assessment")
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    # Weighted: color_grading dominates
    score = (
        0.5 * info["rewards"]["color_grading"] +
        0.25 * info["rewards"]["language_consistency"] +
        0.25 * info["rewards"]["clarity"]
    )
    return score  # 0.0 - 1.0
```

**Why easy**: No math reasoning, no announcement writing, no redistribution logic. Pure perception: read numbers, apply thresholds, output hex codes. A baseline for any agent that can follow instructions.

---

### Task 2: Redirection Announcement (Medium)

**Objective**: Given a single crowd snapshot, produce a **complete redirection response** — polite announcement, recommended platform distribution, and color codes.

**What it tests**: Combines mathematical reasoning (compute redistribution), language generation (polite announcement), structured output formatting, color grading, and sequential direction ordering. Tests all 7 reward dimensions in a single step.

**Configuration**:
- `max_steps = 1` (single step)
- Platform crowd starts with varied values `[15, 85]` per zone — wider range than Task 1
- Train crowd uses all variety patterns (uniform_low, uniform_high, front_heavy, rear_heavy, outliers, random)
- **15% chance of a balanced scenario** — tests no-op detection (agent should say "no changes needed" instead of giving unnecessary directions)
- **Full action format** required:
  ```
  Announcement: "<redirection announcement>"
  Recommended Platform Distribution: [<target %>, ..., <target %>]
  Platform Zone Color Codes: [<hex>, ..., <hex>]
  Train Coach Color Codes: [<hex>, ..., <hex>]
  ```

**Active reward dimensions** (all 7, equal weight):
- `politeness` — polite, non-aggressive language
- `math_accuracy` — proposed redistribution matches capacity-weighted ideal
- `color_grading` — correct hex codes for both platform and train
- `language_consistency` — English only
- `noop_detection` — correctly identify balanced scenarios
- `clarity` — readable, jargon-free, well-structured
- `sequential_direction` — coach-by-coach ordering, no simultaneous directives

**Grader**:

```python
def grade_redirection(env, agent):
    obs = env.reset(task="redirection")
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
    # Equal weight across all 7 dimensions
    score = sum(info["rewards"].values()) / 7.0
    return score  # 0.0 - 1.0
```

**Why medium**: Requires the agent to reason mathematically (redistribution), generate natural language (polite announcement), follow structured output format, handle edge cases (balanced no-op), and order directions sequentially. A significant step up from pure perception, but still single-turn — no temporal adaptation needed.

---

### Task 3: Multi-Train Crowd Management (Hard)

**Objective**: Manage crowd across **8 consecutive train arrivals** as platform crowd evolves. Agent must adapt to changing conditions, handle crisis surges, and maintain quality across all steps.

**What it tests**: Everything from Task 2, plus temporal adaptation — the agent must handle evolving crowd states where its previous response affects the next observation. Tests whether the agent can maintain consistent quality under pressure, handle diverse crowd patterns within a single episode, and correctly detect no-op scenarios amidst chaotic conditions.

**Configuration**:
- `max_steps = 8`
- Platform starts empty (all zeros)
- Train crowd biased toward harder patterns: `"uniform_high"`, `"front_heavy"`, `"rear_heavy"` (70% of steps)
- Rush hour surge active for 50% of steps (2x passenger arrivals)
- Event crowd active for 20% of steps (massive influx at 1-2 zones)
- 15% chance of a balanced step (no-op test amidst chaos)
- **Full action format** required each step
- **Later steps weighted higher** — as the platform fills up, good responses become more critical

**Active reward dimensions** (all 7, equal weight per step):

All 7 dimensions scored each step. Final task score uses step-weighted averaging:

**Grader**:

```python
def grade_multi_train(env, agent):
    obs = env.reset(task="multi_train")
    step_weights = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]  # Later steps matter more
    weighted_reward = 0.0
    total_weight = sum(step_weights)
    for step_idx in range(8):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        weighted_reward += reward * step_weights[step_idx]
        if done:
            break
    return weighted_reward / total_weight  # 0.0 - 1.0
```

**Why hard**: Builds on all Task 2 abilities, but adds temporal complexity. The agent's response at step N affects the crowd state at step N+1 (via compliance-based blending). High crowd densities, rapid accumulation, diverse crisis patterns, and increasing stakes per step. Requires the agent to maintain quality under pressure and handle edge cases (balanced scenarios during peak hours, event surges). Genuinely challenges frontier models because it requires consistent multi-step reasoning, not just one good response.

---

### Task Ability Progression Summary

| Ability | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) |
|---------|:---:|:---:|:---:|
| Color code mapping | **Primary** | Required | Required |
| English consistency | Required | Required | Required |
| Clean formatting | Required | Required | Required |
| Math redistribution | - | **New** | Required |
| Polite announcement | - | **New** | Required |
| Sequential ordering | - | **New** | Required |
| No-op detection | - | **New** | Required |
| Multi-step adaptation | - | - | **New** |
| Crisis handling | - | - | **New** |
| Temporal consistency | - | - | **New** |

---

## Reward Functions — Detailed Mechanism Explanations

All reward functions are **rule-based heuristics** (not LLM-as-judge). Using an LLM judge during RL training would be prohibitively slow — each GRPO step generates multiple completions, and calling an LLM to score each would create a bottleneck. Rule-based rewards are fast (~microseconds per call), deterministic, and sufficient for these dimensions.

Each function signature: `(response_text: str, train_crowd: List[int], platform_crowd: List[int], num_coaches: int) -> float` in `[0.0, 1.0]`.

---

### Reward 1: Politeness & Kindness (`compute_politeness`)

**Why this matters**: The model is addressing a volatile, crowded metro platform. Aggressive or commanding language could escalate tension. Polite language de-escalates.

**Mechanism: Keyword Scoring Heuristic**

**Step 1 — Count polite markers** in the response text (case-insensitive):
```
POLITE_MARKERS = [
    "please", "kindly", "we request", "thank you", "for your comfort",
    "we appreciate", "dear passengers", "may we suggest", "if possible",
    "we recommend", "for your convenience", "would you mind",
    "we encourage", "your cooperation"
]
```
Count how many distinct polite markers appear: `polite_count`

**Step 2 — Count rude/aggressive markers**:
```
RUDE_MARKERS = [
    "move now", "get out", "you must", "immediately leave",
    "stupid", "idiot", "hurry up", "don't stand there",
    "what are you doing", "move it"
]
```
Also flag excessive exclamation marks (>3 total).
Count: `rude_count`

**Step 3 — Compute score**:
```
base_score = min(1.0, polite_count / 3.0)   # 3+ polite markers = full score
penalty = 0.3 * rude_count                   # Each rude marker costs 0.3
score = max(0.0, base_score - penalty)
```

**Edge cases**:
- Empty response -> 0.0 (no politeness detected)
- Response that is only polite pleasantries with no actual directions -> scores high here, but will score 0.0 on math accuracy and clarity, so overall reward stays low

---

### Reward 2: Mathematical Accuracy (`compute_math_accuracy`)

**Why this matters**: The model must propose a redistribution that actually makes mathematical sense — you can't suggest more people move to a zone than exist, and the redistribution should optimize for even load distribution relative to train capacity.

**Mechanism: Percentage-Based Ideal Distribution Comparison**

Since both `train_crowd` and `platform_crowd` are **percentages** (not absolute counts), the calculation works in percentage space throughout.

**Step 1 — Compute the ideal platform distribution (percentage-based)**:

```python
# Available capacity per coach (percentage of empty space in the train)
capacity_pct = [100 - train_crowd[i] for i in range(num_coaches)]
total_capacity_pct = sum(capacity_pct)

# Average platform crowd across all zones
avg_platform_pct = sum(platform_crowd) / num_coaches

# Ideal: redistribute platform percentages weighted by train capacity
ideal_platform_pct = [
    avg_platform_pct * (capacity_pct[i] / (total_capacity_pct / num_coaches))
    for i in range(num_coaches)
]

# Clamp to [0, 100]
ideal_platform_pct = [max(0, min(100, p)) for p in ideal_platform_pct]
```

**Step 2 — Extract the model's proposed percentages from structured output**:

```python
import re

dist_pattern = r'Recommended Platform Distribution\s*:\s*\[([^\]]+)\]'
match = re.search(dist_pattern, response_text, re.IGNORECASE)
if match:
    proposed_pct = [float(x.strip().rstrip('%')) for x in match.group(1).split(',')]
```

**Step 3 — Compute similarity score (percentage-based MAE)**:
```python
if len(proposed_pct) != num_coaches:
    return 0.0

mae_pct = sum(abs(proposed_pct[i] - ideal_platform_pct[i]) for i in range(num_coaches)) / num_coaches
score = max(0.0, 1.0 - (mae_pct / 50.0))
```

**Step 4 — Feasibility check**:
```python
if any(p < 0 or p > 100 for p in proposed_pct):
    score *= 0.5  # 50% penalty for impossible percentages
```

**Edge cases**:
- Model outputs no distribution list -> 0.0
- Model outputs wrong number of values -> 0.0
- Model outputs percentages > 100 or < 0 -> penalized via feasibility check
- Balanced scenario where model correctly says "no changes needed" -> handled by `compute_noop_detection`, this function returns 0.5 (neutral) when no distribution is provided and scenario is balanced

---

### Reward 3: Color Grading Accuracy (`compute_color_grading`)

**Why this matters**: The model should assign hex color codes to each platform zone AND each train coach so that the display system can render crowd density visually.

**Two separate color lists are evaluated**: Platform Zone Color Codes and Train Coach Color Codes.

**Mechanism: Threshold-Based Hex Color Verification**

**Hex color mapping (canonical values)**:
```
#008000  (Green)  — comfortable, <=40% crowd
#FFFF00  (Yellow) — moderate, 40-60% crowd
#FF8C00  (Orange) — crowded, 60-80% crowd
#FF0000  (Red)    — severely overcrowded, >80% crowd
```

**Step 1 — Compute expected colors for Platform Zones**:
```python
def pct_to_hex(pct):
    if pct <= 40:   return "#008000"
    elif pct <= 60: return "#FFFF00"
    elif pct <= 80: return "#FF8C00"
    else:           return "#FF0000"

expected_platform_colors = [pct_to_hex(platform_crowd[i]) for i in range(num_coaches)]
```

**Step 2 — Compute expected colors for Train Coaches**:
```python
expected_train_colors = [pct_to_hex(train_crowd[i]) for i in range(num_coaches)]
```

**Step 3 — Parse the model's hex color outputs from structured response**:
```python
platform_color_pattern = r'Platform Zone Color Codes\s*:\s*\[([^\]]+)\]'
train_color_pattern = r'Train Coach Color Codes\s*:\s*\[([^\]]+)\]'

def parse_hex_list(text, pattern):
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return []
    return [c.strip().upper() for c in match.group(1).split(',')]

model_platform_colors = parse_hex_list(response_text, platform_color_pattern)
model_train_colors = parse_hex_list(response_text, train_color_pattern)
```

**Step 4 — Score both lists**:
```python
def score_color_list(model_colors, expected_colors, num_coaches):
    if len(model_colors) != num_coaches:
        return 0.0
    correct = sum(1 for i in range(num_coaches) if model_colors[i] == expected_colors[i])
    return correct / num_coaches

platform_score = score_color_list(model_platform_colors, expected_platform_colors, num_coaches)
train_score = score_color_list(model_train_colors, expected_train_colors, num_coaches)
score = (platform_score + train_score) / 2.0
```

**Edge cases**:
- Model outputs color names instead of hex -> 0.0
- Model outputs hex in lowercase (`#ff0000`) -> normalize to uppercase before comparison
- Model outputs a non-canonical hex (e.g., `#00FF00` instead of `#008000`) -> partial credit via nearest-threshold matching
- Model omits one of the two color lists -> that list scores 0.0, other list scored normally

---

### Reward 4: Language Consistency (`compute_language_consistency`)

**Why this matters**: Metro systems serve diverse, multilingual populations. The model should stick to English for consistency and universal accessibility.

**Mechanism: Unicode Range Detection + Non-English Word Spotting**

**Step 1 — Detect non-Latin Unicode characters**:
```python
non_latin_pattern = r'[\u0900-\u097F\u0600-\u06FF\u4E00-\u9FFF\u0980-\u09FF\u0A00-\u0A7F]'
non_latin_chars = len(re.findall(non_latin_pattern, response_text))
total_chars = len(response_text)
```

**Step 2 — Spot common non-English words that LLMs tend to inject**:
```python
NON_ENGLISH_WORDS = [
    "kripya", "yahan", "wahan", "bheed",
    "chaliye", "rukiye", "jaldi", "dhanyavaad",
    "bitte", "s'il", "por favor", "gracias",
    "danke", "arigatou", "xie xie"
]
non_english_count = sum(1 for word in response_text.lower().split() if word in NON_ENGLISH_WORDS)
```

**Step 3 — Compute score**:
```python
if total_chars == 0:
    return 0.0
non_english_ratio = (non_latin_chars + non_english_count * 5) / total_chars
if non_english_ratio < 0.01:
    score = 1.0
else:
    score = max(0.0, 1.0 - non_english_ratio * 10)
```

**Edge cases**:
- Station names that happen to be non-English words are proper nouns in the prompt — exclude words from the original prompt_text from the non-English word check.

---

### Reward 5: No-Op Detection (`compute_noop_detection`)

**Why this matters**: If the platform crowd is already roughly balanced, the model should NOT direct people to move.

**Mechanism: Standard Deviation Threshold + Language Detection**

**Step 1 — Determine if scenario is balanced**:
```python
std_dev = np.std(platform_crowd)
is_balanced = std_dev < 10

effective = [(train_crowd[i] + platform_crowd[i]) / 2.0 for i in range(num_coaches)]
effective_std = np.std(effective)
is_balanced = is_balanced or effective_std < 8
```

**Step 2-4 — Detect no-op vs directive language, score using truth table**:

| Scenario | Model Response | Score |
|----------|---------------|-------|
| Balanced + No-op | Correctly identified | **1.0** |
| Balanced + Gives directions | Unnecessary movement | **0.0** |
| Unbalanced + Gives directions | Correct behavior | **1.0** |
| Unbalanced + No-op | Missed opportunity | **0.0** |
| Ambiguous (both no-op + directive) | Partial credit | **0.3** |

---

### Reward 6: Clarity (`compute_clarity`)

**Why this matters**: Real metro announcements must be understood by a diverse public.

**Mechanism: Composite Heuristic (Readability + Jargon + Structure)**

**Component A — Sentence Length Score (30%)**:
- <=15 words/sentence: 1.0
- 15-25 words/sentence: 0.7
- >25: progressively penalized

**Component B — Jargon Penalty (30%)**:
```python
JARGON_TERMS = [
    "redistribution", "optimization", "algorithm", "percentile",
    "standard deviation", "equilibrium", "load balancing",
    "throughput", "utilization", "coefficient", "parameter",
    "infrastructure", "congestion index", "density metric",
    "probabilistic", "stochastic", "heuristic"
]
jargon_score = max(0.0, 1.0 - jargon_count * 0.25)
```

**Component C — Structure Bonus (30%)**: Checks for numbered lists, bullet points, coach headers.

**Component D — Minimum Length Check (10%)**: At least 50 characters.

**Final**: `0.3 * length_score + 0.3 * jargon_score + 0.3 * structure_score + 0.1 * length_ok`

---

### Reward 7: Sequential Direction (`compute_sequential_direction`)

**Why this matters**: Directions must be coach-by-coach ("First, Zone A... Then, Zone B..."), not simultaneous.

**Mechanism: Order Detection + Simultaneous Language Penalty**

- Find zone references in text order, check if they appear in A-F sequence
- Bonus for sequential transition words ("first", "then", "next")
- Penalty for simultaneous language ("everyone move", "all passengers", "at the same time")
- Balanced no-op scenarios return 1.0 (no directives needed)

```python
score = max(0.0, min(1.0, sequential_score + transition_bonus - simultaneous_penalty))
```

---

## Reward Aggregation

In `step()`, the total reward is a **weighted sum** of all 7 dimensions:

```python
weights = {
    'politeness': 1/7,
    'math_accuracy': 1/7,
    'color_grading': 1/7,
    'language_consistency': 1/7,
    'noop_detection': 1/7,
    'clarity': 1/7,
    'sequential_direction': 1/7,
}
total_reward = sum(weights[k] * rewards[k] for k in weights)
```

Equal weights initially. Each per-dimension score is stored in the `info` dict returned by `step()`.

For the **task graders**, the final task score (0.0-1.0) is computed as the (optionally weighted) average of per-step total rewards across all episode steps.

---

## Why Not LLM-as-Judge for Any Reward?

| Factor | Rule-Based | LLM-as-Judge |
|--------|-----------|---------------|
| **Speed** | ~1us per call | ~1-2s per call (API) or ~100ms (local) |
| **Determinism** | Identical scores for identical input | May vary between calls |
| **Cost at scale** | Free | 300 steps x 2 generations x 7 dims = 4,200 LLM calls |
| **Debuggability** | Inspect keyword lists, thresholds | Black box |
| **Suitability** | Perfect for structural/numerical checks | Better for nuanced subjective quality |

All 7 MetroPulse dimensions have objectively verifiable aspects. Rule-based heuristics are the right choice.

---

## OpenEnv Spec Compliance

### `openenv.yaml`

```yaml
name: MetroPulse
description: "Metro station crowd management RL environment. Agent receives train coach occupancy and platform zone crowd percentages, must produce polite redirection announcements with color-coded crowd indicators."
version: "1.0.0"
author: "Giridaran D, Dhiwakar Nagarajan"
tags:
  - openenv
  - crowd-management
  - metro
  - reinforcement-learning
tasks:
  - name: crowd_assessment
    description: "Map crowd percentages to correct color codes for platform zones and train coaches (easy)"
    difficulty: easy
  - name: redirection
    description: "Produce full redirection announcement with redistribution, color codes, and polite directions (medium)"
    difficulty: medium
  - name: multi_train
    description: "Manage crowd across 8 evolving train arrivals with crisis surges and temporal adaptation (hard)"
    difficulty: hard
api:
  step: /step
  reset: /reset
  state: /state
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode, returns initial observation. Accepts `{"task": "single_train\|rush_hour\|peak_crisis"}` |
| `/step` | POST | Submit action, returns observation, reward, done, info |
| `/state` | GET | Returns current episode state for debugging |

### Typed Pydantic Models

All observation, action, and reward models must be fully typed Pydantic BaseModel subclasses. The environment must pass `openenv validate`.

---

## Inference Script (`inference.py`)

**Location**: Root directory of the project (mandatory per hackathon rules).

**Requirements**:
- Uses **OpenAI Client** for all LLM calls
- Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment variables
- Runs all 3 tasks and produces reproducible scores
- Runtime < 20 minutes
- Must run on vcpu=2, memory=8gb

**Mandatory stdout format**:
```
[START] task=<task_name> env=MetroPulse model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

**Script structure**:

```python
import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from MetroPulse import MetroPulseAction, MetroPulseEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS = ["single_train", "rush_hour", "peak_crisis"]
MAX_STEPS_PER_TASK = {"single_train": 1, "rush_hour": 5, "peak_crisis": 8}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a metro station crowd management assistant.
    You receive train coach occupancy and platform zone crowd percentages.
    You must produce polite, clear, coach-by-coach redirection announcements
    with recommended platform distributions and color-coded crowd indicators.
    Follow the exact structured output format requested in each prompt.
""")

def run_task(client, env, task_name):
    obs = env.reset(task=task_name)
    max_steps = MAX_STEPS_PER_TASK[task_name]
    rewards = []

    print(f"[START] task={task_name} env=MetroPulse model={MODEL_NAME}")

    for step_num in range(1, max_steps + 1):
        # Call LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs.prompt_text}
            ],
            temperature=0.7,
            max_tokens=500
        )
        action_text = response.choices[0].message.content
        action = MetroPulseAction(response_text=action_text)

        # Step environment
        obs, reward, done, info = env.step(action)
        error = info.get("last_action_error", None)
        error_str = str(error) if error else "null"

        rewards.append(reward)
        print(f"[STEP] step={step_num} action={action_text[:80]} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}")

        if done:
            break

    success = (sum(rewards) / len(rewards)) >= 0.3  # Threshold for success
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={'true' if success else 'false'} steps={len(rewards)} rewards={rewards_str}")

def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = MetroPulseEnv()

    for task in TASKS:
        run_task(client, env, task)

if __name__ == "__main__":
    main()
```

---

## Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "-m", "MetroPulse.server", "--host", "0.0.0.0", "--port", "7860"]
```

### Hugging Face Spaces

- Space must be tagged with `openenv`
- Must respond to `reset()` ping at the Space URL with HTTP 200
- Containerized via the Dockerfile above
- Environment variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) configured as HF Space secrets

### README.md

Must include per hackathon requirements:
1. Environment description and motivation
2. Action and observation space definitions
3. Task descriptions with expected difficulty
4. Setup and usage instructions
5. Baseline scores from inference.py runs

---

## File Changes Summary

| File | Status | Key Change |
|------|--------|------------|
| `MetroPulse/models.py` | **Modify** | New Action (response_text) + Observation (crowd data, step info, 7 reward fields) |
| `MetroPulse/server/rewards.py` | **Create** | 7 standalone reward computation functions |
| `MetroPulse/server/MetroPulse_environment.py` | **Modify** | Stateful episode management: reset() with empty platform, step() with crowd evolution, state() endpoint, 3 task configs, crowd variety simulation |
| `MetroPulse/client.py` | **Modify** | Updated payloads/parsing for new models, task selection |
| `MetroPulse/server/__init__.py` | **Modify** | Export rewards module |
| `MetroPulse/openenv.yaml` | **Create** | OpenEnv metadata spec (name, tasks, API endpoints) |
| `inference.py` | **Create** | Baseline inference script with mandatory stdout format, OpenAI client, all 3 tasks |
| `Dockerfile` | **Create** | Container build for HF Spaces deployment |
| `README.md` | **Create** | Environment docs (description, spaces, tasks, setup, baseline scores) |
| `MetroPulse/metropulse_train.ipynb` | **Create** | GRPO training notebook (unsloth pattern) |

---

## Training Notebook Structure

Following the `unsloth_2048.ipynb` pattern:

| Cell | Content |
|------|---------|
| 1 | Markdown: Title + problem description |
| 2 | Install deps: `unsloth`, `trackio`, `trl`, `datasets` |
| 3 | Load model: `FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct", load_in_4bit=True, max_seq_length=1024)` |
| 4 | Apply LoRA: `r=8`, target all attention + FFN projections, `lora_alpha=16` |
| 5 | Define system prompt (metro crowd management assistant persona, format instructions, color grading rules) |
| 6 | Start MetroPulse server + connect client |
| 7 | Generate training dataset: call `env.reset()` 500 times, collect diverse scenarios as prompts |
| 8 | Define 7 GRPO reward functions importing from `MetroPulse.server.rewards` |
| 9 | GRPOConfig: `learning_rate=2e-4`, `num_generations=2`, `max_steps=300`, `report_to="trackio"` |
| 10 | Create GRPOTrainer + `trainer.train()` |
| 11 | Save model + export to GGUF for Ollama |
| 12 | Test with Ollama: create modelfile, register, run inference |
| 13 | Evaluate: 10 new scenarios, print reward breakdowns |

---

## Verification Plan

1. **OpenEnv validate**: Run `openenv validate` — must pass (spec compliance, yaml, typed models)
2. **Docker build**: `docker build -t metropulse .` + `docker run -p 7860:7860 metropulse` — must start cleanly
3. **HF Space ping**: Automated ping to Space URL — must return 200 and respond to `reset()`
4. **Environment unit test**: Call `/reset` for each task -> verify valid crowd data with correct initial state (empty platform). Call `/step` with sample text -> verify 7 reward dimensions, crowd evolution, new train generation. Call `/state` -> verify full state returned.
5. **Stateful behavior test**: Run 5 steps on `rush_hour` task -> verify platform crowd accumulates, train crowd changes each step, step counter increments, done=True after step 5.
6. **Reward function tests**: Test each `compute_*` function with known inputs (polite text -> high politeness, jargon-heavy text -> low clarity, etc.)
7. **Task grader tests**: Run all 3 graders, verify scores in 0.0-1.0 range
8. **Inference script test**: Run `inference.py` -> must complete without error, produce stdout in `[START]/[STEP]/[END]` format, scores in 0.0-1.0 range
9. **Notebook dry run**: Cells 1-8 — model loads, LoRA applies, dataset generates, rewards execute
10. **Training smoke test**: 5-10 GRPO steps, verify loss changes and trackio logs

---

## Pre-Submission Checklist

Per hackathon disqualification criteria, ALL must pass:

- [ ] HF Space deploys and returns 200 on ping
- [ ] `openenv validate` passes
- [ ] `docker build && docker run` works
- [ ] `inference.py` runs and produces scores for all 3 tasks
- [ ] 3 tasks with graders, scores in 0.0-1.0 range
- [ ] Graders are deterministic and reproducible
- [ ] Hard task genuinely challenges frontier models
- [ ] Stdout format matches `[START]/[STEP]/[END]` spec exactly
- [ ] Runtime < 20 minutes on vcpu=2, memory=8gb
- [ ] README includes all required sections
- [ ] Environment variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) properly configured
