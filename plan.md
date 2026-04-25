# Plan: Agentic MCP Refactor of MetroCrowdManager for OpenEnv Hackathon Finals

## Context

**Why this change:** We cleared Round 1 with a non-agentic MetroCrowdManager environment (three tasks centered on crowd redirection announcements). For the finals (April 25–26, 2026, Apr '26 OpenEnv Hackathon India), we are re-scoping the environment around agentic workflows using MCP tools. Judges weight Environment Innovation at 40% — this pivot is our main innovation lever. The new tasks model a realistic passenger-journey-as-agentic-workflow: (1) booking a ticket via conversation, (2) issuing a structured ticket with crowd intel, (3) multi-train crowd management. All three require the LLM to orchestrate MCP tools (fetch, calculate, decide) rather than pattern-match a single prompt.

**Intended outcome:** A fully OpenEnv-compliant `MCPEnvironment` hosted on HuggingFace Spaces, with three agentic tasks, a GRPO training script in Colab (Unsloth + TRL, Qwen3-1.7B on T4), visible reward-curve improvement, and a 2-minute demo video showing baseline-vs-trained behavior on the same scenario. The existing 11-dimension reward system is preserved and extended with agentic rewards (tool sequence, tool fidelity, tool economy, format compliance, information sufficiency, payment discipline, ticket schema validity).

**Compute constraint:** T4 / Colab-free (~15 GB VRAM, ~12 h). Plan uses Qwen3-1.7B + QLoRA + Unsloth + short rollouts.

---

## 1. Architecture Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Base class | Switch `Environment` → `MCPEnvironment` (openenv 0.1.13, `openenv/core/env_server/mcp_environment.py`) | Judge criteria explicitly call out `MCPEnvironment / Environment base classes` ([judge.md:158](judge.md#L158)); gives us free `ListToolsAction` / `CallToolAction` routing; lets us register tools via `FastMCP` decorators. |
| Final-response signal | Custom `SubmitResponseAction` handled in `_step_impl` | Clean separation — tool calls via `CallToolAction`, final submission via `SubmitResponseAction`. Rewards computed only at submit time. |
| Passenger simulation (Task 1) | Scripted state machine with randomized phrasing templates | Deterministic → stable rewards. 5–10 phrasing variants per state prevents overfitting. No second LLM in the rollout loop → T4-compatible. |
| Task 3 scope | 3–4 train arrivals (reduced from 8) | Each step now fans out into ~4 tool calls. Keeps rollout budget realistic on T4. |
| `get_ideal_zone` vs `get_ideal_distribution` | Keep as **two separate tools** (per user) | Simpler per-tool logic, less wrong-tool ambiguity. |
| Payment polling | Variable ticks (2–5) + 10–15% failure rate | Makes polling a real decision problem, not a trivial loop. |
| Passenger count | Required before payment, scales ticket cost | Creates a clean `info_sufficiency` reward signal. |
| Task 2 output | Structured JSON ticket | Programmatic verification, hard to game. |
| Task naming | `ticket_booking`, `ticket_issuance`, `crowd_announcement` | Matches new domain; update `openenv.yaml`, inference, training. |
| Reward strategy | Reuse existing 11 dims where applicable, add 7 new agentic rewards | Preserves anti-hacking design, extends with orchestration signals. |
| Demo format | Side-by-side baseline vs trained on same scenario | Strongest evidence signal for the 30% storytelling + 20% improvement criteria. |
| Tool-call format | Follow `demoMCP.py` convention: `<tool_call>{"name":..,"arguments":{..}}</tool_call>` | Works with Qwen chat templates; parser already exists in `demoMCP.py:46–72` to reuse. |

---

## 2. Files to Create / Modify / Delete

### Modify

- **[MetroCrowdManager/openenv.yaml](MetroCrowdManager/openenv.yaml)** — replace 3 task entries with new names + descriptions; bump spec if needed.
- **[MetroCrowdManager/models.py](MetroCrowdManager/models.py)** — redesign `MetrocrowdmanagerAction` / `MetrocrowdmanagerObservation`; add `SubmitResponseAction`; extend `State`.
- **[MetroCrowdManager/server/MetroCrowdManager_environment.py](MetroCrowdManager/server/MetroCrowdManager_environment.py)** — subclass `MCPEnvironment`; implement `_step_impl` for `SubmitResponseAction`; register tools via `FastMCP`; add per-task scenario generators and the scripted passenger state machine.
- **[MetroCrowdManager/server/rewards.py](MetroCrowdManager/server/rewards.py)** — keep existing 11 reward functions; add 7 new reward functions (see §4).
- **[MetroCrowdManager/server/app.py](MetroCrowdManager/server/app.py)** — minor updates to handle new action types if needed (MCP routing is automatic).
- **[MetroCrowdManager/client.py](MetroCrowdManager/client.py)** — update `_step_payload` / `_parse_result` for new action/observation shapes.
- **[MetroCrowdManager/inference.py](MetroCrowdManager/inference.py)** — rewrite rollout loop for agentic multi-turn (tool calls + final submit); reuse parser patterns from [demoMCP.py:46–72](demoMCP.py#L46-L72).
- **[README.md](README.md)** — rewrite to describe new agentic environment, tools, reward plots, HF Space link, demo video link.

### Create

- **`MetroCrowdManager/server/tools.py`** — all MCP tool implementations (12 tools across 3 tasks), registered on a single `FastMCP` instance in the environment constructor.
- **`MetroCrowdManager/server/passenger_sim.py`** — scripted passenger state machine for Task 1 (states, phrasing templates, transition logic).
- **`MetroCrowdManager/server/scenarios.py`** — per-task scenario generators (station layout, passenger goal, crowd state, payment-failure flag).
- **`MetroCrowdManager/server/agentic_rewards.py`** — 7 new reward functions (kept separate from `rewards.py` for clarity; or merged if you prefer one file).
- **`training/train_grpo.py`** — TRL GRPO + Unsloth training script, adapted from [demoMCP.py:286–491](demoMCP.py#L286-L491). Lives outside the env package so the Space container stays small.
- **`training/train_grpo.ipynb`** — Colab notebook wrapping `train_grpo.py` (judge requirement: Colab-runnable).
- **`training/rollout.py`** — shared agentic rollout helper used by both `inference.py` and `train_grpo.py`. Implements the multi-turn tool loop. Deduplicates code between train and infer.
- **`docs/`** — a `plots/` subdirectory for committed PNG reward curves ([judge.md:137](judge.md#L137) — plots must be committed, not only in Colab).

### Delete

- **demoMCP.py** — port useful pieces (parser, system-prompt scaffold, GRPO config, rollout skeleton) into `training/train_grpo.py` and `training/rollout.py`, then remove.
- Current `_evolve_crowd`, `_build_prompt`, `_BASELINE_SCENARIOS` logic in `MetroCrowdManager_environment.py` — replaced by `scenarios.py` + new per-task logic.

---

## 3. MCP Tool Catalog (12 tools)

All tools are registered on a single `FastMCP` instance in the env constructor. All return simulated data from in-memory state; no network calls. Tools are **stateful per-episode** (tied to `self._state`) so that `initiate_payment` → `check_payment_status` can share state.

**Shared across tasks:**
1. `get_platform_for_destination(destination: str) -> dict` — returns `{"platform": int, "found": bool}`. Uses a per-episode station-to-platform map. (Tasks 2, 3.)
2. `get_platform_crowd(platform: int) -> dict` — returns `{"zones": [int; 10]}` (per-zone crowd %). (Tasks 2, 3.)
3. `get_train_crowd_occupation(platform: int) -> dict` — returns `{"coaches": [int; 10]}` (per-coach occupancy %). (Tasks 2, 3.)
4. `get_current_time() -> dict` — returns `{"time": "HH:MM"}`. (Task 2.)

**Task 1 (ticket_booking):**
5. `validate_destination(destination: str) -> dict` — returns `{"valid": bool, "normalized": str | null}`. Valid stations drawn from an 8-station per-episode list; allow fuzzy match for realism.
6. `get_ticket_cost(source: str, destination: str, passenger_count: int) -> dict` — returns `{"cost": float, "currency": "INR"}`. Cost = `base_rate * distance * passenger_count`.
7. `initiate_payment(amount: float, passenger_count: int) -> dict` — returns `{"payment_id": str, "status": "pending"}`. Sets internal state `payment_ticks_remaining = randint(2,5)`, `payment_will_fail = random() < 0.12`.
8. `check_payment_status(payment_id: str) -> dict` — decrements ticks; returns `"pending"` → `"success"` or `"failed"` after ticks exhausted.

**Task 2 (ticket_issuance):**
9. `get_ideal_zone(platform: int, single_passenger: bool = True) -> dict` — returns `{"zone": "A"–"J", "reasoning": str}`. Uses same ideal-distribution algorithm as existing `_compute_ideal()` in [rewards.py:183–216](MetroCrowdManager/server/rewards.py#L183-L216), reduced to a single best zone.

**Task 3 (crowd_announcement):**
10. `get_ideal_distribution(platform: int) -> dict` — returns `{"distribution": [int; 10]}`. Reuses existing `_compute_ideal()` directly.

**Utility:**
11. `list_valid_stations() -> dict` — returns `{"stations": [str]}`. Helps the agent if it needs to re-check a destination. (Tasks 1 primarily.)
12. `get_passenger_request() -> dict` — Task 1 only. Returns the passenger's current utterance (from the scripted state machine). This is how the agent "hears" the passenger; the passenger never speaks outside of tool responses. Alternative: put passenger utterances in observations between turns — decide during implementation. **Leaning toward putting passenger utterances in the observation `metadata.passenger_message`** rather than a tool; tool-driven makes the model fetch dialogue, which is unnatural. (Revisit in implementation.)

**Tool naming guardrails:** None of these conflict with reserved names (`reset`, `step`, `state`, `close`).

**Mode:** All tools registered as `@self.tool(mode="simulation")` to match `MCPEnvironment` semantics (production mode unused — no real metro API).

---

## 4. Reward Design

### Reuse (from `rewards.py`, applicable tasks noted)

| Existing reward | Used in | Notes |
|---|---|---|
| `compute_politeness` | T1 | Small weight (≤5%) — it's hackable; don't over-index. |
| `compute_clarity` | T1, T3 | |
| `compute_language_consistency` | T1, T2, T3 | |
| `compute_color_grading` | T3 | Announcements still use color codes. |
| `compute_distribution_accuracy` | T3 | |
| `compute_conservation_accuracy` | T3 | |
| `compute_feasibility_accuracy` | T3 | |
| `compute_noop_detection` | T3 | Balanced-scenario handling preserved. |
| `compute_factual_accuracy` | T3 | |
| `compute_platform_mention` | T3 | |
| `compute_sequential_direction` | T3 | |

### New (add to `agentic_rewards.py`)

1. **`tool_sequence_reward(turn_history, task_name)`** — Did the agent call tools in a valid order for the task? E.g., for T1: `validate_destination` must precede `get_ticket_cost`, which must precede `initiate_payment`. Returns `1.0` if full expected-order subsequence present, partial credit otherwise.
2. **`tool_fidelity_reward(turn_history)`** — Did the agent use the **actual values** returned by earlier tools in later tool calls? E.g., if `validate_destination` returned `normalized="Central Metro"`, subsequent `get_ticket_cost(destination=...)` must use `"Central Metro"` verbatim, not a hallucinated string. Compares tool call arguments to prior tool results (JSON equality on relevant fields).
3. **`tool_economy_reward(turn_history, task_name)`** — Penalize excess calls. For each task we define `expected_call_count` (T1: ~5 minimum, T2: 5 minimum, T3: ~4 per step). Reward = `min(1.0, expected / actual)` clipped. Prevents spamming `check_payment_status`.
4. **`format_reward(raw_text_per_turn)`** — Did every turn either produce a parseable `<tool_call>{...}</tool_call>` or a final submission with no tool call? Returns fraction of well-formed turns. Reuses parser from [demoMCP.py:46–72](demoMCP.py#L46-L72).
5. **`info_sufficiency_reward(turn_history)`** — T1 specific. Before `initiate_payment`, agent must have (a) successfully called `validate_destination` AND (b) collected passenger count (inferred from `get_ticket_cost` args or conversation). Returns `1.0` if both preconditions met, `0.0` if payment initiated prematurely.
6. **`payment_discipline_reward(turn_history, final_text)`** — T1 specific. (a) On payment failure, did the agent communicate clearly to the user and offer retry/alternative? (b) Was `check_payment_status` polled with reasonable cadence (not 20× in a row, not 0 times)? Composite of two sub-scores averaged.
7. **`ticket_schema_validity(final_text)`** — T2 specific. Is the final output valid JSON with required fields (`time`, `from`, `to`, `price`, `platform`, `ideal_zone`)? Returns `1.0` for valid schema + correct values (cross-check against tool return values), `0.5` for valid schema + wrong values, `0.0` for parse failure.

### Task-specific composition

**T1 (ticket_booking)** — total reward on submit:
```
0.20 * tool_sequence
+ 0.15 * tool_fidelity
+ 0.10 * tool_economy
+ 0.10 * format
+ 0.20 * info_sufficiency          # NEW, heaviest non-sequence signal
+ 0.15 * payment_discipline
+ 0.05 * politeness
+ 0.05 * clarity
```

**T2 (ticket_issuance)**:
```
0.25 * tool_sequence
+ 0.20 * tool_fidelity
+ 0.10 * tool_economy
+ 0.10 * format
+ 0.30 * ticket_schema_validity
+ 0.05 * language_consistency
```

**T3 (crowd_announcement)** — per-step reward, averaged across 3–4 train arrivals:
```
0.15 * tool_sequence
+ 0.10 * tool_fidelity
+ 0.05 * tool_economy
+ 0.05 * format
+ 0.20 * distribution_accuracy
+ 0.10 * conservation_accuracy
+ 0.05 * feasibility_accuracy
+ 0.05 * color_grading
+ 0.05 * factual_accuracy
+ 0.05 * platform_mention
+ 0.05 * noop_detection
+ 0.05 * sequential_direction
+ 0.05 * clarity
```

All weights sum to 1.0. All component rewards ∈ `[0, 1]`.

### Curriculum note

Per [hackathonGuide.md:62–68](hackathonGuide.md#L62-L68), we must make success possible early. **Start training with T2** (most structured, cleanest tool chain, highest floor). Once T2 shows rising reward curves, add T1 (hardest due to multi-turn conversation). T3 last.

---

## 5. Action / Observation Schema

### Actions

- **`CallToolAction`** (inherited from `openenv.core.env_server.mcp_types`) — `tool_name: str`, `arguments: dict`. Routed automatically by `MCPEnvironment.step()`.
- **`ListToolsAction`** (inherited) — for agent to discover tools. Routed automatically.
- **`SubmitResponseAction`** (new, in `models.py`) — `type: Literal["submit_response"]`, `content: str`. Handled by our `_step_impl()`. This is the only action that can terminate an episode (for T1/T2) or advance a step (T3).

### Observations

- **`CallToolObservation`** (inherited) — returned for every tool call.
- **`MetrocrowdmanagerObservation`** (redesigned) — returned after `SubmitResponseAction` or on reset. Fields:
  - `prompt_text: str` — task-specific system+user prompt (including passenger utterance for T1).
  - `task_name: str`
  - `current_step: int`, `max_steps: int`
  - `done: bool`, `reward: float`
  - `metadata: dict` — includes `passenger_message` (T1), `scenario_summary`, `reward_breakdown` (post-submit).

### State

Extend existing `State` with agentic fields:
- `turn_history: list[dict]` — per-turn record (tool calls issued, tool results, text generated).
- `tool_call_log: list[dict]` — chronological log of all tool calls and their results (used by reward functions).
- `passenger_state: str` — T1 state machine state.
- `payment_state: dict` — `{payment_id, ticks_remaining, will_fail, status}`.
- `scenario: dict` — immutable per-episode (station map, crowd ground truth, passenger destination goal).

---

## 6. Rollout Loop (shared by training and inference)

Location: `training/rollout.py`.

```
async def agentic_episode(client, task_name, model_generate_fn, max_turns=10):
    obs = await client.reset(task_name=task_name)
    history = []  # chat messages
    history.append({"role": "system", "content": SYSTEM_PROMPT_FOR_TASK[task_name]})
    history.append({"role": "user", "content": obs.prompt_text})

    for turn in range(max_turns):
        text = await model_generate_fn(history)
        tool_calls = parse_tool_calls(text)  # from demoMCP.py

        if tool_calls:
            for tc in tool_calls:
                tool_obs = await client.step(CallToolAction(tool_name=tc["name"],
                                                            arguments=tc["arguments"]))
                history.append({"role": "assistant", "content": text})
                history.append({"role": "user",
                                "content": f"<tool_result>{tool_obs.result.data}</tool_result>"})
        else:
            # Treat as final submission
            submit_obs = await client.step(SubmitResponseAction(content=text))
            return {
                "reward": submit_obs.reward,
                "reward_breakdown": submit_obs.metadata["reward_breakdown"],
                "history": history,
                "done": submit_obs.done,
            }

    # Force-submit if no final response by max_turns
    submit_obs = await client.step(SubmitResponseAction(content=history[-1]["content"]))
    return {...}
```

For T3, wrap this in an outer loop over 3–4 train arrivals; pass the accumulated state between arrivals.

---

## 7. Training Plan (T4-compatible)

**Stack:** Unsloth `FastLanguageModel` with Qwen3-1.7B-Instruct (4-bit), TRL `GRPOTrainer`, vLLM backend for fast generation, `max_completion_length=256` per turn, `max_prompt_length=2048`.

**Training order (curriculum):**
1. **Phase A (T2 only, ~50 episodes):** Easiest agentic task. Goal: verify the training loop works end-to-end and the model learns `format_reward` + `tool_sequence_reward`. Success criterion: average reward rises from ~0.2 baseline to ≥0.5.
2. **Phase B (T1 only, ~50 episodes):** Harder — multi-turn conversation. Watch `info_sufficiency` and `payment_discipline` closely. Success: rising curve, no premature payment initiation.
3. **Phase C (mixed T1+T2+T3, ~100 episodes):** Combined. Shuffled per batch. This is the production run for demo plots.

**Batch size:** 2 (T4 memory limit). **Gradient accumulation:** 8. **Learning rate:** 5e-6. **Max steps:** 200.

**Checkpoints:** Save LoRA adapters (NOT merged) every 50 steps. Per [hackathonGuide.md:184–190](hackathonGuide.md#L184-L190), do not upcast 4-bit → 16-bit and merge naively — keep adapters separate and load via `from_pretrained` + adapter merge at inference.

**Metrics logged (wandb or CSV):** per-reward-function curves (not just total), `tools_per_episode`, `episode_length`, `format_failure_rate`. Plots committed as PNG to `docs/plots/` — not only in Colab cells per [judge.md:137](judge.md#L137).

**Baseline:** Run `inference.py` with an untrained Qwen3-1.7B on all three tasks, 20 episodes each. Save `baseline_scores.json`. Compare trained-model scores side-by-side.

---

## 8. Deployment to HuggingFace Space

Per [hackathonGuide.md:138–153](hackathonGuide.md#L138-L153) — deploy early. Do this before heavy training.

1. Verify `openenv.yaml` manifest valid (spec_version, app, port).
2. `openenv push` to HF Space (creates Space + Docker image + git repo).
3. Smoke test: client connects to remote Space, reset/list_tools/call_tool/submit works.
4. Pin Space URL in README.
5. README must include Space link, video link, plot PNGs, Colab link per [judge.md:104–108](judge.md#L104-L108).

**Space size note:** Do not commit video files to the Space. Link externally per [judge.md:108](judge.md#L108).

---

## 9. Demo Plan (30% storytelling weight)

**Target: 2-minute YouTube video.**

Structure (rough cut):
1. (0:00–0:15) Hook — "Can an LLM act as a metro station assistant? We trained one via RL, here's the difference."
2. (0:15–0:45) Environment overview — show the 3 tasks, tools available, one scenario from each.
3. (0:45–1:30) Side-by-side comparison — same passenger scenario played through untrained vs trained model, split-screen. Baseline fails (e.g., initiates payment before collecting destination); trained model handles it correctly and politely.
4. (1:30–1:50) Reward curves — Phase A/B/C plots with captions.
5. (1:50–2:00) Call to action — Space URL, GitHub, how to run locally.

README must link the video URL per [judge.md:107](judge.md#L107).

---

## 10. Risk Register & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| T4 OOMs on GRPO + MCP tool loop | High | Reduce batch size to 1 if needed; cap `max_turns` at 8; cap tool-result string length to 512 chars. |
| Reward hacking via `<tool_call>` spam | Medium | `tool_economy_reward` caps per-episode calls; `format_reward` penalizes malformed. |
| Scripted passenger makes model overfit to templates | Medium | ≥5 phrasing variants per state; randomize slot values (destination, count); use held-out variants for eval. |
| `MCPEnvironment` async / event-loop issues during training rollouts | Medium | Use `step_async` on WebSocket path per [mcp_environment.py:588–609](.venv/.../mcp_environment.py#L588-L609); run env locally during training (not over HTTP) to avoid latency. |
| Tool fidelity check too strict (false negatives from string-normalization) | Low | Use fuzzy match (casefold + strip); log mismatches for inspection. |
| Running out of time | High | Phase A alone is a valid submission. Ship T2 first; T1 and T3 are upgrades. |
| LoRA save corrupts model | Medium | Save adapters only (no merge); test `inference.py` with adapters immediately after training. |

---

## 11. Execution Order (Apr 24 planning → Apr 25–26 onsite)

**Pre-onsite (Apr 24):**
- Day 1 (today) — Confirm plan, set up branch, draft tools + passenger sim + scenarios.
- Implement `MCPEnvironment` subclass + `SubmitResponseAction` + new action/obs models.
- Wire tools into `FastMCP`; local smoke test (reset → list_tools → call_tool → submit).
- Port parser and system prompts from `demoMCP.py`; delete `demoMCP.py`.
- Implement all rewards (existing reused + 7 new).
- Rewrite `inference.py` for agentic loop; run baseline on T2 → sanity-check rewards.

**Onsite Day 1 (Apr 25):**
- Deploy to HF Space; verify remote client works.
- Phase A training (T2) — 1–2 hours on Colab T4.
- Inspect generations — look for reward hacking per [hackathonGuide.md:87–101](hackathonGuide.md#L87-L101).
- Phase B training (T1) — 2–3 hours.
- Begin README + video assets.

**Onsite Day 2 (Apr 26):**
- Phase C training (mixed, final).
- Generate plots, commit to `docs/plots/`.
- Record demo video (side-by-side baseline vs trained).
- Write/polish README, mini-blog.
- Final submission: Space URL in submission form.

---

## 12. Verification Checklist

Before declaring done:

- [ ] `MetroCrowdManager/server/MetroCrowdManager_environment.py` subclasses `MCPEnvironment` and implements `_step_impl`.
- [ ] `openenv validate` (or equivalent) passes on `openenv.yaml`.
- [ ] Local smoke: `python -m server.app` → client connects → `ListToolsAction` returns ≥12 tools → each tool callable → `SubmitResponseAction` returns reward + breakdown.
- [ ] Baseline eval: `python MetroCrowdManager/inference.py --task ticket_booking` produces `[START]`/`[STEP]`/`[END]` lines in correct format and score in `[0, 1]`.
- [ ] Training script runs for ≥10 steps without OOM on T4.
- [ ] At least Phase A (T2) shows rising reward curve saved as PNG.
- [ ] Reward breakdown logged per step — inspect for signs of hacking (any single reward pinned at 1.0 while others low = suspicious).
- [ ] HF Space deployed, URL works, client-from-URL works.
- [ ] README has: motivation, task descriptions, tool list, reward design, plots, Space URL, video URL, Colab URL, run-locally instructions.
- [ ] 2-minute video published on YouTube (public, unlisted OK).
- [ ] Colab notebook runnable end-to-end on a fresh T4 (per [judge.md:102](judge.md#L102)).
- [ ] No reserved tool names used (reset, step, state, close).
- [ ] No large video files committed to the Space (link externally).
- [ ] LoRA adapters saved separately; `inference.py` can load them without re-merging.

---

## 13. Critical Files Reference

### To modify
- [MetroCrowdManager/openenv.yaml](MetroCrowdManager/openenv.yaml)
- [MetroCrowdManager/models.py](MetroCrowdManager/models.py)
- [MetroCrowdManager/client.py](MetroCrowdManager/client.py)
- [MetroCrowdManager/server/app.py](MetroCrowdManager/server/app.py)
- [MetroCrowdManager/server/MetroCrowdManager_environment.py](MetroCrowdManager/server/MetroCrowdManager_environment.py)
- [MetroCrowdManager/server/rewards.py](MetroCrowdManager/server/rewards.py)
- [MetroCrowdManager/inference.py](MetroCrowdManager/inference.py)
- [README.md](README.md)

### To create
- `MetroCrowdManager/server/tools.py`
- `MetroCrowdManager/server/passenger_sim.py`
- `MetroCrowdManager/server/scenarios.py`
- `MetroCrowdManager/server/agentic_rewards.py`
- `training/train_grpo.py`
- `training/train_grpo.ipynb`
- `training/rollout.py`
- `docs/plots/` (directory)

### To delete
- [demoMCP.py](demoMCP.py) (after porting parser + GRPO config to `training/`)

### Code to reuse (do not rewrite)
- Tool-call parser: [demoMCP.py:46–72](demoMCP.py#L46-L72)
- System prompt scaffold: [demoMCP.py:125–162](demoMCP.py#L125-L162)
- GRPOConfig + Unsloth setup: [demoMCP.py:408–491](demoMCP.py#L408-L491)
- Multi-reward GRPO integration: [demoMCP.py:286–401](demoMCP.py#L286-L401)
- All 11 reward functions: [MetroCrowdManager/server/rewards.py](MetroCrowdManager/server/rewards.py)
- Ideal-distribution algorithm: [rewards.py:183–216](MetroCrowdManager/server/rewards.py#L183-L216) (called by `get_ideal_zone` and `get_ideal_distribution` tools)
- Inference stdout format: [MetroCrowdManager/inference.py](MetroCrowdManager/inference.py) (keep the `[START]`/`[STEP]`/`[END]` log format; rewrite the loop body)
- Client scaffolding: [MetroCrowdManager/client.py](MetroCrowdManager/client.py) (tweak payload parsing; keep `EnvClient` subclass pattern)
