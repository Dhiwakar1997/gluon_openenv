"""
Inference Script — MetroCrowdManager
=====================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults:
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=crowd_assessment env=MetroCrowdManager model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=Platform Zone Color... reward=0.75 done=true error=null
    [END] success=true steps=1 score=0.750 rewards=0.75
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from MetroCrowdManager.client import MetrocrowdmanagerEnv
from MetroCrowdManager.models import MetrocrowdmanagerAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "MetroCrowdManager"

TASKS = ["crowd_assessment", "redirection", "multi_train"]
MAX_STEPS = {"crowd_assessment": 1, "redirection": 1, "multi_train": 8}
TEMPERATURE = {"crowd_assessment": 0.7, "redirection": 0.2, "multi_train": 0.2}
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.3  # normalized score in [0, 1]

SYSTEM_PROMPT_EASY = textwrap.dedent("""\
    You are a metro station crowd management assistant.
    You receive train coach occupancy and platform zone crowd percentages.
    Map the crowd percentages to the correct hex color codes.
    Follow the exact structured output format requested in each prompt.
    In the announcement you must not include the percentage of the crowd in the announcement, instead you should discribe the crowd in a way that is easy to understand for a human.
""")

SYSTEM_PROMPT_FULL = textwrap.dedent("""\
    You are a metro station crowd management assistant.
    You receive train coach occupancy and platform zone crowd percentages.
    You must produce polite, clear, coach-by-coach redirection announcements
    with recommended platform distributions and color-coded crowd indicators.
    Follow the exact structured output format requested in each prompt.
    In the announcement you must not include the percentage of the crowd in the announcement, instead you should describe the crowd in a way that is easy to understand for a human.

    ## Distribution Calculation (CRITICAL — follow exactly)

    1. For each coach, compute available capacity: capacity_i = 100 - coach_occupancy_i
    2. Sum all capacities: total_capacity = sum of all capacity_i
    3. Sum all current platform crowd values: total_platform = sum of all platform_crowd_i
    4. For each zone, compute: recommended_i = total_platform * (capacity_i / total_capacity)
    5. VERIFY: your recommended values MUST sum to total_platform (conserve passengers).

    ### Worked Example (4 coaches)
    Coach occupancies: [80, 20, 60, 40]
    Platform crowd: [30, 50, 20, 40]
    Step 1 — Capacities: [20, 80, 40, 60] (each is 100 minus occupancy)
    Step 2 — Total capacity: 20+80+40+60 = 200
    Step 3 — Total platform: 30+50+20+40 = 140
    Step 4 — Distribution: [140*(20/200), 140*(80/200), 140*(40/200), 140*(60/200)]
             = [14.0, 56.0, 28.0, 42.0]
    Step 5 — Check: 14+56+28+42 = 140 ✓ (matches total platform)

    ## Announcement Guidelines
    - Begin your announcement by addressing passengers on the specific platform number provided (e.g., "Passengers on Platform 3, may we have your attention please").
    - Reference coaches in alphabetical order: Coach A first, then Coach B, then Coach C, etc.
    - Structure with numbered steps or coach-by-coach headers (e.g., "Coach A: ...").
    - Keep sentences short (under 15 words each) and avoid technical jargon.
    - Be polite: use "please", "kindly", "we recommend", "for your comfort".
    - Describe crowded coaches as "quite busy" or "full" and empty ones as "spacious" or "plenty of room".
    - Start the announcement with a polite greeting and then describe the crowd in a way that is easy to understand for a human and mentions the platform zones in the announcement.
    - Avoid specifying the exact number of passengers in the announcement, instead describe the crowd in a way that is easy to understand for a human.

""")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def get_model_response(
    client: OpenAI,
    messages: List[dict],
    temperature: float,
) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else ""
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, env: MetrocrowdmanagerEnv, task_name: str) -> None:
    """Run a single task and emit stdout logs."""
    max_steps = MAX_STEPS[task_name]
    temperature = TEMPERATURE[task_name]
    system_prompt = SYSTEM_PROMPT_EASY if task_name == "crowd_assessment" else SYSTEM_PROMPT_FULL

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Accumulate conversation history for multi-step tasks
    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        prompt_text = obs.prompt_text

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Add current observation as user message
            messages.append({"role": "user", "content": prompt_text})

            action_text = get_model_response(client, messages, temperature)

            if not action_text:
                # Remove the failed user message so history stays clean
                messages.pop()
                log_step(step=step, action="", reward=0.0, done=step >= max_steps, error="empty model response")
                rewards.append(0.0)
                steps_taken = step
                if step >= max_steps:
                    break
                continue

            # Keep assistant response in history for multi-step context
            messages.append({"role": "assistant", "content": action_text})

            action = MetrocrowdmanagerAction(response_text=action_text)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            # Truncate action text for log readability
            action_log = action_text.replace("\n", " ")[:80]
            log_step(step=step, action=action_log, reward=reward, done=done, error=error)

            # Update prompt for next step
            prompt_text = obs.prompt_text

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        env = await MetrocrowdmanagerEnv.from_docker_image(IMAGE_NAME)
        await run_task(client, env, task)


if __name__ == "__main__":
    asyncio.run(main())
