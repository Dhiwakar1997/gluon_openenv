"""
Inference Script — MetroCrowdManager
=====================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults:
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME   = "Qwen/Qwen2.5-72B-Instruct"

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import textwrap

from openai import OpenAI

# ---------------------------------------------------------------------------
# Import environment directly (no server needed for inference)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MetrocrowdmanagerAction
from server.MetroCrowdManager_environment import MetrocrowdmanagerEnvironment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS = ["crowd_assessment", "redirection", "multi_train"]
MAX_STEPS = {"crowd_assessment": 1, "redirection": 1, "multi_train": 8}

SYSTEM_PROMPT_EASY = textwrap.dedent("""\
    You are a metro station crowd management assistant.
    You receive train coach occupancy and platform zone crowd percentages.
    Map the crowd percentages to the correct hex color codes.
    Follow the exact structured output format requested in each prompt.
""")

SYSTEM_PROMPT_FULL = textwrap.dedent("""\
    You are a metro station crowd management assistant.
    You receive train coach occupancy and platform zone crowd percentages.
    You must produce polite, clear, coach-by-coach redirection announcements
    with recommended platform distributions and color-coded crowd indicators.
    Follow the exact structured output format requested in each prompt.
""")

TEMPERATURE = 0.7
MAX_TOKENS = 500
SUCCESS_THRESHOLD = 0.3


def _bool_str(val: bool) -> str:
    return "true" if val else "false"


def run_task(client: OpenAI, env: MetrocrowdmanagerEnvironment, task_name: str) -> None:
    """Run a single task and emit stdout logs."""
    obs = env.reset(task=task_name)
    max_steps = MAX_STEPS[task_name]
    rewards: list[float] = []

    system_prompt = SYSTEM_PROMPT_EASY if task_name == "crowd_assessment" else SYSTEM_PROMPT_FULL

    print(f"[START] task={task_name} env=MetroCrowdManager model={MODEL_NAME}")

    for step_num in range(1, max_steps + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": obs.prompt_text},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            action_text = response.choices[0].message.content or ""
        except Exception as e:
            action_text = ""
            error_msg = str(e)
            print(
                f"[STEP] step={step_num} action= reward=0.00 "
                f"done={_bool_str(step_num >= max_steps)} error={error_msg}"
            )
            rewards.append(0.0)
            if step_num >= max_steps:
                break
            continue

        action = MetrocrowdmanagerAction(response_text=action_text)
        obs = env.step(action)

        reward = obs.reward if obs.reward is not None else 0.0
        done = obs.done
        error = obs.metadata.get("last_action_error", None)
        error_str = str(error) if error else "null"

        # Truncate action text for log readability
        action_log = action_text.replace("\n", " ")[:80]

        rewards.append(float(reward))
        print(
            f"[STEP] step={step_num} action={action_log} "
            f"reward={reward:.2f} done={_bool_str(done)} error={error_str}"
        )

        if done:
            break

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    success = avg_reward >= SUCCESS_THRESHOLD
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={_bool_str(success)} steps={len(rewards)} rewards={rewards_str}")


def main() -> None:

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = MetrocrowdmanagerEnvironment()

    for task in TASKS:
        run_task(client, env, task)


if __name__ == "__main__":
    main()
