"""
Inference Script — MetroCrowdManager (Agentic MCP edition)
==========================================================

MANDATORY environment variables:
    API_BASE_URL    The OpenAI-compatible API endpoint for the LLM.
    MODEL_NAME      Model identifier.
    HF_TOKEN        Hugging Face / API key.
    LOCAL_IMAGE_NAME  Optional: name of a local Docker image for the env.

Defaults:
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME   = "Qwen/Qwen3-1.7B-Instruct"

The agent now talks to MetroCrowdManager via MCP tool calls. Each
episode runs an agentic loop: the model emits ``<tool_call>...</tool_call>``
blocks, the env runs the tool, results are fed back, and once the model
emits a final answer (no tool call) we submit a ``SubmitResponseAction``.

STDOUT contract (unchanged from Round 1):
    [START] task=<task> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<truncated_text> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List, Optional

from openai import OpenAI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
PACKAGE_ROOT = os.path.dirname(__file__)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from client import MetrocrowdmanagerEnv  # noqa: E402
from training.rollout import (  # noqa: E402
    SYSTEM_PROMPTS,
    agentic_episode_async,
)


IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3-1.7B-Instruct"
BENCHMARK = "MetroCrowdManager"

TASKS = ["ticket_booking", "ticket_issuance", "crowd_announcement"]
MAX_TURNS_PER_STEP: Dict[str, int] = {
    "ticket_booking": 12,
    "ticket_issuance": 8,
    "crowd_announcement": 8,
}
TEMPERATURE = {
    "ticket_booking": 0.4,
    "ticket_issuance": 0.2,
    "crowd_announcement": 0.3,
}
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Logging helpers (preserved stdout contract)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Async generation wrapper around OpenAI SDK
# ---------------------------------------------------------------------------


def _make_generator(client: OpenAI, temperature: float):
    async def generate(messages: List[Dict[str, str]]) -> str:
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover — graceful degradation
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
            return ""

    return generate


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


async def run_task(
    client: OpenAI, env: MetrocrowdmanagerEnv, task_name: str
) -> None:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error: Optional[str] = None

    try:
        result = await agentic_episode_async(
            env,
            task_name,
            generate=_make_generator(client, TEMPERATURE[task_name]),
            max_turns=MAX_TURNS_PER_STEP[task_name],
        )

        rewards = result.per_step_rewards or [result.reward]
        steps_taken = len(rewards)
        for i, (r, breakdown) in enumerate(
            zip(rewards, result.per_step_breakdowns or [{}] * len(rewards)), start=1
        ):
            action_log = (result.final_text or "").replace("\n", " ")[:80]
            done_flag = i == len(rewards)
            log_step(step=i, action=action_log, reward=r, done=done_flag, error=None)

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:  # pragma: no cover
        error = str(exc)
        log_step(step=steps_taken + 1, action="", reward=0.0, done=True, error=error)
    finally:
        try:
            await env.close()
        except Exception as e:  # pragma: no cover
            print(f"[DEBUG] env.close() error: {e}", flush=True)
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
