"""
Local rollout smoke test against a real LLM.

By default this connects to a running OpenEnv FastAPI server (the
container you started with `docker run -p 8000:8000 ...`) so every tool
call hits the actual HTTP/WebSocket path the HF Space will use. Pass
``--in-process`` to skip Docker and use an in-memory env instead.

Usage:
    # Make sure the server is up first:
    docker run -d -p 8000:8000 --name metro-env openenv-metrocrowdmanager

    export HF_TOKEN=hf_...                     # or API_KEY=sk-...
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    python scripts/test_rollout.py ticket_booking
    python scripts/test_rollout.py ticket_issuance --seed 7 --turns 8
    python scripts/test_rollout.py crowd_announcement --show-turns

    # Skip Docker and run against an in-process env:
    python scripts/test_rollout.py ticket_booking --in-process
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MetroCrowdManager"))

from openai import OpenAI

from MetroCrowdManager.client import MetrocrowdmanagerEnv
from training.rollout import agentic_episode_async, SYSTEM_PROMPTS  # noqa: F401


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"


def build_generator(client: OpenAI, temperature: float, max_tokens: int):
    async def generate(messages):
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[gen] error: {exc}", flush=True)
            return ""
    return generate


async def run_rollout(env, args, gen):
    return await agentic_episode_async(
        env,
        args.task,
        gen,
        seed=args.seed,
        max_turns=args.turns,
    )


async def main(args):
    if not API_KEY:
        print("WARN: HF_TOKEN/API_KEY not set — model calls will likely fail.")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    gen = build_generator(client, args.temperature, args.max_tokens)

    if args.in_process:
        from MetroCrowdManager.server.MetroCrowdManager_environment import (
            MetrocrowdmanagerEnvironment,
        )

        print(f"[mode] in-process env (no Docker)")
        env = MetrocrowdmanagerEnvironment()
        result = await run_rollout(env, args, gen)
    else:
        print(f"[mode] HTTP/WS client → {args.base_url}")
        async with MetrocrowdmanagerEnv(base_url=args.base_url) as env:
            result = await run_rollout(env, args, gen)

    print(f"\n=== task={args.task} model={MODEL_NAME} seed={args.seed} ===")
    print(f"reward (avg): {result.reward:.3f}")
    print(f"per-step:     {[round(r, 3) for r in result.per_step_rewards]}")
    print(f"truncated:    {result.truncated}")
    print(f"final_text:   {result.final_text[:200]}{'…' if len(result.final_text) > 200 else ''}")
    print("\nbreakdown:")
    for k, v in result.reward_breakdown.items():
        print(f"  {k}: {v:.3f}")
    if args.show_turns:
        print("\nturns:")
        for i, t in enumerate(result.turn_history):
            tools = ", ".join(c["name"] for c in t.get("tool_calls", []))
            preview = (t.get("text") or "")[:160].replace("\n", " ")
            print(f"  [{i}] tools=[{tools}]  text={preview}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["ticket_booking", "ticket_issuance", "crowd_announcement"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Per-turn generation cap. Bump this if you see truncated tool_call JSON.",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("ENV_BASE_URL", "http://localhost:8000"),
        help="OpenEnv server URL (default http://localhost:8000).",
    )
    p.add_argument(
        "--in-process",
        action="store_true",
        help="Skip Docker and use an in-memory env. Useful for fast iteration.",
    )
    p.add_argument("--show-turns", action="store_true")
    asyncio.run(main(p.parse_args()))
