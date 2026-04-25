"""
GRPO training for MetroCrowdManager (Unsloth + TRL, Qwen3-1.7B on T4).

Phases (run with --phase):
    A   ticket_issuance only — easiest, used to verify the pipeline.
    B   ticket_booking only — multi-turn conversation discipline.
    C   mixed — all three tasks shuffled.

The training loop generates a model rollout against an in-process
`MetrocrowdmanagerEnvironment`, scores it with the env's reward stack,
and feeds the (prompt, completion, reward) triple to a TRL `GRPOTrainer`.

This is intentionally minimal — designed to **run on Colab T4 (~15GB
VRAM, ~12h budget)**, not to scale to multi-GPU. If you have more GPU
budget, increase `--num-episodes`, `--batch-size`, `--max-seq-len`.

Usage:
    # Phase A on T4 (after `pip install unsloth trl`):
    python training/train_grpo.py --phase A --num-episodes 60

    # Resume from adapters:
    python training/train_grpo.py --phase B --num-episodes 60 \
        --resume-adapters ./outputs/phaseA/checkpoint-final
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MetroCrowdManager"))

from training.rollout import (  # noqa: E402
    SYSTEM_PROMPTS,
    agentic_episode_sync,
    format_training_debug_log,
    replay_completion_sync,
)


PHASE_TASKS = {
    "A": ["ticket_issuance"],
    "B": ["ticket_booking"],
    "C": ["ticket_booking", "ticket_issuance", "crowd_announcement"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=list(PHASE_TASKS), default="A")
    p.add_argument("--model", default="unsloth/Qwen3-1.7B-Instruct-bnb-4bit")
    p.add_argument("--num-episodes", type=int, default=60)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-completion-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="outputs/phaseA")
    p.add_argument("--resume-adapters", default=None)
    p.add_argument("--log-csv", default=None,
                   help="Optional CSV path for per-episode reward logs.")
    return p.parse_args()


def build_model_and_tokenizer(args):
    """Lazy-import unsloth so the script can be inspected on machines
    without GPUs. Falls back to plain transformers if unsloth is missing."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
        )
    except ImportError:  # pragma: no cover
        print("[WARN] unsloth not installed — falling back to plain transformers (slow).")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", trust_remote_code=True
        )
    if args.resume_adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapters, is_trainable=True)
    return model, tokenizer


def make_generate_sync(model, tokenizer, *, temperature: float = 0.7, max_new_tokens: int = 512):
    """Return a synchronous generate function compatible with rollout.agentic_episode_sync."""
    import torch

    def generate_sync(messages: List[Dict[str, str]]) -> str:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    return generate_sync


def build_dataset(env_factory, tokenizer, args, num_episodes: int):
    """Pre-roll a small dataset of (prompt_text, expected_reward) pairs.

    GRPO will resample completions from the model online — we just need
    the prompt to be deterministic per dataset row.
    """
    from datasets import Dataset

    rows = []
    rng = random.Random(args.seed)
    tasks = PHASE_TASKS[args.phase]
    for i in range(num_episodes):
        task = rng.choice(tasks)
        env = env_factory()
        obs = env.reset(task=task, seed=rng.randint(0, 1_000_000))
        sys_prompt = SYSTEM_PROMPTS[task]
        user_lines = [obs.prompt_text]
        prompt_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n".join(user_lines)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        rows.append({"task": task, "prompt": prompt_text, "row_seed": rng.randint(0, 1_000_000)})
    return Dataset.from_list(rows)


def make_reward_fn(env_factory, tokenizer, args, log_csv):
    """Return a reward function compatible with TRL GRPOTrainer.

    GRPOTrainer calls `reward_fn(prompts, completions, **kwargs)` and
    expects a list of floats. For each (prompt, completion) we replay the
    rollout against a fresh env: we re-extract tool calls from the
    completion and score the whole trace via the env.
    """
    def _replay(task: str, completion_text: str, seed: int) -> Dict:
        env = env_factory()
        try:
            return replay_completion_sync(
                env,
                task,
                completion_text,
                seed=seed,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    csv_handle = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        csv_handle = open(log_csv, "a", buffering=1)
        if csv_handle.tell() == 0:
            csv_handle.write("step,task,reward,breakdown_json\n")

    step_counter = {"n": 0}

    def reward_fn(prompts, completions, **kwargs):
        rewards: List[float] = []
        tasks = kwargs.get("task") or [None] * len(prompts)
        seeds = kwargs.get("row_seed") or [args.seed] * len(prompts)
        for sample_idx, (prompt, completion, task, seed) in enumerate(
            zip(prompts, completions, tasks, seeds),
            start=1,
        ):
            if isinstance(completion, list):
                completion_text = completion[-1].get("content", "")
            else:
                completion_text = completion
            task_name = task or "ticket_issuance"
            try:
                result = _replay(task_name, completion_text, int(seed))
                r = result["reward"]
                breakdown = result["breakdown"]
            except Exception as exc:  # pragma: no cover
                r = 0.0
                breakdown = {"error": 1.0}
                print(f"[reward_fn] replay error: {exc}", flush=True)
                result = {
                    "reward": r,
                    "breakdown": breakdown,
                    "turn_history": [],
                    "final_text": completion_text,
                    "initial_observation": {},
                    "debug_context": {},
                }
            print(
                format_training_debug_log(
                    step=step_counter["n"],
                    sample_idx=sample_idx,
                    task_name=task_name,
                    replay_result=result,
                ),
                flush=True,
            )
            if csv_handle:
                csv_handle.write(
                    f"{step_counter['n']},{task},{r:.4f},{json.dumps(breakdown)}\n"
                )
            rewards.append(r)
        step_counter["n"] += 1
        return rewards

    return reward_fn


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    from MetroCrowdManager.server.MetroCrowdManager_environment import (
        MetrocrowdmanagerEnvironment,
    )

    def env_factory():
        return MetrocrowdmanagerEnvironment()

    print(f"[train] phase={args.phase} tasks={PHASE_TASKS[args.phase]}")
    print(f"[train] model={args.model}")

    model, tokenizer = build_model_and_tokenizer(args)

    dataset = build_dataset(env_factory, tokenizer, args, args.num_episodes)
    print(f"[train] dataset rows: {len(dataset)}")

    log_csv = args.log_csv or str(Path(args.output_dir) / "rewards.csv")
    reward_fn = make_reward_fn(env_factory, tokenizer, args, log_csv)

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=1,
        max_steps=args.num_episodes,
        save_steps=args.save_steps,
        logging_steps=1,
        bf16=False,
        fp16=True,
        max_prompt_length=args.max_seq_len - args.max_completion_len,
        max_completion_length=args.max_completion_len,
        num_generations=4,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    t0 = time.time()
    trainer.train()
    print(f"[train] done in {time.time() - t0:.1f}s")
    out = Path(args.output_dir) / "final"
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    print(f"[train] adapters saved to {out}")


if __name__ == "__main__":
    main()
