"""
Documented GRPO training script for MetroCrowdManager.

This file is meant to be the readable training entry point for the project.
It explains the whole reinforcement-learning loop in code comments, while
still being runnable on a small GPU setup such as Colab T4 with Unsloth.

What GRPO does here
-------------------
GRPO stands for Group Relative Policy Optimization. For each prompt, the
trainer asks the model to generate several completions. Our reward function
then scores each completion by replaying its tool calls inside the real
MetroCrowdManager environment. The model is updated to prefer completions
that get higher rewards than the other completions in the same group.

For this environment, a high-reward completion is one that:

* asks/uses the required passenger details,
* emits well-formed MCP tool calls,
* calls tools in the correct order,
* uses real tool outputs in later calls,
* avoids premature payment,
* avoids excessive turns/tool spam,
* submits a useful final answer.

Important limitation
--------------------
TRL's GRPOTrainer generates one completion per prompt. It does not naturally
pause generation, execute a tool, feed the tool result back, and continue the
same generation the way `training/rollout.py` does during live inference.

To make GRPO practical, this script trains the model to emit a complete
tool-use trajectory in one completion. The reward function replays every
`<tool_call>...</tool_call>` block against the environment, records the real
tool results in `turn_history`, then submits the trailing plain text as the
final answer. This gives the model reward pressure for the same environment
rules used at inference time.

Recommended commands
--------------------
Run a quick smoke test:

    python training/train_grpo_documented.py \\
        --task-set ticket_booking \\
        --num-episodes 4 \\
        --max-steps 4 \\
        --output-dir outputs/grpo_debug

Train the ticket booking conversation task:

    python training/train_grpo_documented.py \\
        --task-set ticket_booking \\
        --num-episodes 60 \\
        --max-steps 60 \\
        --output-dir outputs/grpo_ticket_booking

Train the structured ticket issuance task:

    python training/train_grpo_documented.py \\
        --task-set ticket_issuance \\
        --num-episodes 60 \\
        --max-steps 60 \\
        --output-dir outputs/grpo_ticket_issuance

Typical Colab dependency install:

    pip install unsloth trl datasets transformers accelerate peft bitsandbytes
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

# The script lives in training/, so add the repository and environment package
# roots to sys.path. This keeps the command simple from the repo root:
# `python training/train_grpo_documented.py ...`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MetroCrowdManager"))

from MetroCrowdManager.server.MetroCrowdManager_environment import (  # noqa: E402
    MetrocrowdmanagerEnvironment,
)
from training.rollout import (  # noqa: E402
    SYSTEM_PROMPTS,
    replay_completion_sync,
)


# ---------------------------------------------------------------------------
# Task sets
# ---------------------------------------------------------------------------

# Start with one task at a time. Ticket issuance is the easiest to debug.
# Ticket booking is your passenger-conversation task.
TASK_SETS = {
    "ticket_booking": ["ticket_booking"],
    "ticket_issuance": ["ticket_issuance"],
    "crowd_announcement": ["crowd_announcement"],
    "mixed": ["ticket_booking", "ticket_issuance", "crowd_announcement"],
}


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MetroCrowdManager with GRPO and environment rewards."
    )
    parser.add_argument("--task-set", choices=list(TASK_SETS), default="ticket_booking")
    parser.add_argument(
        "--model",
        default="unsloth/Qwen3-1.7B-Instruct-bnb-4bit",
        help="Base model or Unsloth 4-bit model name.",
    )
    parser.add_argument(
        "--resume-adapters",
        default=None,
        help="Optional LoRA/PEFT adapter directory to resume from.",
    )
    parser.add_argument("--output-dir", default="outputs/grpo_ticket_booking")
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="GRPO optimizer steps. Defaults to --num-episodes.",
    )
    parser.add_argument("--seed", type=int, default=2026)

    # Model context and generation budget. If tool calls are being truncated,
    # increase --max-completion-len.
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-completion-len", type=int, default=512)

    # Small-GPU defaults. For larger GPUs, increase batch size first.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save-steps", type=int, default=20)

    parser.add_argument(
        "--log-csv",
        default=None,
        help="Reward log path. Defaults to <output-dir>/reward_log.csv.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(args: argparse.Namespace):
    """Load the model and tokenizer.

    We try Unsloth first because it is fast and memory efficient for Colab T4.
    The fallback path uses plain Hugging Face Transformers so the code remains
    readable on machines where Unsloth is not installed.
    """
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
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
        )
    except ImportError:
        print("[warn] unsloth not installed; falling back to transformers.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    if args.resume_adapters:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.resume_adapters,
            is_trainable=True,
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt dataset
# ---------------------------------------------------------------------------


def build_prompt_messages(task: str, seed: int) -> List[Dict[str, str]]:
    """Create the chat prompt for one environment episode.

    The environment is reset with a seed so the prompt and hidden ground truth
    are reproducible. For ticket_booking, the environment exposes the scripted
    passenger messages in observation metadata. We include those messages in
    the prompt so the one-shot GRPO completion can produce a full conversation
    trajectory that our reward function can replay.
    """
    env = MetrocrowdmanagerEnvironment()
    obs = env.reset(task=task, seed=seed)

    user_lines = [obs.prompt_text]

    passenger_messages = (obs.metadata or {}).get("passenger_messages") or []
    if task == "ticket_booking" and passenger_messages:
        user_lines.append("")
        user_lines.append("Passenger messages for this episode, in order:")
        for idx, message in enumerate(passenger_messages, start=1):
            user_lines.append(f"{idx}. Passenger: \"{message}\"")
        user_lines.append("")
        user_lines.append(
            "Write the assistant side of the conversation. Use MCP tool calls "
            "when needed, and end with the final passenger-facing answer."
        )
        user_lines.append(
            "Reward requires this tool order: list_valid_stations, "
            "validate_destination, get_ticket_cost, initiate_payment, "
            "check_payment_status."
        )
    elif getattr(obs, "passenger_message", ""):
        user_lines.append(f'Passenger: "{obs.passenger_message}"')

    return [
        {"role": "system", "content": SYSTEM_PROMPTS[task]},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def build_dataset(tokenizer: Any, args: argparse.Namespace):
    """Build a lightweight prompt dataset for GRPO.

    GRPO does not need target answers in the dataset. It only needs prompts.
    During training the model samples completions, and our reward function
    scores those sampled completions against the environment.
    """
    from datasets import Dataset

    rng = random.Random(args.seed)
    tasks = TASK_SETS[args.task_set]
    rows = []

    for _ in range(args.num_episodes):
        task = rng.choice(tasks)
        row_seed = rng.randint(0, 1_000_000)
        messages = build_prompt_messages(task, row_seed)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append(
            {
                "prompt": prompt,
                "task": task,
                "row_seed": row_seed,
            }
        )

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward replay
# ---------------------------------------------------------------------------


def completion_to_text(completion: Any) -> str:
    """Normalize the different completion shapes TRL versions may return."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def replay_completion(task: str, completion_text: str, seed: int) -> Dict[str, Any]:
    """Replay one model completion against the real environment.

    The completion can contain several tool calls:

        <tool_call>{"name": "validate_destination", "arguments": {...}}</tool_call>

    Every complete tool-call block is executed with `env.step(CallToolAction)`.
    The environment returns real simulated tool results, which are stored in
    `turn_history`. After the last tool call, the remaining plain text becomes
    the final answer submitted through `SubmitResponseAction`.
    """
    return replay_completion_sync(
        MetrocrowdmanagerEnvironment(),
        task,
        completion_text,
        seed=seed,
    )


def make_reward_logger(log_csv: str):
    """Create a CSV writer for reward debugging."""
    path = Path(log_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", newline="", buffering=1)
    writer = csv.DictWriter(
        handle,
        fieldnames=["step", "task", "reward", "breakdown_json", "final_preview"],
    )
    if handle.tell() == 0:
        writer.writeheader()
    atexit.register(handle.close)
    return writer


def make_reward_fn(args: argparse.Namespace):
    """Build the callable used by GRPOTrainer.

    TRL calls this function with batches of prompts and sampled completions.
    We ignore the prompt text itself because the dataset already passes `task`
    and `row_seed`. Those two fields let us reset the same environment scenario
    that produced the prompt, replay the completion, and return one scalar
    reward per completion.
    """
    log_csv = args.log_csv or str(Path(args.output_dir) / "reward_log.csv")
    writer = make_reward_logger(log_csv)
    step_counter = {"value": 0}

    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        tasks = kwargs.get("task") or ["ticket_booking"] * len(completions)
        seeds = kwargs.get("row_seed") or [args.seed] * len(completions)
        rewards: List[float] = []

        for task, seed, completion in zip(tasks, seeds, completions):
            completion_text = completion_to_text(completion)
            try:
                result = replay_completion(task, completion_text, int(seed))
                reward = result["reward"]
                breakdown = result["breakdown"]
                final_preview = result["final_text"][:160].replace("\n", " ")
            except Exception as exc:
                reward = 0.0
                breakdown = {"replay_error": str(exc)}
                final_preview = completion_text[:160].replace("\n", " ")

            writer.writerow(
                {
                    "step": step_counter["value"],
                    "task": task,
                    "reward": f"{reward:.4f}",
                    "breakdown_json": json.dumps(breakdown, sort_keys=True),
                    "final_preview": final_preview,
                }
            )
            rewards.append(reward)

        step_counter["value"] += 1
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.max_steps is None:
        args.max_steps = args.num_episodes

    print(f"[grpo] task_set={args.task_set} tasks={TASK_SETS[args.task_set]}")
    print(f"[grpo] model={args.model}")
    print(f"[grpo] output_dir={args.output_dir}")

    model, tokenizer = build_model_and_tokenizer(args)
    dataset = build_dataset(tokenizer, args)
    reward_fn = make_reward_fn(args)

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=1,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        bf16=False,
        fp16=True,
        max_prompt_length=args.max_seq_len - args.max_completion_len,
        max_completion_length=args.max_completion_len,
        num_generations=args.num_generations,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    started = time.time()
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))

    print(f"[grpo] done in {time.time() - started:.1f}s")
    print(f"[grpo] saved adapters/model to {final_dir}")
    print(f"[grpo] reward log: {args.log_csv or Path(args.output_dir) / 'reward_log.csv'}")


if __name__ == "__main__":
    main()
