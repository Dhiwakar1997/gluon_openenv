"""
HF Jobs GRPO training for MetroCrowdManager — A100 + remote OpenEnv Space + trackio.

Sibling of `local_train_grpo.py` adapted for HF Jobs (`hf jobs run`):

  * Model is loaded with 4-bit QLoRA (`bitsandbytes` NF4 + bf16 compute) so
    `google/gemma-3-27b-it` fits on a single A100 80GB. LoRA adapters on top.
  * Rewards come from a remote MetroCrowdManager OpenEnv Space (deployed
    via `openenv push`); the trainer connects to its WebSocket URL via
    `MetrocrowdmanagerEnv(base_url=...)` — no local Docker required.
  * Training metrics stream to a persistent trackio HF Space dashboard.
  * Final LoRA adapters can be pushed to a Hub repo.

Submit via HF Jobs (see plan file for the full `hf jobs run` invocation).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MetroCrowdManager"))

from training.agentic_rollout_func import make_agentic_rollout_func  # noqa: E402
from training.rollout import (  # noqa: E402
    SYSTEM_PROMPTS,
    format_training_debug_log,
    make_replay_result_from_rollout,
)


PHASE_TASKS = {
    "A": ["ticket_booking"],
    "B": ["ticket_issuance"],
    "C": ["crowd_announcement"],
}


def _validate_grpo_shape(args: argparse.Namespace) -> None:
    if args.num_generations < 2:
        raise SystemExit(
            f"--num-generations must be >=2 (got {args.num_generations}). "
            "GRPO needs at least 2 completions per prompt to compute advantages."
        )
    gen_batch = args.batch_size * args.grad_accum
    if gen_batch % args.num_generations != 0:
        raise SystemExit(
            f"GRPO constraint violated: batch_size ({args.batch_size}) * "
            f"grad_accum ({args.grad_accum}) = {gen_batch} must be divisible by "
            f"num_generations ({args.num_generations})."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=list(PHASE_TASKS), default="A")
    p.add_argument("--model", default="google/gemma-3-27b-it")
    p.add_argument("--env-base-url", default=os.getenv("ENV_BASE_URL"),
                   help="WS/HTTP base URL of the deployed MetroCrowdManager OpenEnv Space.")
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-completion-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="outputs/jobs")
    p.add_argument("--resume-adapters", default=None)
    p.add_argument("--log-csv", default=None)
    p.add_argument("--trackio-project", default="mcm-gemma3-27b")
    p.add_argument("--trackio-space-id", default=os.getenv("TRACKIO_SPACE_ID"),
                   help="HF Space (user/space) hosting the trackio dashboard.")
    p.add_argument("--no-trackio", action="store_true")
    p.add_argument("--push-to-hub-id", default=None,
                   help="Hub repo (user/repo) to push final LoRA adapters to.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model — 4-bit QLoRA for 27B on A100 80GB
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(args):
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    print(f"[hf-jobs-train] loading {args.model} in 4-bit NF4 (bf16 compute)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(args.max_seq_len, 4096)
    if getattr(tokenizer, "truncation_side", None) != "left":
        tokenizer.truncation_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    if args.resume_adapters:
        model = PeftModel.from_pretrained(model, args.resume_adapters, is_trainable=True)
    else:
        model = get_peft_model(model, lora_cfg)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset (prompts only — actual rewards come from the remote env at training)
# ---------------------------------------------------------------------------


def build_dataset(tokenizer, args, num_episodes: int, seed_env):
    """Seed prompts by reset()-ing the *remote* env once per row.

    The remote env Space exposes the same `reset(task, seed) -> obs.prompt_text`
    contract as the in-process server, so we don't ship the server tree in
    the training image — only the WS client.
    """
    from datasets import Dataset

    rng = random.Random(args.seed)
    tasks = PHASE_TASKS[args.phase]
    rows = []
    for _ in range(num_episodes):
        task = rng.choice(tasks)
        row_seed = rng.randint(0, 1_000_000)
        obs = seed_env.reset(task=task, seed=row_seed)
        sys_prompt = SYSTEM_PROMPTS[task]
        user_lines = [getattr(obs, "prompt_text", "") or ""]
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "\n".join(user_lines)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append({"task": task, "prompt": prompt_text, "row_seed": row_seed})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Remote OpenEnv Space client
# ---------------------------------------------------------------------------


def start_remote_env(base_url: str):
    if not base_url:
        raise SystemExit(
            "--env-base-url (or ENV_BASE_URL) is required: point at the deployed "
            "MetroCrowdManager OpenEnv Space, e.g. https://<user>-mcm-env.hf.space"
        )
    from MetroCrowdManager.client import MetrocrowdmanagerEnv

    print(f"[hf-jobs-train] connecting to remote env: {base_url}")
    env = MetrocrowdmanagerEnv(base_url=base_url).sync()
    env.connect()
    print(f"[hf-jobs-train] remote env connected.")
    return env


def stop_remote_env(env) -> None:
    try:
        env.close()
    except Exception as exc:  # pragma: no cover
        print(f"[hf-jobs-train] env.close() error: {exc}")


# ---------------------------------------------------------------------------
# Reward fn — pass-through over rollout_func outputs (same as local_train_grpo)
# ---------------------------------------------------------------------------


def make_remote_reward_fn(args, log_csv: str, trackio_enabled: bool):
    if trackio_enabled:
        import trackio  # noqa: F401

    csv_handle = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        csv_handle = open(log_csv, "a", buffering=1)
        if csv_handle.tell() == 0:
            csv_handle.write("step,task,reward,breakdown_json\n")

    step_counter = {"n": 0}

    def _per_sample(kwargs: Dict[str, Any], idx: int, key: str, default: Any) -> Any:
        values = kwargs.get(key)
        if not values:
            return default
        if idx >= len(values):
            return default
        return values[idx]

    def reward_fn(prompts, completions, **kwargs):
        rewards: List[float] = []
        per_task: Dict[str, List[float]] = {}
        agg_breakdown: Dict[str, List[float]] = {}

        for sample_idx, (prompt, completion) in enumerate(
            zip(prompts, completions), start=1
        ):
            task_name = _per_sample(kwargs, sample_idx - 1, "rollout_task_name", None) \
                or _per_sample(kwargs, sample_idx - 1, "task", None) \
                or "ticket_issuance"

            r = float(_per_sample(kwargs, sample_idx - 1, "rollout_reward", 0.0) or 0.0)
            breakdown = dict(
                _per_sample(kwargs, sample_idx - 1, "rollout_breakdown", {}) or {}
            )
            turn_history = list(
                _per_sample(kwargs, sample_idx - 1, "rollout_turn_history", []) or []
            )
            final_text = str(
                _per_sample(kwargs, sample_idx - 1, "rollout_final_text", "") or ""
            )
            raw_completion = str(
                _per_sample(kwargs, sample_idx - 1, "rollout_raw_completion", "") or ""
            )
            initial_obs = dict(
                _per_sample(kwargs, sample_idx - 1, "rollout_initial_obs", {}) or {}
            )
            system_prompt = str(
                _per_sample(kwargs, sample_idx - 1, "rollout_system_prompt", "") or ""
            )

            replay_result = make_replay_result_from_rollout(
                turn_history=turn_history,
                final_text=final_text,
                reward=r,
                breakdown=breakdown,
                initial_observation=initial_obs,
                raw_completion=raw_completion,
                system_prompt=system_prompt,
            )

            print(
                format_training_debug_log(
                    step=step_counter["n"],
                    sample_idx=sample_idx,
                    task_name=task_name,
                    replay_result=replay_result,
                ),
                flush=True,
            )

            if csv_handle:
                csv_handle.write(
                    f"{step_counter['n']},{task_name},{r:.4f},{json.dumps(breakdown)}\n"
                )
            rewards.append(r)
            per_task.setdefault(task_name, []).append(r)
            for k, v in breakdown.items():
                try:
                    agg_breakdown.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass

        if trackio_enabled:
            log_payload: Dict[str, float] = {
                "reward/mean": sum(rewards) / max(len(rewards), 1),
            }
            for t, vs in per_task.items():
                log_payload[f"reward/{t}"] = sum(vs) / len(vs)
            for k, vs in agg_breakdown.items():
                log_payload[f"breakdown/{k}"] = sum(vs) / len(vs)
            try:
                import trackio
                current_run = None
                try:
                    current_run = trackio.context_vars.current_run.get()
                except Exception:
                    current_run = None
                if current_run is not None:
                    trackio.log(log_payload, step=step_counter["n"])
            except Exception as exc:  # pragma: no cover
                print(f"[reward_fn] trackio.log error: {exc}", flush=True)

        step_counter["n"] += 1
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    _validate_grpo_shape(args)
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    trackio_enabled = not args.no_trackio
    run_name = f"phase{args.phase}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if trackio_enabled:
        try:
            import trackio  # noqa: F401
            print(f"[hf-jobs-train] trackio enabled: project={args.trackio_project} "
                  f"space={args.trackio_space_id} run={run_name}")
        except ImportError:
            print("[hf-jobs-train] trackio not installed — `pip install trackio` to enable.")
            trackio_enabled = False

    print(f"[hf-jobs-train] phase={args.phase} tasks={PHASE_TASKS[args.phase]}")
    print(f"[hf-jobs-train] model={args.model}")

    model, tokenizer = build_model_and_tokenizer(args)

    env = start_remote_env(args.env_base_url)
    dataset = build_dataset(tokenizer, args, args.num_episodes, env)
    print(f"[hf-jobs-train] dataset rows: {len(dataset)}")

    prompt_to_meta: Dict[str, Tuple[str, int]] = {
        row["prompt"]: (row["task"], int(row["row_seed"])) for row in dataset
    }

    def task_resolver(_idx: int, prompt_text: str) -> str:
        return prompt_to_meta.get(prompt_text, (PHASE_TASKS[args.phase][0], args.seed))[0]

    def seed_resolver(_idx: int, prompt_text: str) -> Optional[int]:
        return prompt_to_meta.get(prompt_text, (PHASE_TASKS[args.phase][0], args.seed))[1]

    log_csv = args.log_csv or str(Path(args.output_dir) / "rewards.csv")

    reward_fn = make_remote_reward_fn(args, log_csv, trackio_enabled)
    rollout_func = make_agentic_rollout_func(
        env,
        task_resolver=task_resolver,
        seed_resolver=seed_resolver,
        max_completion_len=args.max_completion_len,
    )

    try:
        from trl import GRPOConfig, GRPOTrainer

        grpo_kwargs = dict(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=1,
            max_steps=args.num_episodes,
            save_steps=args.save_steps,
            logging_steps=1,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            max_completion_length=args.max_completion_len,
            num_generations=args.num_generations,
            seed=args.seed,
            report_to="trackio" if trackio_enabled else "none",
            dataloader_pin_memory=True,
        )
        if trackio_enabled:
            grpo_kwargs["project"] = args.trackio_project
            grpo_kwargs["run_name"] = run_name
            if args.trackio_space_id:
                grpo_kwargs["trackio_space_id"] = args.trackio_space_id
        training_args = GRPOConfig(**grpo_kwargs)

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=dataset,
            reward_funcs=[reward_fn],
            rollout_func=rollout_func,
        )

        t0 = time.time()
        trainer.train()
        print(f"[hf-jobs-train] done in {time.time() - t0:.1f}s")

        out = Path(args.output_dir) / "final"
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out))
        print(f"[hf-jobs-train] adapters saved to {out}")

        if args.push_to_hub_id:
            hf_token = os.environ.get("HF_TOKEN")
            print(f"[hf-jobs-train] pushing adapters to {args.push_to_hub_id}")
            trainer.model.push_to_hub(args.push_to_hub_id, token=hf_token)
            tokenizer.push_to_hub(args.push_to_hub_id, token=hf_token)
            print(f"[hf-jobs-train] push complete.")
    finally:
        stop_remote_env(env)
        if trackio_enabled:
            try:
                import trackio
                trackio.finish()
            except Exception:  # pragma: no cover
                pass


if __name__ == "__main__":
    main()
