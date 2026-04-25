"""
Local GRPO training for MetroCrowdManager — rewards from the Docker image.

This is the laptop-friendly sibling of `train_grpo.py`:

  * Model is loaded with plain `transformers` + `peft` (no Unsloth / bnb-4bit),
    so it works on CUDA, Apple Silicon (MPS), or CPU.
  * Rewards come from the **Docker image** of the env (default tag
    `openenv-mcm`), exercising the same container that scores submissions.
  * Training metrics are streamed to a local `trackio` dashboard, including
    the per-task reward and reward-breakdown components that GRPOTrainer
    doesn't log on its own.

Usage:
    # 1) make sure the image is built (one-time):
    docker build -t openenv-mcm MetroCrowdManager/

    # 2) view the dashboard in another terminal:
    pip install trackio && trackio show

    # 3) run a smoke job:
    python training/local_train_grpo.py --phase A --num-episodes 4 \
        --num-generations 2 --batch-size 1 --grad-accum 1 \
        --max-completion-len 128 --output-dir outputs/local-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "MetroCrowdManager"))

from training.rollout import (  # noqa: E402
    SYSTEM_PROMPTS,
    format_training_debug_log,
    replay_completion_sync,
)


PHASE_TASKS = {
    "A": ["ticket_booking"],
    "B": ["ticket_issuance"],
    "C": ["crowd_announcement"],
}


def _validate_grpo_shape(args: argparse.Namespace) -> None:
    """GRPO requires the generation batch (= per_device_batch * grad_accum,
    single-process) to be divisible by num_generations, and num_generations>=2.
    Catch this here so we fail with an actionable message instead of inside
    GRPOConfig.__post_init__."""
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
            f"num_generations ({args.num_generations}). "
            "Fix by raising --grad-accum (e.g. --grad-accum "
            f"{args.num_generations}) or --batch-size, or lowering --num-generations."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="openenv-mcm",
                   help="Docker image tag for the MetroCrowdManager env.")
    p.add_argument("--phase", choices=list(PHASE_TASKS), default="A")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--max-completion-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="outputs/local")
    p.add_argument("--resume-adapters", default=None)
    p.add_argument("--log-csv", default=None)
    p.add_argument("--trackio-project", default="metrocrowdmanager-grpo")
    p.add_argument("--trackio-space-id", default=None,
                   help="Optional HF Space (user/space) to host the dashboard.")
    p.add_argument("--no-trackio", action="store_true")
    p.add_argument(
        "--open-log-terminal",
        action="store_true",
        help="Open a new Terminal window tailing Docker logs on macOS.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device + model
# ---------------------------------------------------------------------------


def pick_device(requested: str) -> Tuple[str, "torch.dtype"]:  # noqa: F821
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        # MPS fp16 is flaky for many ops; fp32 is the safe default off-CUDA.
        dtype = torch.float32
        if requested == "cpu":
            print("[local-train] WARNING: running on CPU — expect minutes per step.")
    return requested, dtype


def build_model_and_tokenizer(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = pick_device(args.device)
    print(f"[local-train] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len - args.max_completion_len

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
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

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Dataset (prompts only — actual rewards come from docker at training time)
# ---------------------------------------------------------------------------


def build_dataset(tokenizer, args, num_episodes: int):
    from datasets import Dataset

    from MetroCrowdManager.server.MetroCrowdManager_environment import (
        MetrocrowdmanagerEnvironment,
    )

    rng = random.Random(args.seed)
    tasks = PHASE_TASKS[args.phase]
    rows = []
    seed_env = MetrocrowdmanagerEnvironment()
    for _ in range(num_episodes):
        task = rng.choice(tasks)
        row_seed = rng.randint(0, 1_000_000)
        obs = seed_env.reset(task=task, seed=row_seed)
        sys_prompt = SYSTEM_PROMPTS[task]
        user_lines = [obs.prompt_text]
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
# Docker env (single shared client on a dedicated background loop)
# ---------------------------------------------------------------------------


def start_docker_env(image: str, *, open_log_terminal: bool = False):
    from MetroCrowdManager.client import MetrocrowdmanagerEnv
    from openenv.core.containers.runtime.providers import LocalDockerProvider

    print(f"[local-train] starting docker env image={image} ...")
    provider = LocalDockerProvider()
    env = None
    base_url = None
    try:
        base_url = provider.start_container(image)
        provider.wait_for_ready(base_url)
        env = MetrocrowdmanagerEnv(base_url=base_url, provider=provider).sync()
        env.connect()
    except Exception as exc:
        try:
            if env is not None:
                env.close()
            else:
                provider.stop_container()
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to start container from image '{image}'. "
            f"Build it first: docker build -t {image} MetroCrowdManager/"
        ) from exc

    # Surface the container ID and a one-liner for tailing its FastAPI logs.
    container_id = getattr(provider, "_container_id", None)
    ws_url = getattr(env, "_ws_url", None)
    print(f"[local-train] docker env ready.")
    if ws_url:
        # _ws_url is ws://localhost:PORT/ws — convert back to http:// for clarity.
        http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
        if http_url.endswith("/ws"):
            http_url = http_url[:-3]
        print(f"[local-train] base_url={http_url}")
    if container_id:
        short = container_id[:12]
        print(f"[local-train] container_id={short}")
        print(f"[local-train] tail logs:  docker logs -f {short}")
        if open_log_terminal:
            _spawn_log_terminal(f"docker logs -f {short}")
    else:
        print(f"[local-train] tail logs:  docker logs -f $(docker ps --filter ancestor={image} --format '{{{{.ID}}}}' | head -1)")
    return env


def _spawn_log_terminal(cmd: str) -> None:
    if sys.platform != "darwin":
        return
    try:
        subprocess.Popen(
            ["osascript", "-e",
             f'tell application "Terminal" to do script "{cmd}"',
             "-e", 'tell application "Terminal" to activate'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[local-train] opened new Terminal window tailing logs.")
    except Exception as exc:
        print(f"[local-train] could not open log terminal: {exc}")


def stop_docker_env(env) -> None:
    try:
        env.close()
    except Exception as exc:  # pragma: no cover
        print(f"[local-train] env.close() error: {exc}")


# ---------------------------------------------------------------------------
# Reward fn — replay completion's tool calls against the container
# ---------------------------------------------------------------------------


def make_docker_reward_fn(env, args, log_csv: str, trackio_enabled: bool):
    if trackio_enabled:
        import trackio  # noqa: F401  (imported lazily inside log block)

    def _replay(task: str, completion_text: str, seed: int) -> Dict:
        return replay_completion_sync(
            env,
            task,
            completion_text,
            seed=seed,
        )

    csv_handle = None
    if log_csv:
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        csv_handle = open(log_csv, "a", buffering=1)
        if csv_handle.tell() == 0:
            csv_handle.write("step,task,reward,breakdown_json\n")

    step_counter = {"n": 0}

    def reward_fn(prompts, completions, **kwargs):
        rewards: List[float] = []
        per_task: Dict[str, List[float]] = {}
        agg_breakdown: Dict[str, List[float]] = {}

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
            per_task.setdefault(task or "unknown", []).append(r)
            for k, v in breakdown.items():
                agg_breakdown.setdefault(k, []).append(float(v))

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
                # Only log if TrackioCallback has already initialized a run.
                # context_vars is set by trackio.init(); accessing it before
                # the trainer's setup() hook fires would create a stray run
                # under the wrong project.
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
            import trackio  # noqa: F401  — verify install only; init is owned by TrackioCallback
            print(f"[local-train] trackio enabled: project={args.trackio_project} run={run_name}")
        except ImportError:
            print("[local-train] trackio not installed — `pip install trackio` to enable. "
                  "Continuing without tracking.")
            trackio_enabled = False

    print(f"[local-train] phase={args.phase} tasks={PHASE_TASKS[args.phase]}")
    print(f"[local-train] model={args.model}")

    model, tokenizer, device = build_model_and_tokenizer(args)
    dataset = build_dataset(tokenizer, args, args.num_episodes)
    print(f"[local-train] dataset rows: {len(dataset)}")

    log_csv = args.log_csv or str(Path(args.output_dir) / "rewards.csv")

    env = start_docker_env(args.image, open_log_terminal=args.open_log_terminal)
    reward_fn = make_docker_reward_fn(env, args, log_csv, trackio_enabled)

    try:
        from trl import GRPOConfig, GRPOTrainer

        is_cuda = device == "cuda"
        grpo_kwargs = dict(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=1,
            max_steps=args.num_episodes,
            save_steps=args.save_steps,
            logging_steps=1,
            bf16=False,
            fp16=is_cuda,
            max_completion_length=args.max_completion_len,
            num_generations=args.num_generations,
            seed=args.seed,
            report_to="trackio" if trackio_enabled else "none",
            dataloader_pin_memory=is_cuda,
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
        )

        t0 = time.time()
        trainer.train()
        print(f"[local-train] done in {time.time() - t0:.1f}s")

        out = Path(args.output_dir) / "final"
        out.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(out))
        print(f"[local-train] adapters saved to {out}")
    finally:
        stop_docker_env(env)
        if trackio_enabled:
            try:
                import trackio
                trackio.finish()
            except Exception:  # pragma: no cover
                pass


if __name__ == "__main__":
    main()
