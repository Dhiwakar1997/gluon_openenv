#!/usr/bin/env bash
# Full GRPO finetune on HF Jobs A100-80GB for one or all tasks.
# Defaults are tuned for `google/gemma-3-27b-it` + 4-bit QLoRA.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/full_run_hf_job.sh ticket_booking          # one task
#   bash scripts/full_run_hf_job.sh ALL                     # all three sequentially
#
# Override defaults per-invocation:
#   MODEL=google/gemma-3-12b-it STEPS=300 bash scripts/full_run_hf_job.sh ALL
#   FLAVOR=a100x4 STEPS=1000 bash scripts/full_run_hf_job.sh ticket_booking
#   DEBUG_MODE=1 STEPS=50 bash scripts/full_run_hf_job.sh ticket_booking   # warmup eval
#
# Suggested step budgets (per task):
#   50    warmup — confirms reward curve is climbing
#   200   quick eval — initial reward shape
#   500   real finetune — typical GRPO target (default)
#   1000  long run — diminishing returns, mode-collapse risk

set -euo pipefail

PHASE="${1:-ticket_booking}"
REPO_URL="${REPO_URL:-https://github.com/Dhiwakar1997/gluon_openenv}"
ENV_BASE_URL="${ENV_BASE_URL:-https://dhiwakardev-openenv.hf.space}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-DhiwakarDev/mcm-trackio}"
TRACKIO_PROJECT="${TRACKIO_PROJECT:-mcm-gemma3-27b-full}"
MODEL="${MODEL:-google/gemma-3-27b-it}"
FLAVOR="${FLAVOR:-a100-large}"          # 1x A100 80GB. Use a100x4 for multi-GPU.
STEPS="${STEPS:-20}"                    # GRPO update steps (max_steps).
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"  # Per-prompt completions (>=2 for GRPO).
MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LR="${LR:-5e-6}"
SAVE_STEPS="${SAVE_STEPS:-100}"
DEBUG_MODE="${DEBUG_MODE:-0}"
PUSH_TO_HUB_ID="${PUSH_TO_HUB_ID:-DhiwakarDev/mcm-gemma3-27b-grpo}"
TIMEOUT="${TIMEOUT:-12h}"                # Hard cap; HF Jobs kills the run after this.

build_payload() {
  local phase="$1"
  printf '%s' "set -e && \
apt-get update -qq && apt-get install -y -qq --no-install-recommends git && \
pip install -q 'transformers>=4.55,<4.60' 'trl>=1.2,<1.4' 'peft>=0.13,<0.16' 'accelerate>=1.0,<1.5' bitsandbytes datasets trackio 'openenv-core[core]' hf_transfer pydantic websockets httpx && \
git clone ${REPO_URL} /workspace/Gluon && \
cd /workspace/Gluon && \
python training/hf_jobs_train_grpo.py \
  --phase ${phase} \
  --model ${MODEL} \
  --num-episodes ${STEPS} \
  --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} --num-generations ${NUM_GENERATIONS} \
  --max-completion-len ${MAX_COMPLETION_LEN} --max-seq-len ${MAX_SEQ_LEN} \
  --lr ${LR} \
  --save-steps ${SAVE_STEPS} \
  --debug-mode ${DEBUG_MODE} \
  --env-base-url \$ENV_BASE_URL \
  --trackio-space-id \$TRACKIO_SPACE_ID \
  --trackio-project ${TRACKIO_PROJECT} \
  --push-to-hub-id ${PUSH_TO_HUB_ID}-${phase} \
  --output-dir outputs/full-${phase}"
}

submit_phase() {
  local phase="$1"
  echo "=========================================="
  echo "[full] submitting phase=${phase}"
  echo "       model=${MODEL}"
  echo "       steps=${STEPS}  flavor=${FLAVOR}  timeout=${TIMEOUT}"
  echo "       push-to-hub=${PUSH_TO_HUB_ID}-${phase}"
  echo "=========================================="
  hf jobs run \
    --flavor "${FLAVOR}" \
    --secrets HF_TOKEN \
    --timeout "${TIMEOUT}" \
    -e ENV_BASE_URL="${ENV_BASE_URL}" \
    -e TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID}" \
    -e TRL_EXPERIMENTAL_SILENCE=1 \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    -- \
    bash -lc "$(build_payload "${phase}")"
}

if [[ "${PHASE}" == "ALL" ]]; then
  for p in ticket_booking ticket_issuance crowd_announcement; do
    submit_phase "$p"
  done
else
  submit_phase "${PHASE}"
fi
