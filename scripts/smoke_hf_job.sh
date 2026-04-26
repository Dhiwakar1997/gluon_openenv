#!/usr/bin/env bash
# Smoke-test the HF Jobs training pipeline with a tiny model + 10 GRPO steps.
# Uses git-clone-at-runtime instead of a custom Docker image — no local build.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/smoke_hf_job.sh ticket_booking
#   bash scripts/smoke_hf_job.sh ticket_issuance
#   bash scripts/smoke_hf_job.sh crowd_announcement
#   bash scripts/smoke_hf_job.sh ALL          # all three sequentially
#
# Legacy A/B/C aliases also still work:
#   bash scripts/smoke_hf_job.sh A   # -> ticket_booking
#   bash scripts/smoke_hf_job.sh B   # -> ticket_issuance
#   bash scripts/smoke_hf_job.sh C   # -> crowd_announcement

set -euo pipefail

PHASE="${1:-ticket_booking}"
REPO_URL="https://github.com/Dhiwakar1997/gluon_openenv"
ENV_BASE_URL="https://dhiwakardev-openenv.hf.space"
TRACKIO_SPACE_ID="DhiwakarDev/mcm-trackio"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
FLAVOR="a10g-small"
# 1 = print full per-sample rollout trace (USER PROMPT, AGENT RESPONSE,
# TOOL CALLS, etc). 0 = print only breakdown JSON + spotlight reward/loss.
# Override per-invocation: DEBUG_MODE=1 bash scripts/smoke_hf_job.sh A
DEBUG_MODE="${DEBUG_MODE:-0}"

# Build a single-line bash payload. HF Jobs serializes the command list as
# argv to `bash -lc`; multi-line strings get interpreted as a filename, so we
# chain everything with `&&` on one logical line.
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
  --num-episodes 10 \
  --batch-size 1 --grad-accum 2 --num-generations 2 \
  --max-completion-len 256 --max-seq-len 2048 \
  --save-steps 5 \
  --debug-mode ${DEBUG_MODE} \
  --env-base-url \$ENV_BASE_URL \
  --trackio-space-id \$TRACKIO_SPACE_ID \
  --trackio-project mcm-smoketest \
  --output-dir outputs/smoke-${phase}"
}

submit_phase() {
  local phase="$1"
  echo "[smoke] submitting phase=${phase}"
  hf jobs run \
    --flavor "${FLAVOR}" \
    --secrets HF_TOKEN \
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
