#!/usr/bin/env bash
# Push the MetroCrowdManager OpenEnv environment to the Hugging Face Space.
#
# Why a separate script? The standard `openenv push` from inside the env
# directory works, but the README only updates when you actually re-run
# `openenv push` after editing it. If the README on the Space looks stale,
# this script is the one-liner fix.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash scripts/push_env_to_hf_space.sh                          # uses default repo
#   REPO_ID=YourOrg/your-space bash scripts/push_env_to_hf_space.sh
#   FORCE_README=1 bash scripts/push_env_to_hf_space.sh           # also force-uploads README.md alone
#
# After this script runs, visit:
#   https://huggingface.co/spaces/$REPO_ID

set -euo pipefail

REPO_ID="${REPO_ID:-DhiwakarDev/openenv}"
ENV_DIR="${ENV_DIR:-MetroCrowdManager}"
FORCE_README="${FORCE_README:-0}"

if [[ ! -d "$ENV_DIR" ]]; then
  echo "ERROR: env directory '$ENV_DIR' not found. Run from repo root." >&2
  exit 1
fi

echo "=========================================="
echo "[push] env_dir=${ENV_DIR}"
echo "[push] repo_id=${REPO_ID}"
echo "=========================================="

pushd "$ENV_DIR" >/dev/null
openenv push --repo-id "$REPO_ID"
popd >/dev/null

if [[ "$FORCE_README" == "1" ]]; then
  echo
  echo "[push] FORCE_README=1: re-uploading just README.md to overwrite any stale copy"
  hf upload "$REPO_ID" "$ENV_DIR/README.md" README.md \
    --repo-type=space \
    --commit-message "Force-update README from $ENV_DIR/README.md"
fi

echo
echo "Done. Verify the README at:"
echo "  https://huggingface.co/spaces/${REPO_ID}/blob/main/README.md"
echo "Live Space:"
echo "  https://huggingface.co/spaces/${REPO_ID}"
