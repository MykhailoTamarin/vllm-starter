#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash (EXL3) with a compact REAP-sliced DSPark draft.
#
# Builds the K64 draft from the HF-cache snapshot on first run (idempotent),
# then execs `vllm serve "$@"`.  The draft lives OUTSIDE the target snapshot so
# the source checkpoint is never modified.
#
# Env:
#   MODEL_REPO   HF repo id (default: parse --model from args)
#   DRAFT_DIR    output draft dir (must match --speculative-config model path)
#   DSPARK_DRAFT_EXPERTS                 (default 64)
#   DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY (default 32)

set -Eeuo pipefail

model_repo=${MODEL_REPO:-}
draft_dir=${DRAFT_DIR:-}
draft_experts=${DSPARK_DRAFT_EXPERTS:-64}
structured_per_category=${DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY:-32}

if [ -z "$model_repo" ]; then
  prev=""
  for a in "$@"; do
    if [ "$prev" = "--model" ]; then model_repo="$a"; break; fi
    prev="$a"
  done
fi
[ -n "$model_repo" ] || { echo "ERROR: MODEL_REPO or --model is required" >&2; exit 1; }

cache_root="${HF_HOME:-/root/.cache/huggingface}"
hub="${HF_HUB_CACHE:-$cache_root/hub}"
folder="models--${model_repo//\//--}"

# Resolve the HF snapshot directory (pinned revision if present, else latest).
snapshot=""
if [ -f "$hub/$folder/refs/main" ]; then
  rev="$(cat "$hub/$folder/refs/main" 2>/dev/null || true)"
  if [ -n "$rev" ] && [ -d "$hub/$folder/snapshots/$rev" ]; then
    snapshot="$hub/$folder/snapshots/$rev"
  fi
fi
if [ -z "$snapshot" ]; then
  snapshot="$(ls -d "$hub/$folder/snapshots/"*/ 2>/dev/null | head -1 || true)"
fi
[ -n "$snapshot" ] || {
  echo "ERROR: no HF snapshot found for $model_repo under $hub" >&2
  exit 1
}

if [ -z "$draft_dir" ]; then
  draft_dir="/opt/dspark-drafts/${model_repo//\//--}/dspark-draft-k${draft_experts}"
fi

if [ ! -f "$draft_dir/model.safetensors.index.json" ]; then
  echo "Building compact DSPark draft (K${draft_experts}) from $snapshot -> $draft_dir"
  python3 /opt/recipe/build_dspark_draft.py \
    --source "$snapshot" \
    --output "$draft_dir" \
    --experts "$draft_experts" \
    --structured-per-category "$structured_per_category"
else
  echo "DSPark draft already present at $draft_dir (reusing)"
fi

exec vllm serve "$@"
