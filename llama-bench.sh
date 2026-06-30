#!/usr/bin/env bash
# llama-bench.sh — Thin wrapper around forked llama-benchy
# Fork (tools/llama-benchy/) handles wait-idle & internal repeats.
set -euo pipefail

cd "$(dirname "$0")"
set -a; source .env; set +a

API_K="${VLLM_API_KEY:-vllm}"
SSH_H="${SSH_HOST:-localhost}"
MODEL_P="${MODEL_PORT:-8000}"

# ── Fork setup ────────────────────────────────────────────────────────────────
fork_installed=false
fork_correct=false
version_mismatch=false

if uv run --directory tools/llama-benchy llama-benchy --help >/dev/null 2>&1; then
  fork_installed=true
  detected_ver=$(uv run --directory tools/llama-benchy llama-benchy --version 2>/dev/null | awk '{print $2}' || echo "unknown")
  [[ "$detected_ver" == "0.3.8+local.vllm" ]] && fork_correct=true
  [[ "$version_mismatch" != "true" && "$fork_correct" != "true" ]] && version_mismatch=true
fi

if [[ "$fork_correct" != "true" ]]; then
  if [[ -d tools/llama-benchy && -f tools/llama-benchy/pyproject.toml ]]; then
    [[ ! -d tools/llama-benchy/.venv ]] && cd tools/llama-benchy && uv venv >/dev/null 2>&1 && cd ../..
    cd tools/llama-benchy && uv pip install -e . >/dev/null 2>&1 && cd ../..
  else
    echo "❌ tools/llama-benchy not found."
    exit 1
  fi
fi

# ── Parse args ────────────────────────────────────────────────────────────────
args=("$@")
N=${#args[@]}
MODEL_IDX=-1

i=0
while [[ $i -lt $N ]]; do
  if [[ "${args[$i]}" == "--model" ]]; then
    [[ $((i+1)) -lt $N ]] && MODEL_IDX=$((i+1))
  fi
  (( i++ )) || true
done

# ── Resolve model from YAML ─────────────────────────────────────────────────
MODEL_VAL=""
[[ $MODEL_IDX -ge 0 ]] && [[ $MODEL_IDX -lt $N ]] && MODEL_VAL="${args[$MODEL_IDX]}"

MODEL_NAME="${MODEL_VAL:-${MODEL:-}}"
YAML="models/${MODEL_NAME}.yaml"
B_MODEL="$MODEL_NAME"
S_MODEL=""

if [[ -f "$YAML" ]]; then
  B_MODEL=$(grep -- '--model' "$YAML" | head -1 | sed 's/.*--model[[:space:]]*\([^ ]*\).*/\1/')
  S_MODEL=$(grep -- '--served-model-name' "$YAML" | head -1 | sed 's/.*--served-model-name[[:space:]]*\([^ ]*\).*/\1/')
  [[ -z "${B_MODEL:-}" ]] && B_MODEL="$MODEL_NAME"
  YPORT=$(grep '^port:' "$YAML" | head -1 | sed 's/port:[[:space:]]*//')
  [[ -n "${YPORT:-}" ]] && MODEL_P="$YPORT"
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

# Convert space-separated numbers to min-max range
#  → "1 2 4"     → "1-4"
#  → "1"         → "1"
#  → "256 512"   → "256-512"
#  → "0 256 512" → "0-512"
_bench_minmax() {
  local min=$1 max=$1
  shift
  for v in "$@"; do
    [[ $v -lt $min ]] && min=$v
    [[ $v -gt $max ]] && max=$v
  done
  [[ "$min" == "$max" ]] && echo "$min" || echo "${min}-${max}"
}

# ── Generate output filename ────────────────────────────────────────────────
TIMESTAMP=$(date +%d_%m_%y_%H_%M)

# Parse concurrency values (track if explicitly specified)
HAS_CONCURRENCY=false
CONCURRENCY_PART=""
i=0
while [[ $i -lt $N ]]; do
  if [[ "${args[$i]}" == "--concurrency" ]]; then
    HAS_CONCURRENCY=true
    (( i++ )) || true
    local_conc=()
    if [[ $i -lt $N ]]; then
      while [[ $i -lt $N ]]; do
        case "${args[$i]}" in
          --*) break ;;
          *) local_conc+=("${args[$i]}") ;;
        esac
        (( i++ )) || true
      done
    fi
    CONCURRENCY_PART="_c$(_bench_minmax "${local_conc[@]+"${local_conc[@]}"}")"
    break
  fi
  (( i++ )) || true
done

# Parse depth values
i=0
while [[ $i -lt $N ]]; do
  if [[ "${args[$i]}" == "--depth" ]]; then
    (( i++ )) || true
    local_depth=()
    if [[ $i -lt $N ]]; then
      while [[ $i -lt $N ]]; do
        case "${args[$i]}" in
          --*) break ;;
          *) local_depth+=("${args[$i]}") ;;
        esac
        (( i++ )) || true
      done
    fi
    CONCURRENCY_PART="${CONCURRENCY_PART}_d$(_bench_minmax "${local_depth[@]+"${local_depth[@]}"}")"
    break
  fi
  (( i++ )) || true
done

# Default to _c1 if --concurrency wasn't explicitly provided
[[ "$HAS_CONCURRENCY" != "true" ]] && CONCURRENCY_PART="_c1${CONCURRENCY_PART}"

BENCH_DIR="$(pwd)/models/benchmarks/${MODEL_NAME}"
mkdir -p "$BENCH_DIR"
SAVE_PATH="${BENCH_DIR}/benchmark_${TIMESTAMP}${CONCURRENCY_PART}"

# ── Build command ─────────────────────────────────────────────────────────────
cmd=(uv run --directory tools/llama-benchy llama-benchy)
cmd+=(--base-url "http://$SSH_H:$MODEL_P/v1")
cmd+=(--api-key "$API_K")
cmd+=(--model "$B_MODEL")
[[ -n "${S_MODEL:-}" ]] && cmd+=(--served-model-name "$S_MODEL")

# Add formats
cmd+=(--format json,md,png)
cmd+=(--save-result "$SAVE_PATH")

# Pass through user args, skipping --model and its value
i=0
while [[ $i -lt $N ]]; do
  if [[ "${args[$i]}" == "--model" ]]; then
    (( i += 2 )) || true
  else
    cmd+=("${args[$i]}")
    (( i++ )) || true
  fi
done

echo "---"

"${cmd[@]}"