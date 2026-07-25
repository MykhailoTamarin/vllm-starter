#!/usr/bin/env bash
# llama-bench.sh — Thin wrapper around upstream llama-benchy + PNG post-process.
set -euo pipefail

cd "$(dirname "$0")"
set -a; source .env; set +a

API_K="${VLLM_API_KEY:-vllm}"
SSH_H="${SSH_HOST:-localhost}"
MODEL_P="${MODEL_PORT:-8000}"

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
  YPORT=$(grep -- '--port' "$YAML" | head -1 | sed 's/.*--port[[:space:]]*\([0-9]*\).*/\1/')
  [[ -n "${YPORT:-}" ]] && MODEL_P="$YPORT"
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

# Convert space-separated numbers to min-max range
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

[[ "$HAS_CONCURRENCY" != "true" ]] && CONCURRENCY_PART="_c1${CONCURRENCY_PART}"

BENCH_DIR="$(pwd)/models/benchmarks/${MODEL_NAME}"
mkdir -p "$BENCH_DIR"
SAVE_PATH="${BENCH_DIR}/benchmark_${TIMESTAMP}${CONCURRENCY_PART}"

# ── Build command ─────────────────────────────────────────────────────────────
cmd=(uvx llama-benchy)
cmd+=(--base-url "http://$SSH_H:$MODEL_P/v1")
cmd+=(--api-key "$API_K")
cmd+=(--model "$B_MODEL")
[[ -n "${S_MODEL:-}" ]] && cmd+=(--served-model-name "$S_MODEL")

cmd+=(--format json)
cmd+=(--save-result "$SAVE_PATH")

# Pass through user args, skipping --model and its value,
# and flags that don't exist in upstream llama-benchy
skip_flags=("--idle-wait" "--idle-interval" "--idle-max-retries")
i=0
while [[ $i -lt $N ]]; do
  if [[ "${args[$i]}" == "--model" ]]; then
    (( i += 2 )) || true
  elif [[ " ${skip_flags[*]} " == *" ${args[$i]} "* ]]; then
    # Skip the flag and its value arg if present
    (( i++ )) || true
    # Check if next arg is a value (not a flag)
    if [[ $i -lt $N && ! "${args[$i]}" == --* ]]; then
      (( i++ )) || true
    fi
  else
    cmd+=("${args[$i]}")
    (( i++ )) || true
  fi
done

echo "---"

# ── Run benchmark ────────────────────────────────────────────────────────────
if ! "${cmd[@]}"; then
  echo "llama-benchy failed" >&2
  exit 1
fi

# Upstream saves without .json extension — rename so gitignore catches it
if [[ -f "$SAVE_PATH" && ! "$SAVE_PATH" == *.json ]]; then
  mv "$SAVE_PATH" "${SAVE_PATH}.json"
fi
JSON_PATH="${SAVE_PATH}.json"

# ── Post-process: generate MD + PNG from JSON ──────────────────────────────
if [[ -f "$JSON_PATH" ]]; then
  python3 scripts/bench-process.py "$JSON_PATH"
else
  echo "No JSON output found at $SAVE_PATH — skipping post-process" >&2
fi
