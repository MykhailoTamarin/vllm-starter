#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── Check llama-benchy binary ─────────────────────────────────────────────────
if ! command -v llama-benchy &> /dev/null; then
  echo "❌ llama-benchy not found."
  echo "   Install with: uvx llama-benchy"
  echo "   Or:        git clone https://github.com/eugr/llama-benchy && cd llama-benchy && pip install -e ."
  exit 1
fi

# ── Read .env ─────────────────────────────────────────────────────────────────
set -a; source .env; set +a

API_KEY="${VLLM_API_KEY:-vllm}"
SSH_HOST="${SSH_HOST:-localhost}"
MODEL_PORT="${MODEL_PORT:-8000}"

# Export HF_TOKEN so llama-benchy can pull from HF if needed
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

# ── Parse flags by index ──────────────────────────────────────────────────────
args=("$@")
N=${#args[@]}

MODEL_VAL_IDX=-1        # index of value after --model
CONC_START=-1           # start index of --concurrency values
CONC_END=-1             # end index (exclusive) of --concurrency values

i=0
while [[ $i -lt $N ]]; do
  case "${args[$i]}" in
    --model)
      [[ $((i+1)) -lt $N ]] && MODEL_VAL_IDX=$((i+1))
      ;;
    --concurrency)
      [[ $CONC_START -eq -1 ]] && CONC_START=$((i+1))
      CONC_END=$((i+2))
      while [[ $CONC_END -lt $N ]] && [[ "${args[$CONC_END]}" != --* ]]; do
        (( CONC_END++ )) || true
      done
      ;;
  esac
  (( i++ )) || true
done

# ── Resolve model ─────────────────────────────────────────────────────────────
MODEL_FROM_CLI=""
if [[ $MODEL_VAL_IDX -ge 0 ]] && [[ $MODEL_VAL_IDX -lt $N ]]; then
  MODEL_FROM_CLI="${args[$MODEL_VAL_IDX]}"
fi
MODEL="${MODEL_FROM_CLI:-${MODEL:-}}"

if [[ -z "$MODEL" ]]; then
  echo "❌ No --model specified (use --model or set MODEL in .env)" >&2
  exit 1
fi

BENCH_MODEL=""
SERVED_NAME=""

model_file="models/${MODEL}.yaml"
if [[ -f "$model_file" ]]; then
  BENCH_MODEL=$(awk '/^args:/{f=1} f && /--model /{print $2; exit}' "$model_file")
  SERVED_NAME=$(awk '/^args:/{f=1} f && /--served-model-name /{print $2; exit}' "$model_file")
  YAML_PORT=$(awk '/^port:/{print $2; exit}' "$model_file")
  [[ -n "${YAML_PORT:-}" ]] && MODEL_PORT="$YAML_PORT"
else
  BENCH_MODEL="$MODEL"
fi

BASE_URL="http://${SSH_HOST}:${MODEL_PORT}/v1"

[[ -z "${BENCH_MODEL:-}" ]] && { echo "❌ No --model found inside ${model_file}" >&2; exit 1; }

# ── Collect remaining args (skip consumed ranges) ────────────────────────────
BENCH_ARGS=()
i=0
while [[ $i -lt $N ]]; do
  skip=0
  if [[ "${args[$i]}" == "--model" ]]; then
    skip=1
  elif [[ $MODEL_VAL_IDX -ge 0 ]] && [[ $i -eq $MODEL_VAL_IDX ]]; then
    skip=1
  fi
  if [[ "${args[$i]}" == "--concurrency" ]]; then
    skip=1
  elif [[ $CONC_START -ge 0 ]] && [[ $i -ge $CONC_START ]] && [[ $i -lt $CONC_END ]]; then
    skip=1
  fi
  [[ $skip -eq 0 ]] && BENCH_ARGS+=("${args[$i]}")
  (( i++ )) || true
done

# ── Build concurrency array ──────────────────────────────────────────────────
CONC_ARR=()
if [[ $CONC_START -ge 0 ]]; then
  for (( j=CONC_START; j<CONC_END; j++ )); do
    CONC_ARR+=("${args[$j]}")
  done
fi

# ── Auto-generate result path ────────────────────────────────────────────────
# models/benchmarks/{yaml_name}/benchmark_dd_mm_yy_HH_mm[_c{concurrency}].md
BENCH_DIR="models/benchmarks/${MODEL}"
TIMESTAMP=$(date +"%d_%m_%y_%H_%M")
CONC_LIST="${CONC_ARR[*]:-1}"
CONC_PART="${CONC_LIST// /_}"
CONC_POSTFIX="_c${CONC_PART}"
BENCH_FILE="${BENCH_DIR}/benchmark_${TIMESTAMP}${CONC_POSTFIX}.md"

mkdir -p "${BENCH_DIR}"

# ── Build benchy command ──────────────────────────────────────────────────────
benchy_cmd=(
  llama-benchy
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --model "${BENCH_MODEL}"
  --save-result "${BENCH_FILE}"
)

[[ -n "${SERVED_NAME:-}" ]] && benchy_cmd+=(--served-model-name "${SERVED_NAME}")
[[ ${#CONC_ARR[@]} -gt 0 ]] && benchy_cmd+=(--concurrency "${CONC_ARR[@]}")

[[ ${#BENCH_ARGS[@]} -gt 0 ]] && benchy_cmd+=("${BENCH_ARGS[@]}")

# ── Run ───────────────────────────────────────────────────────────────────────
echo "▶ llama-benchy — model: ${BENCH_MODEL}"
echo "  base_url: ${BASE_URL}"
echo "  api_key:  ${API_KEY}"
[[ -n "${SERVED_NAME:-}" ]] && echo "  served_name: ${SERVED_NAME}"
[[ ${#CONC_ARR[@]} -gt 0 ]] && echo "  concurrency:  ${CONC_ARR[*]}"
echo "  results:    ${BENCH_FILE}"
echo

"${benchy_cmd[@]}"
