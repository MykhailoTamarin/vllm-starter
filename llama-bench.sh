#!/usr/bin/env bash
# llama-bench.sh — Run llama-benchy benchmarks with optional wait-for-idle gating
# ───────────────────────────────────────────────────────────────────────────────
# Standard mode (default): single benchy call, save MD to benchmark_*.md
# wait-idle mode:          sequential benchy calls with idle gate, save JSONs
# ───────────────────────────────────────────────────────────────────────────────
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
[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

# ── Parse flags by index ─────────────────────────────────────────────────────
args=("$@")
N=${#args[@]}

MODEL_IDX=-1          # index of value after --model
DEPTH_START=-1        # start index of first --depth value
DEPTH_END=-1          # end index (exclusive) of last --depth value
CONC_START=-1         # start index of first --concurrency value
CONC_END=-1           # end index (exclusive) of last --concurrency value
WAIT_IDLE=false       # --wait-idle flag
RUN_DEFAULT=1         # iterations per {C×D} in wait-idle mode
REPEAT=1              # number of times to run the entire suite
REPEAT_IDX=-1         # index of value after --repeat

i=0
while [[ $i -lt $N ]]; do
  case "${args[$i]}" in
    --model)      [[ $((i+1)) -lt $N ]] && MODEL_IDX=$((i+1)) ;;
    --depth)      [[ $DEPTH_START -eq -1 ]] && DEPTH_START=$((i+1)); DEPTH_END=$((i+2)); while [[ $DEPTH_END -lt $N ]] &&  [[ "${args[$DEPTH_END]}" != --* ]]; do (( DEPTH_END++ )) || true; done ;;
    --concurrency) [[ $CONC_START -eq -1 ]] && CONC_START=$((i+1)); CONC_END=$((i+2)); while [[ $CONC_END -lt $N ]] && [[ "${args[$CONC_END]}" != --* ]]; do (( CONC_END++ )) || true; done ;;
    --wait-idle)  WAIT_IDLE=true; RUN_DEFAULT=1 ;;
    --repeat)     [[ $((i+1)) -lt $N ]] && REPEAT_IDX=$((i+1)) && REPEAT="${args[$((i+1))]}" ;;
  esac
  (( i++ )) || true
done

# ── Build value arrays ───────────────────────────────────────────────────────
MODEL_VAL=""
if [[ $MODEL_IDX -ge 0 ]] && [[ $MODEL_IDX -lt $N ]]; then
  MODEL_VAL="${args[$MODEL_IDX]}"
fi
DEPTHS=()
[[ $DEPTH_START -ge 0 ]] && [[ $DEPTH_START -lt $DEPTH_END ]] && \
  for (( j=DEPTH_START; j<DEPTH_END; j++ )); do DEPTHS+=("${args[$j]}"); done
CONCS=()
[[ $CONC_START -ge 0 ]] && [[ $CONC_START -lt $CONC_END ]] && \
  for (( j=CONC_START; j<CONC_END; j++ )); do CONCS+=("${args[$j]}"); done

# Defaults
MODEL="${MODEL_VAL:-${MODEL:-}}"
[[ -z "$MODEL" ]] && { echo "❌ No --model (use --model or set MODEL in .env)" >&2; exit 1; }
[[ ${#CONCS[@]} -eq 0 ]] && CONCS=(1)
[[ ${#DEPTHS[@]} -eq 0 ]] && DEPTHS=(1024)

# ── Resolve model from YAML ─────────────────────────────────────────────────
model_yaml="models/${MODEL}.yaml"
BENCH_MODEL="${MODEL}"
SERVED_MODEL=""

if [[ -f "$model_yaml" ]]; then
  BENCH_MODEL=$(awk '/^args:/{f=1} f && /--model/{print $2; exit}' "$model_yaml")
  SERVED_MODEL=$(awk '/^args:/{f=1} f && /--served-model-name/{print $2; exit}' "$model_yaml")
  yaml_port=$(awk '/^port:/{print $2; exit}' "$model_yaml")
  [[ -n "${yaml_port:-}" ]] && MODEL_PORT="$yaml_port"
fi
[[ -z "${BENCH_MODEL:-}" ]] && BENCH_MODEL="$MODEL"

# ── Collect non-consumed bench args ──────────────────────────────────────────
bench_extra=()
i=0
while [[ $i -lt $N ]]; do
  skip=0
  case "${args[$i]}" in
  --model) skip=1 ;;
  --depth) skip=1 ;;
  --concurrency) skip=1 ;;
  --wait-idle) skip=1 ;;
  --repeat) skip=1 ;;
  esac
  if [[ $skip -eq 0 ]]; then
    [[ $MODEL_IDX -gt 0 ]] && [[ $i -eq $MODEL_IDX ]] && skip=1
    [[ $DEPTH_START -ge 0 ]] && [[ $i -ge $DEPTH_START ]] && [[ $i -lt $DEPTH_END ]] && skip=1
    [[ $CONC_START -ge 0 ]] && [[ $i -ge $CONC_START ]] && [[ $i -lt $CONC_END ]] && skip=1
    [[ $REPEAT_IDX -gt 0 ]] && [[ $i -eq $REPEAT_IDX ]] && skip=1
    [[ $skip -eq 0 ]] && bench_extra+=("${args[$i]}")
  fi
  (( i++ )) || true
done

BASE_URL="http://${SSH_HOST}:${MODEL_PORT}/v1"

# ══════════════════════════════════════════════════════════════════════════════
# wait-idle mode
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$WAIT_IDLE" == "true" ]]; then
  SCRIPT_DIR="$(dirname "$0")"
  WFI="${SCRIPT_DIR}/scripts/wait-for-idle.sh"
  if [[ ! -x "$WFI" ]]; then
    echo "❌ wait-for-idle.sh not found at ${WFI}" >&2
    exit 1
  fi

  CONC_PART="c${CONCS[*]}"
  CONC_PART="${CONC_PART// /_}"
  DEPTH_PART="d${DEPTHS[*]}"
  DEPTH_PART="${DEPTH_PART// /_}"
  out_dir="models/benchmarks/${MODEL}/${CONC_PART}_${DEPTH_PART}"
  mkdir -p "$out_dir"

  echo "▶ llama-benchy (wait-idle mode) — model: ${BENCH_MODEL}"
  echo "  output:      ${out_dir}"
  echo "  concurrency: ${CONCS[*]}"
  echo "  depths:      ${DEPTHS[*]}"
  echo "  repeat:      ${REPEAT}x"
  echo

  suites=$(( ${#DEPTHS[@]} * ${#CONCS[@]} ))
  total=$(( suites * ${RUN_DEFAULT} * ${REPEAT} ))
  count=0

  for rep in $(seq 1 "$REPEAT"); do
    echo "━━━━━ Suite #${rep} / ${REPEAT} ━━━━━"

    for d in "${DEPTHS[@]}"; do
      for c in "${CONCS[@]}"; do
        for r in $(seq 1 "$RUN_DEFAULT"); do
          (( count++ )) || true
          echo "── ${count}/${total} C=${c} d=${d} r=${r} ──"

          save_file="${out_dir}/c${c}_d${d}_r${r}_s${rep}.json"

          echo "  idle-check..."
          "$WFI" "$BASE_URL" || {
            echo "  ⚠ idle timeout — continuing" >&2
          }

          bc=(llama-benchy --base-url "$BASE_URL" --api-key "$API_KEY" --model "$BENCH_MODEL"
              --depth "$d" --concurrency "$c" --save-result "$save_file" --format json
              --no-cache --runs 1 --no-results-on-fail)
          [[ -n "${SERVED_MODEL:-}" ]] && bc+=(--served-model-name "$SERVED_MODEL")
          [[ ${#bench_extra[@]} -gt 0 ]] && bc+=("${bench_extra[@]}")

          "${bc[@]}" || { echo "⚠️ benchy failed" >&2; continue; }
          echo "  done"
          echo
        done
      done
    done
  done

  echo "✅ Done. ${count} runs in ${out_dir}"

  # Auto-parse and save results — follow same naming as standard mode
  SCRIPT_DIR="$(dirname "$0")"
  if [[ -x "$SCRIPT_DIR/scripts/bench-parse.sh" ]]; then
    TIMESTAMP="$(date +%d_%m_%y_%H_%M)"
    RESULT_MD="models/benchmarks/${MODEL}/benchmark_${TIMESTAMP}_${CONC_PART}_${DEPTH_PART}.md"
    echo
    echo "📊 Generating results..."
    "$SCRIPT_DIR/scripts/bench-parse.sh" -d "$out_dir" -o "$RESULT_MD" 2>/dev/null && echo "  → ${RESULT_MD}" || true
  fi

  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Standard mode — single benchy call (original flow)
# ══════════════════════════════════════════════════════════════════════════════
CONC_PART="${CONCS[*]}"
CONC_PART="${CONC_PART// /_}"
BENCH_FILE="models/benchmarks/${MODEL}/benchmark_$(date +%d_%m_%y_%H_%M)_c${CONC_PART}.md"
mkdir -p "models/benchmarks/${MODEL}"

echo "▶ llama-benchy — model: ${BENCH_MODEL}"
echo "  output:   ${BENCH_FILE}"
echo

bc=(llama-benchy --base-url "$BASE_URL" --api-key "$API_KEY" --model "$BENCH_MODEL"
    --save-result "${BENCH_FILE}"
    --no-cache --runs 3 --no-results-on-fail)
[[ -n "${SERVED_MODEL:-}" ]] && bc+=(--served-model-name "$SERVED_MODEL")
[[ ${#CONCS[@]} -gt 0 ]] && bc+=(--concurrency "${CONCS[@]}")
[[ ${#bench_extra[@]} -gt 0 ]] && bc+=("${bench_extra[@]}")

"${bc[@]}"