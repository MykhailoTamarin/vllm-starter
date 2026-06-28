#!/usr/bin/env bash
# wait-for-idle.sh — wait for vLLM to become idle (no running/waiting requests)
#                    Reports KV cache usage after idle.
# Usage: ./wait-for-idle.sh <BASE_URL> [--max-retries N] [--interval N]
set -euo pipefail

BASE_URL="${1:-}"
MAX_RETRIES=300
INTERVAL=2

if [[ -z "$BASE_URL" ]]; then
  echo "Usage: $0 <base_url> [--max-retries N] [--interval N]" >&2
  exit 1
fi

# Parse optional flags after BASE_URL
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --interval)    INTERVAL="$2"; shift 2 ;;
    *)             echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Trim trailing slash if present
BASE_URL="${BASE_URL%/}"

echo -n "⏳ Waiting for idle..."

retries=0
while true; do
  metrics=$(curl -s --max-time 5 "${BASE_URL}/metrics" 2>/dev/null) || {
    (( retries++ )) || true
    if [[ $retries -gt $MAX_RETRIES ]]; then
      echo ""
      echo "❌ Timeout after ${MAX_RETRIES} retries (${MAX_RETRIES}×${INTERVAL}s)" >&2
      exit 1
    fi
    sleep "$INTERVAL"
    continue
  }

  # Parse vllm:num_requests_running, num_requests_waiting, kv_cache_usage_perc
  running=$(echo "$metrics" | grep '^vllm:num_requests_running' | tail -1 | awk '{print $NF}')
  waiting=$(echo "$metrics" | grep '^vllm:num_requests_waiting' | tail -1 | awk '{print $NF}')
  cache=$(echo "$metrics" | grep '^vllm:kv_cache_usage_perc' | tail -1 | awk '{print $NF}')

  # Ensure values are numeric (default to 1 if unparseable — keeps waiting)
  [[ "$running" == "0" ]] || running="1"
  [[ "$waiting" == "0" ]] || waiting="1"
  [[ -z "$cache" ]] && cache="1"

  if [[ "$running" == "0" ]] && [[ "$waiting" == "0" ]]; then
    cache_pct=$(python3 -c "print(f'{100*$cache:.1f}')" 2>/dev/null || echo "$cache")
    echo " ✅ Idle (cache: ${cache_pct}%)"
    exit 0
  fi

  (( retries++ )) || true
  if [[ $retries -gt $MAX_RETRIES ]]; then
    echo ""
    echo "❌ Timeout after ${MAX_RETRIES} retries (${MAX_RETRIES}×${INTERVAL}s)" >&2
    exit 1
  fi

  sleep "$INTERVAL"
done
