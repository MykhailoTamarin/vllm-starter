#!/usr/bin/env bash
# wait-for-idle.sh — wait for vLLM to become idle (no running requests)
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

echo -n "⏳ Waiting for idle (no running requests)..."

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

  # Parse vllm:num_requests_running
  # May appear with or without labels: vllm:num_requests_running 0
  #   or  vllm:num_requests_running{queue_size="0"} 0
  current_value=$(echo "$metrics" | grep '^vllm:num_requests_running' | tail -1)

  if [[ -n "${current_value:-}" ]]; then
    # Extract numeric value (last field after the last space)
    running=$(echo "$current_value" | awk '{print $NF}')
    if [[ "$running" == "0" ]]; then
      echo " ✅ Idle"
      exit 0
    fi
  fi

  (( retries++ )) || true
  if [[ $retries -gt $MAX_RETRIES ]]; then
    echo ""
    echo "❌ Timeout after ${MAX_RETRIES} retries (${MAX_RETRIES}×${INTERVAL}s)" >&2
    exit 1
  fi

  sleep "$INTERVAL"
done
