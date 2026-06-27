#!/usr/bin/env bash
# bench-parse.sh - Parse llama-benchy JSON outputs into a markdown table
# Usage: bench-parse.sh [-d <dir>] [-o <out>] [-p <pp>]
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [-d <dir>] [-o <out>] [-p <pp>]"
  echo "  Parse benchmark JSON files into a markdown table"
  echo ""
  echo "Options:"
  echo "  -d <dir>   Directory with benchmark JSON files (default: .)"
  echo "  -o <out>   Save MD to file (default: stdout)"
  echo "  -p <pp>    Prefill prompt size (default: 2048)"
  exit 1
}

BENCH_DIR="."
OUTPUT=""
PP_VALUE=2048

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) BENCH_DIR="$2"; shift 2 ;;
    -o) OUTPUT="$2"; shift 2 ;;
    -p) PP_VALUE="$2"; shift 2 ;;
    *) usage ;;
  esac
done

if [ ! -d "$BENCH_DIR" ]; then
  echo "❌ Directory not found: $BENCH_DIR" >&2; exit 1
fi

SCRIPT_DIR="$(dirname "$0")"
if [ -n "$OUTPUT" ]; then
  exec python3 "$SCRIPT_DIR/bench-parse.py" -d "$BENCH_DIR" -o "$OUTPUT" -p "$PP_VALUE"
else
  exec python3 "$SCRIPT_DIR/bench-parse.py" -d "$BENCH_DIR" -p "$PP_VALUE"
fi
