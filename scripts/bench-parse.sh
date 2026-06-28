#!/usr/bin/env bash
# bench-parse.sh - Parse llama-benchy JSON outputs into a markdown table
#                  Also generates PNG graph (with same naming pattern)
# Usage: bench-parse.sh [-d <dir>] [-o <out>] [-p <pp>] [-g|-G]
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [-d <dir>] [-o <out>] [-p <pp>] [-g|-G]"
  echo "  Parse benchmark JSON files into a markdown table and PNG graph"
  echo ""
  echo "Options:"
  echo "  -d <dir>   Directory with benchmark JSON files (default: .)"
  echo "  -o <out>   Save MD to file (default: stdout)"
  echo "  -p <pp>    Prefill prompt size (default: 2048)"
  echo "  -g         Generate PNG graph (default when -o is used)"
  echo "  -G         Skip graph generation"
  exit 1
}

BENCH_DIR="."
OUTPUT=""
PP_VALUE=2048
GRAPH="true"  # default: generate graphs when -o is used

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) BENCH_DIR="$2"; shift 2 ;;
    -o) OUTPUT="$2"; shift 2 ;;
    -p) PP_VALUE="$2"; shift 2 ;;
    -g) GRAPH="true"; shift ;;
    -G) GRAPH="false"; shift ;;
    *) usage ;;
  esac
done

if [ ! -d "$BENCH_DIR" ]; then
  echo "❌ Directory not found: $BENCH_DIR" >&2; exit 1
fi

SCRIPT_DIR="$(dirname "$0")"

# ── Output mode (file): always write MD, optionally graph ─────────────────
if [ -n "$OUTPUT" ]; then
  python3 "$SCRIPT_DIR/bench-parse.py" -d "$BENCH_DIR" -o "$OUTPUT" -p "$PP_VALUE"
  if [ "$GRAPH" != "false" ]; then
    GRAPH_OUT="${OUTPUT%.md}.png"
    python3 "$SCRIPT_DIR/bench-graph.py" -d "$BENCH_DIR" -o "$GRAPH_OUT" && echo "  → ${GRAPH_OUT}" || echo "  ⚠ graph generation failed" >&2
  fi
else
  exec python3 "$SCRIPT_DIR/bench-parse.py" -d "$BENCH_DIR" -p "$PP_VALUE"
fi
