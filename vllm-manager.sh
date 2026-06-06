#!/usr/bin/env bash
# vLLM Model Manager - structured start/stop/restart/logs/list/delete/status
# Each model has a YAML config in models/<name>.yaml
#
# Usage: ./vllm-manager.sh [flags] <command> [model-name]
# Flags: --remote  --local  --model <name>  --follow
#
# Remote execution via SSH when DRY_RUN is not set in .env.
# Use --local to force local, --remote to force remote (when DRY_RUN is set).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

# ── load .env (auto-exports variables) ──────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

# ── SSH remote config validation ────────────────────────────────────────────
validate_ssh_config() {
  [ -n "${SSH_USER:-}" ] || die "SSH_USER not set in .env"
  [ -n "${SSH_HOST:-}" ] || die "SSH_HOST not set in .env"
  [ -n "${SSH_KEY:-}" ]  || die "SSH_KEY not set in .env"
  [ -f "${SSH_KEY}" ]    || die "SSH key not found: ${SSH_KEY}"
}

# ── remote execution helper ─────────────────────────────────────────────────
run_remote() {
  validate_ssh_config
  local cmd="$1"; shift
  local ssh_cmd="ssh"
  [ -n "${SSH_PORT:-}" ] && ssh_cmd+=" -p ${SSH_PORT}"
  [ -n "${SSH_KEY:-}" ]  && ssh_cmd+=" -i ${SSH_KEY}"
  ssh_cmd+=" ${SSH_USER}@${SSH_HOST}"
  # Use SSH_DIR on the remote side so it finds its own .env and models/
  # VLLM_REMOTE=0 prevents recursive SSH calls on the remote side
  ssh_cmd+=" 'VLLM_REMOTE=0 cd \"${SSH_DIR}\" && ./vllm-manager.sh $cmd \"$@\"'"
  eval "$ssh_cmd"
}

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "❌ $*" >&2; exit 1; }
info() { printf "  📌 %s\n" "$*"; }
ok()   { echo "✅ $*"; }
warn() { printf "  ⚠️  %s\n" "$*"; }

usage() {
  cat <<'USAGE'

  📦 vLLM Model Manager

  Usage: ./vllm-manager.sh [flags] <command> --model <name>

  Flags:
    --remote          Force remote execution via SSH
    --local           Force local execution (opt-out from SSH)
    --model <name>    Model name (required; falls back to .env MODEL)
    --follow          Live log follow (for logs command)

  Commands:
    start    --model <name>  Stop any running model & start this one
    stop     --model <name>  Stop & remove a model container
    stop-all                   Stop & remove ALL model containers
    restart  --model <name>  Restart a model
    logs     --model <name>  Show logs for a model
    status                     Show docker ps output
    list                       Show all models with status
    delete   --model <name>  Remove stopped container entirely
    update                     Commit, push to develop, and pull on remote
    pull                       Pull latest from develop on remote only

  Remote execution:
    When DRY_RUN is NOT set in .env, commands run remotely via SSH by default.
    When DRY_RUN IS set, commands run locally (dry run). Use --remote to force SSH.

  If --model is omitted, falls back to MODEL=<name> from .env.

  Configs live in: models/*.yaml

  Examples:
    ./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4
    ./vllm-manager.sh --remote start --model qwen3.6-35b-a3b-nvfp4
    ./vllm-manager.sh --remote logs --model qwen3.6-35b-a3b-nvfp4 --follow
    ./vllm-manager.sh --remote stop-all
    ./vllm-manager.sh status
    ./vllm-manager.sh list
    ./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4
    ./vllm-manager.sh stop --model qwen3.6-35b-a3b-nvfp4
    ./vllm-manager.sh stop-all
    # With MODEL=qwen3.6-35b-a3b-nvfp4 in .env:
    ./vllm-manager.sh start       # uses default model
    ./vllm-manager.sh --model other-model start  # explicit flag overrides
USAGE
  exit 1
}

# ── YAML config loader ──────────────────────────────────────────────────────
# Uses Python stdlib only - no pip dependencies.

tmpconfig="/tmp/vllm-config-$$"
trap "rm -f $tmpconfig" EXIT

load_model_config() {
  local name="$1"
  local config_file="$MODELS_DIR/${name}.yaml"
  [ -f "$config_file" ] || die "No config found: $config_file (try: ls models/)"

  rm -f "$tmpconfig"

  python3 - "$config_file" >> "$tmpconfig" <<'PYEOF'
import sys, shlex

path = sys.argv[1]
env_vars = {}
model_args = []
volumes = []
image = ""
port = ""

section = None

with open(path) as f:
    for raw in f:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if not line[0:1].isspace():
            section = None
            if stripped.startswith("image:"):
                val = stripped.split(":", 1)[1].strip()
                for q in ('"', "'"):
                    val = val.strip(q)
                image = val
            elif stripped.startswith("port:"):
                val = stripped.split(":", 1)[1].strip()
                for q in ('"', "'"):
                    val = val.strip(q)
                port = val
            elif stripped.startswith("env:"):
                section = "env"
                # Handle inline env: "env: KEY=value"
                inline = stripped[len("env:"):].strip()
                if inline and "=" in inline:
                    k, v = inline.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip('"').strip("'")
            elif stripped.startswith("args:"):
                section = "args"
                # Handle inline args: "args: --model foo --port 8000"
                inline = stripped[len("args:"):].strip()
                if inline:
                    parts = shlex.split(inline)
                    model_args.extend(parts)
            elif stripped.startswith("volumes:"):
                section = "volumes"
                # Handle inline volume: "volumes: /host:/container"
                inline = stripped[len("volumes:"):].strip()
                if inline:
                    if inline.startswith("- "):
                        inline = inline[2:]
                    inline = inline.strip().strip('"').strip("'")
                    if inline:
                        volumes.append(inline)
            continue

        if section == "env" and "=" in stripped and not stripped.startswith("--"):
            k, v = stripped.split("=", 1)
            env_vars[k.strip()] = v.strip().strip('"').strip("'")
        elif section == "args" and stripped.startswith("--"):
            parts = shlex.split(stripped)
            model_args.extend(parts)
        elif section == "volumes":
            v = stripped.strip().strip('"').strip("'")
            # YAML list items start with "- "; strip the prefix
            if v.startswith("- "):
                v = v[2:]
            v = v.strip()
            if v:
                volumes.append(v)

print(f'IMAGE={shlex.quote(image)}')
port = port or "8000"
print(f'PORT={shlex.quote(port)}')
print(f'NUM_VOLS={len(volumes)}')
for i, v in enumerate(volumes):
    print(f'VOL_{i}={shlex.quote(v)}')

print(f'NUM_ARGS={len(model_args)}')
for i, a in enumerate(model_args):
    print(f'ARG_{i}={shlex.quote(a)}')

print(f'NUM_ENV={len(env_vars)}')
for i, (k, v) in enumerate(env_vars.items()):
    print(f'ENV_{i}={shlex.quote(f"{k}={v}")}')
PYEOF

  source "$tmpconfig"
}

# ── parse args into a quick-info map ────────────────────────────────────────
# Scans margs to extract key details for the display line.
parse_model_info() {
  local -a _args=("$@")
  local _j=0
  _minfo_model=""
  _minfo_quant=""
  _minfo_dtype=""
  _minfo_maxlen=""
  _minfo_attn=""
  _minfo_moe=""
  _minfo_extra=""
  _minfo_port="8000"
  local _key=""

  for (( _j=0; _j<${#_args[@]}; _j++ )); do
    local _arg="${_args[$_j]}"
    case "$_arg" in
      --model)                  _key="MODEL" ;;
      --quantization|-q)        _key="QUANT" ;;
      --dtype)                  _key="DTYPE" ;;
      --max-model-len)          _key="MAXLEN" ;;
      --attention-backend)      _key="ATTN" ;;
      --moe-backend)            _key="MOE" ;;
      --tensor-parallel-size)   _key="TP" ;;
      --port)                   _key="PORT" ;;
      *)
        if [ -n "$_key" ]; then
          case "$_key" in
            MODEL) _minfo_model="$_arg" ;;
            QUANT) _minfo_quant="$_arg" ;;
            DTYPE) _minfo_dtype="$_arg" ;;
            MAXLEN) _minfo_maxlen="$_arg" ;;
            ATTN)  _minfo_attn="$_arg" ;;
            MOE)   _minfo_moe="$_arg" ;;
            TP)    _minfo_extra+=" TP=$_arg" ;;
            PORT)  _minfo_port="$_arg" ;;
          esac
          _key=""
        fi
        ;;
    esac
  done

  [ -z "$_minfo_model" ] && _minfo_model="unknown"
  return 0
}

# ── stop helpers ────────────────────────────────────────────────────────────

stop_one() {
  local name="$1"
  local container="vllm-${name}"
  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$container"; then
    info "Stopping ${container} ..."
    docker stop "$container" >/dev/null 2>&1 && docker rm "$container" >/dev/null 2>&1
    ok "${container} stopped"
  else
    info "No container ${container} found"
  fi
}

cmd_stop()  { stop_one "$1"; }

cmd_stop_all() {
  local containers
  containers=$(docker ps -a --filter "name=vllm-" --format '{{.Names}}' 2>/dev/null || true)
  if [ -z "$containers" ]; then
    info "No vllm containers found"
    return
  fi
  echo "$containers" | while read -r c; do
    info "Stopping ${c} ..."
    docker stop "$c" >/dev/null 2>&1 && docker rm "$c" >/dev/null 2>&1
  done
  ok "All vllm containers stopped"
}

# ── start ───────────────────────────────────────────────────────────────────

cmd_start() {
  local name="$1"
  load_model_config "$name"

  # Stop any running model first
  local existing
  existing=$(docker ps --filter "name=vllm-" --format '{{.Names}}' 2>/dev/null || true)
  if [ -n "$existing" ]; then
    warn "Something else is running:"
    echo "$existing" | sed 's/^/  /'
    warn "Stopping it first..."
    cmd_stop_all 2>/dev/null || true
    sleep 2
  fi

  local container="vllm-${name}"
  PORT="${PORT:-8000}"

  # Build docker run flags
  local -a dr=()
  dr+=(-d --name "$container" --gpus all --ipc=host --restart unless-stopped)
  dr+=(-p "${PORT}:8000")

  # Mount host .cache → /root/.cache (covers HF cache, torch.compile, flashinfer, etc.)
  if [ -d "$HOME/.cache" ]; then
    dr+=(-v "${HOME}/.cache:/root/.cache")
  fi

  # Timezone sync
  [ -f /etc/localtime ] && dr+=(-v /etc/localtime:/etc/localtime:ro)
  [ -f /etc/timezone ] && dr+=(-v /etc/timezone:/etc/timezone:ro)

  # Extra volumes from config
  local i
  for ((i = 0; i < NUM_VOLS; i++)); do
    local vol_var="VOL_$i"
    local vol_str="${!vol_var}"
    # Expand $HOME and ${HOME} in volume paths
    vol_str=$(eval echo "$vol_str")
    dr+=(-v "$vol_str")
  done

  # Env vars from config (in addition to HF_TOKEN and VLLM_API_KEY)
  for ((i = 0; i < NUM_ENV; i++)); do
    local env_var="ENV_$i"
    dr+=(-e "${!env_var}")
  done

  # Always set these from shell env / defaults
  dr+=(-e "HF_TOKEN=${HF_TOKEN:-}")
  dr+=(-e "VLLM_API_KEY=${VLLM_API_KEY:-vllm}")
  dr+=(-e "SERVICE_NAME=${SERVICE_NAME:-vllm}")

  # Loki logging driver (only when LOKI_URL is set)
  if [ -n "${LOKI_URL:-}" ]; then
    dr+=(--log-driver=loki)
    dr+=(--log-opt "loki-url=${LOKI_URL}/loki/api/v1/push")
    dr+=(--log-opt "loki-external-labels=job=dockerlogs,stack=${SERVICE_NAME:-vllm},model=${name:-unknown}")
  fi

  # Image
  local img="$IMAGE"

  # Model args into array
  local -a margs=()
  for ((i = 0; i < NUM_ARGS; i++)); do
    local arg_var="ARG_$i"
    margs+=("${!arg_var}")
  done

  # ── DRY_RUN: skip everything, just show what would happen ──────────────
  if [ "${DRY_RUN:-false}" = "true" ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    local lower_name
    lower_name=$(echo "$name" | awk '{print tolower($0)}')
    echo "║              🚀 Starting vLLM: ${lower_name}               ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    info "Image  : ${img}"
    info "Port   : ${PORT}"
    info "Args   : ${#margs[@]} flags from config"

    echo ""
    echo "Command (simulated - no docker commands will run):"
    echo "  docker run ${dr[*]}"
    echo "  ${img} ${margs[*]}"
    echo ""
    ok "Dry run complete (no container created, no pull, no network)"
    return 0
  fi

  # ── Normal mode: always pull latest image ────────────────────────────

  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  local lower_name
  lower_name=$(echo "$name" | awk '{print tolower($0)}')
  echo "║              🚀 Starting vLLM: ${lower_name}               ║"
  echo "╚════════════════════════════════════════════════════════════╝"
  echo ""
  info "Image  : ${img}"
  info "Port   : ${PORT}"
  info "Args   : ${#margs[@]} flags from config"

  info "Pulling latest image ${img} …"
  docker pull "$img"

  # Show full command
  echo ""
  echo "Command:"
  echo "  docker run ${dr[*]}"
  echo "  ${img} ${margs[*]}"
  echo ""

  # Run
  docker run "${dr[@]}" "$img" "${margs[@]}"
}

cmd_restart() { stop_one "$1"; cmd_start "$1"; }

cmd_logs() {
  if [ "${2:-}" = "--follow" ]; then
    docker logs -f "vllm-$1"
  else
    docker logs --tail 100 "vllm-$1"
  fi
}

cmd_status() {
  docker ps --filter "name=vllm-" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
}

cmd_update() {
  # Step 1: commit and push locally
  info "Committing and pushing to develop..."
  git add -A
  git commit -m "auto-update: $(date '+%Y-%m-%d %H:%M:%S')" >/dev/null 2>&1 || info "Nothing to commit"
  git push origin develop >/dev/null 2>&1 || die "Failed to push to develop"
  ok "Pushed to origin/develop"

  # Step 2: pull on remote
  info "Pulling on remote ${SSH_USER}@${SSH_HOST}..."
  validate_ssh_config
  local ssh_cmd="ssh"
  [ -n "${SSH_PORT:-}" ] && ssh_cmd+=" -p ${SSH_PORT}"
  [ -n "${SSH_KEY:-}" ]  && ssh_cmd+=" -i ${SSH_KEY}"
  ssh_cmd+=" ${SSH_USER}@${SSH_HOST}"
  ssh_cmd+=" 'cd \"${SSH_DIR}\" && git pull origin develop'"
  eval "$ssh_cmd"
  ok "Remote updated"
}

cmd_pull() {
  info "Pulling latest on remote ${SSH_USER}@${SSH_HOST}..."
  validate_ssh_config
  local ssh_cmd="ssh"
  [ -n "${SSH_PORT:-}" ] && ssh_cmd+=" -p ${SSH_PORT}"
  [ -n "${SSH_KEY:-}" ]  && ssh_cmd+=" -i ${SSH_KEY}"
  ssh_cmd+=" ${SSH_USER}@${SSH_HOST}"
  ssh_cmd+=" 'cd \"${SSH_DIR}\" && git pull origin develop'"
  eval "$ssh_cmd"
  ok "Remote pulled"
}

cmd_delete() {
  docker rm "vllm-$1" 2>/dev/null && ok "Removed vllm-$1" || info "No such container"
}

cmd_list() {
  echo ""
  echo "╔════════════════════════════════════════════════════════════╗"
  echo "║              📊 vLLM Model Status                         ║"
  echo "╚════════════════════════════════════════════════════════════╝"
  echo ""

  # Running containers
  local running
  running=$(docker ps --filter "name=vllm-" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true)
  if [ -n "$running" ]; then
    info "Running:"
    echo "$running" | sed 's/^/  /'
    echo ""
  fi

  # Stopped containers
  local stopped
  stopped=$(docker ps -a --filter "name=vllm-" --filter "status=exited" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' 2>/dev/null || true)
  if [ -n "$stopped" ]; then
    info "Stopped:"
    echo "$stopped" | sed 's/^/  /'
    echo ""
  fi

  # Config files
  info "Available configs:"
  if compgen -G "$MODELS_DIR/*.yaml" >/dev/null 2>&1; then
    for f in "$MODELS_DIR"/*.yaml; do
      [ -f "$f" ] && info "$(basename "$f" .yaml)  ←  $f"
    done
  fi
  echo ""
}

# ── main ─────────────────────────────────────────────────────────────────────

[ $# -ge 1 ] || usage

# ── Parse flags before command ──────────────────────────────────────────────
REMOTE=false
LOCAL=false
FOLLOW=false
MODEL_FLAG=""

# First pass: extract flags
remaining=()
_pos=0
while _pos=$(( _pos + 1 )); do
  _arg="${!_pos:-}"
  [ -n "$_arg" ] || break
  case "$_arg" in
    --remote)
      REMOTE=true
      ;;
    --local)
      LOCAL=true
      ;;
    --model)
      _pos=$(( _pos + 1 ))
      MODEL_FLAG="${!_pos:-}"
      continue
      ;;
    --follow)
      FOLLOW=true
      ;;
    *)
      remaining+=("$_arg")
      ;;
  esac
done

# Determine execution mode
if [ "$LOCAL" = true ]; then
  REMOTE=false
elif [ "${VLLM_REMOTE:-}" = "0" ]; then
  # Remote .env sets VLLM_REMOTE=0 to disable SSH recursion
  REMOTE=false
elif [ "$REMOTE" = true ]; then
  : # remote forced
elif [ -z "${DRY_RUN:-}" ]; then
  # DRY_RUN not set → default to remote
  REMOTE=true
fi

# Show remote indicator
if [ "$REMOTE" = true ]; then
  echo "🌐 Remote → ${SSH_USER:-?}@${SSH_HOST:-?}${SSH_PORT:+:$SSH_PORT}"
fi

# Resolve model name: --model flag > .env MODEL
if [ -n "$MODEL_FLAG" ]; then
  MODEL_RESOLVED="$MODEL_FLAG"
else
  MODEL_RESOLVED="${MODEL:-}"
fi

[ -n "$MODEL_RESOLVED" ] || die "No model specified. Usage: $0 --model <name> (or set MODEL=<name> in .env)"

# Get command from remaining args
cmd="${remaining[0]:-}"
[ -n "$cmd" ] || usage

# ── Route commands ──────────────────────────────────────────────────────────
case "$cmd" in
  start)
    if [ "$REMOTE" = true ]; then
      run_remote "start" "--model" "$MODEL_RESOLVED"
    else
      cmd_start "$MODEL_RESOLVED"
    fi
    ;;
  stop)
    if [ "$REMOTE" = true ]; then
      run_remote "stop" "--model" "$MODEL_RESOLVED"
    else
      cmd_stop "$MODEL_RESOLVED"
    fi
    ;;
  stop-all)
    if [ "$REMOTE" = true ]; then
      run_remote "stop-all"
    else
      cmd_stop_all
    fi
    ;;
  restart)
    if [ "$REMOTE" = true ]; then
      run_remote "restart" "--model" "$MODEL_RESOLVED"
    else
      cmd_restart "$MODEL_RESOLVED"
    fi
    ;;
  logs)
    if [ "$REMOTE" = true ]; then
      if [ "$FOLLOW" = true ]; then
        run_remote "logs" "--model" "$MODEL_RESOLVED" "--follow"
      else
        run_remote "logs" "--model" "$MODEL_RESOLVED"
      fi
    else
      if [ "$FOLLOW" = true ]; then
        cmd_logs "$MODEL_RESOLVED" "--follow"
      else
        cmd_logs "$MODEL_RESOLVED"
      fi
    fi
    ;;
  status)
    if [ "$REMOTE" = true ]; then
      run_remote "status"
    else
      cmd_status
    fi
    ;;
  update)
    cmd_update
    ;;
  pull)
    cmd_pull
    ;;
  list)
    if [ "$REMOTE" = true ]; then
      run_remote "list"
    else
      cmd_list
    fi
    ;;
  delete)
    if [ "$REMOTE" = true ]; then
      run_remote "delete" "--model" "$MODEL_RESOLVED"
    else
      cmd_delete "$MODEL_RESOLVED"
    fi
    ;;
  *)
    usage
    ;;
esac
