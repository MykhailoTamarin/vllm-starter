# Agents — vLLM Model Manager

This repo manages vLLM model containers on a DGX Spark. Each model is a YAML config in `models/`, controlled by `vllm-manager.sh`.

---

## ⚠️ Critical Rules

1. **Never commit or push unless explicitly asked.**
2. **Always work on `develop` branch. Never push to `main`.**
3. **Always pull `main` before starting work.**
4. **Always test with `DRY_RUN=true` before committing.**

---

## Git Workflow

```bash
# 1. Start fresh
git pull origin main
git switch develop
git pull origin develop

# 2. Make changes...

# 3. When explicitly asked to commit & push:
git add -A
git commit -m "your message here"
git push origin develop
```

---

## Project Structure

```
.
├── vllm-manager.sh          # Main controller (start/stop/restart/logs/list/delete/status)
├── .env                      # Config: HF_TOKEN, SSH keys, DRY_RUN, MODEL, etc.
├── models/
│   ├── template.yaml         # Full template with all options documented
│   └── *.yaml                # One per model (clean, no comments)
├── README.md                 # User-facing docs
├── FOR_AGENTS.md             # Legacy agent guide (deprecated)
└── AGENTS.md                 # This file
```

---

## Manager Commands

Model name is always specified via `--model <name>` flag (or `.env MODEL`).

| Command | Description |
|---------|-------------|
| `start --model <name>` | Stop all running models, then start this one |
| `stop --model <name>` | Stop & remove container |
| `stop-all` | Stop & remove all containers |
| `restart --model <name>` | Stop then start a model |
| `logs --model <name>` | Show last 100 lines (local only supports `--follow` for live) |
| `status` | Show docker ps for vllm containers |
| `list` | Show all models with status |
| `delete --model <name>` | Remove stopped container entirely |
| `update` | Commit, push to develop, pull on remote |
| `pull` | Pull latest from develop on remote only |

### Flags

| Flag | Description |
|------|-------------|
| `--remote` | Force remote execution via SSH |
| `--local` | Force local execution (opt-out SSH) |
| `--model <name>` | Model name (required; falls back to `.env MODEL`) |
| `--follow` | Live log follow (local only, not supported over SSH) |

### Execution Mode

| `DRY_RUN` | `--remote` | `--local` | Result |
|-----------|------------|-----------|--------|
| `true` | absent | absent | Local dry run (no docker) |
| `true` | `--remote` | absent | Remote via SSH |
| `true` | absent | `--local` | Local dry run |
| *unset* | absent | absent | Remote via SSH |
| *unset* | `--remote` | absent | Remote via SSH |
| *unset* | absent | `--local` | Local dry run |

### Examples

```bash
# Local (when DRY_RUN=true or --local)
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh --local status
./vllm-manager.sh --local list

# Remote (when DRY_RUN unset or --remote)
./vllm-manager.sh --remote start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4 --follow  # local only
./vllm-manager.sh --remote stop-all

# With MODEL=qwen3.6-35b-a3b-nvfp4 in .env
./vllm-manager.sh start          # uses default model
./vllm-manager.sh --model other start  # explicit flag overrides
```

---

## Adding a New Model

### Step 1: Create config

```bash
cp models/template.yaml models/<name>.yaml
```

### Step 2: Fill in the YAML

Required fields:
- `image:` — Docker image (e.g. `vllm/vllm-openai:latest`)
- `args:` — At minimum `--model <repo-id>`

Common fields:
- `port:` — Host port (default 8000)
- `env:` — Container env vars
- `volumes:` — Extra host mounts

**Rule:** Model YAML files must be clean — no comments in `env:` or `args:` sections. Comments stay only in `template.yaml`.

### Step 3: Test with DRY_RUN

```bash
# .env already has DRY_RUN=true
./vllm-manager.sh start --model <name>
```

Verify the output contains:
- ✅ Correct image tag
- ✅ Correct port mapping (`-p 8000:8000`)
- ✅ `N flags from config` where N > 0
- ✅ Env vars from YAML (`-e KEY=VALUE`)
- ✅ HF cache mount (`-v /home/.../.cache:/root/.cache`)
- ✅ Full docker run command shown

If you see `0 flags from config` or missing env vars, the YAML parser is not reading the config correctly.

### Step 4: Run for real

```bash
# Set DRY_RUN=false in .env (or remove it)
./vllm-manager.sh start --model <name>
```

---

## YAML Config Reference

### Minimal config

```yaml
image: vllm/vllm-openai:latest
args:
  --model Qwen/Qwen3-8B
  --tensor-parallel-size 1
```

### Full config (from template.yaml)

```yaml
image: vllm/vllm-openai:latest
port: 8000
hf_cache: /path/to/hf/cache          # optional, default: $HOME/.cache/huggingface
volumes:                             # optional extra mounts
  - /data/models:/models

env:                                 # container env vars
  VLLM_ATTENTION_BACKEND=FLASHINFER

args:                                # vLLM CLI flags
  --model MODEL_NAME_HERE
  --port 8000
  --tensor-parallel-size 1
  --dtype auto
  --max-model-len 32768
  --max-num-seqs 16
  --max-num-batched-tokens 8192
  --enable-chunked-prefill
  --async-scheduling
  --enable-prefix-caching
  --gpu-memory-utilization 0.9
  --trust-remote-code
  --enable-auto-tool-choice
  --disable-access-log-for-endpoints /health,/metrics,/ping
```

### Key args

| Arg | Description | Common values |
|-----|-------------|---------------|
| `--model` | HuggingFace repo ID | `Qwen/Qwen3-8B`, `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| `--tensor-parallel-size` | GPU count | `1`, `2`, `4` |
| `--dtype` | Data type | `auto`, `bfloat16`, `float16` |
| `--quantization` | Quant method | `modelopt`, `fp8`, `awq`, `gptq` |
| `--max-model-len` | Context window | `32768`, `65536`, `131072` |
| `--gpu-memory-utilization` | VRAM fraction | `0.85`, `0.9`, `0.95` |
| `--attention-backend` | Attention engine | `flashinfer`, `sdpa`, `flash_attn` |
| `--moe-backend` | MoE backend | `marlin`, `triton` |
| `--kv-cache-dtype` | KV cache type | `fp8`, `fp16`, `auto` |

### Docker image

| Tag | Use case |
|-----|----------|
| `:latest` | Stable release (default) |
| `:nightly` | New features (NVFP4, etc.) |

See: https://hub.docker.com/r/vllm/vllm-openai/tags

---

## Model Inspection

```bash
# Model info (size, files, tags)
hf models info meta-llama/Llama-3.1-8B-Instruct

# List repo files
hf repo-list unsloth/Qwen3.6-27B-NVFP4

# Model card (README)
hf models metadata Qwen/Qwen3-8B

# Download to local cache (optional)
hf download Qwen/Qwen3-8B
```

Models are cached under `$HOME/.cache/huggingface` (mounted into every container).

---

## Environment Variables

### Required (.env)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace auth token |
| `VLLM_API_KEY` | API key (default: `vllm`) |
| `DRY_RUN` | `true` to simulate, unset for real docker |
| `MODEL` | Default model name (used when `--model` omitted) |

### Optional (.env)

| Variable | Description |
|----------|-------------|
| `LOKI_URL` | Loki log forwarding URL |
| `SERVICE_NAME` | Loki label (default: `vllm`) |
| `SSH_USER` | Remote SSH username |
| `SSH_HOST` | Remote host IP/hostname |
| `SSH_PORT` | SSH port (default: 22) |
| `SSH_KEY` | SSH private key path |
| `SSH_DIR` | Remote project directory |
| `VLLM_REMOTE` | Set to `0` on remote `.env` to prevent recursion |

---

## Container Naming

Pattern: `vllm-<model-name>`

Example: `vllm-qwen3.6-35b-a3b-nvfp4`

---

## API Access

Once started, the model is available at:

```
http://localhost:<port>/v1/chat/completions
```

For remote models, use the remote host's IP:

```
http://<remote-host>:<port>/v1/chat/completions
```

---

## Update Workflow

When making YAML changes:

1. Edit the model YAML locally
2. Push changes: `./vllm-manager.sh update` (commits, pushes, pulls on remote)
3. Restart: `./vllm-manager.sh --remote restart --model <name>`

---

## Quick Reference

```bash
# Start a model
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4

# Check status
./vllm-manager.sh list
./vllm-manager.sh status

# Watch logs
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4 --follow

# Stop
./vllm-manager.sh stop --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh stop-all

# Restart
./vllm-manager.sh restart --model qwen3.6-35b-a3b-nvfp4

# Add a new model
cp models/template.yaml models/my-model.yaml
# edit my-model.yaml
./vllm-manager.sh start --model my-model

# Test changes (DRY_RUN)
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4
# → verify output, then set DRY_RUN=false and run again

# Push changes
./vllm-manager.sh update
```
