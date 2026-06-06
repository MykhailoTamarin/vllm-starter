# vLLM Model Manager

Easy model management on a single DGX Spark. Every config is tuned for a coding agent's best experience — max context window, proper quantization, and optimized inference params. Want to test a new model? Ask an AI agent to read its model card and generate the YAML config — you're ready to benchmark in seconds.

## Quick Start

```bash
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4   # start NVFP4 model
./vllm-manager.sh list                                    # see what's running
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4    # last 100 lines
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4-mtp --follow  # live tail
./vllm-manager.sh stop --model qwen3.6-35b-a3b-nvfp4    # tear it down
./vllm-manager.sh stop-all                              # nuke everything
```

## Commands

| Command             | Description                     |
|---------------------|---------------------------------|
| `start --model <name>` | Stop all running models, then start this one |
| `stop --model <name>`  | Stop & remove container         |
| `stop-all`          | Stop & remove all containers    |
| `restart --model <name>` | Restart a model             |
| `logs --model <name>`  | Show last 100 lines           |
| `list`              | Show status of all models       |
| `delete --model <name>`| Remove stopped container      |

## Adding a New Model

1. Copy the template:
   ```bash
   cp models/template.yaml models/my-model.yaml
   ```

2. Edit the YAML with your model's image, args, env vars, and optional volumes

3. Download the model to local cache (optional - vLLM will pull on first run):
   ```bash
   hf download Qwen/Qwen3-8B
   ```

4. Start it:
   ```bash
   ./vllm-manager.sh start --model my-model
   ```

## Downloading Models

Models are cached under `$HOME/.cache/huggingface` (mounted into every container).

```bash
# Download a model
hf download unsloth/Qwen3.6-27B-NVFP4

# Inspect before downloading
hf models info meta-llama/Llama-3.1-8B-Instruct
hf models ls Qwen/Qwen3-8B
```

## Model Config Format (YAML)

```yaml
image: vllm/vllm-openai:latest   # Docker image (latest | nightly)
port: 8001                        # Host port (default 8000)
hf_cache: /path/to/hf/cache       # Optional custom HF cache path
volumes:                          # Optional extra host volumes
  - /data/models:/models

env:                              # Environment variables
  HF_TOKEN=${HF_TOKEN}
  VLLM_API_KEY=${VLLM_API_KEY}
  VLLM_USE_FLASHINFER_MOE_FP4=0

args:                             # vLLM arguments (one per line)
  --model Qwen/Qwen3-8B
  --tensor-parallel-size 1
  --dtype auto
  --gpu-memory-utilization 0.9
  --enable-auto-tool-choice
```

### Available commands

```bash
./vllm-manager.sh start --model <name>      # Start a model (stops any running first)
./vllm-manager.sh stop --model <name>       # Stop & remove a container
./vllm-manager.sh stop-all                  # Stop & remove ALL containers
./vllm-manager.sh restart --model <name>    # Restart a model
./vllm-manager.sh logs --model <name>       # Show last 100 lines
./vllm-manager.sh logs --model <name> --follow  # Live log follow
./vllm-manager.sh status                    # Show docker ps output
./vllm-manager.sh list                      # Show all models & status
./vllm-manager.sh delete --model <name>     # Remove stopped container
./vllm-manager.sh update                    # Commit, push, and pull on remote
./vllm-manager.sh pull                      # Pull latest on remote only
```

### Environment Variables (.env)

The manager auto-loads `.env` from the project directory. Required variables:

| Variable | Description | Default |
|----------|-------------|---------|  
| `HF_TOKEN` | HuggingFace auth token | — |
| `VLLM_API_KEY` | API key for authenticated requests | `vllm` |
| `DRY_RUN` | Set `true` to simulate without running docker | `false` |
| `MODEL` | Default model name (used when `--model` is omitted) | — |
| `LOKI_URL` | Loki log forwarding URL | — |
| `SERVICE_NAME` | Service name for Loki labels | `vllm` |

### Remote Commands Execution (SSH)

Configure SSH settings in `.env` for remote command execution:

| Variable | Description | Example |
|----------|-------------|---------|  
| `SSH_USER` | Remote SSH username | `administrator` |
| `SSH_HOST` | Remote host IP/hostname | `192.168.88.57` |
| `SSH_PORT` | SSH port (22 if not set) | `22` |
| `SSH_KEY` | Path to SSH private key | `~/.ssh/id_rsa` |
| `SSH_DIR` | Remote project directory path | `/home/administrator/vllm-starters` |
| `VLLM_REMOTE` | Set to `0` on remote `.env` to prevent recursion | `0` |

#### Flags

| Flag | Description |
|------|-------------|
| `--remote` | Force remote execution via SSH |
| `--local` | Force local execution (opt-out) |
| `--model <name>` | Model name (required; falls back to `.env MODEL`) |
| `--follow` | Live log follow (for `logs` command) |

Flags can be placed before or after the command:

```bash
./vllm-manager.sh --remote start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh start --remote --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh --remote logs --model qwen3.6-35b-a3b-nvfp4-mtp --follow
./vllm-manager.sh --remote logs --model qwen3.6-35b-a3b-nvfp4-mtp      # last 100 lines
./vllm-manager.sh --remote stop-all
./vllm-manager.sh --local status
```

### API Access

Once started, the model is available at:

```
http://localhost:<port>/v1/chat/completions
```

For remote models, use the remote host's IP:

```
http://<remote-host>:<port>/v1/chat/completions
```

### Container Naming

Each container follows the pattern: `vllm-<model-name>` (e.g., `vllm-qwen3.6-35b-a3b-nvfp4`, `vllm-qwen3.6-35b-a3b-nvfp4-mtp`)
