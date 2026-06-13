# vLLM Model Manager

Easy model management on a single DGX Spark. Every config is tuned for a coding agent's best experience — max context window, proper quantization, and optimized inference params. Built-in [llama-benchy](https://github.com/eugr/llama-benchy) wrapper handles URL, API key, and model resolution automatically — just run `./llama-bench.sh --model <name>` and results save to `models/benchmarks/`. Want to test a new model? Ask an AI agent to read its model card, generate the YAML config, and kick off a benchmark in seconds.

## Quick Start

```bash
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4   # start NVFP4 model
./vllm-manager.sh list                                    # see what's running
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4    # last 100 lines
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4-mtp --follow  # live tail (local only)
./vllm-manager.sh stop --model qwen3.6-35b-a3b-nvfp4    # tear it down
./vllm-manager.sh stop-all                              # nuke everything
```

## Available Models

All configs live in `models/*.yaml`. Benchmark results measured on DGX Spark with llama-benchy (generation latency mode, concurrency 1, 3 runs per config).

| Model                                    | Quant            | Params     | Model size | Attention  | Max Len |      Prefill |                                   Gen t/s | TTFT @ 64k | Status                                                                        |
| ---------------------------------------- | ---------------- | ---------- | ---------- | ---------- | ------- | -----------: | ----------------------------------------: | ---------: | ----------------------------------------------------------------------------- |
| **qwen3.6-35b-a3b-nvfp4-mtp**            | NVFP4 (modelopt) | 35B / 3B   | 21.9G | flashinfer | 256k    | 2.7–6.3k t/s | 151–203 t/s (C4: 98 @ 8k, ~351 t/s total) | 16.9s | ✅ **Tested** |
| **qwen3.6-27b-nvfp4-mtp**                 | NVFP4 | 27B / — | 20.2G | flashinfer+MTP | 262k    | 1.3–2.5k t/s | 26.5–30.7 t/s | 50.4s | ✅ **Tested** |
| **minimax-m2.7-reap-nvfp4**              | NVFP4            | 172B / ~10B | 98.9G | flashinfer | 64k     | 1.4–2.3k t/s | 16.8–22.8 t/s | 25.7s (at 32k) | ✅ **Tested** |
| **nemotron-3-super-120b-a12b-nvfp4-mtp** | NVFP4            | 120B / 12B | 74.9G | marlin+MTP | 256k    | 1.5–2.0k t/s |    21–28 t/s (C8: 12 @ 8k, ~93 t/s total) |      38.6s | ✅ **Tested**                                                                  |
| **nex-n2-mini-nvfp4** | NVFP4 | 35B / — | 22.1G | flashinfer+cutlass MoE | 262k | 4.2–7.4k t/s | 38.4–40.5 t/s (C2: ~61–69 req t/s) | 16.2s | ✅ **Tested**                                                                  |
| **step3p7-flash-148b**                   | NVFP4 (modelopt) | 148B / ~11B | 90.1G | flashinfer | 128k    | 1.6–2.2k t/s | 12.3–13.4 t/s (C2: ~7–10 t/s, ~6.1–15.7 t/s total) | 43.0s | ✅ **Tested** |
| **mistral-small-4-119b-nvfp4**             | NVFP4            | 119B / 6.5B | —     | triton_mla | 256k    |            — |                                         — |          — | ⬜ Untested                                                                    |

## Commands

| Command                  | Description                                  |
| ------------------------ | -------------------------------------------- |
| `start --model <name>`   | Stop all running models, then start this one |
| `stop --model <name>`    | Stop & remove container                      |
| `stop-all`               | Stop & remove all containers                 |
| `restart --model <name>` | Restart a model                              |
| `logs --model <name>`    | Show last 100 lines                          |
| `list`                   | Show status of all models                    |
| `delete --model <name>`  | Remove stopped container                     |

## Benchmarking

Run `llama-bench.sh` to benchmark a model against its live endpoint. It reads `.env` to auto-build the API URL and API key, resolves the model name from YAML config, and saves results to `models/benchmarks/`.

### Prerequisites

Install [llama-benchy](https://github.com/eugr/llama-benchy):

```bash
# Quick (via uvx)
uvx llama-benchy

# Or install from source
uv pip install git+https://github.com/eugr/llama-benchy --system
```

### Quick Usage

```bash
# Benchmark using .env MODEL (default) with depth 0, 4096, 8192
./llama-bench.sh --depth 0 4096 8192 --latency-mode generation

# Explicit model via YAML config name and single client throughput
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 16384 32768 65536 131072 --latency-mode generation

# Concurrency test — compare single vs multi-client throughput
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 4096 8192 16384 32768 65536 --concurrency 1 2 4 --latency-mode generation
```

### How It Works

1. Reads `VLLM_API_KEY` and `SSH_HOST` from `.env` → builds `--base-url` and `--api-key`
2. Resolves `--model <yaml-name>` → reads `models/<yaml-name>.yaml` → extracts `--model` and `--served-model-name` from `args:` section
3. Passes all remaining args through to `llama-benchy`
4. Auto-saves JSON results to `models/benchmarks/<model-name>/benchmark_<timestamp>.json`

### Arguments

| Argument                    | Description                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--model <name>`            | Model YAML name (e.g. `qwen3.6-35b-a3b-nvfp4-mtp`) or direct HF model name                           |
| `--depth 0 4096 8192`       | Context depths to benchmark (default: `[0]`)                                                         |
| `--concurrency 1 2 4`       | Number of parallel clients per test (default: `[1]`). Produces `t/s (total)` and `t/s (req)` columns |
| `--latency-mode generation` | Measure server latency via 1-token generation (recommended)                                          |
| `--no-warmup`               | Skip the warmup phase                                                                                |
| `--runs N`                  | Number of runs per test (default: 3)                                                                 |

All `llama-benchy` flags are supported — see its [README](https://github.com/eugr/llama-benchy) for the full list.

### Adding a New Model

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
./vllm-manager.sh logs --model <name> --follow  # Live log follow (local only)
./vllm-manager.sh status                    # Show docker ps output
./vllm-manager.sh list                      # Show all models & status
./vllm-manager.sh delete --model <name>     # Remove stopped container
./vllm-manager.sh update                    # Commit, push, and pull on remote
./vllm-manager.sh pull                      # Pull latest on remote only
```

### Environment Variables (.env)

The manager auto-loads `.env` from the project directory. Required variables:

| Variable       | Description                                         | Default |
| -------------- | --------------------------------------------------- | ------- |
| `HF_TOKEN`     | HuggingFace auth token                              | —       |
| `VLLM_API_KEY` | API key for authenticated requests                  | `vllm`  |
| `DRY_RUN`      | Set `true` to simulate without running docker       | `false` |
| `MODEL`        | Default model name (used when `--model` is omitted) | —       |
| `LOKI_URL`     | Loki log forwarding URL                             | —       |
| `SERVICE_NAME` | Service name for Loki labels                        | `vllm`  |

### Remote Commands Execution (SSH)

Configure SSH settings in `.env` for remote command execution:

| Variable      | Description                                      | Example                             |
| ------------- | ------------------------------------------------ | ----------------------------------- |
| `SSH_USER`    | Remote SSH username                              | `administrator`                     |
| `SSH_HOST`    | Remote host IP/hostname                          | `192.168.88.57`                     |
| `SSH_PORT`    | SSH port (22 if not set)                         | `22`                                |
| `SSH_KEY`     | Path to SSH private key                          | `~/.ssh/id_rsa`                     |
| `SSH_DIR`     | Remote project directory path                    | `/home/administrator/vllm-starters` |
| `VLLM_REMOTE` | Set to `0` on remote `.env` to prevent recursion | `0`                                 |

#### Flags

| Flag             | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `--remote`       | Force remote execution via SSH                       |
| `--local`        | Force local execution (opt-out)                      |
| `--model <name>` | Model name (required; falls back to `.env MODEL`)    |
| `--follow`       | Live log follow (local only, not supported over SSH) |

Flags can be placed before or after the command:

```bash
./vllm-manager.sh --remote start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh start --remote --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh --remote logs --model qwen3.6-35b-a3b-nvfp4-mtp      # last 100 lines (no --follow over SSH)
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
