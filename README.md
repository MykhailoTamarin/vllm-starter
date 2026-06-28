# vLLM Model Manager

Easy model management on a single DGX Spark. Every config is tuned for agent coding — stable throughput across the full context window, NVFP4 quantization, and concurrency up to 4 so your orchestrator can run subagents without dropping performance. Not chasing peak t/s at small context or high concurrency: the goal is predictable, usable speed for multi-agent coding workflows. Built-in [llama-benchy](https://github.com/eugr/llama-benchy) wrapper handles URL, API key, and model resolution automatically — just run `./llama-bench.sh --model <name>` and results save to `models/benchmarks/`. Want to test a new model? Ask an AI agent to read its model card, generate the YAML config, and kick off a benchmark in seconds.

## Quick Start

```bash
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4-mtp   # start NVFP4 model
./vllm-manager.sh list                                    # list all available models
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4-mtp    # last 100 lines
./vllm-manager.sh logs --model qwen3.6-35b-a3b-nvfp4-mtp --follow  # live tail (local only)
./vllm-manager.sh stop --model qwen3.6-35b-a3b-nvfp4-mtp    # tear it down
./vllm-manager.sh stop-all                              # nuke everything
```

## Available Models

All configs live in `models/*.yaml`. Benchmarks measured on DGX Spark with llama-benchy (generation latency mode, 3 runs per config). The goal is stable throughput for agent coding — so we look at t/s across the context range (not just zero-context peak), and concurrency up to 4 for subagent support. Multi-concurrency tests cap at 16k depth (beyond that, concurrency is impractical). Single concurrency tests go to full context (253k).

| Model                                       | Params      | Model size | Max Len | Max Concurrency | Prefill        | Gen t/s                                  | TTFT @ 64k     | Status       |
| ------------------------------------------- | ----------- | ---------- | ------- | --------------: | -------------- | ---------------------------------------- | -------------- | ------------ |
| **qwen3.6-35b-a3b-nvfp4-mtp** | 35B / 3B | 21.9G | 256k | 13.65x | 2.8–6.0k t/s | 129–247 t/s (C2: ~296 @ 2k, ~207 @ 4k, ~69 @ 8k; C4: ~345 @ d0, ~214 @ 1k, ~116 @ 2k) | 47.0s | ✅ **Tested** |
| **qwopus3.5-122b-a10b-kimi-k2.6-nvfp4-mtp** | 122B / ~10B | 75.9G      | 256k    |           4.25x | 1.0–2.3k t/s   | 24–30 t/s (C2: ~27 @ 4k)                 | 47.0s          | ✅ **Tested** |
| **qwopus3.6-35b-a3b-nvfp4-mtp**             | 35B / 3B    | —          | 256k    |           7.12x | 2.7–5.9k t/s   | 51–84 t/s (C2: ~117 @ 4k, C4: ~54 @ 4k)  | 17.9s          | ✅ **Tested** |
| **qwen3.6-27b-nvfp4-mtp**                   | 27B / —     | 20.2G      | 262k    |           5.28x | 1.0–2.7k t/s   | 23–30 t/s (C2: ~29 @ 4k, C4: ~29 @ 4k)   | 47.0s          | ✅ **Tested** |
| **qwopus3.6-27b-v2-nvfp4-mtp**              | 27B / —     | 26G        | 262k    |           4.64x | 797–2.1k t/s   | 12–20 t/s (C2: ~27 @ 4k, C4: ~26 @ 4k)     | 66.8s          | ✅ **Tested** |
| **nemotron-3-super-120b-a12b-nvfp4-mtp**    | 120B / 12B  | 74.9G      | 1000k   |           5.53x | 0.97–2.08k t/s | 14–33 t/s (C2: ~30 @ 4k, C4: ~16 @ 4k)   | 38.9s          | ✅ **Tested** |
| **deepseek-v4-flash-nvfp4-mtp** | 180B / 13B | 96G | 262k | 1.68x | 452–908 t/s | 18–26 t/s | 105.1s | ✅ **Tested** |
| **ornith-1.0-35b-nvfp4**                    | 35B / ~8.6B | 21.9G      | 262k    | —             | 2.9–7.4k t/s   | 47–67 t/s (C2: ~87 @ 4k, C4: ~55 @ 4k)   | 16.9s          | ✅ **Tested** |
| **mistral-small-4-119b-nvfp4**              | 119B / 6.5B | —          | 256k    | —             | —              | —                                        | —              | ⬜ Untested   |
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

> **Recommended:** Always use `--wait-idle` for accurate results. It prevents concurrency overlap by waiting for the vLLM to be idle between each {C×D} test.

```bash
# ✅ Recommended: sequential single-concurrency, full depth (3 reps averaged)
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --wait-idle --depth 0 1024 2048 4096 8192 16384 32768 65536 131072 --repeat 3

# ✅ Recommended: sequential multi-concurrency with idle gates (caps at 16k depth)
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --wait-idle --depth 0 1024 2048 4096 8192 --concurrency 1 2 4 --repeat 3

# Legacy: default benchy logic
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 16384 32768 65536 --latency-mode generation
```

### Benchmark Output

Each wait-idle batch creates a subfolder in `models/benchmarks/<model>/` with JSON files per {C×D×run}:

```
models/benchmarks/<model>/c1_2_d0_1024/        (gitignored)
  c1_d0_r1_s1.json   # C=1, d=0, run=1, suite=1
  c2_d0_r1_s1.json   # C=2, d=0, run=1, suite=1
  c1_d1024_r1_s2.json # C=1, d=1024, run=1, suite=2
  ...

models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_2_d0_1024.md  (tracked)
  # Auto-generated parsed table after batch completes
```

Legacy mode writes directly to `benchmark_*.md` (tracked).

#### Parsing

Auto-generated after each wait-idle run. Manual:
```bash
./scripts/bench-parse.sh -d models/benchmarks/<model>/<folder>/ -o results.md
```

### How It Works

**wait-idle mode** (recommended): benchy runs sequentially with an idle-gate between each test (checks `vllm:num_requests_running == 0`). Prevents concurrency overlap that skews results. Each test saves to its own JSON file.

**Standard mode** (legacy): single benchy call, all flags passed through → saves one MD result file.

1. Reads `VLLM_API_KEY` and `SSH_HOST` from `.env` → builds `--base-url` and `--api-key`
2. Resolves `--model <yaml-name>` → reads `models/<yaml-name>.yaml` → extracts `--model` and `--served-model-name` from `args:` section
3. Passes all remaining args through to `llama-benchy`
4. **wait-idle mode**: creates `models/benchmarks/<model>/<c>_<d>/` with individual JSON files per {C×D×run}
5. **Standard mode**: writes `models/benchmarks/<model>/benchmark_*.md`
6. Auto-runs `scripts/bench-parse.sh` after wait-idle batch completes → generates parsed table at `models/benchmarks/<model>/benchmark_<date>_*.md`

### Arguments

| Argument                     | Description                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--model <name>`             | Model YAML name (e.g. `qwen3.6-35b-a3b-nvfp4-mtp`) or direct HF model name                           |
| `--depth <d1> <d2> ...`      | Context depths to benchmark (default: `[1024]`). Examples below show single-concurrency (full depth, 253k) vs multi-concurrency (caps at 16k). |
| `--concurrency <c1> <c2> ...`| Number of parallel clients per test (default: `[1]`). Produces `t/s (total)` and `t/s (req)` columns |
| `--latency-mode generation`  | Measure server latency via 1-token generation (recommended)                                          |
| `--no-warmup`                | Skip the warmup phase                                                                                |
| `--runs N`                   | Number of runs per test (default: 3)                                                                 |
| `--wait-idle`                | Sequential mode — waits for GPU idle between each {C×D} test                                        |
| `--repeat N`                 | Run the entire benchmark suite N times (default: 1). Generates files with `_s<N>` suffix.            |

### Where to find results

- **Raw JSONs** (gitignored):
  ```
  models/benchmarks/<model>/c1_d0_1024_2048/
    c1_d0_r1_s1.json      # C=1, d=0, run=1, suite=1
    c2_d0_r1_s2.json      # C=2, d=0, run=1, suite=2
    ...
  ```
  Each file has: `{benchmarks: [{pp_throughput: {mean, std}, tg_throughput: {mean, std}, ...}]}`

- **Parsed MD** (tracked by git):
  ```
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_d0_1024.md
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_c1_2_4_d1024_2048.md
  ```
  Auto-generated by `scripts/bench-parse.sh` after each wait-idle run. Contains the markdown table from the benchmark output.

  Manual parse:
  ```bash
  ./scripts/bench-parse.sh -d models/benchmarks/<model>/<folder> -o results.md
  ```

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
