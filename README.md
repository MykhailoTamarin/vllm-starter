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
| **qwen3.6-35b-a3b-nvfp4-mtp** | 35B / 3B | 21.9G | 256k | 13.65x | 1.7–6.1k t/s | 134–270 t/s (C2: ~156–253 @ d0-4k, ~32–64 @ 8k-65k; C4: ~240 @ d0, ~230 @ d1k, ~208 @ d2k, ~65 @ 4k, ~34 @ 8k) | 17.0s | ✅ **Tested** |
| **qwen3.6-27b-nvfp4-mtp**                   | 27B / —     | 20.2G      | 262k    |           5.28x | 1.0–2.7k t/s   | 23–30 t/s (C2: ~29 @ 4k, C4: ~29 @ 4k)   | 47.0s          | ✅ **Tested** |
| **nemotron-3-super-120b-a12b-nvfp4-mtp**    | 120B / 12B  | 74.9G      | 1000k   |           5.53x | 0.97–2.08k t/s | 14–33 t/s (C2: ~30 @ 4k, C4: ~16 @ 4k)   | 38.9s          | ✅ **Tested** |
| **deepseek-v4-flash-nvfp4-mtp** | 180B / 13B | 96G | 262k | 1.68x | 452–908 t/s | 18–26 t/s | 105.1s | ✅ **Tested** |

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

Run `llama-bench.sh` to benchmark a model against its live endpoint. It uses our [forked llama-benchy](https://github.com/eugr/llama-benchy) which adds:

- **vLLM idle-check** via `/metrics` endpoint — prevents concurrency overlap that skews results
- **Multiple report generation** — automatically creates JSON (raw), MD (parsed summary), and PNG (graph) in one run

Auto-builds base-url from `.env SSH_HOST` + `VLLM_API_KEY`, resolves model from YAML config, and saves results to `models/benchmarks/`.

```bash
# Required: llama-benchy installed (via uvx or from source)
uvx llama-benchy
```

### Running Benchmarks

> **Recommended:** Always use `--idle-wait`. The vLLM `/metrics` check between each {C×D} test prevents concurrency overlap that skews results.

#### Benchmark output structure

Each wait-idle benchmark run creates files with the same base name but different extensions:

```
models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_<depths>.json  # Raw data (gitignored)
models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_<depths>.md    # Parsed summary (tracked)
models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_<depths>.png   # Graph (gitignored)
```

Where `<concurrencies>` and `<depths>` use min-max ranges (e.g., `_c1_d0_256`, `_c1-4_d256-16384`).

#### Single concurrency, full depth

```bash
# C=1 only, full context — 3 reps each
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --idle-wait --depth 0 256 512 1024 2048 4096 8192 16384 32768 65536 131072 --runs 3
```

`benchmark_<timestamp>_c<concurrencies>_d<depths>.md` (tracked)

#### Multi-concurrency with idle gates (caps at 16k depth)

```bash
# C1, C2, C4 across multiple depths — 3 reps each
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --idle-wait --depth 0 256 512 1024 2048 4096 8192 16384 --concurrency 1 2 4 --runs 3
```

`benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_d<depths>.png` (ignored by agents)

#### Legacy Mode (original behavior)

Single benchy call, no vLLM idle check, no PNG output. Quick single-pass only.

```bash
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 --latency-mode generation
```

`benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_d<depths>_{json,md}` (MD tracked)

### Report Formats & Agent Usage

| Format | Description | Git | Agent Use |
| ------ | ----------- | --- | --------- |
| **JSON** | Full raw benchmark data (all metrics, timestamps, etc.) | ✗ Ignored | Deep inspection only |
| **MD**   | Parsed markdown table with key metrics | ✓ Tracked | ✅ **Source of truth** |
| **PNG**  | Visualization graph (prefill + generation curves + TTFT) | ✗ Ignored | ⛔ **NEVER analyze** |

> **Concurrency rule for agents:** When analyzing benchmark results, always compare C1 files against C1 only. Do NOT mix C-only concurrency files (e.g., `benchmark_..._c1_d0_256.md`) with multi-concurrency files (e.g., `benchmark_..._c1-4_d0_256.md`). Each benchmark file represents a specific concurrency suite — use the C1-only files whenever you need C1-specific metrics (prefill throughput, generation t/s, TTFT).

### Arguments

| Argument                     | Description                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--model <name>`             | Model YAML name (e.g. `qwen3.6-35b-a3b-nvfp4-mtp`) or direct HF model name                           |
| `--depth <d1> <d2> ...`      | Context depths to benchmark (default: `[1024]`). Single-concurrency tests go to full context (253k). Multi-concurrency caps at 16k. |
| `--concurrency <c1> <c2> ...`| Number of parallel clients per test (default: `[1]`). Produces `t/s (total)` and `t/s (req)` columns |
| `--format <f1>,<f2>...`      | Output format(s), comma-separated (default: `json,md,png`)                                            |
| `--latency-mode generation`  | Measure server latency via 1-token generation (recommended)                                          |
| `--no-warmup`                | Skip the warmup phase                                                                                |
| `--runs N`                   | Number of runs per test (default: 3)                                                                 |
| `--idle-wait`                | Sequential mode — waits for vLLM `/metrics` to be idle between each {C×D} test                      |
| `--repeat N`                 | Run the entire benchmark suite N times (default: 1). Generates files with `_s<N>` suffix.            |

### Where to find results

- **Raw JSONs** (gitignored, use only for deep inspection):
  ```
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_d<depths>.json
  ```
  Contains: `{benchmarks: [{pp_throughput: {mean, std}, tg_throughput: {mean, std}, ...}]}`

- **Parsed MD** (tracked by git — **source of truth**):
  ```
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_d<depths>.md
  ```
  Auto-generated markdown table. Key patterns in the `test` column:
  - `pp2048` — prefill throughput (2048 tokens input)
  - `tg32` — generation throughput (32 tokens output)
  - `pp2048 @ d4096` — prefill at 4096 token context depth
  - `tg32 (cN)` — generation throughput at concurrency N (multi-concurrency files only)

  Values are always formatted as `mean ± stddev` — use the `mean` value.

- **PNG graphs** (gitignored, **never analyze**):
  ```
  models/benchmarks/<model>/benchmark_<dd_mm_yy_HH_mm>_<concurrencies>_d<depths>.png
  ```
  Publication-quality visualization. Prefill uses circle markers with dashed lines, Generation uses square markers with solid lines.

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
