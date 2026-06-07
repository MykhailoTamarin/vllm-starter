# Agents — vLLM Model Manager

Manages vLLM model containers on a DGX Spark. Each model is a YAML config in `models/`, controlled by `vllm-manager.sh`.

## Critical Rules

1. Never commit or push unless explicitly asked.
2. Always work on `develop` — never push to `main`.
3. Always pull `main` before starting work.
4. Always test with `DRY_RUN=true` before committing.

## Git Workflow

```bash
git pull origin main && git switch develop && git pull origin develop
# ... make changes ...
git add -A && git commit -m "your message here" && git push origin develop
```

## Project Structure

```
.
├── vllm-manager.sh          # Main controller
├── llama-bench.sh           # llama-benchy wrapper (auto-saves to models/benchmarks/)
├── .env                     # Config: HF_TOKEN, SSH, DRY_RUN, MODEL
├── models/
│   ├── template.yaml        # Full template (all options documented)
│   └── *.yaml               # One per model (no comments)
├── README.md
├── FOR_AGENTS.md            # Deprecated
└── AGENTS.md
```

## Manager Commands

Model name via `--model <name>` or `.env MODEL`.

| Command                          | Description                            |
| -------------------------------- | -------------------------------------- |
| `start --model <name>`           | Stop all, then start this model        |
| `stop --model <name>`            | Stop & remove container                |
| `stop-all`                       | Stop & remove all                      |
| `restart --model <name>`         | Stop then start                        |
| `logs --model <name> [--follow]` | Last 100 lines; `--follow` local only  |
| `status`                         | docker ps for vllm containers          |
| `list`                           | All models with status                 |
| `delete --model <name>`          | Remove stopped container               |
| `update`                         | Commit, push develop, pull remote      |
| `pull`                           | Pull latest from develop (remote only) |

### Flags

| Flag             | Description                             |
| ---------------- | --------------------------------------- |
| `--remote`       | Force SSH execution                     |
| `--local`        | Force local execution                   |
| `--model <name>` | Model name (falls back to `.env MODEL`) |
| `--follow`       | Live logs (local only)                  |

### Execution Mode

- **Default**: remote SSH
- `--local` or `DRY_RUN=true` → local dry run (no docker)
- `--remote` → remote SSH (overrides DRY_RUN)
- `DRY_RUN=true --remote` → remote SSH (not a dry run)

### Examples

```bash
# Local dry run
./vllm-manager.sh start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh --local status

# Remote
./vllm-manager.sh --remote start --model qwen3.6-35b-a3b-nvfp4
./vllm-manager.sh --remote stop-all

# Default model from .env
./vllm-manager.sh start
./vllm-manager.sh --model other start  # override
```

## Benchmarking

`llama-bench.sh` wraps [llama-benchy](https://github.com/eugr/llama-benchy) — auto-builds base-url from `.env SSH_HOST` + `VLLM_API_KEY`, resolves model from YAML config.

**Required:** `llama-benchy` installed (`uvx llama-benchy` or `pip install git+https://github.com/eugr/llama-benchy`).

| Command                         | Description                                                                            |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| `llama-bench.sh --model <name>` | Run benchmark (auto-saves to `models/benchmarks/<name>/benchmark_dd_mm_yy_HH_mm.json`) |
| `+ --depth 0 4096 8192`         | Context depths to test                                                                 |
| `+ --concurrency 1 2 4`         | Parallel client counts (shows `t/s (total)` vs `t/s (req)`)                            |
| `+ --latency-mode generation`   | Measure server-side latency (recommended)                                              |

```bash
# YAML reference (reads --model and --served-model-name from config)
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 8192 --latency-mode generation

# Direct model name
./llama-bench.sh --model nvidia/Qwen3.6-35B-A3B-NVFP4 --depth 0 4096 --latency-mode generation

# Concurrency test (parallel load)
./llama-bench.sh --model qwen3.6-35b-a3b-nvfp4-mtp --depth 0 4096 --concurrency 1 2 4

# Default from .env MODEL
./llama-bench.sh --depth 4096 --latency-mode generation
```

Results auto-save to `models/benchmarks/<yaml-name>/benchmark_<timestamp>.json` (gitignored).

---

## Updating the README Models Table

The **Available Models** table in `README.md` has benchmark results inline. When benchmarking a model, always update the table with the new results.

### Table columns

| Column     | Source                                                                             | Format                                                                                                                                           |
| ---------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model      | YAML filename (without `.yaml`)                                                    | `qwen3.6-35b-a3b-nvfp4-mtp`                                                                                                                      |
| Quant      | YAML header `# Key specs` line, or `hf models card` `tags`/`quantization`          | `NVFP4 (modelopt)`, `modelopt`, `—`                                                                                                              |
| TP         | YAML `args: --tensor-parallel-size`                                                | `1`, `2`, `4`, `—`                                                                                                                               |
| Attention  | YAML `args: --attention-backend` + `--moe-backend`                                 | `flashinfer`, `marlin`, `flashinfer+MTP`, `—`                                                                                                    |
| Max Len    | YAML `args: --max-model-len` or HF card "Context length"                           | `32k`, `128k`, `262k`, `256k`, `—`                                                                                                               |
| Prefill    | Benchmark `benchmarks[].pp_throughput.mean` across all context sizes               | `4.1–6.2k t/s` (range, k suffix for thousands, `—` if untested)                                                                                  |
| Gen t/s    | Single-client `tg_throughput.mean` across context sizes; concurrency data separate | `116–197 t/s` (range only — append concurrency notes only if report provided, e.g. `116–197 t/s (C8: 72 @ 8k, ~470 t/s total)`), `—` if untested |
| TTFT @ 64k | Benchmark `ttft.mean` for the largest context_size, in seconds                     | `16.7s` (convert ms → s, `—` if untested)                                                                                                        |
| Status     | Whether benchmark has been run                                                     | `✅ **Tested**` or `⬜ Untested`                                                                                                                   |

### YAML name → model name mapping

The table **Model** column always matches the YAML filename (no `.yaml` extension):

| YAML file                                          | Table Model column                     |
| -------------------------------------------------- | -------------------------------------- |
| `models/qwen3.6-35b-a3b-nvfp4-mtp.yaml`            | `qwen3.6-35b-a3b-nvfp4-mtp`            |
| `models/nemotron-3-super-120b-a12b-nvfp4-mtp.yaml` | `nemotron-3-super-120b-a12b-nvfp4-mtp` |

### Extracting benchmark data from JSON

Benchmark results are in `models/benchmarks/<yaml-name>/benchmark_*.json`:

```json
{
  "benchmarks": [
    {
      "context_size": 4096,
      "pp_throughput": { "mean": 6046.5, ... },   // ← prefill t/s
      "tg_throughput": { "mean": 126.2, ... },     // ← generation t/s
      "ttft": { "mean": 1084.7, ... },             // ← TTFT in ms
      "peak_throughput": { "mean": 130.3, ... }    // ← peak gen t/s
    }
  ]
}
```

**Steps to fill a table row from benchmark JSON:**

1. **Prefill**: collect all `pp_throughput.mean` values → format as `min–max` → if max ≥ 1000, use `k` suffix (e.g. `4.1–6.2k t/s`)
2. **Gen t/s**: collect all `tg_throughput.mean` values → format as `min–max t/s` (always < 1000, no k suffix). **Only** append concurrency notes if a concurrency report is provided: `(C<n>: <per-req t/s> @ <depth>, ~<total> t/s total)` — otherwise just the range
3. **TTFT @ 64k**: find the entry with the largest `context_size` → take `ttft.mean` → convert ms → s (divide by 1000) → format as `X.Xs`
4. **Quant**: read from YAML header comment line (after `# ──` block), or from `hf models card` tags (e.g. `nvfp4` → `NVFP4`, `modelopt` → add `(modelopt)` if quantization tag present)
5. **Status**: if benchmark JSON exists → `✅ **Tested**`, else → `⬜ Untested`

### When adding a new model (no benchmark yet)

Fill what you know from the YAML config and HF model card:

```markdown
| minimax-m2.7-reap-nvfp4 | NVFP4 | 1 | flashinfer | 128k | — | — | — | ⬜ Untested |
```

Leave `Prefill`, `Gen t/s`, and `TTFT @ 64k` as `—`.

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

**Every model must have a header block** above `image:` following this exact format:

```yaml
# ── <Short Title> ─────────────────────────────────────────────────────────
# <One-line description — architecture, model family>
# <Key specs — params, active params, size, quantization>
#
# Recommended for: <use-cases>
# ⚠️ <Warnings if any (nightly required, special tags, etc.)>
#
# Container: vllm-<model-name>
# API:       http://localhost:<port>/v1/chat/completions
# HF:        https://huggingface.co/<owner>/<repo-id>
#
# ──────────────────────────────────────────────────────────────────────────
```

All fields are required except `⚠️` — omit the warning line if none applies.

**Exception args** — inline on a specific `args:` line, when a parameter must be present for stability or correctness and the reason is non-obvious:

```yaml
# FlashIner JIT autotuner loops infinitely on dense models; disable it
--kernel-config '{"enable_flashinfer_autotune":false}'
```

No other inline comments. Don't comment every arg — only where removal breaks the model.

**When editing an existing model**, always check its [HuggingFace model card](https://huggingface.co) for the latest config, architecture notes, and recommended generation parameters before modifying `args:`.

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

| Arg                        | Description         | Common values                                   |
| -------------------------- | ------------------- | ----------------------------------------------- |
| `--model`                  | HuggingFace repo ID | `Qwen/Qwen3-8B`, `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| `--tensor-parallel-size`   | GPU count           | `1`, `2`, `4`                                   |
| `--dtype`                  | Data type           | `auto`, `bfloat16`, `float16`                   |
| `--quantization`           | Quant method        | `modelopt`, `fp8`, `awq`, `gptq`                |
| `--max-model-len`          | Context window      | `32768`, `65536`, `131072`                      |
| `--gpu-memory-utilization` | VRAM fraction       | `0.85`, `0.9`, `0.95`                           |
| `--attention-backend`      | Attention engine    | `flashinfer`, `sdpa`, `flash_attn`              |
| `--moe-backend`            | MoE backend         | `marlin`, `triton`                              |
| `--kv-cache-dtype`         | KV cache type       | `fp8`, `fp16`, `auto`                           |

### Docker image

| Tag        | Use case                   |
| ---------- | -------------------------- |
| `:latest`  | Stable release (default)   |
| `:nightly` | New features (NVFP4, etc.) |

See: https://hub.docker.com/r/vllm/vllm-openai/tags

---

## Model Inspection

**Use the `hf` CLI to inspect models before creating or editing a YAML config.**

**⛔ NEVER use web_fetch for model cards or HF metadata.** The `hf` CLI is the only correct source. If `hf` seems broken, verify it's installed (`which hf`) — never fall back to web_fetch.

```bash
# Model card (README + YAML frontmatter) — architecture, specs, tags, license
hf models card owner/Model-Name
#→ Markdown with YAML frontmatter (--- ... ---) containing: pipeline_tag,
#  license, tags, base_model, library_name, language, etc.
#  Followed by the full model card markdown with architecture details,
#  benchmarks, usage examples, and warnings.

# Download to local cache (optional, populates $HOME/.cache/huggingface)
hf download owner/Model-Name
```

**Extract key fields from `hf models card` output:**

| YAML frontmatter field | Use for                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `pipeline_tag`         | Inference type — `text-generation`, `image-text-to-text`, etc. |
| `license`              | License (e.g. `apache-2.0`, `other`)                           |
| `tags`                 | Quick scan — `nvfp4`, `moe`, `vlm`, `mamba`, `quantized`, etc. |
| `base_model`           | Original model this variant is based on                        |
| `library_name`         | Runtime library — `transformers`, `Model Optimizer`, etc.      |

**Extract architecture/details from the model card body:**

| What to look for             | Where                                                                  |
| ---------------------------- | ---------------------------------------------------------------------- |
| Total params / active params | "Model Details" table — "Total Parameters" or "Number of Parameters"   |
| Architecture                 | "Model Details" table — "Architecture" or card title                   |
| Context length               | "Input" section — "Context length" or "Max Sequence Length"            |
| Quantization method          | "Model Details" table — "Quantization" or card body                    |
| Disk size                    | "Model Details" table — "Size on Disk"                                 |
| Hardware requirements        | "Target Hardware" or "Software Integration" section                    |
| License                      | YAML frontmatter `license` field                                       |
| Warnings / gotchas           | Card body — look for "Gotchas", "Limitations", "⚠️", "⚠"                |
| Usage examples               | "Usage" or "Running on" section — shows `vllm serve` / docker commands |

**Example — parsing a card:**

```bash
hf models card nvidia/Qwen3.6-35B-A3B-NVFP4 2>/dev/null
```

Read the YAML frontmatter (between `---` markers) for license, tags, pipeline_tag. Then read the card body for architecture specs, context length, quantization details, and any warnings. Use this to fill the model YAML header block accurately.

Models are cached under `$HOME/.cache/huggingface` (mounted into every container).

---

## Environment Variables

### Required (.env)

| Variable       | Description                                      |
| -------------- | ------------------------------------------------ |
| `HF_TOKEN`     | HuggingFace auth token                           |
| `VLLM_API_KEY` | API key (default: `vllm`)                        |
| `DRY_RUN`      | `true` to simulate, unset for real docker        |
| `MODEL`        | Default model name (used when `--model` omitted) |

### Optional (.env)

| Variable       | Description                                      |
| -------------- | ------------------------------------------------ |
| `LOKI_URL`     | Loki log forwarding URL                          |
| `SERVICE_NAME` | Loki label (default: `vllm`)                     |
| `SSH_USER`     | Remote SSH username                              |
| `SSH_HOST`     | Remote host IP/hostname                          |
| `SSH_PORT`     | SSH port (default: 22)                           |
| `SSH_KEY`      | SSH private key path                             |
| `SSH_DIR`      | Remote project directory                         |
| `VLLM_REMOTE`  | Set to `0` on remote `.env` to prevent recursion |

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


