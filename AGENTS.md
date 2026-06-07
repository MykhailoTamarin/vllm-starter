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

| Command | Description |
|---------|-------------|
| `start --model <name>` | Stop all, then start this model |
| `stop --model <name>` | Stop & remove container |
| `stop-all` | Stop & remove all |
| `restart --model <name>` | Stop then start |
| `logs --model <name> [--follow]` | Last 100 lines; `--follow` local only |
| `status` | docker ps for vllm containers |
| `list` | All models with status |
| `delete --model <name>` | Remove stopped container |
| `update` | Commit, push develop, pull remote |
| `pull` | Pull latest from develop (remote only) |

### Flags

| Flag | Description |
|------|-------------|
| `--remote` | Force SSH execution |
| `--local` | Force local execution |
| `--model <name>` | Model name (falls back to `.env MODEL`) |
| `--follow` | Live logs (local only) |

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

| YAML frontmatter field | Use for |
|------------------------|---------|
| `pipeline_tag` | Inference type — `text-generation`, `image-text-to-text`, etc. |
| `license` | License (e.g. `apache-2.0`, `other`) |
| `tags` | Quick scan — `nvfp4`, `moe`, `vlm`, `mamba`, `quantized`, etc. |
| `base_model` | Original model this variant is based on |
| `library_name` | Runtime library — `transformers`, `Model Optimizer`, etc. |

**Extract architecture/details from the model card body:**

| What to look for | Where |
|------------------|-------|
| Total params / active params | "Model Details" table — "Total Parameters" or "Number of Parameters" |
| Architecture | "Model Details" table — "Architecture" or card title |
| Context length | "Input" section — "Context length" or "Max Sequence Length" |
| Quantization method | "Model Details" table — "Quantization" or card body |
| Disk size | "Model Details" table — "Size on Disk" |
| Hardware requirements | "Target Hardware" or "Software Integration" section |
| License | YAML frontmatter `license` field |
| Warnings / gotchas | Card body — look for "Gotchas", "Limitations", "⚠️", "⚠" |
| Usage examples | "Usage" or "Running on" section — shows `vllm serve` / docker commands |

**Example — parsing a card:**

```bash
hf models card nvidia/Qwen3.6-35B-A3B-NVFP4 2>/dev/null
```

Read the YAML frontmatter (between `---` markers) for license, tags, pipeline_tag. Then read the card body for architecture specs, context length, quantization details, and any warnings. Use this to fill the model YAML header block accurately.

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


