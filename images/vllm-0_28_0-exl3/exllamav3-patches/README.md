# exllamav3 ARM64 patches

These patches make ExLlamaV3 compile and link cleanly on **ARM64** (aarch64),
e.g. the DGX Spark (GB10 / SM121). They are applied at image build time to a
fresh upstream checkout:

```bash
git clone --branch v1.4.6 https://github.com/turboderp-org/exllamav3.git /tmp/exllamav3
cd /tmp/exllamav3
for p in exllamav3-patches/*.patch; do
    git apply "$p"
done
pip install --no-build-isolation --no-cache-dir /tmp/exllamav3
```

## Upstream baseline

| Field | Value |
|---|---|
| Repo | `https://github.com/turboderp-org/exllamav3` |
| Tag / commit | `v1.4.6` == `499890c75d20d8e7c9d061f37189ae611a5c9f0b` |
| `exllamav3/version.py` | `1.4.6` |

> The v26 image (v1.3.0) applied the same ARM64 changes in-tree (setup.py
> x86-exclusion + AVX guards in `avx2/avx512_target.cpp` and `moe_mul1.cpp`).
> They were re-derived for v1.4.1 and, for the first time, extracted into
> standalone `.patch` files so they can be applied to any future upstream bump
> without re-doing the exercise. v1.4.1 also gained a 4th patch
> (`bindings.cpp`) that fixes a latent undefined-symbol bug present in the v26
> image. For **v1.4.6** (118 commits past v1.4.1), patch 4 (`moe_mul1.cpp`) was
> re-anchored: the file gained an `#include <limits>` and worker-pool
> participant fields, shifting the immintrin-guard context; the other four
> patches carried over untouched.

## Patch list

| # | Patch | File(s) | What / why |
|---|---|---|---|
| 1 | `0001-setup.py-*.patch` | `setup.py` | On `aarch64`/`arm64`, exclude x86-only sources from the CUDA extension build: `avx2_target.cpp`, `avx512_target.cpp`, `all_reduce_cpu_avx2.cpp`, `all_reduce_cpu_avx512.cpp`, `moe_mul1.cpp`, `moe_handoff.cu`, `all_reduce_cpu.cu`. These files use `immintrin.h` / AVX intrinsics / `__builtin_ia32_pause()` that do not exist on ARM64. |
| 2 | `0002-avx2_target.cpp-*.patch` | `avx2_target.cpp` | Guard the `is_avx2_supported()` / `is_f16c_supported()` CPUID bodies with `#if defined(__x86_64__) || defined(__i386__)`; return `false` otherwise. (Defensive — the file is already excluded by patch 1.) |
| 3 | `0003-avx512_target.cpp-*.patch` | `avx512_target.cpp` | Same guard for `is_avx512_supported()`. |
| 4 | `0004-moe_mul1.cpp-*.patch` | `cpu/moe_mul1.cpp` | Guard `#include <immintrin.h>` and the `detect_isa()` body; return `Isa::Scalar` on ARM64. |
| 5 | `0005-bindings.cpp-*.patch` | `bindings.cpp` | **NEW in v1.4.1 port.** Wrap the CPU-MoE (`exl3_moe_cpu_*`, `exl3_moe_flag_*`) and CPU all-reduce (`pg_all_reduce_cpu`, `run_cpu_reduce_jobs`, `end_cpu_reduce_jobs`) `m.def` registrations in `#if defined(__x86_64__) || defined(__i386__)`. Without this the shared library still references symbols from the excluded sources → `import exllamav3_ext` fails with `undefined symbol`. **This bug existed in the v26 image** (its `.so` was un-importable; vLLM's EXL3 overlay tolerated it via a stub fallback). |

## Why these specific files?

ExLlamaV3's `exllamav3_ext` CUDA extension is compiled once from all
`.cpp`/`.cu` files under `exllamav3/exllamav3_ext/`. A handful of those are
x86-host-only:

- `avx2_target.cpp`, `avx512_target.cpp` — CPUID runtime detection (x86-only
  intrinsics).
- `all_reduce_cpu_avx2.cpp`, `all_reduce_cpu_avx512.cpp` — CPU all-reduce
  kernels compiled with `-mavx2` / AVX-512 target attributes.
- `moe_mul1.cpp` — CPU MoE mul1 kernels with `M1_TARGET_*` AVX target
  attributes and `__builtin_ia32_pause()`.
- `moe_handoff.cu` — CPU-MoE worker handoff, uses `__builtin_ia32_pause()`.
- `all_reduce_cpu.cu` — CPU-assisted all-reduce, uses AVX2 + pause.

The GPU-side compute entry points the vLLM EXL3 overlay actually needs —
`exl3_gemm`, `exl3_moe`, `exl3_moe_max_concurrency`, `hgemm`, `rope`, etc. —
live in non-excluded `.cu` files and are unaffected.

## Regenerating after an upstream bump

1. Clone the new tag:
   ```bash
   git clone --branch <new-tag> https://github.com/turboderp-org/exllamav3.git /tmp/exllamav3-new
   cd /tmp/exllamav3-new
   ```
2. Re-apply patches 1–4 by hand (they are small; `avx2/avx512_target.cpp` are
   typically byte-identical between releases, so 2 and 3 carry over untouched).
3. Check whether `bindings.cpp` gained/lost any `exl3_moe_cpu_*` / CPU
   all-reduce registrations and update patch 5's guard region.
4. `git format-patch <upstream-commit>..HEAD -o <repo>/exllamav3-patches/`
   and update this README.
