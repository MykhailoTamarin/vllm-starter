# 05 — Maintenance & Risks

Findings from this review that are not throughput, but correct-ness/robustness/upgrade hygiene.

## 1. `verify_exl3.py` is a near no-op

It only imports the patched module (smoke test). It will **not** catch a failed patch anchor (e.g. an upstream bump that changed the anchor text and made a `str.replace` silently no-op or bail). The image's real protection is each patch script's own fail-closed `if old not in content: FAIL... raise SystemExit(1)`. Recommendation: extend `verify_exl3.py` to grep each patched target for a *post-patch* sentinel string (e.g. `route_active` in tiny_decode, `expert_counts` in `_impl.py`, `(216, 256, 384)` in dsv4_topk, `_fill_short_context_topk_indices` in attention.py).

## 2. Dead / unreachable code in `exl3.py`

- **Parity path is runtime-dead.** `VLLM_EXL3_TRELLIS_MIN_M=1` ⇒ every `m≥1` is trellis/prefill; the `exl3_moe` argsort tail (`_apply_rank_sliced`) and its staging buffers (`tg/tu/ig/iu`, `token_sorted`/`weight_sorted`, `flat_token`, `ones`) are unreachable in the served config. They still reserve memory and add a large, hard-to-follow alternate path. Candidates for a future cleanup (keep the parity path only if you intend to use `min_trellis_m>1`).
- **`_fp8_moe_config()` (exl3.py:592) is never called**; `Fp8MoEMethod` imported-but-unused (README already documents the real routing: FP8 linears → `Fp8LinearMethod`, non-EXL3 MoE → `Mxfp4MoEMethod`). Delete the dead helper.
- **Global `VLLM_EXL3_TRELLIS_MIN_M=1` vs draft default.** `_rank_sliced_runtime` defaults draft layers to `MIN_CAPTURABLE_TRELLIS_M=1` automatically; the YAML's global `VLLM_EXL3_TRELLIS_MIN_M=1` is therefore redundant for drafts (it matters only to force the **target** window to start at 1, which is what unlocks FULL-cudagraph decode capture). No change needed, but the config comment could note it's required for target, not draft.

## 3. `patch_version.py` writes a malformed version tuple

It stamps a 4-tuple `__version_tuple__` with a **string 4th element** — a latent comparison footgun (`(0,26,0,'exl3...')` vs upstream int). Low urgency; the banner string is what users see.

## 4. Patch-anchor fragility on upgrade

Every `patch_*.py` is a `str.replace` against dist-packages text. Anchors shift when the base image bumps. The `vllm-version-upgrade` skill protocol applies on any `v0.26.0 → vX` move: re-verify every anchor via `git diff <old>..<new> -- <target>` and **never copy upstream scripts raw**. Also note `patch_dsv4_topk_k216.py` is now permanently required (upstream whitelist unchanged — see `04` §6), so it must survive any bump.

## 5. Draft builder coupling

`build_dspark_draft.py` (mounted, not baked) depends on the checkpoint's `mtp.{0,1,2}` stage layout and the vendored `REAP_K216_PLAN.json`. If the upstream checkpoint drops/changes the plan again (it already did once — the vendor file exists for that reason), the draft build breaks independently of the image. Keep the vendored plan current.

## 6. README / doc drift

- Image-dir `README.md` "Status" still says decode ~18.5 t/s and K64-era text, while the YAML is on K192 draft and benchmarks show 24–34 t/s; the "compact draft K64" prose (§Compact draft) predates the config. The root `README.md` benchmark row (0.98–1.13k / 24–33) is current. Drift is cosmetic but can mislead future tuning.
- `docs` for env knobs — several `VLLM_EXL3_*` knobs exist in code but are only documented piecemeal (`TRELLIS_BLOCK_M`, `PREFILL_CHUNK`, `PREFILL_BLOCK_M`). Consider a single "tuning knobs" table.

## 7. ExLlamaV3 bump hygiene

`exllamav3-patches/README.md` documents re-derivation (re-apply 1–4, re-check `bindings.cpp` for new CPU-MoE/all-reduce `m.def` entries). The v1.4.1 port fixed a latent undefined-symbol bug the v26 image silently tolerated via the stub fallback — keep that patch set's `bindings.cpp` guard in sync on any bump or `import exllamav3_ext` will regress.

## 8. Risks summary

| Risk | Severity | Mitigation |
|---|---|---|
| Anchor drift on a future base bump | High | vllm-version-upgrade protocol; keep 216-router patch |
| `verify_exl3.py` blind | Medium | add post-patch sentinel greps |
| Dead parity path masking a regression | Low | cleanup later; keep if `min_m>1` intended |
| `swiglu_limit`/shared-expert handling unverified | Low–Med (quality) | confirm (see `03` L4) |
| Draft/build coupling to upstream plan | Medium | keep vendored `REAP_K216_PLAN.json` current |
| Doc drift (README "Status", K64 prose) | Low | sync with K192 config |

---

*End of series (00–05). All analysis is from the repository + vLLM v0.26.0 sources + HF `config.json` (no weights); no runtime was launched and no file was modified.*
