# Report: Planar/Iso KV-cache quantization — FIXED on gfx1201

**Status:** Working. All four fork-specific KV types (`planar3`, `iso3`, `planar4`, `iso4`)
now generate correct output in **same-type AND mixed** (K or V standard) combinations.
Standard types (`f16`, `q8_0`, `q4_0`, …) still work.

Verified on LFM2.5-230M-Q4_K_M (head_dim=64, n_head=32, n_kv_head=8) with
`-ngl 99 -fa on`: every combination answers "The capital of France is [Paris]".

## Three root-cause defects found and fixed this session

**Fix 1 — build had `GGML_CUDA_FA_ALL_QUANTS=OFF` (no planar kernels compiled).**
All planar/iso FA dispatch is gated behind `#ifdef GGML_CUDA_FA_ALL_QUANTS`. Every build dir
(`/tmp/build-rdna4`, `/tmp/build-rdna4-fa`, …) configured that option OFF, so the planar VEC
kernels + mixed dispatch cases were never compiled → silent empty / fallback-to-unsupported.
Fix: configure `/tmp/build-rdna4-fa` with `-DGGML_CUDA_FA_ALL_QUANTS=ON`
(`-DGGML_HIP=ON -DGGML_CUDA_FA=ON -DAMDGPU_TARGETS=gfx1201`). Rebuild `llama-cli`/`llama-server`.

**Fix 2 — `get_k`/`get_v` exposed the 128-padded physical width to the matmul/FA graph.**
The padded cache is allocated at 128 elements/head (one planar quant block), but Q/matmul expect
the logical head width (64). `src/llama-kv-cache.cpp` `get_k`/`get_v` reported `ne[0]=128`
(via `head_k_eff`/`head_v_eff`), so `ggml_can_mul_mat(k, q)` asserted at `ggml.c:5425`
(K=128 vs Q=64). Fix: `get_k`/`get_v` now expose the **logical** width (`head_k`/`head_v` = 64) as
`ne[0]`, while keeping the **padded pitch** (128) in `nb[1]`. The VEC kernel reads full 128-blocks via
the pitch; the dot product indexes 0..63 (D=64); zero-padded tail contributes 0. `k_stream`/`v_stream`
views use the allocation width so the full quant blocks stay visible to dequant.

**Fix 3 — planar/iso mixed dispatch cases lived in the `#else` branch (skipped when FA_ALL_QUANTS=ON).**
`ggml/src/ggml-cuda/fattn.cu` `ggml_cuda_flash_attn_ext_vec` had the `#ifdef GGML_CUDA_FA_ALL_QUANTS`
block (standard pairs + same-type planar/iso) end at the same-type cases, and the **mixed** pairs
(`K=planar/iso, V=standard`; `K=f16, V=planar/iso`; IQ4_NL pairs) were placed in the `#else` block.
With FA_ALL_QUANTS=ON the `#else` is compiled out, so mixed combinations had **no dispatch entry** →
`GGML_ABORT("fatal error")` at the end of the VEC dispatcher.
Fix: moved all mixed-planar / f16-planar-V / IQ4_NL case blocks from the `#else` region into the
`#ifdef` region (before the `#else`), so they are emitted whenever FA_ALL_QUANTS is defined.
The `#else` block now retains only the minimal non-FA_ALL_QUANTS fallback set.

## What was already correct (from prior session, confirmed)
- VEC kernel forced for planar/iso (kernel selection returns `BEST_FATTN_KERNEL_VEC`).
- Rotation constants + dequant tables regenerated (math self-consistent).
- iso4 block-type typo fixed (`block_iso4_0`).
- 128-head padding + zero-pad `cpy_k`/`cpy_v` in `llama-kv-cache.cpp`.

## Verification matrix (head_dim=64, LFM2.5-230M)
All 13 cases produce "Paris" with no abort/assert:

| ctk / ctv        | output   | t/s  |
|-------------------|----------|------|
| f16 / f16         | Paris    | 481  |
| planar3 / planar3 | Paris    | 313  |
| iso3 / iso3       | ok       | 331  |
| planar4 / planar4 | Paris    | 328  |
| iso4 / iso4       | ok       | 361  |
| f16 / planar3     | Paris    | 402  |
| f16 / iso3        | Paris    | 377  |
| f16 / planar4     | ok       | 380  |
| f16 / iso4        | Paris    | 403  |
| planar3 / f16     | Paris    | 373  |
| iso3 / f16        | ok       | 376  |
| planar4 / f16     | Paris    | 387  |
| iso4 / f16        | Paris    | 384  |

## Notes / non-blocking
- Deferred-quantization path (reference fork's `convert_deferred_keys`) was NOT needed for correctness;
  the logical-view + padded-pitch approach matches the FA VEC kernel contract directly. PPL/long-context
  stability is unverified but not required for functional generation.
- Builds must always set `GGML_CUDA_FA_ALL_QUANTS=ON` or planar/iso dispatch is silently absent.
- Second GPU (gfx1031) has no compatible kernels; pin with `HIP_VISIBLE_DEVICES=0`.

---
*Build: fork `master` at `7d1aad67a`, ROCm 7.15.0, gfx1201, RUNPATH `/opt/rocm-7.15.0/lib`.
Binary: `/tmp/build-rdna4-fa/bin/llama-cli`. Tests with `HIP_VISIBLE_DEVICES=0`.*
