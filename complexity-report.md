# Complexity Report — llama.cpp-rdna4 (RDNA4 fork)

Generated: 2026-07-06
Base: ggml-org/llama.cpp `b9859` (`4fc4ec554`) + RDNA4 patches (PlanarQuant, WMMA, stream-k, FA iq4_nl)

---

## Summary

| Metric | Value |
|--------|-------|
| C/C++ files | 471 (383,620 lines) |
| Python files | 198 (~40,000 lines) |
| Total source | ~424,000 lines across 669 files |
| Main inference engine | 20,172 lines (src/llama-*.cpp) |
| GGML core | 279,297 lines |
| CUDA/HIP backend ops | 40,769 lines (ggml-cuda/) |
| Graphify nodes/edges | 40,483 nodes · 108,669 edges · 1,116 communities |

### Python Complexity Scan (198 .py files)

| Pattern | Count | Severity |
|---------|-------|----------|
| `nested_loop` | 162 | MEDIUM — mostly O(n×m) on model conversion metadata |
| `str_concat` (string `+=` in loop) | 102 | LOW — string building for docs/reports, small n |
| `redundant_keys` (`d.keys()` → `for k in d`) | 47 | LOW — style only, no perf impact |
| `O(n²)` (`.index()` / `.remove()` in loop) | 29 | LOW to MEDIUM — on small conversion tensors |
| `no_context` (`open()` without `with`) | 12 | LOW — all in scripts/tools |
| `bare_except` | 1 | LOW |

**Python verdict:** The Python code is almost entirely model conversion scripts, GGUF metadata tools, and test harnesses. None of it is in the hot path. The `str_concat` patterns are document-generation helpers with small N. No action needed.

---

## Hot Paths (C++ — where performance actually matters)

> The 80/20 of this codebase: 279K lines in GGML, 40K in CUDA/HIP backends, 20K in `src/llama-*.cpp`.

### Tier 1: MatMul (dominant cost by far)

This is **the** hot path. ~95%+ of inference time is here.

| Operation | File | Complexity | Data Size (per token) | Est. % of inference time | Risk |
|-----------|------|------------|----------------------|--------------------------|------|
| `ggml_mul_mat()` | `ggml/src/ggml-cuda/mmq.cuh` | O(n²) | N=4096..8192 (hidden dim) | **60-80%** | CRITICAL |
| `mmq_mfma()` (WMMA path) | `ggml/src/ggml-cuda/mma.cuh` | O(n²) | tile 16×16, 32×32 | 50-60% of matmul | CRITICAL |
| `ggml_mul_mat_vec_q()` (MMV) | `ggml/src/ggml-cuda/mmvq.cu` | O(n) | hidden dim × batch | 5-15% | HIGH |
| `ggml_mul_mat_id()` (MoE) | `ggml/src/ggml-cuda/ggml-cuda.cu` | O(n × n_experts) | top_k experts | 10-30% (MoE models) | HIGH |

**MatMul complexity note:** This is O(n²) in hidden dimension *per layer*. A 70B model with hidden_dim=8192 does ~160 layers × 8192² ≈ 10.7B multiply-adds per token. The MMQ kernels use `stream-k` scheduling (already patched in) to minimize this via tile-level parallelism on RDNA4.

### Tier 2: Attention

| Operation | File | Complexity | Note | Risk |
|-----------|------|------------|------|------|
| `FlashAttention` (ROCWMMA) | `ggml-cuda/fattn-mma-f16.cuh` (2033 lines) | O(n² ÷ SM) | n = seq_len, SM = streaming multiprocessors | HIGH |
| `FlashAttention` (WMMA) | `ggml-cuda/fattn-wmma-f16.cu` (698 lines) | O(n² ÷ SM) | Fallback for non-ROCWMMA | HIGH |
| `FlashAttention` (Tile) | `ggml-cuda/fattn-tile.cuh` (1361 lines) | O(n² ÷ SM) | Tile-level approach | HIGH |
| `soft_max` | `ggml-cuda/norm.cu` (698 lines) | O(n) | per-attention head | MEDIUM |
| `rope` (RoPE) | `ggml-cuda/rope.cu` (672 lines) | O(n) | per-token position encoding | LOW |
| `KQ_mask` | `ggml-cuda/ggml-cuda.cu` | O(n²) | seq_len² matrix | MEDIUM |

**Attention complexity:** FlashAttention reduces the constant factor dramatically vs. O(n²) softmax attention by tiling and avoiding materializing the full attention matrix on-chip. The ROCWMMA path (`-DGGML_HIP_ROCWMMA_FATTN=ON`) is the primary RDNA4 attention codepath.

### Tier 3: Quantization

| Operation | File | Complexity | Note | Risk |
|-----------|------|------------|------|------|
| `dequantize()` per type | `ggml-cuda/dequantize-planar-iso.cuh` (661 lines) | O(n ÷ vec_width) | **PlanarQuant** (custom RDNA4 patch) | HIGH |
| `quantize()` | `src/llama-quant.cpp` (lots) | O(n) per tensor | Model-load-time only | LOW |
| `imatrix` importance matrix | `common/imatrix-loader.cpp` | O(n²) | Load-time weight scaling | LOW |

### Tier 4: Graph & Model Execution

| Function | File | Complexity | Note | Risk |
|---------|------|------------|------|------|
| `llama_decode()` | `src/llama-context.cpp:4058` | O(n_layers × matmul) | Main decode entrypoint | CRITICAL |
| `build_llama_graph()` | `src/llama-graph.cpp:2646` | O(n_layers) | Builds compute graph each time | MEDIUM |
| `graph_compute()` | `src/llama-context.cpp:1364` | O(ops) | GGMl graph scheduler | MEDIUM |
| `process_tokens()` | `src/llama.cpp` | O(batch_size × seq_len) | Token processing | MEDIUM |
| `llama_model_loader` | `src/llama-model-loader.cpp` | O(n_tensors) | Load-time only | LOW |

### Tier 5: KV Cache

| Operation | File | Complexity | Note | Risk |
|-----------|------|------------|------|------|
| KV cache read/write | `src/llama-kv-cache.cpp` (2658 lines) | O(1) per cell | Direct buffer access | LOW |
| Cache defrag/shift | `src/llama-kv-cache.cpp` | O(n_cells) | Context-shift ops | LOW |
| PlanarQuant V compression | `src/llama-kv-cache-dsa.cpp` | O(n_kv) | Custom: Givens rotation 3-bit | MEDIUM |

### Tier 6: Sampler

| Operation | File | Complexity | Note | Risk |
|-----------|------|------------|------|------|
| `llama_sample_top_p()` | `src/llama-sampler.cpp` (3883 lines) | O(vocab log vocab) | Sorting | LOW (vocab=128K, once per token) |
| `llama_sample_temp()` | `src/llama-sampler.cpp` | O(vocab) | Per-token scaling | LOW |
| `llama_sample_grammar()` | `src/llama-grammar.cpp` (1510 lines) | O(stack_depth) | GBNF parser state machine | LOW |

---

## O(n²) or Worse — C++ Hot Path

| Function | File | Complexity | Loop Depth | Est. Impact | Recommendation |
|----------|------|------------|------------|-------------|----------------|
| `mmq_mfma()` kernel tile loops | `ggml-cuda/mma.cuh` | O(n²) | 3 (tiles, warp, element) | **CRITICAL** | Already optimized via stream-k + WMMA intrinsics |
| `fattn_mma_f16()` | `ggml-cuda/fattn-mma-f16.cuh` | O(n²/SM) | 3 (blocks, warps, tiles) | **CRITICAL** | ROCWMMA tensor core path — as good as it gets |
| Attention `Q @ K^T` | `ggml-cuda/fattn-common.cuh` | O(n²) | 2 | CRITICAL | FlashAttention tiling reduces by SM count |
| `kq_soft_max` | `ggml-cuda/norm.cu` | O(n) | 1 | HIGH | Linear in seq_len |
| Context shift in KV | `src/llama-kv-cache.cpp` | O(n_kv) | 1 | LOW | Rarely triggered |
| `ggml_cuda_cpy()` | `ggml-cuda/cpy.cu` | O(n) | 1 | MEDIUM | Host↔device transfers |
| `dequantize_planar_iso()` | `ggml-cuda/dequantize-planar-iso.cuh` | O(n/vec) | 2 | MEDIUM | Custom 3-bit planar decompress |

---

## Memory Analysis

| Allocation | Size | Location | Lifetime | Risk |
|------------|------|----------|----------|------|
| Model weights (GGUF) | Model-dependent (7B ≈ 4GB Q4, 70B ≈ 40GB Q4) | GPU VRAM | Process lifetime | CRITICAL |
| KV cache | n_layers × n_kv_heads × seq_len × d_head × 2 (K+V) | GPU VRAM | Per context | HIGH — O(seq_len) |
| Compute graph | ~100MB per graph build | CPU/GPU | Per decode call | MEDIUM |
| Intermediate tensors | ~5-50GB depending on batch x seq | GPU VRAM | Per operation | HIGH |
| GGML allocator pool | Pre-allocated arena | GPU VRAM | Process lifetime | MEDIUM |

**Key memory concern:** The GGML CUDA backend uses an arena allocator (`ggml_cuda_pool`) that holds GPU memory across decode calls. This avoids cudaMalloc/cudaFree per op but means peak usage = maximum simultaneous allocations. On a 16GB RX 9070 XT, fitting a 12B+ model with KV cache requires careful sizing — this is the hard constraint on RDNA4.

---

## Data Structure Choices

| Pattern | Location | Recommendation | Priority |
|---------|----------|---------------|----------|
| `std::vector` for tensors | Everywhere | Correct — contiguous storage | OK |
| `std::unordered_map` for type dispatch | `ggml-cuda.cu` | O(1) expected, O(n) worst — use `switch` for hot paths | LOW — switch already exists for quant types |
| `GGML_TENSOR_BINARY_OP` macro pattern | `ggml.c` | C macro metaprogramming reduces code but complicates debug | OK |
| Per-layer graph rebuild | `src/llama-graph.cpp` | Each decode call rebuilds the graph | MEDIUM — could cache unmodified graph segments |
| `cudaMemcpyAsync` with stream | `ggml-cuda.cu` | Correct async pattern | OK |
| `std::map` in sampler config | `src/llama-sampler.cpp` | O(log n) per lookup | LOW — sampler is not hot |
| Bare `int` for tensor extents | `include/llama.h` | Fine for current model sizes | OK |
| Manual CUDA/HIP kernel launch configs | `mmq.cuh`, `mma.cuh` | Hand-tuned per GPU gen | HIGH — fragile, needs per-GPU tuning |

---

## Python-specific Findings (Noteworthy)

These are from `conversion/` and `gguf-py/` — not in the hot path, but worth flagging for maintenance:

| Issue | File | Line | Impact |
|-------|------|------|--------|
| 5× `index()` in loop | `conversion/base.py` | 2287-2299 | O(n²) on weight list — ~10K tensors max → 10M ops, trivial |
| 3× `remove()` in loop | `gguf-py/gguf/metadata.py` | 330, 346 | O(n²) on metadata list |
| `string +=` in doc gen | `gguf-py/gguf/gguf_writer.py` | 152, 246, 264, 268, 1416 | O(n²) string building — small strings, negligible |
| Nested loops (depth 2) | `tests/test-tokenizer-random.py` | 258, 270 | Test generators only |

---

## Key Complexity Observations

1. **MatMul dominates.** Everything else is rounding error. The RDNA4 patches (stream-k, WMMA, ROCWMMA FA) are aimed at exactly this — to maximize ALU utilization on the 9070 XT's 64 CU.

2. **FlashAttention tiling is the critical path in attention.** The O(n²) attention complexity is mitigated by on-chip tiling (ROCWMMA/WMMA tensor cores), making it effectively O(n² ÷ SM) for the matmul portion and O(n) for softmax.

3. **PlanarQuant adds a decompression step** in the KV cache read path. The `dequantize_planar_iso()` kernel runs per KV access, adding O(n) overhead to attention reads. On a 16GB card this tradeoff (memory vs. compute) is worthwhile.

4. **No TOP_K sampler** on gfx1201 is a known RDNA4 limitation — omitting `--top-k` in server config avoids any attempt.

5. **Python code is cold path.** 198 .py files, all conversion scripts, tests, and tooling. None runs during inference. The O(n²) patterns found there are on small datasets (model config tensors) and represent no runtime risk.

Reference: [RDNA4 llama.cpp Fork Build & Porting Guide](projects/rdna4-llamacpp-fork-build)
