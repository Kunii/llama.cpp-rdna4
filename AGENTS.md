# llama.cpp-rdna4 — Project Rules

**Repo:** `github.com/Kunii/llama.cpp-rdna4` (private fork)
**Base:** ggml-org/llama.cpp `b9859` (`4fc4ec554`)
**GPU target:** RDNA4 gfx1201 (RX 9070 XT, ROCm 7.2)
**CI:** Disabled entirely. No GitHub Actions, no workflows.

---

## Codebase navigation: graphify FIRST

**HARD RULE: Before ANY file search, grep, or code reading, check if `graphify-out/graph.json` exists. If yes, use `graphify query "..."` to answer questions about the codebase.**

The graph stores the complete file structure, AST relationships (function calls, type references, includes), and community clusters. The graph is always more efficient than file-by-file searching for:

- "What files touch the KV cache?"
- "Trace the dispatch chain for a new quant type"
- "Which files include ggml-cuda.cuh?"
- "How does FlashAttention dispatch work?"
- "List every file that references GGML_TYPE_IQ4_NL"

### Key community map (for targeted queries)

| Community | Area | Example nodes |
|---|---|---|
| 65, 27, 204, 180 | **FlashAttention** | fattn-vec, fattn-wmma-f16, fattn-mma-f16, fattn-tile |
| 17, 43, 137, 15 | **Quantization types** | iq4_nl, q4_0, block structs, dequantize |
| 123, 487, 724, 188 | **MMQ kernels** | mmq.cuh, mul_mat_q, stream-k |
| 83, 21, 107, 347 | **CUDA dispatch** | cpy.cu, set-rows.cu, ggml-cuda.cu |
| 44, 65, 180 | **WMMA / tensor cores** | wmma-f16, mma-f16 |
| 400 | **PlanarQuant** (once applied) | planar3 types |

### Graph maintenance

```bash
# Full rebuild (after major changes):
cd /path/to/llama.cpp-rdna4
export OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' ~/.hermes/.env | head -1 | cut -d= -f2-)"
graphify . --mode deep --backend openrouter --model "@preset/dsv4f-ds"

# Fast incremental update (after small commits):
graphify . --update

# Query the existing graph:
graphify query "your question about the codebase"
```

### When to NOT use graphify

- Reading file contents (use `read_file`)
- Writing/modifying files (use `write_file` / `patch`)
- Small single-file lookups where you know the exact path

---

## Implementation workflow

1. **Plan first** — write plan to `.hermes/plans/`. Bite-sized tasks, exact file paths, complete code.
2. **Query graph** — before implementing, understand the relevant subsystem via `graphify query`.
3. **Implement** — one commit per logical change. Each commit independently buildable.
4. **Test on RDNA4** — build with `build-rdna4/` cmake config, run smoke test with `llama-server`.
5. **Graph update** — after push, run `graphify . --update` to keep graph current.

---

## Build config (RDNA4 / ROCm 7.2)

```bash
mkdir -p build-rdna4 && cd build-rdna4
cmake .. \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++ \
  -DAMDGPU_TARGETS=gfx1201 \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc) --target llama-server
```

---

## Key RDNA4 patches applied

| Patch | Files | Purpose |
|---|---|---|
| WMMA flag | `ggml/src/ggml-hip/CMakeLists.txt` | `-DRDNA4` for `AMD_WMMA_AVAILABLE` |
| FA iq4_nl | `ggml/src/ggml-cuda/fattn.cu` | `case GGML_TYPE_IQ4_NL:` |
| stream-k | `mmq.cu` + `mmq.cuh` | Enable stream-k scheduling on RDNA4 |
| PlanarQuant KV | ~18 files | Givens rotation 3-bit KV cache compression |

---

## Wiki reference

Full E2E breakdown of the fork, all patches, PlanarQuant architecture, and build guide:

**`projects/rdna4-llamacpp-fork-build`** — RDNA4 llama.cpp Fork Build & Porting Guide

Derived Hermes skill for future KV type additions:

**`mlops/llama-cpp-new-kv-type`** — Add a new KV cache quant type to llama.cpp for HIP

---

## Git rules

- Commits pushed directly to `master` (private fork, no reviews)
- One logical change per commit
- Commit messages: concise, no unicode (ASCII only)

---

## Performance

### IU4 WMMA mul_mat_q (q4_0/q4_1 on gfx1201)

| Test | IU4 (t/s) | IU8 (t/s) | Delta |
|------|-----------|-----------|-------|
| pp64  | 2212 | 1904 | **+16.2%** |
| pp128 | 3031 | 2953 | **+2.6%** |
| pp256 | 3604 | 3317 | **+8.6%** |
| tg64  | 84.5 | 84.1 | flat |

Build with `-DMMQ_IU4_ENABLE=OFF` to disable and compare.

### Non-FA planar/iso KV types

All 4 planar/iso KV types work with `--flash-attn off` + f16 V. Throughput ~44-50% of f16 FA baseline. Commit `53d113002`.

---

## Known RDNA4 Limitations

- **No TOP_K sampler op** on gfx1201. Always omit `--top-k` in `llama-server` / `llama-cli`.
- **AMDGPU workqueue hogging**: `svm_range_deferred_list_work [amdgpu]` can cause kernel panics under sustained GPU load. Kernel suggests switching to `WQ_UNBOUND`.
- **Symmetric quantized V** (q4_0+q4_0 etc.) requires FlashAttention (`V cache quantization requires flash_attn`).
