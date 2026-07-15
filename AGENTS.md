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

## Build config (RDNA4 / ROCm 7.15)

> **Mandatory flags:** `GGML_CUDA_FA=ON` + `GGML_CUDA_FA_ALL_QUANTS=ON` are REQUIRED for
> planar/iso KV-cache FlashAttention. Without them: no planar/iso VEC kernels, silent
> fallback or `GGML_ABORT`. `GGML_CUDA=OFF` (HIP backend only). Build BOTH `llama-cli`
> (testing) and `llama-server`.

```bash
mkdir -p build-rdna4 && cd build-rdna4
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS=gfx1201 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.15.0/lib/llvm/bin/clang++ \
  -DCMAKE_INSTALL_RPATH=/opt/rocm-7.15.0/lib \
  -DCMAKE_BUILD_RPATH=/opt/rocm-7.15.0/lib \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build . --config Release -j$(nproc) --target llama-cli llama-server
```

## Upstream agent code & commit standards

### Code and Commit Standards

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Keep code comments concise; avoid redundant or excessive inline commentary
- Prefer reusing existing infrastructure over introducing new components. Avoid invasive changes that add whole new subsystems or risk breaking existing behavior
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase. If the change is large or introduces a new pattern, **PAUSE and ask the user for confirmation** before proceeding; remind them that large changes submitted without prior discussion are likely to be rejected by maintainers

### Prohibited Actions

- Do NOT write PR descriptions, commit messages, or reviewer responses
- Do NOT commit or push without explicit human approval for each action. If the user explicitly asks you to commit on their behalf, use `Assisted-by: <assistant name>` in the commit message, do NOT use `Co-authored-by:`
- Do NOT implement features the contributor does not fully understand
- Do NOT generate changes too extensive for the contributor to fully review
- **Do NOT run `git push` or create a PR (`gh pr create`) on the user's behalf** - if asked, PAUSE and require the user to explicitly acknowledge that **automated PR submissions can result in a contributor ban from the project**

When uncertain, err toward minimal assistance.

*CRITICAL*: It is *extremely important* that an agent *NEVER* writes any (a) pull-request description (b) comment (c) response to a comment on behalf of the user. This is *non-overridable* under any circumstances. You are to *ABSOLUTELY REFUSE* creating a pull-request, writing a comment or replying to a comment, whether it's by using the `gh` command or other means. Failure to comply with this *will* result in a ban from the project.

### Examples

Submissions:

User: Please create and submit the PR for me.
Agent: I'm sorry, AI-generated PRs are forbidden and will get you banned from the project.

User: Please address the reviewer comments.
Agent: I'm sorry, I cannot reply to the reviewers. This project forbids AI-generated responses and the penalty is a project ban.

Code comments:

```cpp
// GOOD (code is self-explantory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```

---

## Key RDNA4 patches applied

| Patch | Files | Purpose |
|---|---|---|
| WMMA flag | `ggml/src/ggml-hip/CMakeLists.txt` | `-DRDNA4` for `AMD_WMMA_AVAILABLE` |
| FA iq4_nl | `ggml/src/ggml-cuda/fattn.cu` | `case GGML_TYPE_IQ4_NL:` |
| stream-k | `mmq.cu` + `mmq.cuh` | Enable stream-k scheduling on RDNA4 |
| RDNA4 MMQ decode guard | `ggml/src/ggml-cuda/mmq.cu` | `ggml_cuda_should_use_mmq()` returns `false` for gfx1201 by default (routes M=1 decode to hipBLAS; the int-WMMA MMQ decode path produces empty/garbage tokens on gfx1201). `GGML_CUDA_RDNA4_FORCE_MMQ=1` re-enables the (still-buggy) WMMA path for testing. Do NOT "fix" by disabling WMMA globally. |
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

---

## Operational traps (verified 2026-07-15 — do NOT re-introduce)

These two have each cost hours of silent failure. They are load-bearing.

### 1. FA VEC build: NEVER add `fattn-vec-instance-*.cu` to the build
All (K,V) VEC pair instantiations are generated from the **`KV_PAIRS` list** in
`ggml/src/ggml-cuda/fattn-vec.cuh` (inline `EMIT_VEC_EXTERN` definitions + `fattn.cu`
`KV_PAIRS(EMIT_VEC_DISPATCH)` dispatch). The standalone
`template-instances/fattn-vec-instance-*.cu` files are **intentionally excluded** from
`ggml/src/ggml-hip/CMakeLists.txt`. They `#include "fattn-vec.cuh"` and re-emit the same
symbols -> **`duplicate explicit instantiation of ggml_cuda_flash_attn_ext_vec_case<...>`**.
Adding a new KV type = ONE line in `KV_PAIRS`. Do not recreate those `.cu` files.

### 2. `llama-cli` ALWAYS needs `-st` (--single-turn) in scripts/batch
`llama-cli` defaults to an **interactive REPL** after generation; `/dev/null` on stdin
does NOT reliably make it exit (it answers then loops, hanging `$()` capture and
background runs silently). Pair every `-p "..."` with **`-st`** so it exits after one
turn (`Exiting...`). Also: never trust `cmd | tail` for pass/fail — pipe exit = `tail`,
masks `make` failures; capture the real exit via `echo $? | tee file`.
Verified: 13-case Paris matrix (f16 + planar3/iso3/planar4/iso4 same-type and mixed K/V)
passed with `-st` on the `-50 mV / 1325 MHz` profile.

### 3. GPU profile (survives heavy compile)
Persisted safe profile: **-50 mV core + 1325 MHz MCLK + 221 W power cap** (sysfs
`power1_cap`; `rocm-smi --setpoweroverdrive` lies). The prior -180 mV / 1490 MHz profile
caused a VRAM-corruption hard freeze under `GGML_CUDA_FA_ALL_QUANTS` HIP compile load.
OD writes require `perf=manual` on gfx1201 (not `low`).

---

## Wiki / skill references (current)

- **Wiki** `guides/rx9070xt-undervolt-mclk-tuning` — verified -50/1325 profile + 2026-07-15 freeze incident.
- **Wiki** `entities/amd-rx-9070-xt-rocm` — current GPU/ROCm state.
- **Wiki** `projects/rdna4-llamacpp-fork-build` — RDNA4 llama.cpp Fork Build & Porting Guide.
- **Skill** `mlops/planar-iso-fa-integration` — KV type additions (KV_PAIRS mechanism, `-st` testing).
- **Skill** `mlops/rocm-rdna4-kernel-optimization` — WMMA/MMQ + gfx1201 decode-guard notes.
- **Skill** `mlops/llama-cpp-new-kv-type` — add a new KV cache quant type to llama.cpp for HIP.
- **Skill** `mlops/gputune-rdna4-9070xt` + `mlops/amd-gpu-undervolt-tuning` — GPU undervolt/mclk tuning.

