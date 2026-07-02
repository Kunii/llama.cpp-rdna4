"""Apply IU4 WMMA changes to mmq.cuh.
Reads original, applies all changes, validates pre-conditions, writes result.
Must compile on gfx1201 with MMQ_IU4_ENABLE."""

import re

MMQ_PATH = '/home/hush/llama.cpp-rdna4/ggml/src/ggml-cuda/mmq.cuh'

with open(MMQ_PATH, 'r') as f:
    content = f.read()

# ============================================================
# 1. MMQ_MMA_TILE_X_K_IU4 constant (after MMQ_MMA_TILE_X_K_Q6_K)
# ============================================================
old_const = """#define MMQ_MMA_TILE_X_K_Q6_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7)

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.")"""

# 32 + 32/8 + 0 = 36. 36 % 8 = 4. ✓ (no extra padding needed unlike 2x types)
new_const = """#define MMQ_MMA_TILE_X_K_Q6_K  (2*MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI6_K   + MMQ_TILE_NE_K/8 + 7)
#if defined(MMQ_IU4_ENABLE)
#define MMQ_MMA_TILE_X_K_IU4   (MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_0 + 0)
static_assert(MMQ_MMA_TILE_X_K_IU4 % 8 == 4, "Wrong padding for IU4. 36 %% 8 = 4.");
#endif

static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.")"""
assert old_const in content, f"CONST MARKER NOT FOUND"

content = content.replace(old_const, new_const, 1)

# ============================================================
# 2. mmq_get_mma_tile_x_k returns MMQ_MMA_TILE_X_K_IU4 for q4_0/q4_1
# ============================================================
old_getter = """        case GGML_TYPE_Q4_0:    return MMQ_MMA_TILE_X_K_Q8_0;
        case GGML_TYPE_Q4_1:    return MMQ_MMA_TILE_X_K_Q8_1;"""

new_getter = """        case GGML_TYPE_Q4_0:
#if defined(MMQ_IU4_ENABLE)
            return MMQ_MMA_TILE_X_K_IU4;
#else
            return MMQ_MMA_TILE_X_K_Q8_0;
#endif
        case GGML_TYPE_Q4_1:
#if defined(MMQ_IU4_ENABLE)
            return MMQ_MMA_TILE_X_K_IU4;
#else
            return MMQ_MMA_TILE_X_K_Q8_1;
#endif"""

assert old_getter in content, "GETTER MARKER NOT FOUND"
content = content.replace(old_getter, new_getter, 1)

# ============================================================
# 3. IU4 load_tiles function (after load_tiles_q4_0, before vec_dot_q4_0_q8_1_dp4a)
# ============================================================
iu4_load = """

#if defined(MMQ_IU4_ENABLE)
// IU4 variant: copy q4_0 nibbles as packed 4-bit (no INT8 expansion, half stride 40 vs 76)
template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0_iu4(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K);

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int kbx  = threadIdx.x / QI4_0;
    const int kqsx = threadIdx.x % QI4_0;

    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows*8) {
        int i = i0 + (threadIdx.y*nrows + threadIdx.x/threads_per_row);
        if (need_check) { i = min(i, i_max); }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbx;
        const int qs_val = ((const int *)bxi->qs)[kqsx];
        x_qs[i*MMQ_MMA_TILE_X_K_IU4 + kbx*4 + kqsx] = qs_val;
    }
    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 8 * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;
        if (need_check) { i = min(i, i_max); }
        const block_q4_0 * bxi = (const block_q4_0 *) x + kbx0 + i*stride + kbxd;
        x_df[i*MMQ_MMA_TILE_X_K_IU4 + kbxd] = bxi->d;
    }
}
#endif // MMQ_IU4_ENABLE
"""

# Insert before the start of vec_dot_q4_0_q8_1_dp4a
insert_marker = "\ntemplate <int mmq_x, int mmq_y>\nstatic __device__ __forceinline__ void vec_dot_q4_0_q8_1_dp4a("
assert insert_marker in content, "INSERT AFTER load_tiles MARKER NOT FOUND"
content = content.replace(insert_marker, iu4_load + insert_marker, 1)

# ============================================================
# 4. IU4 vec_dot function (after vec_dot_q8_0_q8_1_mma, BEFORE #endif)
# ============================================================
# The vec_dot_q8_0_q8_1_mma function ends with:
#   }  (closing brace of for loop)
# }  (closing brace of function)
# #endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
# }  (closing brace of else)
# 
# We insert the IU4 vec_dot AFTER the closing #endif line but BEFORE the trailing }
# Actually, looking at the code, the structure is:
# ... code ...
# } // close for j0
#     } // close for k01
# #endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
# } // close function
# 
# We insert after the #endif, before the }

iu4_vec = """
#if defined(MMQ_IU4_ENABLE)
// IU4 vec_dot: uses IU4 X tile stride, mma() resolves to IU4 when MMQ_IU4_ENABLE
template <int mmq_x, int mmq_y, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_iu4(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = granularity;
    constexpr int ntx = rows_per_warp/tile_C::I;

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;
        tile_A A[ntx];
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*MMQ_MMA_TILE_X_K_IU4 + k0, MMQ_MMA_TILE_X_K_IU4);
        }
        #pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            float dB;
            const int j = j0 + tile_C::get_j(0);
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            } else {
                dB = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);
                #pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    const float dA = x_df[i*MMQ_MMA_TILE_X_K_IU4 + k0/QI8_0];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA*dB;
                }
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}
#endif // MMQ_IU4_ENABLE
"""

# Find the closing of vec_dot_q8_0_q8_1_mma:
# Pattern: end of AMD_MFMA_AVAILABLE block (nested in AMD_WMMA_AVAILABLE)
# The function ends with:
#   }  // close j0+K loops
# #endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
# }  // close function
#
# We want to insert AFTER the #endif but BEFORE the closing }

# Find: "#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)\n}"
# This pattern is unique to the vec_dot_q8_0_q8_1_mma function
orig_close = "#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)\n}"
assert orig_close in content, "VEC_DOT_END NOT FOUND"

# Replace with: #endif + IU4 block + }
content = content.replace(orig_close, orig_close + iu4_vec, 1)

# ============================================================
# 5. Type traits: override vec_dot_mma for q4_0 and q4_1 when IU4 enabled
# ============================================================
old_q40 = """struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y>;
};"""

new_q40 = """struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_0> {
    static constexpr int              vdr          = VDR_Q4_0_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_0<mmq_y, need_check>;
#if defined(MMQ_IU4_ENABLE)
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_iu4<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>;
#else
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>;
#endif
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_0_q8_1_dp4a<mmq_x, mmq_y>;
};"""

assert old_q40 in content, "Q40_TRAITS NOT FOUND"
content = content.replace(old_q40, new_q40, 1)

# Same for Q4_1
old_q41 = """struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr          = VDR_Q4_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y>;
};"""

new_q41 = """struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q4_1> {
    static constexpr int              vdr          = VDR_Q4_1_Q8_1_MMQ;
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_q4_1<mmq_y, need_check>;
#if defined(MMQ_IU4_ENABLE)
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_iu4<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>;
#else
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_1_q8_1_mma<mmq_x, mmq_y>;
#endif
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q4_1_q8_1_dp4a<mmq_x, mmq_y>;
};"""

assert old_q41 in content, "Q41_TRAITS NOT FOUND"
content = content.replace(old_q41, new_q41, 1)

# ============================================================
# 6. Write updated file
# ============================================================
with open(MMQ_PATH, 'w') as f:
    f.write(content)

# Verify
print("Changes applied:")
print(f"  1. MMQ_MMA_TILE_X_K_IU4 = 36 constant + static_assert")
print(f"  2. mmq_get_mma_tile_x_k returns IU4 stride for q4_0/q4_1")
print(f"  3. load_tiles_q4_0_iu4 (packed 4-bit, stride 36)")
print(f"  4. vec_dot_q8_0_q8_1_iu4 (IU4 X stride, Y stays 36)")
print(f"  5. Q4_0 type traits: vec_dot_mma -> _iu4")
print(f"  6. Q4_1 type traits: vec_dot_mma -> _iu4")
