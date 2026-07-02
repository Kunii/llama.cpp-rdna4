// IU4 activation quantizer for RDNA4: f32 -> packed INT4 in block_q8_1_mmq format
// Each 128 values: 16 int32 packed INT4 data + 4 float scales, same 132-int total stride
// Thread: 4 f32 -> 2 bytes (4x4-bit packed), write 1 int32 at iqs/4 in qs[]

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"

#define IU4_BLOCK_VALS 128

__global__ void quantize_mmq_q4_0_kernel(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const float * x_row = x + i3 * s03 + i2 * s02 + (ids ? ids[i1] : i1) * s01;
    block_q8_1_mmq * y = (block_q8_1_mmq *)vy;

    const int64_t ib = (i0 / IU4_BLOCK_VALS) * ne1 + blockIdx.x;
    const int64_t iqs = i0 % IU4_BLOCK_VALS;

    const float4 xi = i0 < ne00 ?
        ((const float4 *)(x_row + i0))[0] : make_float4(0,0,0,0);

    float amax = fmaxf(fabsf(xi.x), fmaxf(fabsf(xi.y),
                       fmaxf(fabsf(xi.z), fabsf(xi.w))));
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));

    const float d_inv = 7.0f / fmaxf(amax, 1e-10f);

    int v0 = max(-7, min(7, __float2int_rn(xi.x * d_inv))) & 0xF;
    int v1 = max(-7, min(7, __float2int_rn(xi.y * d_inv))) & 0xF;
    int v2 = max(-7, min(7, __float2int_rn(xi.z * d_inv))) & 0xF;
    int v3 = max(-7, min(7, __float2int_rn(xi.w * d_inv))) & 0xF;

    reinterpret_cast<int *>(y[ib].qs)[iqs / 4] = v0 | (v1 << 4) | (v2 << 8) | (v3 << 12);

    if (iqs % 32 == 0)
        y[ib].d4[iqs / 32] = 1.0f / d_inv;
}

void quantize_mmq_q4_0_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4 * IU4_BLOCK_VALS) == 0);
    const int64_t block_num_y = (ne0 + 4 * 128 - 1) / (4 * 128);
    const dim3 grid(ne1, block_num_y, ne2 * ne3);
    const dim3 block(128, 1, 1);
    quantize_mmq_q4_0_kernel<<<grid, block, 0, stream>>>(
        x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
    GGML_UNUSED(type_src0);
}
