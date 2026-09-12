// Q4_0 GEMV and dequantization kernels for the PyTorch extension.
//
// The GEMV kernels are 19-llm-engine's, generalized from one activation vector
// to a batch: block x selects a group of 8 output rows (one warp each), block y
// selects the batch element, so a batch of B runs B times as many warps rather
// than B launches.

#include "q4_kernels.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace q4 {
namespace {

constexpr int QK = 32;
constexpr unsigned FULL = 0xffffffffu;
constexpr int kRowsPerBlock = 8;   // 256 threads = 8 warps = 8 output rows

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FULL, v, off);
    return v;
}
__device__ __forceinline__ int nib(unsigned int p, int i) {
    return static_cast<int>((p >> (4 * i)) & 0xfu) - 8;
}
__device__ __forceinline__ int pack4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}

__global__ void k_gemv_float(const unsigned char* __restrict__ qs, const float* __restrict__ ws_all,
                             const float* __restrict__ x_all, float* __restrict__ y_all,
                             int M, int K) {
    const int row = blockIdx.x * kRowsPerBlock + (threadIdx.x / 32);
    if (row >= M) return;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int ngroups = K / QK;
    const unsigned char* w = qs + size_t(row) * (K / 2);
    const float* ws = ws_all + size_t(row) * ngroups;
    const float* x = x_all + size_t(b) * K;

    float acc = 0.0f;
    for (int g = lane; g < ngroups; g += 32) {
        const unsigned char* bytes = w + g * (QK / 2);
        const float* xv = x + g * QK;
        float s = 0.0f;
        for (int e = 0; e < QK; ++e)
            s += float(int((bytes[e >> 1] >> (4 * (e & 1))) & 0xf) - 8) * xv[e];
        acc += s * ws[g];
    }
    acc = warp_sum(acc);
    if (lane == 0) y_all[size_t(b) * M + row] = acc;
}

__global__ void k_quant_act(const float* __restrict__ x_all, signed char* __restrict__ xq_all,
                            float* __restrict__ xs_all, int groups) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int b = blockIdx.y;
    const float* v = x_all + (size_t(b) * groups + g) * QK;
    float amax = 0.0f;
    for (int j = 0; j < QK; ++j) amax = fmaxf(amax, fabsf(v[j]));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    signed char* q = xq_all + (size_t(b) * groups + g) * QK;
    for (int j = 0; j < QK; ++j)
        q[j] = static_cast<signed char>(fminf(127.0f, fmaxf(-127.0f, floorf(v[j] * inv + 0.5f))));
    xs_all[size_t(b) * groups + g] = scale;
}

__global__ void k_gemv_int8(const unsigned char* __restrict__ qs, const float* __restrict__ ws_all,
                            const signed char* __restrict__ xq_all, const float* __restrict__ xs_all,
                            float* __restrict__ y_all, int M, int K) {
    const int row = blockIdx.x * kRowsPerBlock + (threadIdx.x / 32);
    if (row >= M) return;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int ngroups = K / QK;
    const unsigned char* w = qs + size_t(row) * (K / 2);
    const float* ws = ws_all + size_t(row) * ngroups;
    const signed char* xq = xq_all + size_t(b) * K;
    const float* xs = xs_all + size_t(b) * ngroups;

    float acc = 0.0f;
    for (int g = lane; g < ngroups; g += 32) {
        uint4 p = *reinterpret_cast<const uint4*>(w + g * (QK / 2));
        const int4* xp = reinterpret_cast<const int4*>(xq + g * QK);
        int4 x0 = xp[0], x1 = xp[1];
        unsigned int q[4] = {p.x, p.y, p.z, p.w};
        int xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        int s = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int lo = pack4(nib(q[j], 0), nib(q[j], 1), nib(q[j], 2), nib(q[j], 3));
            int hi = pack4(nib(q[j], 4), nib(q[j], 5), nib(q[j], 6), nib(q[j], 7));
            s = __dp4a(lo, xv[2 * j], s);
            s = __dp4a(hi, xv[2 * j + 1], s);
        }
        acc += float(s) * ws[g] * xs[g];
    }
    acc = warp_sum(acc);
    if (lane == 0) y_all[size_t(b) * M + row] = acc;
}

__global__ void k_dequant(const unsigned char* __restrict__ qs, const float* __restrict__ ws_all,
                          float* __restrict__ w_out, int M, int K) {
    const int row = blockIdx.y;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int ngroups = K / QK;
    if (row >= M || g >= ngroups) return;
    const unsigned char* bytes = qs + size_t(row) * (K / 2) + g * (QK / 2);
    const float scale = ws_all[size_t(row) * ngroups + g];
    float* dst = w_out + size_t(row) * K + g * QK;
    for (int e = 0; e < QK; ++e)
        dst[e] = float(int((bytes[e >> 1] >> (4 * (e & 1))) & 0xf) - 8) * scale;
}

thread_local char g_error[256] = "";

void record_error() {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        snprintf(g_error, sizeof g_error, "kernel launch failed: %s", cudaGetErrorName(e));
    else
        g_error[0] = '\0';
}

dim3 row_grid(int rows, int batch) {
    return dim3((rows + kRowsPerBlock - 1) / kRowsPerBlock, batch);
}

}  // namespace

void gemv_float(const std::uint8_t* nibbles, const float* scales, const float* x, float* y,
                int batch, int out_features, int in_features, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    k_gemv_float<<<row_grid(out_features, batch), dim3(32 * kRowsPerBlock, 1), 0, s>>>(
        nibbles, scales, x, y, out_features, in_features);
    record_error();
}

void gemv_int8(const std::uint8_t* nibbles, const float* scales, const float* x, float* y,
               signed char* xq, float* xs, int batch, int out_features, int in_features,
               void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int groups = in_features / QK;
    k_quant_act<<<dim3((groups + 255) / 256, batch), dim3(256, 1), 0, s>>>(x, xq, xs, groups);
    record_error();
    if (g_error[0]) return;
    k_gemv_int8<<<row_grid(out_features, batch), dim3(32 * kRowsPerBlock, 1), 0, s>>>(
        nibbles, scales, xq, xs, y, out_features, in_features);
    record_error();
}

void dequantize(const std::uint8_t* nibbles, const float* scales, float* w, int out_features,
                int in_features, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    const int groups = in_features / QK;
    k_dequant<<<dim3((groups + 255) / 256, out_features), dim3(256, 1), 0, s>>>(
        nibbles, scales, w, out_features, in_features);
    record_error();
}

const char* last_error() { return g_error; }

}  // namespace q4
