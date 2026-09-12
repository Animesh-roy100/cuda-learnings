// Inference kernels: Q4_0 GEMV via __dp4a, RMSNorm, and fused RMSNorm+RoPE.

#include "kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace llm {
namespace {

constexpr int QK = 32;   // weights per quantization group

__device__ __forceinline__ int nib(unsigned int p, int i) {
    return static_cast<int>((p >> (4 * i)) & 0xfu) - 8;   // stored biased by +8
}
__device__ __forceinline__ int pack4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// W4A8 GEMV. One warp per output row. 32 packed weights arrive as ONE 16-byte
// load and become 8 __dp4a instructions.
// ---------------------------------------------------------------------------
__global__ void k_gemv_q4(const unsigned char* __restrict__ qs,
                          const float* __restrict__ scales,
                          const signed char* __restrict__ xq, float xscale,
                          float* __restrict__ y, int M, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;
    int lane = threadIdx.x & 31;

    const int ngroups = K / QK;
    const unsigned char* w = qs + static_cast<size_t>(row) * (K / 2);
    const float* ws = scales + static_cast<size_t>(row) * ngroups;

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
        acc += static_cast<float>(s) * ws[g];
    }
    acc = warp_sum(acc) * xscale;
    if (lane == 0) y[row] = acc;
}

__global__ void k_gemv_f32(const float* __restrict__ W, const float* __restrict__ x,
                           float* __restrict__ y, int M, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;
    int lane = threadIdx.x & 31;
    const float4* w4 = reinterpret_cast<const float4*>(W + static_cast<size_t>(row) * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int i = lane; i < K / 4; i += 32) {
        float4 a = w4[i], b = x4[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// Per-tensor symmetric INT8 quantization of the activation vector.
__global__ void k_quant_act(const float* __restrict__ x, signed char* __restrict__ q,
                            float inv_scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = __float2int_rn(x[i] * inv_scale);
    q[i] = static_cast<signed char>(max(-127, min(127, v)));
}

// ---------------------------------------------------------------------------
// RMSNorm. One block, whole vector in registers + shared reduction.
// ---------------------------------------------------------------------------
__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ out, int n, float eps) {
    __shared__ float ssum[32];
    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];

    local = warp_sum(local);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ssum[wid] = local;
    __syncthreads();

    float total = 0.0f;
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float v = (lane < nw) ? ssum[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) ssum[0] = v;
    }
    __syncthreads();
    total = ssum[0];

    // RMSNorm, not LayerNorm: no mean subtraction, so one pass suffices.
    float inv = rsqrtf(total / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * inv * w[i];
}

// RoPE applied to an already-normalised vector laid out [heads][head_dim].
__global__ void k_rope(float* __restrict__ v, int n_heads, int head_dim, int pos,
                       float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total = n_heads * half;
    if (idx >= total) return;

    int h = idx / half;
    int i = idx % half;
    float freq = powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
    float angle = pos * freq;
    float c = cosf(angle), s = sinf(angle);

    float* base = v + static_cast<size_t>(h) * head_dim;
    float a = base[i], b = base[i + half];
    base[i] = a * c - b * s;
    base[i + half] = a * s + b * c;
}

// ---------------------------------------------------------------------------
// Fused RMSNorm + RoPE: one kernel, one round trip.
//
// One block handles the whole vector, so the normalisation sum and the rotation
// both happen while the data is already in registers/shared. The unfused pair
// writes the normalised vector to global memory and reads it straight back --
// pure waste at decode time, where the vector is a few KB and the arithmetic
// is trivial.
// ---------------------------------------------------------------------------
__global__ void k_rmsnorm_rope(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ out, int n, float eps,
                               int n_heads, int head_dim, int pos, float theta) {
    extern __shared__ float buf[];
    __shared__ float ssum[32];

    float local = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        buf[i] = v;                     // stage once, reuse twice
        local += v * v;
    }
    local = warp_sum(local);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ssum[wid] = local;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float v = (lane < nw) ? ssum[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) ssum[0] = v;
    }
    __syncthreads();

    float inv = rsqrtf(ssum[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) buf[i] = buf[i] * inv * w[i];
    __syncthreads();

    // Rotate pairs (i, i+half) within each head.
    int half = head_dim / 2;
    int total = n_heads * half;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int h = idx / half, i = idx % half;
        float freq = powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
        float angle = pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float a = buf[h * head_dim + i];
        float b = buf[h * head_dim + i + half];
        out[h * head_dim + i] = a * c - b * s;
        out[h * head_dim + i + half] = a * s + b * c;
    }
}

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(std::size_t bytes) { CU_CHECK(cudaMalloc(&p, bytes)); }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    template <typename T> T* as() { return static_cast<T*>(p); }
};

}  // namespace

// ---------------------------------------------------------------------------
Q4Matrix quantize_q4(const std::vector<float>& w, int rows, int cols) {
    if (cols % QK != 0) throw std::invalid_argument("cols must be a multiple of 32");
    if (static_cast<std::size_t>(rows) * cols != w.size())
        throw std::invalid_argument("weight size does not match rows*cols");

    Q4Matrix m;
    m.rows = rows;
    m.cols = cols;
    const int ng = cols / QK;
    m.qs.assign(static_cast<std::size_t>(rows) * (cols / 2), 0);
    m.scales.assign(static_cast<std::size_t>(rows) * ng, 0.0f);

    for (int r = 0; r < rows; ++r) {
        for (int g = 0; g < ng; ++g) {
            const float* src = &w[static_cast<std::size_t>(r) * cols + g * QK];
            float amax = 0.0f;
            for (int i = 0; i < QK; ++i) amax = std::fmax(amax, std::fabs(src[i]));
            // Symmetric INT4 spans [-8,7]; divide by 7 so nothing clips high.
            float sc = amax / 7.0f;
            if (sc == 0.0f) sc = 1e-12f;
            m.scales[static_cast<std::size_t>(r) * ng + g] = sc;
            for (int i = 0; i < QK; ++i) {
                int v = static_cast<int>(std::lrintf(src[i] / sc));
                v = std::max(-8, std::min(7, v));
                unsigned nibble = static_cast<unsigned>(v + 8) & 0xfu;
                std::size_t byte = static_cast<std::size_t>(r) * (cols / 2) + (g * QK + i) / 2;
                if (i % 2 == 0) m.qs[byte] |= static_cast<std::uint8_t>(nibble);
                else            m.qs[byte] |= static_cast<std::uint8_t>(nibble << 4);
            }
        }
    }
    return m;
}

std::vector<float> dequantize_q4(const Q4Matrix& m) {
    std::vector<float> out(static_cast<std::size_t>(m.rows) * m.cols);
    const int ng = m.cols / QK;
    for (int r = 0; r < m.rows; ++r)
        for (int g = 0; g < ng; ++g) {
            float sc = m.scales[static_cast<std::size_t>(r) * ng + g];
            for (int i = 0; i < QK; ++i) {
                std::size_t byte = static_cast<std::size_t>(r) * (m.cols / 2) + (g * QK + i) / 2;
                int nibble = (i % 2 == 0) ? (m.qs[byte] & 0xf) : (m.qs[byte] >> 4);
                out[static_cast<std::size_t>(r) * m.cols + g * QK + i] = (nibble - 8) * sc;
            }
        }
    return out;
}

std::vector<float> gemv_q4(const Q4Matrix& w, const std::vector<float>& x, float* elapsed_ms) {
    if (static_cast<int>(x.size()) != w.cols)
        throw std::invalid_argument("x length must equal matrix cols");
    const int M = w.rows, K = w.cols;
    const int ng = K / QK;

    float amax = 0.0f;
    for (float v : x) amax = std::fmax(amax, std::fabs(v));
    float xs = (amax == 0.0f) ? 1e-12f : amax / 127.0f;

    DevBuf d_qs(w.qs.size());
    DevBuf d_sc(w.scales.size() * sizeof(float));
    DevBuf d_x(x.size() * sizeof(float));
    DevBuf d_xq(x.size());
    DevBuf d_y(static_cast<std::size_t>(M) * sizeof(float));

    CU_CHECK(cudaMemcpy(d_qs.p, w.qs.data(), w.qs.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_sc.p, w.scales.data(), w.scales.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int T = 256;
    k_quant_act<<<(K + T - 1) / T, T>>>(d_x.as<float>(), d_xq.as<signed char>(), 1.0f / xs, K);
    CU_CHECK_KERNEL();

    const int WPB = 8;
    cu::EventTimer t;
    t.start();
    k_gemv_q4<<<(M + WPB - 1) / WPB, WPB * 32>>>(d_qs.as<unsigned char>(), d_sc.as<float>(),
                                                 d_xq.as<signed char>(), xs, d_y.as<float>(),
                                                 M, K);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> y(M);
    CU_CHECK(cudaMemcpy(y.data(), d_y.p, static_cast<std::size_t>(M) * sizeof(float),
                        cudaMemcpyDeviceToHost));
    (void)ng;
    return y;
}

std::vector<float> gemv_f32(const std::vector<float>& w, int rows, int cols,
                            const std::vector<float>& x, float* elapsed_ms) {
    if (static_cast<int>(x.size()) != cols) throw std::invalid_argument("x length mismatch");
    if (cols % 4 != 0) throw std::invalid_argument("cols must be a multiple of 4");

    DevBuf d_w(w.size() * sizeof(float));
    DevBuf d_x(x.size() * sizeof(float));
    DevBuf d_y(static_cast<std::size_t>(rows) * sizeof(float));
    CU_CHECK(cudaMemcpy(d_w.p, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int WPB = 8;
    cu::EventTimer t;
    t.start();
    k_gemv_f32<<<(rows + WPB - 1) / WPB, WPB * 32>>>(d_w.as<float>(), d_x.as<float>(),
                                                     d_y.as<float>(), rows, cols);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> y(rows);
    CU_CHECK(cudaMemcpy(y.data(), d_y.p, static_cast<std::size_t>(rows) * sizeof(float),
                        cudaMemcpyDeviceToHost));
    return y;
}

std::vector<float> gemv_q4_cpu(const Q4Matrix& w, const std::vector<float>& x) {
    if (static_cast<int>(x.size()) != w.cols)
        throw std::invalid_argument("x length must equal matrix cols");
    const int M = w.rows, K = w.cols, ng = K / QK;

    // Identical activation quantization to the kernel path.
    float amax = 0.0f;
    for (float v : x) amax = std::fmax(amax, std::fabs(v));
    const float xs = (amax == 0.0f) ? 1e-12f : amax / 127.0f;
    std::vector<int> xq(K);
    for (int i = 0; i < K; ++i)
        xq[i] = std::max(-127, std::min(127, static_cast<int>(std::lrintf(x[i] / xs))));

    std::vector<float> y(M, 0.0f);
    for (int r = 0; r < M; ++r) {
        float acc = 0.0f;
        for (int g = 0; g < ng; ++g) {
            int s = 0;
            for (int i = 0; i < QK; ++i) {
                std::size_t byte = static_cast<std::size_t>(r) * (K / 2) + (g * QK + i) / 2;
                int nibble = (i % 2 == 0) ? (w.qs[byte] & 0xf) : (w.qs[byte] >> 4);
                s += (nibble - 8) * xq[g * QK + i];
            }
            acc += static_cast<float>(s) * w.scales[static_cast<std::size_t>(r) * ng + g];
        }
        y[r] = acc * xs;
    }
    return y;
}

std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& weight,
                           float eps) {
    const int n = static_cast<int>(x.size());
    if (weight.size() != x.size()) throw std::invalid_argument("weight length mismatch");

    DevBuf d_x(n * sizeof(float)), d_w(n * sizeof(float)), d_o(n * sizeof(float));
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_w.p, weight.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    k_rmsnorm<<<1, 256>>>(d_x.as<float>(), d_w.as<float>(), d_o.as<float>(), n, eps);
    CU_CHECK_KERNEL();

    std::vector<float> out(n);
    CU_CHECK(cudaMemcpy(out.data(), d_o.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<float> rmsnorm_rope(const std::vector<float>& x, const std::vector<float>& weight,
                                float eps, int n_heads, int head_dim, int pos, float theta,
                                float* elapsed_ms) {
    const int n = static_cast<int>(x.size());
    if (n != n_heads * head_dim) throw std::invalid_argument("x must be n_heads*head_dim");
    if (head_dim % 2 != 0) throw std::invalid_argument("head_dim must be even");
    if (weight.size() != x.size()) throw std::invalid_argument("weight length mismatch");

    DevBuf d_x(n * sizeof(float)), d_w(n * sizeof(float)), d_o(n * sizeof(float));
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_w.p, weight.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cu::EventTimer t;
    t.start();
    k_rmsnorm_rope<<<1, 256, n * sizeof(float)>>>(d_x.as<float>(), d_w.as<float>(),
                                                  d_o.as<float>(), n, eps, n_heads,
                                                  head_dim, pos, theta);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> out(n);
    CU_CHECK(cudaMemcpy(out.data(), d_o.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<float> rmsnorm_then_rope_unfused(const std::vector<float>& x,
                                             const std::vector<float>& weight, float eps,
                                             int n_heads, int head_dim, int pos, float theta,
                                             float* elapsed_ms) {
    const int n = static_cast<int>(x.size());
    DevBuf d_x(n * sizeof(float)), d_w(n * sizeof(float)), d_o(n * sizeof(float));
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_w.p, weight.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int half_total = n_heads * (head_dim / 2);
    cu::EventTimer t;
    t.start();
    k_rmsnorm<<<1, 256>>>(d_x.as<float>(), d_w.as<float>(), d_o.as<float>(), n, eps);
    k_rope<<<(half_total + 255) / 256, 256>>>(d_o.as<float>(), n_heads, head_dim, pos, theta);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> out(n);
    CU_CHECK(cudaMemcpy(out.data(), d_o.p, n * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

// ---------------------------------------------------------------------------
std::vector<float> rmsnorm_cpu(const std::vector<float>& x, const std::vector<float>& weight,
                               float eps) {
    double ss = 0.0;
    for (float v : x) ss += double(v) * v;
    float inv = static_cast<float>(1.0 / std::sqrt(ss / x.size() + eps));
    std::vector<float> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] * inv * weight[i];
    return out;
}

std::vector<float> rmsnorm_rope_cpu(const std::vector<float>& x,
                                    const std::vector<float>& weight, float eps,
                                    int n_heads, int head_dim, int pos, float theta) {
    auto v = rmsnorm_cpu(x, weight, eps);
    const int half = head_dim / 2;
    std::vector<float> out = v;
    for (int h = 0; h < n_heads; ++h)
        for (int i = 0; i < half; ++i) {
            float freq = std::pow(theta, -2.0f * float(i) / float(head_dim));
            float angle = pos * freq;
            float c = std::cos(angle), s = std::sin(angle);
            float a = v[h * head_dim + i];
            float b = v[h * head_dim + i + half];
            out[h * head_dim + i] = a * c - b * s;
            out[h * head_dim + i + half] = a * s + b * c;
        }
    return out;
}

void softmax_cpu(std::vector<float>& v) {
    if (v.empty()) return;
    float m = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (auto& e : v) { e = std::exp(e - m); sum += e; }
    for (auto& e : v) e = static_cast<float>(e / sum);
}

}  // namespace llm
