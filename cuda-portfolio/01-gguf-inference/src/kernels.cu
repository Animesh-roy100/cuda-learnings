// Inference kernels: Q4_0 GEMV (via __dp4a, or portable), RMSNorm, and fused
// RMSNorm+RoPE, behind wrappers that validate everything before a launch.

#include "kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "cu/check.hpp"
#include "cu/timer.hpp"
#include "gguf.h"

namespace llm {
namespace {

constexpr int QK = 32;   // weights per quantization group

// ------------------------------------------------------------ host validation

[[noreturn]] void bad(const char* op, const std::string& why) {
    throw std::invalid_argument(std::string(op) + ": " + why);
}

std::size_t checked_mul(const char* op, std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) bad(op, "size overflows size_t");
    return a * b;
}

int checked_int(const char* op, std::size_t v, const char* what) {
    if (v > static_cast<std::size_t>(INT_MAX)) bad(op, std::string(what) + " exceeds INT_MAX");
    return static_cast<int>(v);
}

void check_eps(const char* op, float eps) {
    if (!std::isfinite(eps) || eps < 0.0f) bad(op, "epsilon must be finite and >= 0");
}

void check_norm_args(const char* op, const std::vector<float>& x, const std::vector<float>& w, float eps) {
    if (x.empty()) bad(op, "input is empty");
    if (w.size() != x.size())
        bad(op, "weight length " + std::to_string(w.size()) + " != input length " + std::to_string(x.size()));
    checked_int(op, checked_mul(op, x.size(), sizeof(float)), "input byte count");
    check_eps(op, eps);
}

void check_rope_args(const char* op, const std::vector<float>& x, int n_heads, int head_dim, int pos, float theta) {
    if (n_heads <= 0) bad(op, "n_heads must be > 0");
    if (head_dim <= 0) bad(op, "head_dim must be > 0");
    if (head_dim % 2 != 0) bad(op, "head_dim must be even");
    if (checked_mul(op, std::size_t(n_heads), std::size_t(head_dim)) != x.size())
        bad(op, "input length " + std::to_string(x.size()) + " != n_heads*head_dim (" +
                    std::to_string(n_heads) + "*" + std::to_string(head_dim) + ")");
    if (!std::isfinite(theta) || theta <= 0.0f) bad(op, "theta must be finite and > 0");
    if (pos < 0) bad(op, "position must be >= 0");
    if (pos > (1 << 24)) bad(op, "position must be <= 2^24 (FP32 exactness)");
}

// ------------------------------------------------------------ device limits

struct Limits {
    int max_threads_per_block = 0;
    int max_grid_x = 0;
    int arch = 0;
};

Limits device_limits() {
    int dev = 0;
    CU_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    CU_CHECK(cudaGetDeviceProperties(&p, dev));
    return {p.maxThreadsPerBlock, p.maxGridSize[0], p.major * 10 + p.minor};
}

void check_launch(const char* op, std::size_t blocks, int threads) {
    const Limits l = device_limits();
    if (threads > l.max_threads_per_block || blocks > static_cast<std::size_t>(l.max_grid_x))
        throw DeviceUnsupported(std::string(op) + ": launch of " + std::to_string(blocks) + " blocks x " +
                                std::to_string(threads) + " threads exceeds the device (grid " +
                                std::to_string(l.max_grid_x) + ", block " +
                                std::to_string(l.max_threads_per_block) + ")");
}

// ------------------------------------------------------------ device code

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
//
// Per group the integer sum is exact: |s| <= 32 * 8 * 127 = 32512. Groups are
// accumulated in FP32, lane by lane, then summed across the warp -- a fixed
// order for a fixed launch, so results are deterministic run to run.
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

// The fallback for devices without __dp4a (below sm_61): the same integers,
// summed with ordinary multiply-adds, accumulated in the same order.
__global__ void k_gemv_q4_portable(const unsigned char* __restrict__ qs,
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
        const unsigned char* b = w + g * (QK / 2);
        const signed char* xg = xq + g * QK;
        int s = 0;
        for (int i = 0; i < QK; ++i) {
            int n = ((i & 1) ? (b[i >> 1] >> 4) : (b[i >> 1] & 0xf)) - 8;
            s += n * xg[i];
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

// Per-tensor symmetric INT8 quantization of the activation vector: round to
// nearest (ties to even), saturate to [-127, 127].
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

constexpr int kNormThreads = 256;   // one block; the reductions assume 8 warps

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(std::size_t bytes) { CU_CHECK(cudaMalloc(&p, bytes)); }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    template <typename T> T* as() { return static_cast<T*>(p); }
};

float half_to_float(std::uint16_t h) {
    const int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
    float v;
    if (exp == 0) v = std::ldexp(float(mant), -24);                       // subnormal
    else if (exp == 31) v = mant ? std::numeric_limits<float>::quiet_NaN()
                                 : std::numeric_limits<float>::infinity();
    else v = std::ldexp(float(mant + 0x400), exp - 25);
    return sign ? -v : v;
}

float activation_scale(const char* op, const std::vector<float>& x) {
    float amax = 0.0f;
    for (float v : x) {
        if (!std::isfinite(v)) bad(op, "activations must be finite");
        amax = std::fmax(amax, std::fabs(v));
    }
    return amax == 0.0f ? 1e-12f : amax / 127.0f;
}

}  // namespace

// ---------------------------------------------------------------------------
void validate(const Q4Matrix& m) {
    const char* op = "Q4Matrix";
    if (m.rows <= 0) bad(op, "rows must be > 0");
    if (m.cols <= 0) bad(op, "cols must be > 0");
    if (m.cols % QK != 0) bad(op, "cols (" + std::to_string(m.cols) + ") must be a multiple of 32");
    const std::size_t qs = checked_mul(op, std::size_t(m.rows), std::size_t(m.cols / 2));
    const std::size_t sc = checked_mul(op, std::size_t(m.rows), std::size_t(m.cols / QK));
    if (m.qs.size() != qs)
        bad(op, "qs holds " + std::to_string(m.qs.size()) + " bytes, shape needs " + std::to_string(qs));
    if (m.scales.size() != sc)
        bad(op, "scales holds " + std::to_string(m.scales.size()) + ", shape needs " + std::to_string(sc));
    for (float s : m.scales)
        if (!std::isfinite(s)) bad(op, "non-finite scale");
}

Q4Matrix quantize_q4(const std::vector<float>& w, int rows, int cols) {
    const char* op = "quantize_q4";
    if (rows <= 0 || cols <= 0) bad(op, "rows and cols must be > 0");
    if (cols % QK != 0) bad(op, "cols must be a multiple of 32");
    if (checked_mul(op, std::size_t(rows), std::size_t(cols)) != w.size())
        bad(op, "weight size does not match rows*cols");
    for (float v : w)
        if (!std::isfinite(v)) bad(op, "weights must be finite");

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
    validate(m);
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

Q4Matrix q4_from_gguf_q4_0(const std::uint8_t* blocks, std::size_t bytes, int rows, int cols) {
    const char* op = "q4_from_gguf_q4_0";
    if (rows <= 0 || cols <= 0) bad(op, "rows and cols must be > 0");
    if (cols % QK != 0) bad(op, "cols (" + std::to_string(cols) + ") must be a multiple of 32");
    const std::size_t nblocks = checked_mul(op, std::size_t(rows), std::size_t(cols / QK));
    const std::size_t need = checked_mul(op, nblocks, 18);
    if (bytes != need) bad(op, std::to_string(bytes) + " bytes, shape needs " + std::to_string(need));
    if (!blocks) bad(op, "null block pointer");

    Q4Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.qs.assign(checked_mul(op, std::size_t(rows), std::size_t(cols / 2)), 0);
    m.scales.resize(nblocks);
    for (std::size_t b = 0; b < nblocks; ++b) {
        const std::uint8_t* blk = blocks + b * 18;
        std::uint16_t d16;
        std::memcpy(&d16, blk, 2);
        const float d = half_to_float(d16);
        if (!std::isfinite(d)) bad(op, "block " + std::to_string(b) + " has a non-finite scale");
        m.scales[b] = d;
        std::uint8_t* dst = m.qs.data() + b * (QK / 2);
        for (int i = 0; i < QK; ++i) {
            // GGUF: weight i < 16 is the low nibble of byte i, weight i >= 16
            // the high nibble of byte i-16. Internal: interleaved.
            const unsigned n = i < 16 ? (blk[2 + i] & 0xf) : (blk[2 + i - 16] >> 4);
            dst[i >> 1] |= static_cast<std::uint8_t>((i & 1) ? n << 4 : n);
        }
    }
    return m;
}

Q4Matrix q4_from_gguf(const GgufFile& file, const GgufTensor& t) {
    const std::string op = "q4_from_gguf(" + t.name + ")";
    if (t.type != GgmlType::Q4_0)
        bad(op.c_str(), std::string("tensor type is ") + type_traits(t.type).name + ", expected Q4_0");
    if (t.dims.size() != 2) bad(op.c_str(), "tensor must be two-dimensional");
    if (t.dims[0] > std::uint64_t(INT_MAX) || t.dims[1] > std::uint64_t(INT_MAX))
        bad(op.c_str(), "dimension exceeds INT_MAX");
    // ggml order: dims[0] is the fastest-varying index -- the column.
    return q4_from_gguf_q4_0(static_cast<const std::uint8_t*>(file.tensor_data(t)), std::size_t(t.num_bytes()),
                             int(t.dims[1]), int(t.dims[0]));
}

bool device_supports_dp4a() { return device_limits().arch >= 61; }

std::vector<float> gemv_q4(const Q4Matrix& w, const std::vector<float>& x, float* elapsed_ms, Q4Kernel kernel) {
    const char* op = "gemv_q4";
    validate(w);
    if (x.size() != std::size_t(w.cols))
        bad(op, "x length " + std::to_string(x.size()) + " != matrix cols " + std::to_string(w.cols));
    const int M = w.rows, K = w.cols;
    const float xs = activation_scale(op, x);

    const int WPB = 8, T = 256;
    check_launch(op, (std::size_t(M) + WPB - 1) / WPB, WPB * 32);
    check_launch(op, (std::size_t(K) + T - 1) / T, T);
    const bool dp4a = device_supports_dp4a();
    if (kernel == Q4Kernel::Dp4a && !dp4a)
        throw DeviceUnsupported("gemv_q4: __dp4a needs sm_61 or newer; use Q4Kernel::Portable or Auto");
    const bool use_dp4a = kernel == Q4Kernel::Dp4a || (kernel == Q4Kernel::Auto && dp4a);

    DevBuf d_qs(w.qs.size());
    DevBuf d_sc(checked_mul(op, w.scales.size(), sizeof(float)));
    DevBuf d_x(checked_mul(op, x.size(), sizeof(float)));
    DevBuf d_xq(x.size());
    DevBuf d_y(checked_mul(op, std::size_t(M), sizeof(float)));

    CU_CHECK(cudaMemcpy(d_qs.p, w.qs.data(), w.qs.size(), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_sc.p, w.scales.data(), w.scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));

    k_quant_act<<<(K + T - 1) / T, T>>>(d_x.as<float>(), d_xq.as<signed char>(), 1.0f / xs, K);
    CU_CHECK_KERNEL();

    cu::EventTimer t;
    t.start();
    if (use_dp4a)
        k_gemv_q4<<<(M + WPB - 1) / WPB, WPB * 32>>>(d_qs.as<unsigned char>(), d_sc.as<float>(),
                                                     d_xq.as<signed char>(), xs, d_y.as<float>(), M, K);
    else
        k_gemv_q4_portable<<<(M + WPB - 1) / WPB, WPB * 32>>>(d_qs.as<unsigned char>(), d_sc.as<float>(),
                                                              d_xq.as<signed char>(), xs, d_y.as<float>(), M, K);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> y(M);
    CU_CHECK(cudaMemcpy(y.data(), d_y.p, std::size_t(M) * sizeof(float), cudaMemcpyDeviceToHost));
    return y;
}

std::vector<float> gemv_f32(const std::vector<float>& w, int rows, int cols,
                            const std::vector<float>& x, float* elapsed_ms) {
    const char* op = "gemv_f32";
    if (rows <= 0 || cols <= 0) bad(op, "rows and cols must be > 0");
    if (cols % 4 != 0) bad(op, "cols must be a multiple of 4");
    if (w.size() != checked_mul(op, std::size_t(rows), std::size_t(cols)))
        bad(op, "weight length " + std::to_string(w.size()) + " != rows*cols");
    if (x.size() != std::size_t(cols)) bad(op, "x length " + std::to_string(x.size()) + " != cols");
    const int WPB = 8;
    check_launch(op, (std::size_t(rows) + WPB - 1) / WPB, WPB * 32);

    DevBuf d_w(checked_mul(op, w.size(), sizeof(float)));
    DevBuf d_x(x.size() * sizeof(float));
    DevBuf d_y(std::size_t(rows) * sizeof(float));
    CU_CHECK(cudaMemcpy(d_w.p, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));

    cu::EventTimer t;
    t.start();
    k_gemv_f32<<<(rows + WPB - 1) / WPB, WPB * 32>>>(d_w.as<float>(), d_x.as<float>(),
                                                     d_y.as<float>(), rows, cols);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> y(rows);
    CU_CHECK(cudaMemcpy(y.data(), d_y.p, std::size_t(rows) * sizeof(float), cudaMemcpyDeviceToHost));
    return y;
}

std::vector<float> gemv_q4_cpu(const Q4Matrix& w, const std::vector<float>& x) {
    const char* op = "gemv_q4_cpu";
    validate(w);
    if (x.size() != std::size_t(w.cols)) bad(op, "x length must equal matrix cols");
    const int M = w.rows, K = w.cols, ng = K / QK;

    // Identical activation quantization to the kernel path.
    const float xs = activation_scale(op, x);
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

std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& weight, float eps) {
    const char* op = "rmsnorm";
    check_norm_args(op, x, weight, eps);
    check_launch(op, 1, kNormThreads);
    const int n = static_cast<int>(x.size());
    const std::size_t bytes = std::size_t(n) * sizeof(float);

    DevBuf d_x(bytes), d_w(bytes), d_o(bytes);
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_w.p, weight.data(), bytes, cudaMemcpyHostToDevice));

    k_rmsnorm<<<1, kNormThreads>>>(d_x.as<float>(), d_w.as<float>(), d_o.as<float>(), n, eps);
    CU_CHECK_KERNEL();

    std::vector<float> out(n);
    CU_CHECK(cudaMemcpy(out.data(), d_o.p, bytes, cudaMemcpyDeviceToHost));
    return out;
}

std::size_t rmsnorm_rope_fused_capacity() {
    std::size_t avail = 0;
    CU_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(&avail, k_rmsnorm_rope, 1, kNormThreads));
    return avail / sizeof(float);
}

std::vector<float> rmsnorm_rope(const std::vector<float>& x, const std::vector<float>& weight,
                                float eps, int n_heads, int head_dim, int pos, float theta,
                                float* elapsed_ms, bool* fused) {
    const char* op = "rmsnorm_rope";
    check_norm_args(op, x, weight, eps);
    check_rope_args(op, x, n_heads, head_dim, pos, theta);
    check_launch(op, 1, kNormThreads);

    // The fused kernel stages the vector in one block's shared memory. Past
    // what the device leaves for it, the launch would fail -- so take the
    // global-memory path instead, which computes the same thing.
    if (x.size() > rmsnorm_rope_fused_capacity()) {
        if (fused) *fused = false;
        return rmsnorm_then_rope_unfused(x, weight, eps, n_heads, head_dim, pos, theta, elapsed_ms);
    }
    if (fused) *fused = true;

    const int n = static_cast<int>(x.size());
    const std::size_t bytes = std::size_t(n) * sizeof(float);
    DevBuf d_x(bytes), d_w(bytes), d_o(bytes);
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_w.p, weight.data(), bytes, cudaMemcpyHostToDevice));

    cu::EventTimer t;
    t.start();
    k_rmsnorm_rope<<<1, kNormThreads, bytes>>>(d_x.as<float>(), d_w.as<float>(), d_o.as<float>(), n, eps,
                                                n_heads, head_dim, pos, theta);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> out(n);
    CU_CHECK(cudaMemcpy(out.data(), d_o.p, bytes, cudaMemcpyDeviceToHost));
    return out;
}

std::vector<float> rmsnorm_then_rope_unfused(const std::vector<float>& x,
                                             const std::vector<float>& weight, float eps,
                                             int n_heads, int head_dim, int pos, float theta,
                                             float* elapsed_ms) {
    const char* op = "rmsnorm_then_rope_unfused";
    check_norm_args(op, x, weight, eps);
    check_rope_args(op, x, n_heads, head_dim, pos, theta);
    const int n = static_cast<int>(x.size());
    const int half_total = n_heads * (head_dim / 2);
    const int T = 256;
    check_launch(op, 1, kNormThreads);
    check_launch(op, (std::size_t(half_total) + T - 1) / T, T);

    const std::size_t bytes = std::size_t(n) * sizeof(float);
    DevBuf d_x(bytes), d_w(bytes), d_o(bytes);
    CU_CHECK(cudaMemcpy(d_x.p, x.data(), bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_w.p, weight.data(), bytes, cudaMemcpyHostToDevice));

    cu::EventTimer t;
    t.start();
    k_rmsnorm<<<1, kNormThreads>>>(d_x.as<float>(), d_w.as<float>(), d_o.as<float>(), n, eps);
    k_rope<<<(half_total + T - 1) / T, T>>>(d_o.as<float>(), n_heads, head_dim, pos, theta);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> out(n);
    CU_CHECK(cudaMemcpy(out.data(), d_o.p, bytes, cudaMemcpyDeviceToHost));
    return out;
}

// ---------------------------------------------------------------------------
std::vector<float> rmsnorm_cpu(const std::vector<float>& x, const std::vector<float>& weight, float eps) {
    check_norm_args("rmsnorm_cpu", x, weight, eps);
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
    check_rope_args("rmsnorm_rope_cpu", x, n_heads, head_dim, pos, theta);
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
    std::size_t n_posinf = 0;
    bool any_nan = false, any_finite_or_posinf = false;
    float m = -std::numeric_limits<float>::infinity();
    for (float e : v) {
        if (std::isnan(e)) any_nan = true;
        else if (e == std::numeric_limits<float>::infinity()) ++n_posinf;
        else if (e != -std::numeric_limits<float>::infinity()) m = std::max(m, e);
        if (!std::isnan(e) && e != -std::numeric_limits<float>::infinity()) any_finite_or_posinf = true;
    }
    if (any_nan) {
        std::fill(v.begin(), v.end(), std::numeric_limits<float>::quiet_NaN());
        return;
    }
    if (!any_finite_or_posinf) throw std::invalid_argument("softmax_cpu: every logit is -inf");
    if (n_posinf) {
        for (auto& e : v) e = e == std::numeric_limits<float>::infinity() ? 1.0f / float(n_posinf) : 0.0f;
        return;
    }
    double sum = 0.0;
    std::vector<double> ex(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) sum += ex[i] = std::exp(double(v[i]) - m);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(ex[i] / sum);
}

}  // namespace llm
