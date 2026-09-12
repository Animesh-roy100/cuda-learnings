// Llama forward pass on the device, and the token loop around it.

#include "engine.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cu/check.hpp"
#include "gguf.h"

namespace llm {
namespace {

constexpr int QK = 32;   // weights per Q4_0 group
constexpr int HD = 64;   // head dimension the attention kernel is compiled for
constexpr unsigned FULL = 0xffffffffu;

// ===========================================================================
// Kernels
//
// Every kernel that depends on the sequence position reads it from device
// memory (`dpos`) instead of taking it as an argument. That keeps each token's
// launches byte-identical to the last, which is what lets a CUDA graph replay
// them.
// ===========================================================================

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

// RMSNorm of one vector with a single warp: each lane sums the squares at the
// positions congruent to it, a reduction tree collects the total into lane 0,
// and a broadcast hands it back to every lane.
__global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ out, int n, float eps) {
    const int lane = threadIdx.x;
    float ss = 0.0f;
    for (int i = lane; i < n; i += 32) ss += x[i] * x[i];
    ss = warp_sum(ss);
    const float total = __shfl_sync(FULL, ss, 0);
    const float inv = rsqrtf(total / float(n) + eps);
    for (int i = lane; i < n; i += 32) out[i] = x[i] * inv * w[i];
}

// Per-group symmetric int8 quantization of an activation vector, one thread
// per group of 32 -- the same grouping as the weights, so each weight group is
// multiplied against an activation group with its own scale.
__global__ void k_quant_act(const float* __restrict__ x, signed char* __restrict__ xq,
                            float* __restrict__ xs, int groups) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const float* v = x + g * QK;
    float amax = 0.0f;
    for (int j = 0; j < QK; ++j) amax = fmaxf(amax, fabsf(v[j]));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int j = 0; j < QK; ++j)
        xq[g * QK + j] = static_cast<signed char>(fminf(127.0f, fmaxf(-127.0f, floorf(v[j] * inv + 0.5f))));
    xs[g] = scale;
}

// W4A8 GEMV, the 01-gguf-inference kernel with per-group activation scales.
// One warp per output row; each lane handles the groups congruent to it. Four
// weight words (32 nibbles) arrive as one 16-byte load and become eight __dp4a.
__global__ void k_gemv_q4_a8(const unsigned char* __restrict__ qs, const float* __restrict__ ws_all,
                             const signed char* __restrict__ xq, const float* __restrict__ xs,
                             float* __restrict__ y, int M, int K) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;
    const int lane = threadIdx.x & 31;
    const int ngroups = K / QK;
    const unsigned char* w = qs + static_cast<size_t>(row) * (K / 2);
    const float* ws = ws_all + static_cast<size_t>(row) * ngroups;

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
        acc += static_cast<float>(s) * ws[g] * xs[g];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// W4A16 GEMV: the same weights against float activations. Exact up to
// rounding, so this is the path validated against the host reference.
__global__ void k_gemv_q4_f(const unsigned char* __restrict__ qs, const float* __restrict__ ws_all,
                            const float* __restrict__ x, float* __restrict__ y, int M, int K) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;
    const int lane = threadIdx.x & 31;
    const int ngroups = K / QK;
    const unsigned char* w = qs + static_cast<size_t>(row) * (K / 2);
    const float* ws = ws_all + static_cast<size_t>(row) * ngroups;

    float acc = 0.0f;
    for (int g = lane; g < ngroups; g += 32) {
        const unsigned char* b = w + g * (QK / 2);
        const float* xv = x + g * QK;
        float s = 0.0f;
        for (int e = 0; e < QK; ++e) {
            const int q = int((b[e >> 1] >> (4 * (e & 1))) & 0xf) - 8;
            s += float(q) * xv[e];
        }
        acc += s * ws[g];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// Rotary embedding on n_heads heads of 64, adjacent pairs (i, i+1) -- GGUF's
// llama conversion permutes the Q and K weights so that this pairing matches
// the original model's rotate-half form. One thread per head.
__global__ void k_rope(float* __restrict__ v, int n_heads, const int* __restrict__ dpos,
                       const float* __restrict__ cos_t, const float* __restrict__ sin_t) {
    const int h = threadIdx.x;
    if (h >= n_heads) return;
    const int pos = *dpos;
    const float* c = cos_t + pos * (HD / 2);
    const float* s = sin_t + pos * (HD / 2);
    float* x = v + h * HD;
    for (int i = 0; i < HD / 2; ++i) {
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c[i] - b * s[i];
        x[2 * i + 1] = a * s[i] + b * c[i];
    }
}

__global__ void k_kv_store(const float* __restrict__ k, const float* __restrict__ v,
                           float* __restrict__ K, float* __restrict__ V, int kv_dim,
                           const int* __restrict__ dpos) {
    const int lane = threadIdx.x;
    const size_t base = size_t(*dpos) * kv_dim;
    for (int i = lane; i < kv_dim; i += 32) {
        K[base + i] = k[i];
        V[base + i] = v[i];
    }
}

// Decode attention with grouped-query heads and the online softmax from
// 18-flash-attention: one warp, one lane per query head, one pass over the
// cached keys. Query head h reads key/value head h / (n_heads / n_kv).
__global__ void k_attn_decode(const float* __restrict__ q, const float* __restrict__ K,
                              const float* __restrict__ V, float* __restrict__ out,
                              const int* __restrict__ dpos, int n_heads, int n_kv, float scale) {
    __shared__ float Qs[32][HD + 1];   // width 65: see 16-layout-advanced
    __shared__ float Os[32][HD + 1];
    const int h = threadIdx.x;
    const bool live = h < n_heads;
    const int kvh = live ? h / (n_heads / n_kv) : 0;
    const int len = *dpos + 1;

    for (int c = 0; c < HD; ++c) {
        Qs[h][c] = live ? q[h * HD + c] : 0.0f;
        Os[h][c] = 0.0f;
    }
    __syncthreads();

    float m = -FLT_MAX, s = 0.0f;
    const int last = live ? len : 0;
    for (int j = 0; j < last; ++j) {
        const float* kr = K + (size_t(j) * n_kv + kvh) * HD;
        const float* vr = V + (size_t(j) * n_kv + kvh) * HD;
        float x = 0.0f;
        for (int c = 0; c < HD; ++c) x += Qs[h][c] * kr[c];
        x *= scale;
        const float mn = fmaxf(m, x);
        const float e_old = __expf(m - mn), e_new = __expf(x - mn);
        const float sn = s * e_old + e_new;
        const float a = s * e_old / sn, b = e_new / sn;
        for (int c = 0; c < HD; ++c) Os[h][c] = Os[h][c] * a + vr[c] * b;
        s = sn;
        m = mn;
    }
    __syncthreads();
    if (live)
        for (int c = 0; c < HD; ++c) out[h * HD + c] = Os[h][c];
}

// Decode attention, one warp per query head.
//
// Launched as one block of 32 threads per head. Lane L handles the keys
// congruent to L, so a head's keys are split 32 ways and all heads run at once.
// The softmax can no longer be computed online key by key, so it is done in
// passes, each ending in a reduction across the warp:
//   A  scores into S[h][j], and each lane's max -> warp max    -> M
//   B  each lane's sum of exp(score - M)       -> warp sum    -> Z
//   C  each lane's sum of exp(score - M)/Z * v into Os[lane]
//   D  lanes stride over output dims, summing Os[0..31][c]    -> out
// Every lane reads the same query row and the same key/value head's rows, so
// the loads coalesce.
__global__ void k_attn_warp_per_head(const float* __restrict__ q, const float* __restrict__ K,
                                     const float* __restrict__ V, float* __restrict__ S,
                                     float* __restrict__ out, const int* __restrict__ dpos,
                                     int n_heads, int n_kv, float scale, int ctx) {
    __shared__ float Os[32][HD + 1];   // lane-first indexing: width 65, not 64
    const int h = blockIdx.x;
    const int lane = threadIdx.x;
    const int len = *dpos + 1;
    const int kvh = h / (n_heads / n_kv);
    const float* qh = q + h * HD;
    float* Sh = S + size_t(h) * ctx;

    float m = -FLT_MAX;
    for (int j = lane; j < len; j += 32) {
        const float* kr = K + (size_t(j) * n_kv + kvh) * HD;
        float x = 0.0f;
        for (int c = 0; c < HD; ++c) x += qh[c] * kr[c];
        x *= scale;
        Sh[j] = x;
        m = fmaxf(m, x);
    }
    // Max-reduction tree. Lane 0 only ever consumes lanes whose own partners
    // were in range, so whatever an out-of-range shuffle returns never reaches it.
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(FULL, m, off));
    const float M = __shfl_sync(FULL, m, 0);

    float z = 0.0f;
    for (int j = lane; j < len; j += 32) z += __expf(Sh[j] - M);
    z = warp_sum(z);
    const float Z = __shfl_sync(FULL, z, 0);
    const float invZ = 1.0f / Z;

    for (int c = 0; c < HD; ++c) Os[lane][c] = 0.0f;
    for (int j = lane; j < len; j += 32) {
        const float w = __expf(Sh[j] - M) * invZ;
        const float* vr = V + (size_t(j) * n_kv + kvh) * HD;
        for (int c = 0; c < HD; ++c) Os[lane][c] += w * vr[c];
    }
    __syncthreads();

    for (int c = lane; c < HD; c += 32) {
        float t = 0.0f;
        for (int l = 0; l < 32; ++l) t += Os[l][c];
        out[h * HD + c] = t;
    }
}

// SwiGLU's gate: out = silu(gate) * up.
__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = g[i];
    out[i] = x / (1.0f + __expf(-x)) * u[i];
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// ===========================================================================
// Host-side helpers
// ===========================================================================

template <typename T>
struct DevBuf {
    T* p = nullptr;
    std::size_t n = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        std::swap(p, o.p);   // the old buffer is freed by o's destructor
        std::swap(n, o.n);
        return *this;
    }
    ~DevBuf() { cudaFree(p); }
    void alloc(std::size_t count) {
        cudaFree(p);
        p = nullptr;
        n = 0;
        CU_CHECK(cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)));
        n = count;
    }
    void upload(const T* h, std::size_t count) {
        alloc(count);
        CU_CHECK(cudaMemcpy(p, h, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::size_t bytes() const { return n * sizeof(T); }
};

struct DevQ4 {
    DevBuf<unsigned char> nib;   // interleaved: weight e at byte e/2, shift 4*(e&1)
    DevBuf<float> scale;
    int rows = 0, cols = 0;
};

struct Layer {
    DevBuf<float> attn_norm, ffn_norm;
    DevQ4 q, k, v, o, gate, up, down;
    DevBuf<float> K, V;   // [context][kv_dim]
};

const GgufTensor& need(const GgufFile& f, const std::string& name) {
    const auto* t = f.find(name);
    if (!t) throw std::runtime_error("model: missing tensor " + name);
    return *t;
}

// GGUF stores Q4_0 nibbles in split halves (weights 0..15 in the low nibbles,
// 16..31 in the high); the __dp4a kernel reads them interleaved. Re-packing on
// load is the only place the two conventions meet.
DevQ4 load_q4(const GgufFile& f, const std::string& name, std::size_t& total) {
    const auto& t = need(f, name);
    if (t.type != GgmlType::Q4_0) throw std::runtime_error("model: " + name + " is not Q4_0");
    if (t.dims.size() != 2) throw std::runtime_error("model: " + name + " is not a matrix");
    DevQ4 m;
    m.cols = static_cast<int>(t.dims[0]);
    m.rows = static_cast<int>(t.dims[1]);
    if (m.cols % QK) throw std::runtime_error("model: " + name + " width not a multiple of 32");
    const std::size_t blocks = std::size_t(m.rows) * (m.cols / QK);
    const auto* src = static_cast<const std::uint8_t*>(f.tensor_data(t));

    std::vector<unsigned char> nib(blocks * 16);
    std::vector<float> scale(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        const std::uint8_t* blk = src + b * 18;
        std::uint16_t d16;
        std::memcpy(&d16, blk, 2);
        scale[b] = fp16_to_float(d16);
        const std::uint8_t* q = blk + 2;
        auto elem = [q](int e) { return e < 16 ? (q[e] & 0xf) : (q[e - 16] >> 4); };
        for (int by = 0; by < 16; ++by)
            nib[b * 16 + by] = static_cast<unsigned char>(elem(2 * by) | (elem(2 * by + 1) << 4));
    }
    m.nib.upload(nib.data(), nib.size());
    m.scale.upload(scale.data(), scale.size());
    total += m.nib.bytes() + m.scale.bytes();
    return m;
}

void cublas_check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS ") + what + " failed: status " +
                                 std::to_string(static_cast<int>(s)));
}

}  // namespace

// ===========================================================================

struct Engine::Impl {
    GgufFile file;
    EngineConfig cfg;
    ModelInfo info;
    std::unique_ptr<Tokenizer> tok;

    std::vector<Layer> layers;
    DevBuf<float> out_norm, out_w;     // output projection, dequantized FP32
    DevBuf<float> cos_t, sin_t;
    const GgufTensor* embd = nullptr;

    // Activations. Sized for the larger of dim and ffn.
    DevBuf<float> x, h, q, k, v, att, o, gate, up, mix, down, logits;
    DevBuf<float> scores;              // [heads][context], WarpPerHead only
    DevBuf<signed char> xq;
    DevBuf<float> xs;
    int* dpos = nullptr;              // device-side position, read by kernels
    std::vector<float> host_embd;

    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool captured = false;

    int pos = 0;
    std::size_t weight_bytes = 0, cache_bytes = 0;

    ~Impl() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
        cudaFree(dpos);
    }

    // ------------------------------------------------------------ launches
    void rmsnorm(const DevBuf<float>& in, const DevBuf<float>& w, DevBuf<float>& out, int n) {
        k_rmsnorm<<<1, 32, 0, stream>>>(in.p, w.p, out.p, n, info.rms_eps);
    }

    void gemv(const DevQ4& m, const DevBuf<float>& in, DevBuf<float>& out) {
        const int threads = 256;   // 8 rows per block
        const int blocks = (m.rows + 7) / 8;
        if (cfg.activations == Activations::Int8) {
            const int groups = m.cols / QK;
            k_quant_act<<<(groups + 255) / 256, 256, 0, stream>>>(in.p, xq.p, xs.p, groups);
            k_gemv_q4_a8<<<blocks, threads, 0, stream>>>(m.nib.p, m.scale.p, xq.p, xs.p, out.p,
                                                          m.rows, m.cols);
        } else {
            k_gemv_q4_f<<<blocks, threads, 0, stream>>>(m.nib.p, m.scale.p, in.p, out.p, m.rows,
                                                         m.cols);
        }
    }

    template <typename Launch>
    void pointwise(int n, Launch&& launch) {
        const int t = 1024;
        launch((n + t - 1) / t, t);
    }

    // Every launch of one token's forward pass up to the final norm. Static
    // arguments only: this is exactly what the CUDA graph captures.
    void run_layers() {
        const int D = info.dim, F = info.ffn, kv_dim = info.kv_heads * HD;
        const float scale = 1.0f / std::sqrt(float(HD));
        for (auto& L : layers) {
            rmsnorm(x, L.attn_norm, h, D);
            gemv(L.q, h, q);
            gemv(L.k, h, k);
            gemv(L.v, h, v);
            k_rope<<<1, 32, 0, stream>>>(q.p, info.heads, dpos, cos_t.p, sin_t.p);
            k_rope<<<1, 32, 0, stream>>>(k.p, info.kv_heads, dpos, cos_t.p, sin_t.p);
            k_kv_store<<<1, 32, 0, stream>>>(k.p, v.p, L.K.p, L.V.p, kv_dim, dpos);
            if (cfg.attention == DecodeAttention::WarpPerHead)
                k_attn_warp_per_head<<<dim3(info.heads, 1), dim3(32, 1), 0, stream>>>(
                    q.p, L.K.p, L.V.p, scores.p, att.p, dpos, info.heads, info.kv_heads, scale,
                    cfg.context);
            else
                k_attn_decode<<<1, 32, 0, stream>>>(q.p, L.K.p, L.V.p, att.p, dpos, info.heads,
                                                    info.kv_heads, scale);
            gemv(L.o, att, o);
            pointwise(D, [&](int b, int t) { k_add<<<b, t, 0, stream>>>(x.p, o.p, D); });

            rmsnorm(x, L.ffn_norm, h, D);
            gemv(L.gate, h, gate);
            gemv(L.up, h, up);
            pointwise(F, [&](int b, int t) { k_silu_mul<<<b, t, 0, stream>>>(gate.p, up.p, mix.p, F); });
            gemv(L.down, mix, down);
            pointwise(D, [&](int b, int t) { k_add<<<b, t, 0, stream>>>(x.p, down.p, D); });
        }
        rmsnorm(x, out_norm, h, D);
    }

    void project_logits() {
        // Row-major W [vocab x dim] read column-major is W^T [dim x vocab];
        // logits = W h = (W^T)^T h.
        const float one = 1.0f, zero = 0.0f;
        cublas_check(cublasSgemv(blas, CUBLAS_OP_T, info.dim, info.vocab, &one, out_w.p,
                                 info.dim, h.p, 1, &zero, logits.p, 1),
                     "logits");
    }

    std::vector<float> step(int token) {
        if (token < 0 || token >= info.vocab) throw std::out_of_range("step: token id out of range");
        if (pos >= cfg.context) throw std::length_error("step: context full");

        // Embedding lookup on the host: one Q4_0 row, 64 blocks, dequantized
        // straight out of the mapped file.
        dequantize_q4_0(static_cast<const std::uint8_t*>(file.tensor_data(*embd)) +
                            std::size_t(token) * (info.dim / QK) * 18,
                        info.dim, host_embd.data());
        CU_CHECK(cudaMemcpyAsync(x.p, host_embd.data(), info.dim * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
        CU_CHECK(cudaMemcpyAsync(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice, stream));
        CU_CHECK(cudaStreamSynchronize(stream));

        if (cfg.cuda_graphs) {
            if (!captured) {
                // One ordinary pass first, so every kernel has been launched and
                // warmed before the capture records it.
                run_layers();
                CU_CHECK_KERNEL();
                CU_CHECK(cudaMemcpyAsync(x.p, host_embd.data(), info.dim * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
                CU_CHECK(cudaStreamSynchronize(stream));
                CU_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                run_layers();
                CU_CHECK(cudaStreamEndCapture(stream, &graph));
                CU_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                captured = true;
                // Re-stage this token's input: whether or not capture executed
                // the launches it recorded, the replay must start from the
                // embedding, not from a residual stream already added to.
                CU_CHECK(cudaMemcpyAsync(x.p, host_embd.data(), info.dim * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
            }
            CU_CHECK(cudaGraphLaunch(exec, stream));
        } else {
            run_layers();
        }
        project_logits();
        CU_CHECK(cudaStreamSynchronize(stream));
        CU_CHECK(cudaGetLastError());

        std::vector<float> out(info.vocab);
        CU_CHECK(cudaMemcpy(out.data(), logits.p, out.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
        ++pos;
        return out;
    }
};

Engine::Engine(const std::string& path, const EngineConfig& cfg) : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.file = GgufFile::open(path);
    I.cfg = cfg;
    auto& f = I.file;

    if (f.meta_string("general.architecture").value_or("") != "llama")
        throw std::runtime_error("model: only the llama architecture is supported");
    auto req = [&](const char* key) {
        auto v = f.meta_int(key);
        if (!v) throw std::runtime_error(std::string("model: missing ") + key);
        return static_cast<int>(*v);
    };
    auto& m = I.info;
    m.layers = req("llama.block_count");
    m.dim = req("llama.embedding_length");
    m.heads = req("llama.attention.head_count");
    m.kv_heads = req("llama.attention.head_count_kv");
    m.ffn = req("llama.feed_forward_length");
    m.trained_context = req("llama.context_length");
    m.head_dim = m.dim / m.heads;
    m.rope_base = float(f.meta_float("llama.rope.freq_base").value_or(10000.0));
    m.rms_eps = float(f.meta_float("llama.attention.layer_norm_rms_epsilon").value_or(1e-5));
    I.tok = std::make_unique<Tokenizer>(f);
    m.vocab = I.tok->vocab_size();

    if (m.head_dim != HD) throw std::runtime_error("model: attention kernel is compiled for head_dim 64");
    if (m.heads > 32) throw std::runtime_error("model: decode attention uses one warp, so at most 32 heads");
    if (m.heads % m.kv_heads) throw std::runtime_error("model: heads must be a multiple of kv_heads");
    if (cfg.context <= 0 || cfg.context > m.trained_context)
        throw std::invalid_argument("engine: context must be in (0, trained context]");

    const int kv_dim = m.kv_heads * HD;
    I.layers.resize(m.layers);
    for (int l = 0; l < m.layers; ++l) {
        auto& L = I.layers[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        auto norm = [&](const std::string& n, DevBuf<float>& dst) {
            const auto& t = need(f, p + n);
            if (t.type != GgmlType::F32) throw std::runtime_error("model: " + n + " is not F32");
            dst.upload(static_cast<const float*>(f.tensor_data(t)), t.num_elements());
            I.weight_bytes += dst.bytes();
        };
        norm("attn_norm.weight", L.attn_norm);
        norm("ffn_norm.weight", L.ffn_norm);
        L.q = load_q4(f, p + "attn_q.weight", I.weight_bytes);
        L.k = load_q4(f, p + "attn_k.weight", I.weight_bytes);
        L.v = load_q4(f, p + "attn_v.weight", I.weight_bytes);
        L.o = load_q4(f, p + "attn_output.weight", I.weight_bytes);
        L.gate = load_q4(f, p + "ffn_gate.weight", I.weight_bytes);
        L.up = load_q4(f, p + "ffn_up.weight", I.weight_bytes);
        L.down = load_q4(f, p + "ffn_down.weight", I.weight_bytes);
        L.K.alloc(std::size_t(cfg.context) * kv_dim);
        L.V.alloc(std::size_t(cfg.context) * kv_dim);
        I.cache_bytes += L.K.bytes() + L.V.bytes();
    }

    const auto& on = need(f, "output_norm.weight");
    I.out_norm.upload(static_cast<const float*>(f.tensor_data(on)), on.num_elements());
    const auto& ow = need(f, "output.weight");
    {
        std::vector<float> w(ow.num_elements());
        if (ow.type == GgmlType::Q6_K) {
            dequantize_q6_k(static_cast<const std::uint8_t*>(f.tensor_data(ow)), w.size(), w.data());
        } else if (ow.type == GgmlType::Q4_0) {
            dequantize_q4_0(static_cast<const std::uint8_t*>(f.tensor_data(ow)), w.size(), w.data());
        } else if (ow.type == GgmlType::F32) {
            std::memcpy(w.data(), f.tensor_data(ow), w.size() * sizeof(float));
        } else {
            throw std::runtime_error("model: unsupported output.weight type");
        }
        I.out_w.upload(w.data(), w.size());
        I.weight_bytes += I.out_w.bytes() + I.out_norm.bytes();
    }
    I.embd = &need(f, "token_embd.weight");
    if (I.embd->type != GgmlType::Q4_0) throw std::runtime_error("model: token_embd is not Q4_0");

    // RoPE tables for every position the cache can hold.
    {
        std::vector<float> c(std::size_t(cfg.context) * (HD / 2)), s(c.size());
        for (int p = 0; p < cfg.context; ++p)
            for (int i = 0; i < HD / 2; ++i) {
                const double theta = p * std::pow(double(m.rope_base), -2.0 * i / HD);
                c[std::size_t(p) * (HD / 2) + i] = float(std::cos(theta));
                s[std::size_t(p) * (HD / 2) + i] = float(std::sin(theta));
            }
        I.cos_t.upload(c.data(), c.size());
        I.sin_t.upload(s.data(), s.size());
    }

    const int big = std::max(m.dim, m.ffn);
    for (auto* b : {&I.x, &I.h, &I.q, &I.att, &I.o, &I.down}) b->alloc(m.dim);
    I.k.alloc(kv_dim);
    I.v.alloc(kv_dim);
    for (auto* b : {&I.gate, &I.up, &I.mix}) b->alloc(m.ffn);
    I.logits.alloc(m.vocab);
    I.scores.alloc(std::size_t(m.heads) * cfg.context);
    I.xq.alloc(big);
    I.xs.alloc(big / QK);
    CU_CHECK(cudaMalloc(&I.dpos, sizeof(int)));
    I.host_embd.resize(m.dim);

    CU_CHECK(cudaStreamCreate(&I.stream));
    cublas_check(cublasCreate(&I.blas), "create");
    cublas_check(cublasSetStream(I.blas, I.stream), "set stream");
}

Engine::~Engine() = default;

const ModelInfo& Engine::info() const { return impl_->info; }
const EngineConfig& Engine::config() const { return impl_->cfg; }
const Tokenizer& Engine::tokenizer() const { return *impl_->tok; }
std::size_t Engine::device_bytes() const { return impl_->weight_bytes + impl_->cache_bytes; }
void Engine::reset() { impl_->pos = 0; }
int Engine::position() const { return impl_->pos; }
std::vector<float> Engine::step(int token) { return impl_->step(token); }

namespace {

int sample(const std::vector<float>& logits, const Sampling& s, std::mt19937_64& rng) {
    if (s.temperature <= 0.0f)
        return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<int> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int k = std::clamp(s.top_k, 1, int(logits.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<double> p(k);
    const double mx = logits[idx[0]] / s.temperature;
    double z = 0.0;
    for (int i = 0; i < k; ++i) z += p[i] = std::exp(logits[idx[i]] / s.temperature - mx);
    double cum = 0.0;
    int keep = k;
    for (int i = 0; i < k; ++i) {
        cum += p[i] / z;
        if (cum >= s.top_p) {
            keep = i + 1;
            break;
        }
    }
    std::discrete_distribution<int> d(p.begin(), p.begin() + keep);
    return idx[d(rng)];
}

}  // namespace

std::string Engine::generate(const std::string& prompt, int max_new_tokens, const Sampling& s,
                             GenerationStats* stats,
                             const std::function<void(const std::string&)>& on_text) {
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    reset();
    const auto ids = impl_->tok->encode(prompt, true);
    if (int(ids.size()) + max_new_tokens > impl_->cfg.context)
        throw std::length_error("generate: prompt plus output exceeds the context length");

    GenerationStats st;
    st.prompt_tokens = int(ids.size());
    auto t0 = clock::now();
    std::vector<float> logits;
    for (int id : ids) logits = step(id);
    auto t1 = clock::now();
    st.prefill_ms = ms(t0, t1);

    std::mt19937_64 rng(s.seed);
    std::vector<int> out;
    std::string text;
    for (int i = 0; i < max_new_tokens; ++i) {
        const int next = sample(logits, s, rng);
        if (next == impl_->tok->eos()) {
            st.stopped_on_eos = true;
            break;
        }
        out.push_back(next);
        const std::string piece = impl_->tok->piece(next);
        text += piece;
        if (on_text) on_text(piece);
        ++st.generated_tokens;
        if (i + 1 < max_new_tokens) logits = step(next);
    }
    st.decode_ms = ms(t1, clock::now());
    if (stats) *stats = st;
    // The first generated piece carries SentencePiece's leading space.
    if (!text.empty() && text[0] == ' ') text.erase(0, 1);
    return text;
}

double Engine::perplexity(const std::string& text, int max_tokens) {
    reset();
    auto ids = impl_->tok->encode(text, true);
    const int n = std::min({int(ids.size()), max_tokens, impl_->cfg.context});
    if (n < 2) throw std::invalid_argument("perplexity: need at least two tokens");
    double nll = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        const auto logits = step(ids[i]);
        const double mx = *std::max_element(logits.begin(), logits.end());
        double z = 0.0;
        for (float l : logits) z += std::exp(double(l) - mx);
        nll += (mx + std::log(z)) - logits[ids[i + 1]];
    }
    return std::exp(nll / (n - 1));
}

std::string chat_prompt(const std::string& user_message) {
    return "<|user|>\n" + user_message + "</s>\n<|assistant|>\n";
}

std::string default_model_path() {
    if (const char* env = std::getenv("CUDA_PORTFOLIO_MODEL")) return env;
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    return std::string(home ? home : ".") + "/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
}

}  // namespace llm
