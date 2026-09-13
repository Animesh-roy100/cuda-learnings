// The production runtime: batched, paged, graph-replayed Llama decoding.

#include "runtime.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "cu/timer.hpp"
#include "gguf.h"
#include "page_allocator.h"
#include "quant.h"

namespace llm {

namespace detail {
struct KvPoolState {
    PageAllocator pages;
    int page_tokens;
    int context;
    std::atomic<int> live{0};
    std::atomic<int> next_id{0};
    KvPoolState(int total, int pt, int ctx) : pages(total), page_tokens(pt), context(ctx) {}
};
}  // namespace detail

namespace {

constexpr int QK = 32;
constexpr int HD = 64;
constexpr unsigned FULL = 0xffffffffu;
constexpr int kRowsPerBlock = 8;   // GEMV: 256 threads = 8 warps = 8 output rows

// ===========================================================================
// Kernels. Every one is batched: grid y selects the sequence. Anything that
// varies per token -- positions, page tables -- is read from device memory,
// so each step's launches are identical and a CUDA graph can replay them.
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

__global__ void k_rmsnorm(const float* __restrict__ x_all, const float* __restrict__ w,
                          float* __restrict__ out_all, int n, float eps) {
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const float* x = x_all + size_t(b) * n;
    float* out = out_all + size_t(b) * n;
    float ss = 0.0f;
    for (int i = lane; i < n; i += 32) ss += x[i] * x[i];
    ss = warp_sum(ss);
    const float total = __shfl_sync(FULL, ss, 0);
    const float inv = rsqrtf(total / float(n) + eps);
    for (int i = lane; i < n; i += 32) out[i] = x[i] * inv * w[i];
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

__global__ void k_gemv_float(const unsigned char* __restrict__ qs, const float* __restrict__ ws_all,
                             const float* __restrict__ x_all, float* __restrict__ y_all, int M, int K) {
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

__global__ void k_rope(float* __restrict__ v_all, int n_heads, const int* __restrict__ positions,
                       const float* __restrict__ cos_t, const float* __restrict__ sin_t) {
    const int h = threadIdx.x;
    if (h >= n_heads) return;
    const int b = blockIdx.y;
    const int pos = positions[b];
    const float* c = cos_t + size_t(pos) * (HD / 2);
    const float* s = sin_t + size_t(pos) * (HD / 2);
    float* x = v_all + (size_t(b) * n_heads + h) * HD;
    for (int i = 0; i < HD / 2; ++i) {
        const float a = x[2 * i], bb = x[2 * i + 1];
        x[2 * i] = a * c[i] - bb * s[i];
        x[2 * i + 1] = a * s[i] + bb * c[i];
    }
}

// Writes this token's key and value into its page slot.
__global__ void k_kv_store(const float* __restrict__ k_all, const float* __restrict__ v_all,
                           float* __restrict__ K, float* __restrict__ V, int kv_dim,
                           const int* __restrict__ positions, const int* __restrict__ table,
                           int max_pages, int page_tokens) {
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const int pos = positions[b];
    const int page = table[size_t(b) * max_pages + pos / page_tokens];
    const size_t base = (size_t(page) * page_tokens + pos % page_tokens) * kv_dim;
    const float* k = k_all + size_t(b) * kv_dim;
    const float* v = v_all + size_t(b) * kv_dim;
    for (int i = lane; i < kv_dim; i += 32) {
        K[base + i] = k[i];
        V[base + i] = v[i];
    }
}

// Paged decode attention, one warp per (sequence, query head). Lane L handles
// the keys congruent to L; each key is found through the sequence's page
// table, so pages need not be contiguous or even private to the sequence.
__global__ void k_attn_paged(const float* __restrict__ q_all, const float* __restrict__ K,
                             const float* __restrict__ V, float* __restrict__ S,
                             float* __restrict__ out_all, const int* __restrict__ positions,
                             const int* __restrict__ table, int n_heads, int n_kv, float scale,
                             int ctx, int max_pages, int page_tokens) {
    __shared__ float Os[32][HD + 1];   // lane-first indexing: width 65
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int lane = threadIdx.x;
    const int len = positions[b] + 1;
    const int kvh = h / (n_heads / n_kv);
    const int kv_dim = n_kv * HD;
    const float* qh = q_all + (size_t(b) * n_heads + h) * HD;
    const int* pt = table + size_t(b) * max_pages;
    float* Sh = S + (size_t(b) * n_heads + h) * ctx;

    float m = -FLT_MAX;
    for (int j = lane; j < len; j += 32) {
        const size_t slot = size_t(pt[j / page_tokens]) * page_tokens + j % page_tokens;
        const float* kr = K + slot * kv_dim + kvh * HD;
        float x = 0.0f;
        for (int c = 0; c < HD; ++c) x += qh[c] * kr[c];
        x *= scale;
        Sh[j] = x;
        m = fmaxf(m, x);
    }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(FULL, m, off));
    const float M = __shfl_sync(FULL, m, 0);

    float z = 0.0f;
    for (int j = lane; j < len; j += 32) z += __expf(Sh[j] - M);
    z = warp_sum(z);
    const float invZ = 1.0f / __shfl_sync(FULL, z, 0);

    for (int c = 0; c < HD; ++c) Os[lane][c] = 0.0f;
    for (int j = lane; j < len; j += 32) {
        const size_t slot = size_t(pt[j / page_tokens]) * page_tokens + j % page_tokens;
        const float* vr = V + slot * kv_dim + kvh * HD;
        const float w = __expf(Sh[j] - M) * invZ;
        for (int c = 0; c < HD; ++c) Os[lane][c] += w * vr[c];
    }
    __syncthreads();

    float* out = out_all + (size_t(b) * n_heads + h) * HD;
    for (int c = lane; c < HD; c += 32) {
        float t = 0.0f;
        for (int l = 0; l < 32; ++l) t += Os[l][c];
        out[c] = t;
    }
}

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
// Host helpers
// ===========================================================================

std::string cuda_name(cudaError_t e) { return cudaGetErrorName(e); }

ErrorContext device_ctx(const std::string& op, int device, cudaError_t e = cudaSuccess) {
    ErrorContext c;
    c.operation = op;
    c.device = device;
    if (e != cudaSuccess) c.cuda_error = cuda_name(e);
    return c;
}

template <typename T>
struct DevBuf {
    T* p = nullptr;
    std::size_t n = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept { std::swap(p, o.p); std::swap(n, o.n); return *this; }
    ~DevBuf() { cudaFree(p); }
    void alloc(std::size_t count, const std::string& what, int device) {
        cudaFree(p);
        p = nullptr;
        n = 0;
        const cudaError_t e = cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T));
        if (e != cudaSuccess) {
            cudaGetLastError();
            auto c = device_ctx("load", device, e);
            c.tensor = what;
            c.shape = {count};
            throw Error(ErrorKind::OutOfMemory, "device allocation failed", c);
        }
        n = count;
    }
    std::size_t bytes() const { return n * sizeof(T); }
};

template <typename T>
struct PinnedBuf {
    T* p = nullptr;
    PinnedBuf() = default;
    PinnedBuf(const PinnedBuf&) = delete;
    PinnedBuf& operator=(const PinnedBuf&) = delete;
    ~PinnedBuf() { if (p) cudaFreeHost(p); }
    void alloc(std::size_t count, int device) {
        const cudaError_t e = cudaMallocHost(&p, std::max<std::size_t>(count, 1) * sizeof(T));
        if (e != cudaSuccess) {
            cudaGetLastError();
            throw Error(ErrorKind::OutOfMemory, "pinned host allocation failed",
                        device_ctx("load", device, e));
        }
    }
};

struct DevQ4 {
    DevBuf<unsigned char> nib;
    DevBuf<float> scale;
    int rows = 0, cols = 0;
};

struct Layer {
    DevBuf<float> attn_norm, ffn_norm;
    DevQ4 q, k, v, o, gate, up, down;
    DevBuf<float> K, V;   // [pages][page_tokens][kv_dim]
};

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};

}  // namespace

// ===========================================================================

GgufModel::~GgufModel() = default;

DeviceInfo query_device_info(int ordinal) {
    DeviceInfo d;
    d.ordinal = ordinal;
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || ordinal < 0 || ordinal >= count) {
        cudaGetLastError();
        throw Error(ErrorKind::Device, "no CUDA device " + std::to_string(ordinal),
                    device_ctx("query device", ordinal, e));
    }
    e = cudaSetDevice(ordinal);
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw Error(ErrorKind::Device, "cannot select device", device_ctx("query device", ordinal, e));
    }
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, ordinal);
    d.name = p.name;
    d.compute_capability = p.major * 10 + p.minor;
    cudaMemGetInfo(&d.free_bytes, &d.total_bytes);
    cudaDriverGetVersion(&d.driver_version);
    d.driver_release = "unknown";
    if (nvmlInit_v2() == NVML_SUCCESS) {
        char release[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
        if (nvmlSystemGetDriverVersion(release, sizeof release) == NVML_SUCCESS) d.driver_release = release;
        nvmlShutdown();
    }
    cudaRuntimeGetVersion(&d.runtime_version);
    d.dp4a = d.compute_capability >= 61;
    return d;
}

MemoryPlan Runtime::plan_memory(const GgufModel& model, const RuntimeOptions& o) {
    const auto& c = model.config();
    const std::uint64_t B = o.max_batch, D = c.hidden_size, F = c.intermediate_size,
                        V = c.vocab_size, KV = c.kv_dim(), H = c.attention_heads, ctx = o.context;
    const std::uint64_t max_pages = (ctx + o.page_tokens - 1) / o.page_tokens;
    const std::uint64_t pages = o.kv_pages > 0 ? std::uint64_t(o.kv_pages) : B * max_pages;
    const std::uint64_t wide = std::max(D, F);

    MemoryPlan m;
    m.weights = weight_device_bytes(model.file(), c);
    m.kv_cache = std::uint64_t(c.layers) * 2 * pages * o.page_tokens * KV * sizeof(float);
    m.workspace = B * (7 * D + 2 * KV + 3 * F + V) * sizeof(float)   // activations and logits
                  + B * wide * (1 + sizeof(float) / double(QK))       // int8 activations + scales
                  + B * H * ctx * sizeof(float)                       // attention scores
                  + B * (max_pages + 1) * sizeof(int)                 // page tables, positions
                  + ctx * HD * sizeof(float)                          // RoPE tables
                  + (std::uint64_t(64) << 20);                        // cuBLAS workspace allowance
    return m;
}

struct Runtime::Impl {
    std::atomic<bool> stepping{false};   // step() is single-threaded; detect violations
    DeviceInfo dev;
    Activations act = Activations::Float;
    MemoryPlan plan;
    std::shared_ptr<detail::KvPoolState> pool;
    int max_pages = 0;

    std::vector<Layer> layers;
    DevBuf<float> out_norm, out_w, cos_t, sin_t;

    DevBuf<float> x, h, q, k, v, att, o, gate, up, mix, down, logits, scores;
    DevBuf<signed char> xq;
    DevBuf<float> xs;
    DevBuf<int> positions, table;

    PinnedBuf<float> h_embd, h_logits;
    PinnedBuf<int> h_positions, h_table;

    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::map<int, Graph> graphs;

    StageTimes times;
    std::vector<std::unique_ptr<cu::EventTimer>> timers;

    ~Impl() {
        for (auto& [b, g] : graphs) {
            if (g.exec) cudaGraphExecDestroy(g.exec);
            if (g.graph) cudaGraphDestroy(g.graph);
        }
        if (blas) cublasDestroy(blas);
        if (stream) cudaStreamDestroy(stream);
    }
};

namespace {

DevQ4 load_q4(const GgufFile& f, const std::string& name, int device) {
    const auto& t = *f.find(name);   // presence, type and shape validated at open
    DevQ4 m;
    m.cols = int(t.dims[0]);
    m.rows = int(t.dims[1]);
    const std::size_t blocks = std::size_t(m.rows) * (m.cols / QK);
    const auto* src = static_cast<const std::uint8_t*>(f.tensor_data(t));
    std::vector<unsigned char> nib(blocks * 16);
    std::vector<float> scale(blocks);
    for (std::size_t bl = 0; bl < blocks; ++bl) {
        const std::uint8_t* blk = src + bl * 18;
        std::uint16_t d16;
        std::memcpy(&d16, blk, 2);
        scale[bl] = fp16_to_float(d16);
        const std::uint8_t* qb = blk + 2;
        auto elem = [qb](int e) { return e < 16 ? (qb[e] & 0xf) : (qb[e - 16] >> 4); };
        for (int by = 0; by < 16; ++by)
            nib[bl * 16 + by] = static_cast<unsigned char>(elem(2 * by) | (elem(2 * by + 1) << 4));
    }
    m.nib.alloc(nib.size(), name, device);
    m.scale.alloc(scale.size(), name, device);
    cudaMemcpy(m.nib.p, nib.data(), nib.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m.scale.p, scale.data(), scale.size() * sizeof(float), cudaMemcpyHostToDevice);
    return m;
}

void upload(DevBuf<float>& dst, const float* src, std::size_t n, const std::string& what, int device) {
    dst.alloc(n, what, device);
    const cudaError_t e = cudaMemcpy(dst.p, src, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaGetLastError();
        auto c = device_ctx("load", device, e);
        c.tensor = what;
        throw Error(ErrorKind::Device, "upload failed", c);
    }
}

void blas_check(cublasStatus_t s, const char* what, int device) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw Error(ErrorKind::Device,
                    std::string("cuBLAS ") + what + " failed with status " + std::to_string(int(s)),
                    device_ctx("step", device));
}

}  // namespace

Runtime::Runtime(std::shared_ptr<const GgufModel> model, const RuntimeOptions& options)
    : model_(std::move(model)), options_(options), impl_(std::make_unique<Impl>()) {
    if (!model_) throw Error(ErrorKind::InvalidArgument, "Runtime: null model");
    auto& I = *impl_;
    const auto& c = model_->config();
    const auto& o = options_;
    ErrorContext lc;
    lc.operation = "load";
    lc.model = model_->path();

    // ---- options
    std::vector<std::string> bad;
    if (o.context <= 0 || o.context > c.context_length)
        bad.push_back("context must be in (0, " + std::to_string(c.context_length) + "]");
    if (o.max_batch <= 0 || o.max_batch > 64) bad.push_back("max_batch must be in [1, 64]");
    if (o.page_tokens <= 0 || o.page_tokens > o.context) bad.push_back("page_tokens must be in [1, context]");
    if (o.kv_pages < 0) bad.push_back("kv_pages must be >= 0");
    if (!(o.vram_headroom >= 0.0 && o.vram_headroom < 1.0)) bad.push_back("vram_headroom must be in [0, 1)");
    if (!bad.empty()) {
        std::string msg = "invalid runtime options:";
        for (auto& b : bad) msg += " " + b + ";";
        throw Error(ErrorKind::InvalidArgument, msg, lc);
    }

    // ---- device and capabilities
    I.dev = query_device_info(o.device);
    lc.device = o.device;
    if (o.activations == Activations::Int8 && !I.dev.dp4a)
        throw Error(ErrorKind::Unsupported,
                    "int8 activations need __dp4a (compute capability 6.1+); this device is " +
                        std::to_string(I.dev.compute_capability),
                    lc);
    I.act = o.activations == Activations::Auto ? (I.dev.dp4a ? Activations::Int8 : Activations::Float)
                                               : o.activations;

    // ---- memory budget, before any allocation
    I.plan = plan_memory(*model_, o);
    I.plan.free_at_load = I.dev.free_bytes;
    const double headroom = o.vram_headroom * double(I.dev.total_bytes);
    if (double(I.plan.total()) + headroom > double(I.dev.free_bytes)) {
        auto mb = [](double b) { return std::to_string(std::uint64_t(b / (1 << 20))); };
        throw Error(ErrorKind::OutOfMemory,
                    "memory plan " + mb(double(I.plan.total())) + " MB (weights " + mb(double(I.plan.weights)) +
                        ", KV cache " + mb(double(I.plan.kv_cache)) + ", workspace " +
                        mb(double(I.plan.workspace)) + ") plus " + mb(headroom) + " MB headroom exceeds " +
                        mb(double(I.dev.free_bytes)) + " MB free; reduce context, max_batch or kv_pages",
                    lc);
    }

    I.max_pages = (o.context + o.page_tokens - 1) / o.page_tokens;
    const int total_pages = o.kv_pages > 0 ? o.kv_pages : o.max_batch * I.max_pages;
    I.pool = std::make_shared<detail::KvPoolState>(total_pages, o.page_tokens, o.context);

    // ---- weights (every Impl member is RAII: a throw below frees what exists)
    const auto& f = model_->file();
    const int D = c.hidden_size, F = c.intermediate_size, KV = c.kv_dim(), V = c.vocab_size,
              B = o.max_batch, H = c.attention_heads;
    I.layers.resize(c.layers);
    for (int l = 0; l < c.layers; ++l) {
        auto& L = I.layers[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        auto norm = [&](const std::string& n, DevBuf<float>& dst) {
            const auto& t = *f.find(p + n);
            upload(dst, static_cast<const float*>(f.tensor_data(t)), t.num_elements(), p + n, o.device);
        };
        norm("attn_norm.weight", L.attn_norm);
        norm("ffn_norm.weight", L.ffn_norm);
        L.q = load_q4(f, p + "attn_q.weight", o.device);
        L.k = load_q4(f, p + "attn_k.weight", o.device);
        L.v = load_q4(f, p + "attn_v.weight", o.device);
        L.o = load_q4(f, p + "attn_output.weight", o.device);
        L.gate = load_q4(f, p + "ffn_gate.weight", o.device);
        L.up = load_q4(f, p + "ffn_up.weight", o.device);
        L.down = load_q4(f, p + "ffn_down.weight", o.device);
        const std::size_t kv_floats = std::size_t(total_pages) * o.page_tokens * KV;
        L.K.alloc(kv_floats, p + "kv_cache.K", o.device);
        L.V.alloc(kv_floats, p + "kv_cache.V", o.device);
    }
    {
        const auto& on = *f.find("output_norm.weight");
        upload(I.out_norm, static_cast<const float*>(f.tensor_data(on)), on.num_elements(),
               "output_norm.weight", o.device);
        const auto& ow = *f.find("output.weight");
        std::vector<float> w(ow.num_elements());
        const auto* src = static_cast<const std::uint8_t*>(f.tensor_data(ow));
        if (ow.type == GgmlType::Q6_K) dequantize_q6_k(src, w.size(), w.data());
        else if (ow.type == GgmlType::Q4_0) dequantize_q4_0(src, w.size(), w.data());
        else std::memcpy(w.data(), src, w.size() * sizeof(float));
        upload(I.out_w, w.data(), w.size(), "output.weight", o.device);
    }
    {
        std::vector<float> cs(std::size_t(o.context) * (HD / 2)), sn(cs.size());
        for (int pos = 0; pos < o.context; ++pos)
            for (int i = 0; i < HD / 2; ++i) {
                const double theta = pos * std::pow(double(c.rope_theta), -2.0 * i / HD);
                cs[std::size_t(pos) * (HD / 2) + i] = float(std::cos(theta));
                sn[std::size_t(pos) * (HD / 2) + i] = float(std::sin(theta));
            }
        upload(I.cos_t, cs.data(), cs.size(), "rope.cos", o.device);
        upload(I.sin_t, sn.data(), sn.size(), "rope.sin", o.device);
    }

    // ---- workspace for the largest batch
    for (auto* buf : {&I.x, &I.h, &I.q, &I.att, &I.o, &I.down}) buf->alloc(std::size_t(B) * D, "workspace", o.device);
    I.k.alloc(std::size_t(B) * KV, "workspace", o.device);
    I.v.alloc(std::size_t(B) * KV, "workspace", o.device);
    for (auto* buf : {&I.gate, &I.up, &I.mix}) buf->alloc(std::size_t(B) * F, "workspace", o.device);
    I.logits.alloc(std::size_t(B) * V, "workspace.logits", o.device);
    I.scores.alloc(std::size_t(B) * H * o.context, "workspace.scores", o.device);
    I.xq.alloc(std::size_t(B) * std::max(D, F), "workspace", o.device);
    I.xs.alloc(std::size_t(B) * std::max(D, F) / QK, "workspace", o.device);
    I.positions.alloc(B, "workspace.positions", o.device);
    I.table.alloc(std::size_t(B) * I.max_pages, "workspace.page_table", o.device);
    I.h_embd.alloc(std::size_t(B) * D, o.device);
    I.h_logits.alloc(std::size_t(B) * V, o.device);
    I.h_positions.alloc(B, o.device);
    I.h_table.alloc(std::size_t(B) * I.max_pages, o.device);

    cudaError_t e = cudaStreamCreate(&I.stream);
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw Error(ErrorKind::Device, "stream creation failed", device_ctx("load", o.device, e));
    }
    blas_check(cublasCreate(&I.blas), "create", o.device);
    blas_check(cublasSetStream(I.blas, I.stream), "set stream", o.device);
}

Runtime::~Runtime() = default;

Activations Runtime::activations() const { return impl_->act; }
const DeviceInfo& Runtime::device() const { return impl_->dev; }
const MemoryPlan& Runtime::memory() const { return impl_->plan; }
StageTimes Runtime::stage_times() const { return impl_->times; }
void Runtime::reset_stage_times() { impl_->times = {}; }

KvStats Runtime::kv_stats() const {
    KvStats s;
    s.total_pages = impl_->pool->pages.total();
    s.pages_in_use = impl_->pool->pages.in_use();
    s.page_tokens = impl_->pool->page_tokens;
    s.live_sequences = impl_->pool->live.load();
    return s;
}

std::unique_ptr<Sequence> Runtime::new_sequence() {
    auto* s = new Sequence(impl_->pool, impl_->pool->next_id++);
    ++impl_->pool->live;
    return std::unique_ptr<Sequence>(s);
}

// ---------------------------------------------------------------- Sequence

Sequence::~Sequence() {
    for (int p : pages_) pool_->pages.release(p);
    --pool_->live;
}

int Sequence::capacity() const { return pool_->context; }

std::unique_ptr<Sequence> Sequence::fork() const {
    auto* s = new Sequence(pool_, pool_->next_id++);
    for (int p : pages_) pool_->pages.retain(p);
    s->pages_ = pages_;
    s->position_ = position_;
    ++pool_->live;
    return std::unique_ptr<Sequence>(s);
}

void Sequence::truncate(int new_position) {
    if (new_position < 0 || new_position > position_)
        throw Error(ErrorKind::InvalidArgument, "truncate: position must be in [0, " +
                                                    std::to_string(position_) + "]");
    const int keep = (new_position + pool_->page_tokens - 1) / pool_->page_tokens;
    while (int(pages_.size()) > keep) {
        pool_->pages.release(pages_.back());
        pages_.pop_back();
    }
    position_ = new_position;
}

// ------------------------------------------------------------------- step

std::vector<std::vector<float>> Runtime::step(const std::vector<Sequence*>& seqs,
                                              const std::vector<int>& tokens) {
    auto& I = *impl_;
    const auto& c = model_->config();
    const auto& o = options_;
    const int B = static_cast<int>(seqs.size());
    const int D = c.hidden_size, F = c.intermediate_size, KV = c.kv_dim(), V = c.vocab_size,
              H = c.attention_heads, PT = o.page_tokens;
    ErrorContext sc;
    sc.operation = "step";
    sc.model = model_->path();
    sc.device = o.device;

    // One thread drives a runtime: steps share one workspace and one stream.
    // A second concurrent caller is refused, not allowed to corrupt the first.
    if (I.stepping.exchange(true))
        throw Error(ErrorKind::InvalidArgument, "step() called concurrently on one Runtime", sc);
    struct Release {
        std::atomic<bool>& flag;
        ~Release() { flag = false; }
    } release{I.stepping};

    // ---- validation: nothing below may change state until all of it passes
    if (B == 0) throw Error(ErrorKind::InvalidArgument, "empty batch", sc);
    if (B > o.max_batch)
        throw Error(ErrorKind::InvalidArgument,
                    "batch of " + std::to_string(B) + " exceeds max_batch " + std::to_string(o.max_batch), sc);
    if (int(tokens.size()) != B)
        throw Error(ErrorKind::InvalidArgument, "one token per sequence required", sc);
    std::set<const Sequence*> unique;
    for (int b = 0; b < B; ++b) {
        auto ctx = sc;
        if (!seqs[b]) throw Error(ErrorKind::InvalidArgument, "null sequence", ctx);
        ctx.sequence = seqs[b]->id();
        ctx.position = seqs[b]->position();
        if (seqs[b]->pool_ != I.pool)
            throw Error(ErrorKind::InvalidArgument, "sequence belongs to another runtime", ctx);
        if (!unique.insert(seqs[b]).second)
            throw Error(ErrorKind::InvalidArgument, "sequence appears twice in one batch", ctx);
        if (tokens[b] < 0 || tokens[b] >= V)
            throw Error(ErrorKind::InvalidArgument, "token id " + std::to_string(tokens[b]) + " out of range", ctx);
        if (seqs[b]->position() >= o.context)
            throw Error(ErrorKind::ContextFull, "sequence is at its context length", ctx);
    }

    // ---- pages: allocate everything this step needs, all or nothing
    std::vector<char> needs_new(B, 0), needs_cow(B, 0);
    int wanted = 0;
    for (int b = 0; b < B; ++b) {
        const auto* s = seqs[b];
        if (s->position() % PT == 0) needs_new[b] = 1;
        else if (I.pool->pages.refcount(s->pages_.back()) > 1) needs_cow[b] = 1;
        wanted += needs_new[b] + needs_cow[b];
    }
    auto fresh = I.pool->pages.allocate_many(wanted);
    if (!fresh) {
        auto ctx = sc;
        throw Error(ErrorKind::OutOfMemory,
                    "KV cache exhausted: step needs " + std::to_string(wanted) + " page(s), " +
                        std::to_string(I.pool->pages.free_count()) + " free of " +
                        std::to_string(I.pool->pages.total()),
                    ctx);
    }

    // Apply, remembering how to undo it if the device fails mid-step.
    std::vector<std::pair<Sequence*, int>> appended;          // (seq, page) pushed
    std::vector<std::tuple<Sequence*, int, int>> replaced;     // (seq, old, new) copy-on-write
    auto rollback = [&]() {
        for (auto& [s, page] : appended) {
            s->pages_.pop_back();
            I.pool->pages.release(page);
        }
        for (auto& [s, old_page, new_page] : replaced) {
            s->pages_.back() = old_page;
            I.pool->pages.retain(old_page);
            I.pool->pages.release(new_page);
        }
    };
    std::size_t next = 0;
    try {
        for (int b = 0; b < B; ++b) {
            auto* s = seqs[b];
            if (needs_new[b]) {
                const int page = (*fresh)[next++];
                s->pages_.push_back(page);
                appended.emplace_back(s, page);
            } else if (needs_cow[b]) {
                const int page = (*fresh)[next++];
                const int old_page = s->pages_.back();
                const std::size_t bytes = std::size_t(PT) * KV * sizeof(float);
                for (auto& L : I.layers) {
                    for (auto* buf : {L.K.p, L.V.p}) {
                        const cudaError_t e = cudaMemcpy(buf + std::size_t(page) * PT * KV,
                                                         buf + std::size_t(old_page) * PT * KV, bytes,
                                                         cudaMemcpyDeviceToDevice);
                        if (e != cudaSuccess) {
                            cudaGetLastError();
                            auto ctx = device_ctx("step (copy-on-write)", o.device, e);
                            ctx.sequence = s->id();
                            // Release the page not yet recorded before unwinding.
                            I.pool->pages.release(page);
                            ++next;
                            throw Error(ErrorKind::Device, "page copy failed", ctx);
                        }
                    }
                }
                s->pages_.back() = page;
                I.pool->pages.release(old_page);
                replaced.emplace_back(s, old_page, page);
            }
        }

        // ---- host-side inputs
        const auto* embd_t = model_->file().find("token_embd.weight");
        const auto* embd = static_cast<const std::uint8_t*>(model_->file().tensor_data(*embd_t));
        for (int b = 0; b < B; ++b) {
            dequantize_q4_0(embd + std::size_t(tokens[b]) * (D / QK) * 18, D, I.h_embd.p + std::size_t(b) * D);
            I.h_positions.p[b] = seqs[b]->position();
            const auto& pages = seqs[b]->pages_;
            for (int i = 0; i < I.max_pages; ++i)
                I.h_table.p[std::size_t(b) * I.max_pages + i] = i < int(pages.size()) ? pages[i] : 0;
        }

        const bool profile = o.profile_stages;
        auto& T = I.times;
        auto stage = [&](double& acc, auto&& body) {
            if (!profile) {
                body();
                return;
            }
            cu::EventTimer t;
            t.start(I.stream);
            body();
            acc += t.stop(I.stream);
        };

        auto upload_inputs = [&]() {
            cudaMemcpyAsync(I.x.p, I.h_embd.p, std::size_t(B) * D * sizeof(float), cudaMemcpyHostToDevice, I.stream);
            cudaMemcpyAsync(I.positions.p, I.h_positions.p, std::size_t(B) * sizeof(int), cudaMemcpyHostToDevice, I.stream);
            cudaMemcpyAsync(I.table.p, I.h_table.p, std::size_t(B) * I.max_pages * sizeof(int),
                            cudaMemcpyHostToDevice, I.stream);
        };
        stage(T.embed_upload, upload_inputs);

        const bool int8 = I.act == Activations::Int8;
        const float scale = 1.0f / std::sqrt(float(HD));
        auto s = I.stream;
        auto gemv = [&](const DevQ4& m, const DevBuf<float>& in, DevBuf<float>& out) {
            const dim3 grid((m.rows + kRowsPerBlock - 1) / kRowsPerBlock, B);
            if (int8) {
                const int groups = m.cols / QK;
                k_quant_act<<<dim3((groups + 255) / 256, B), dim3(256, 1), 0, s>>>(in.p, I.xq.p, I.xs.p, groups);
                k_gemv_int8<<<grid, dim3(32 * kRowsPerBlock, 1), 0, s>>>(m.nib.p, m.scale.p, I.xq.p, I.xs.p,
                                                                        out.p, m.rows, m.cols);
            } else {
                k_gemv_float<<<grid, dim3(32 * kRowsPerBlock, 1), 0, s>>>(m.nib.p, m.scale.p, in.p, out.p,
                                                                         m.rows, m.cols);
            }
        };
        auto pointwise = [&](int n) { return dim3((n + 1023) / 1024, 1); };

        auto run_layers = [&]() {
            for (auto& L : I.layers) {
                stage(T.norms, [&] { k_rmsnorm<<<dim3(1, B), dim3(32, 1), 0, s>>>(I.x.p, L.attn_norm.p, I.h.p, D, c.rms_epsilon); });
                stage(T.projections, [&] {
                    gemv(L.q, I.h, I.q);
                    gemv(L.k, I.h, I.k);
                    gemv(L.v, I.h, I.v);
                });
                stage(T.rope_and_cache, [&] {
                    k_rope<<<dim3(1, B), dim3(32, 1), 0, s>>>(I.q.p, H, I.positions.p, I.cos_t.p, I.sin_t.p);
                    k_rope<<<dim3(1, B), dim3(32, 1), 0, s>>>(I.k.p, c.kv_heads, I.positions.p, I.cos_t.p, I.sin_t.p);
                    k_kv_store<<<dim3(1, B), dim3(32, 1), 0, s>>>(I.k.p, I.v.p, L.K.p, L.V.p, KV, I.positions.p,
                                                                  I.table.p, I.max_pages, PT);
                });
                stage(T.attention, [&] {
                    k_attn_paged<<<dim3(H, B), dim3(32, 1), 0, s>>>(I.q.p, L.K.p, L.V.p, I.scores.p, I.att.p,
                                                                    I.positions.p, I.table.p, H, c.kv_heads,
                                                                    scale, o.context, I.max_pages, PT);
                });
                stage(T.projections, [&] { gemv(L.o, I.att, I.o); });
                stage(T.activation, [&] { k_add<<<pointwise(B * D), 1024, 0, s>>>(I.x.p, I.o.p, B * D); });
                stage(T.norms, [&] { k_rmsnorm<<<dim3(1, B), dim3(32, 1), 0, s>>>(I.x.p, L.ffn_norm.p, I.h.p, D, c.rms_epsilon); });
                stage(T.projections, [&] {
                    gemv(L.gate, I.h, I.gate);
                    gemv(L.up, I.h, I.up);
                });
                stage(T.activation, [&] { k_silu_mul<<<pointwise(B * F), 1024, 0, s>>>(I.gate.p, I.up.p, I.mix.p, B * F); });
                stage(T.projections, [&] { gemv(L.down, I.mix, I.down); });
                stage(T.activation, [&] { k_add<<<pointwise(B * D), 1024, 0, s>>>(I.x.p, I.down.p, B * D); });
            }
            stage(T.norms, [&] { k_rmsnorm<<<dim3(1, B), dim3(32, 1), 0, s>>>(I.x.p, I.out_norm.p, I.h.p, D, c.rms_epsilon); });
        };

        if (o.cuda_graphs && !profile) {
            auto it = I.graphs.find(B);
            if (it == I.graphs.end()) {
                // Warm every kernel once at this batch size, then capture. The
                // replay always starts from freshly uploaded inputs, so it does
                // not matter whether capture itself executed the launches.
                run_layers();
                cudaStreamSynchronize(s);
                Graph g;
                cudaError_t e = cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
                if (e == cudaSuccess) {
                    run_layers();
                    e = cudaStreamEndCapture(s, &g.graph);
                }
                if (e == cudaSuccess) e = cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0);
                if (e != cudaSuccess) {
                    cudaGetLastError();
                    if (g.graph) cudaGraphDestroy(g.graph);
                    throw Error(ErrorKind::Device, "CUDA graph capture failed", device_ctx("step", o.device, e));
                }
                it = I.graphs.emplace(B, g).first;
                upload_inputs();
            }
            const cudaError_t e = cudaGraphLaunch(it->second.exec, s);
            if (e != cudaSuccess) {
                cudaGetLastError();
                throw Error(ErrorKind::Device, "graph launch failed", device_ctx("step", o.device, e));
            }
        } else {
            run_layers();
        }

        stage(T.logits, [&] {
            // Row-major logits [B][V] = h W^T; column-major that is W h^T.
            const float one = 1.0f, zero = 0.0f;
            blas_check(cublasSgemm(I.blas, CUBLAS_OP_T, CUBLAS_OP_N, V, B, D, &one, I.out_w.p, D, I.h.p, D,
                                   &zero, I.logits.p, V),
                       "logits", o.device);
        });
        stage(T.download, [&] {
            cudaMemcpyAsync(I.h_logits.p, I.logits.p, std::size_t(B) * V * sizeof(float), cudaMemcpyDeviceToHost, s);
        });
        const cudaError_t sync = cudaStreamSynchronize(s);
        const cudaError_t launch = cudaGetLastError();
        if (sync != cudaSuccess || launch != cudaSuccess) {
            auto ctx = device_ctx("step", o.device, sync != cudaSuccess ? sync : launch);
            throw Error(ErrorKind::Device, "kernel execution failed", ctx);
        }
        if (profile) ++T.steps;
    } catch (...) {
        // Nothing has advanced yet: undo the page changes and propagate.
        for (std::size_t i = next; i < fresh->size(); ++i) I.pool->pages.release((*fresh)[i]);
        rollback();
        throw;
    }

    std::vector<std::vector<float>> out(B);
    for (int b = 0; b < B; ++b) {
        out[b].assign(I.h_logits.p + std::size_t(b) * V, I.h_logits.p + std::size_t(b + 1) * V);
        ++seqs[b]->position_;
    }
    return out;
}

}  // namespace llm
