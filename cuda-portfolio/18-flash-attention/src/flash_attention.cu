// Scaled dot-product attention: materialized, fused, and fused-tiled.

#include "flash_attention.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace fa {
namespace {

constexpr int D = kFusedHeadDim;

// ---------------------------------------------------------------------------
// The online softmax update, shared by both fused kernels.
//
// After j keys, a query holds
//   m = max logit so far
//   s = sum_k exp(x_k - m)
//   o = sum_k exp(x_k - m) / s * v_k      -- the softmax-weighted sum so far
// and one more key with logit x and value v updates all three exactly:
//   m' = max(m, x)
//   s' = s * exp(m - m') + exp(x - m')
//   o' = o * (s * exp(m - m')) / s'  +  v * exp(x - m') / s'
//
// Every exponent is <= 0, so nothing overflows in FP32 however large the
// logits get -- the same reason a materialized softmax subtracts the row max.
// It is branch-free on purpose: a branch on x > m would split the warp's lanes
// on data, key after key.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void online_weights(float x, float& m, float& s, float& a, float& b) {
    const float mn = fmaxf(m, x);
    const float e_old = __expf(m - mn);
    const float e_new = __expf(x - mn);
    const float sn = s * e_old + e_new;
    a = s * e_old / sn;
    b = e_new / sn;
    s = sn;
    m = mn;
}

// ---------------------------------------------------------------------------
// FusedGlobal. One warp per 32 queries of one head; keys and values read from
// global. Every lane reads the SAME key row at each step, so the warp's 32 key
// loads coalesce into one 256-byte transaction -- memory-bound, but not strided.
//
// Q and the running output are staged into shared memory. Both are indexed
// [lane][c] and declared with width 65: at width 64, element c of every lane
// would land in bank c % 32 and the whole warp would serialize on it.
// ---------------------------------------------------------------------------
__global__ void k_fused_global(const float* __restrict__ Q, const float* __restrict__ K,
                               const float* __restrict__ V, float* __restrict__ O,
                               int N, int causal, float scale, int head) {
    __shared__ float Qs[32][D + 1];
    __shared__ float Os[32][D + 1];

    const int lane = threadIdx.x;
    const int i = blockIdx.x * 32 + lane;
    const bool live = i < N;
    const std::size_t hb = std::size_t(head) * N;

    for (int c = 0; c < D; ++c) {
        Qs[lane][c] = live ? Q[(hb + i) * D + c] : 0.0f;
        Os[lane][c] = 0.0f;
    }
    __syncthreads();

    float m = -FLT_MAX, s = 0.0f;
    // A lane past the end of the sequence still runs the loop shape (with no
    // keys), so it reaches every barrier its warp does.
    const int last = !live ? 0 : (causal ? i + 1 : N);
    for (int j = 0; j < last; ++j) {
        const float* k = K + (hb + j) * D;
        const float* v = V + (hb + j) * D;
        float x = 0.0f;
        for (int c = 0; c < D; ++c) x += Qs[lane][c] * k[c];
        float a, b;
        online_weights(x * scale, m, s, a, b);
        for (int c = 0; c < D; ++c) Os[lane][c] = Os[lane][c] * a + v[c] * b;
    }
    __syncthreads();
    if (live)
        for (int c = 0; c < D; ++c) O[(hb + i) * D + c] = Os[lane][c];
}

// ---------------------------------------------------------------------------
// FusedTiled. As above, but TILE keys and values at a time are staged into
// shared memory first, so the inner loop reads nothing from DRAM.
//
// The key/value tiles are indexed [r][c] with every lane reading the same row,
// so width D needs no padding. Staging them, each lane copies the columns
// congruent to its lane number -- two per row at D = 64 -- so each row arrives
// as one contiguous run and no two lanes write the same bank.
//
// Shared memory per block is 2 x 32 x 65 floats for Q and O plus
// 2 x TILE x 64 floats for the tile: 16.6 KB + TILE x 512 B. The widest tile
// that fits the 48 KB per-kernel limit (measured in 17-tc-gemm) is 61; 48 is
// the largest compiled.
// ---------------------------------------------------------------------------
template <int TILE>
__global__ void k_fused_tiled(const float* __restrict__ Q, const float* __restrict__ K,
                              const float* __restrict__ V, float* __restrict__ O,
                              int N, int causal, float scale, int head) {
    __shared__ float Qs[32][D + 1];
    __shared__ float Os[32][D + 1];
    __shared__ float Ks[TILE][D];
    __shared__ float Vs[TILE][D];

    const int lane = threadIdx.x;
    const int i = blockIdx.x * 32 + lane;
    const bool live = i < N;
    const std::size_t hb = std::size_t(head) * N;

    for (int c = 0; c < D; ++c) {
        Qs[lane][c] = live ? Q[(hb + i) * D + c] : 0.0f;
        Os[lane][c] = 0.0f;
    }
    __syncthreads();

    float m = -FLT_MAX, s = 0.0f;
    for (int j0 = 0; j0 < N; j0 += TILE) {
        const int t = min(TILE, N - j0);
        for (int r = 0; r < t; ++r) {
            const float* k = K + (hb + j0 + r) * D;
            const float* v = V + (hb + j0 + r) * D;
            for (int c = lane; c < D; c += 32) {
                Ks[r][c] = k[c];
                Vs[r][c] = v[c];
            }
        }
        __syncthreads();

        // Keys past i are masked for a causal query; the lane still takes
        // part in every staging round and barrier.
        const int rmax = !live ? 0 : (causal ? max(0, min(t, i - j0 + 1)) : t);
        for (int r = 0; r < rmax; ++r) {
            float x = 0.0f;
            for (int c = 0; c < D; ++c) x += Qs[lane][c] * Ks[r][c];
            float a, b;
            online_weights(x * scale, m, s, a, b);
            for (int c = 0; c < D; ++c) Os[lane][c] = Os[lane][c] * a + Vs[r][c] * b;
        }
        __syncthreads();
    }
    if (live)
        for (int c = 0; c < D; ++c) O[(hb + i) * D + c] = Os[lane][c];
}

#define FA_FOR_EACH_TILE(X) X(8) X(16) X(24) X(32) X(48)

// ---------------------------------------------------------------------------
// Naive: softmax each row of the materialized score matrix in place.
// Three passes over N floats per row -- max, normalizer, write -- for every
// row, all of it a round trip to DRAM.
// ---------------------------------------------------------------------------
__global__ void k_softmax_rows(float* __restrict__ S, int N, int causal, int head) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float* row = S + std::size_t(head) * N * N + std::size_t(i) * N;
    const int len = causal ? i + 1 : N;

    float m = -FLT_MAX;
    for (int j = 0; j < len; ++j) m = fmaxf(m, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < len; ++j) sum += __expf(row[j] - m);
    const float inv = 1.0f / sum;
    for (int j = 0; j < len; ++j) row[j] = __expf(row[j] - m) * inv;
    for (int j = len; j < N; ++j) row[j] = 0.0f;
}

void cublas_check(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS ") + what + " failed: status " +
                                 std::to_string(static_cast<int>(st)));
}

template <typename T>
struct DevBuf {
    T* p = nullptr;
    std::size_t n = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { cudaFree(p); }
    void alloc(std::size_t count) {
        cudaFree(p);
        p = nullptr;
        n = 0;
        CU_CHECK(cudaMalloc(&p, count * sizeof(T)));
        n = count;
    }
};

}  // namespace

const char* to_string(Kernel k) {
    switch (k) {
        case Kernel::NaiveCublas: return "naive (cuBLAS, materialized)";
        case Kernel::FusedGlobal: return "fused, from global";
        case Kernel::FusedTiled: return "fused, tiled";
    }
    return "?";
}

std::vector<int> tile_sizes() {
    std::vector<int> v;
#define FA_PUSH(T) v.push_back(T);
    FA_FOR_EACH_TILE(FA_PUSH)
#undef FA_PUSH
    return v;
}

struct Attention::Impl {
    Shape shape;
    std::size_t vec_floats = 0;    // heads * seq * head_dim
    bool have_inputs = false;
    DevBuf<float> q, k, v, o, scores;
    cublasHandle_t blas = nullptr;

    ~Impl() {
        if (blas) cublasDestroy(blas);
    }

    void run(Kernel kern, int tile) {
        if (!have_inputs) throw std::logic_error("Attention: set_qkv() first");
        const int N = shape.seq, H = shape.heads, Dh = shape.head_dim;
        const float scale = 1.0f / std::sqrt(float(Dh));

        if (kern == Kernel::NaiveCublas) {
            const std::size_t need = std::size_t(H) * N * N;
            if (scores.n != need) scores.alloc(need);
            // Row-major Q [N x Dh] read column-major is Q^T. The row-major
            // scores S = Q K^T are, column-major, S^T = K Q^T = (K^T)^T (Q^T):
            // op T on the column-major view of K, op N on the view of Q.
            //
            // One head per call rather than one strided-batched call: at 16 heads
            // x 8192 tokens the score matrix is 4.3 GB, and on Windows it spills
            // into system RAM, where a single call spanning all of it could run
            // past the display-driver timeout.
            const float alpha = scale, one = 1.0f, beta = 0.0f;
            const std::size_t vec = std::size_t(N) * Dh, mat = std::size_t(N) * N;
            for (int h = 0; h < H; ++h) {
                cublas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, N, Dh, &alpha,
                                         k.p + h * vec, Dh, q.p + h * vec, Dh, &beta,
                                         scores.p + h * mat, N),
                             "scores");
                CU_CHECK(cudaDeviceSynchronize());
                const int t = 256;
                k_softmax_rows<<<(N + t - 1) / t, t>>>(scores.p, N, shape.causal ? 1 : 0, h);
                CU_CHECK_KERNEL();
                // Row-major O = S V; column-major O^T = V^T S^T.
                cublas_check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, Dh, N, N, &one,
                                         v.p + h * vec, Dh, scores.p + h * mat, N, &beta,
                                         o.p + h * vec, Dh),
                             "output");
                CU_CHECK(cudaDeviceSynchronize());
            }
            return;
        }

        if (Dh != D)
            throw std::invalid_argument("fused attention is compiled for head_dim " +
                                        std::to_string(D));
        const int blocks = (N + 31) / 32;
        const int causal = shape.causal ? 1 : 0;
        // One head per launch: a long sequence with many heads in a single
        // launch could run past the Windows display-driver timeout.
        for (int h = 0; h < H; ++h) {
            if (kern == Kernel::FusedGlobal) {
                k_fused_global<<<blocks, 32>>>(q.p, k.p, v.p, o.p, N, causal, scale, h);
            } else {
                switch (tile) {
#define FA_CASE(T)                                                                          \
    case T:                                                                                 \
        k_fused_tiled<T><<<blocks, 32>>>(q.p, k.p, v.p, o.p, N, causal, scale, h);          \
        break;
                    FA_FOR_EACH_TILE(FA_CASE)
#undef FA_CASE
                    default:
                        throw std::invalid_argument("fused attention: no kernel for tile " +
                                                    std::to_string(tile));
                }
            }
            CU_CHECK_KERNEL();
        }
    }
};

Attention::Attention(const Shape& s) : impl_(new Impl) {
    try {
        if (s.heads <= 0 || s.seq <= 0 || s.head_dim <= 0)
            throw std::invalid_argument("Attention: heads, seq and head_dim must be positive");
        impl_->shape = s;
        impl_->vec_floats = std::size_t(s.heads) * s.seq * s.head_dim;
        impl_->q.alloc(impl_->vec_floats);
        impl_->k.alloc(impl_->vec_floats);
        impl_->v.alloc(impl_->vec_floats);
        impl_->o.alloc(impl_->vec_floats);
        cublas_check(cublasCreate(&impl_->blas), "create");
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

Attention::~Attention() { delete impl_; }

const Shape& Attention::shape() const { return impl_->shape; }

void Attention::set_qkv(const std::vector<float>& q, const std::vector<float>& k,
                        const std::vector<float>& v) {
    const std::size_t n = impl_->vec_floats;
    if (q.size() != n || k.size() != n || v.size() != n)
        throw std::invalid_argument("Attention::set_qkv: tensors must be heads*seq*head_dim");
    auto up = [n](DevBuf<float>& d, const std::vector<float>& h) {
        CU_CHECK(cudaMemcpy(d.p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };
    up(impl_->q, q);
    up(impl_->k, k);
    up(impl_->v, v);
    impl_->have_inputs = true;
}

std::vector<float> Attention::forward(Kernel k, int tile) {
    impl_->run(k, tile);
    std::vector<float> out(impl_->vec_floats);
    CU_CHECK(cudaMemcpy(out.data(), impl_->o.p, out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
    return out;
}

float Attention::time(Kernel k, int tile, int iterations, int warmup) {
    return cu::benchmark([&] { impl_->run(k, tile); }, iterations, warmup).median_ms;
}

std::size_t Attention::device_bytes(Kernel k) const {
    const std::size_t vecs = 4 * impl_->vec_floats * sizeof(float);
    if (k != Kernel::NaiveCublas) return vecs;
    const std::size_t N = impl_->shape.seq;
    return vecs + std::size_t(impl_->shape.heads) * N * N * sizeof(float);
}

int Attention::blocks_per_sm(Kernel k, int tile) const {
    const void* fn = nullptr;
    if (k == Kernel::FusedGlobal) {
        fn = reinterpret_cast<const void*>(k_fused_global);
    } else if (k == Kernel::FusedTiled) {
        switch (tile) {
#define FA_PTR(T)                                                  \
    case T:                                                        \
        fn = reinterpret_cast<const void*>(k_fused_tiled<T>);      \
        break;
            FA_FOR_EACH_TILE(FA_PTR)
#undef FA_PTR
            default:
                throw std::invalid_argument("no kernel for tile " + std::to_string(tile));
        }
    } else {
        throw std::invalid_argument("blocks_per_sm: only the fused kernels are custom kernels");
    }
    int blocks = 0;
    CU_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, 32, 0));
    return blocks;
}

std::vector<float> reference_attention(const Shape& s, const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v) {
    const int H = s.heads, N = s.seq, Dh = s.head_dim;
    const double scale = 1.0 / std::sqrt(double(Dh));
    std::vector<float> out(std::size_t(H) * N * Dh, 0.0f);
    std::vector<double> logits(N), acc(Dh);
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < N; ++i) {
            const std::size_t qi = (std::size_t(h) * N + i) * Dh;
            const int len = s.causal ? i + 1 : N;
            double mx = -1e300;
            for (int j = 0; j < len; ++j) {
                const std::size_t kj = (std::size_t(h) * N + j) * Dh;
                double x = 0.0;
                for (int c = 0; c < Dh; ++c) x += double(q[qi + c]) * k[kj + c];
                logits[j] = x * scale;
                mx = std::max(mx, logits[j]);
            }
            double z = 0.0;
            for (int j = 0; j < len; ++j) z += std::exp(logits[j] - mx);
            // Accumulate in double and narrow once: a float accumulator would
            // throw away exactly the precision a reference is for.
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < len; ++j) {
                const double w = std::exp(logits[j] - mx) / z;
                const std::size_t vj = (std::size_t(h) * N + j) * Dh;
                for (int c = 0; c < Dh; ++c) acc[c] += w * v[vj + c];
            }
            for (int c = 0; c < Dh; ++c) out[qi + c] = float(acc[c]);
        }
    return out;
}

}  // namespace fa
