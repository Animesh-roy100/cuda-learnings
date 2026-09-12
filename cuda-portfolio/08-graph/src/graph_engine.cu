// Graph processing engine -- implementation.
//
// The defining problem of GPU graph work is IRREGULARITY. Neighbour lists have
// wildly different lengths, and the node ids inside them are scattered, so:
//
//   * one thread per node   -> a warp runs as slowly as its highest-degree node
//   * push (scatter)        -> needs atomics, and hub nodes contend hard
//   * pull (gather)         -> atomic-free, but reads scattered source values
//
// Both directions and both balancing strategies are implemented so the
// trade-off can be measured rather than argued about.

#include "graph_engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>

#include "cu/check.hpp"

namespace graph {
namespace {

using i32 = std::int32_t;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// PageRank, pull direction. Each node gathers contributions from its
// IN-neighbours, so every write is private -- no atomics at all.
// ---------------------------------------------------------------------------
__global__ void k_pagerank_pull_thread(const i32* __restrict__ in_off,
                                       const i32* __restrict__ in_idx,
                                       const i32* __restrict__ out_deg,
                                       const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       i32 n, float damping, float base) {
    i32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    float sum = 0.0f;
    for (i32 e = in_off[v]; e < in_off[v + 1]; ++e) {
        i32 u = in_idx[e];
        i32 d = out_deg[u];
        if (d > 0) sum += src[u] / static_cast<float>(d);
    }
    dst[v] = base + damping * sum;
}

// One WARP per node. A hub with 10000 in-edges is split across 32 lanes instead
// of stalling one thread while the other 31 idle -- the single most effective
// fix for degree skew.
__global__ void k_pagerank_pull_warp(const i32* __restrict__ in_off,
                                     const i32* __restrict__ in_idx,
                                     const i32* __restrict__ out_deg,
                                     const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     i32 n, float damping, float base) {
    i32 warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= n) return;

    i32 begin = in_off[warp_id], end = in_off[warp_id + 1];
    float sum = 0.0f;
    for (i32 e = begin + lane; e < end; e += 32) {
        i32 u = in_idx[e];
        i32 d = out_deg[u];
        if (d > 0) sum += src[u] / static_cast<float>(d);
    }
    sum = warp_sum(sum);
    if (lane == 0) dst[warp_id] = base + damping * sum;
}

// ---------------------------------------------------------------------------
// PageRank, push direction. Each node scatters its rank to out-neighbours.
// Correct, but every update is an atomicAdd and hub nodes serialise on them.
// ---------------------------------------------------------------------------
__global__ void k_pagerank_push_warp(const i32* __restrict__ out_off,
                                     const i32* __restrict__ out_idx,
                                     const float* __restrict__ src,
                                     float* __restrict__ dst, i32 n) {
    i32 warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= n) return;

    i32 begin = out_off[warp_id], end = out_off[warp_id + 1];
    i32 deg = end - begin;
    if (deg == 0) return;
    float share = src[warp_id] / static_cast<float>(deg);
    for (i32 e = begin + lane; e < end; e += 32) {
        atomicAdd(&dst[out_idx[e]], share);
    }
}

__global__ void k_finish_push(float* dst, i32 n, float damping, float base) {
    i32 v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) dst[v] = base + damping * dst[v];
}

__global__ void k_fill(float* p, i32 n, float v) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// L1 residual between iterations, for the convergence check.
__global__ void k_l1_diff(const float* a, const float* b, i32 n, double* out) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    i32 stride = gridDim.x * blockDim.x;
    float s = 0.0f;
    for (; i < n; i += stride) s += fabsf(a[i] - b[i]);
    s = warp_sum(s);
    __shared__ float w[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) w[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float v = (lane < nw) ? w[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, static_cast<double>(v));
    }
}

// Dangling nodes (out-degree 0) leak rank out of the system. Their mass has to
// be collected and redistributed or the vector stops summing to 1.
__global__ void k_dangling_mass(const float* rank, const i32* out_deg, i32 n, double* out) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    i32 stride = gridDim.x * blockDim.x;
    float s = 0.0f;
    for (; i < n; i += stride) if (out_deg[i] == 0) s += rank[i];
    s = warp_sum(s);
    __shared__ float w[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) w[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float v = (lane < nw) ? w[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, static_cast<double>(v));
    }
}

// ---------------------------------------------------------------------------
// SSSP by Bellman-Ford style relaxation, warp-per-node, with a global "did
// anything change" flag. Simple, correct, and the iteration count equals the
// hop-depth of the graph, which makes frontier behaviour easy to observe.
// ---------------------------------------------------------------------------
__global__ void k_relax_warp(const i32* __restrict__ off, const i32* __restrict__ idx,
                             const float* __restrict__ wgt, bool weighted,
                             float* __restrict__ dist, i32 n, int* changed) {
    i32 warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= n) return;

    float du = dist[warp_id];
    if (du >= 3.4e38f) return;              // not reached yet; nothing to push

    i32 begin = off[warp_id], end = off[warp_id + 1];
    for (i32 e = begin + lane; e < end; e += 32) {
        float w = weighted ? wgt[e] : 1.0f;
        float nd = du + w;
        i32 v = idx[e];
        // atomicMin on float via integer bit pattern is valid for non-negative
        // floats, whose IEEE754 encoding is monotone as unsigned ints.
        int old = atomicMin(reinterpret_cast<int*>(&dist[v]), __float_as_int(nd));
        if (__int_as_float(old) > nd) *changed = 1;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
CsrGraph CsrGraph::transpose() const {
    CsrGraph t;
    const i32 n = num_nodes();
    t.row_offsets.assign(n + 1, 0);
    for (std::size_t e = 0; e < col_indices.size(); ++e) t.row_offsets[col_indices[e] + 1]++;
    for (i32 i = 0; i < n; ++i) t.row_offsets[i + 1] += t.row_offsets[i];

    t.col_indices.resize(col_indices.size());
    if (!weights.empty()) t.weights.resize(weights.size());
    std::vector<i32> cursor(t.row_offsets.begin(), t.row_offsets.end() - 1);
    for (i32 u = 0; u < n; ++u) {
        for (i32 e = row_offsets[u]; e < row_offsets[u + 1]; ++e) {
            i32 v = col_indices[e];
            i32 pos = cursor[v]++;
            t.col_indices[pos] = u;
            if (!weights.empty()) t.weights[pos] = weights[e];
        }
    }
    return t;
}

CsrGraph CsrGraph::random_power_law(i32 n, int avg_degree, unsigned seed, bool weighted) {
    std::mt19937 rng(seed);
    // Preferential-attachment-ish: targets drawn with a bias toward low ids,
    // which become the hubs.
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<std::vector<i32>> adj(n);
    for (i32 v = 0; v < n; ++v) {
        int deg = 1 + static_cast<int>(u01(rng) * (2 * avg_degree - 1));
        for (int k = 0; k < deg; ++k) {
            // x^3 skews hard toward 0 -> a handful of very high in-degree hubs.
            float x = u01(rng);
            i32 t = static_cast<i32>(x * x * x * n);
            if (t >= n) t = n - 1;
            if (t != v) adj[v].push_back(t);
        }
        std::sort(adj[v].begin(), adj[v].end());
        adj[v].erase(std::unique(adj[v].begin(), adj[v].end()), adj[v].end());
    }

    CsrGraph g;
    g.row_offsets.resize(n + 1, 0);
    for (i32 v = 0; v < n; ++v) g.row_offsets[v + 1] = g.row_offsets[v] + static_cast<i32>(adj[v].size());
    g.col_indices.reserve(g.row_offsets[n]);
    for (i32 v = 0; v < n; ++v)
        for (i32 t : adj[v]) g.col_indices.push_back(t);
    if (weighted) {
        g.weights.resize(g.col_indices.size());
        for (auto& w : g.weights) w = 1.0f + u01(rng) * 9.0f;
    }
    return g;
}

CsrGraph CsrGraph::grid_2d(i32 w, i32 h, bool weighted) {
    CsrGraph g;
    const i32 n = w * h;
    g.row_offsets.assign(n + 1, 0);
    std::vector<std::vector<i32>> adj(n);
    for (i32 y = 0; y < h; ++y)
        for (i32 x = 0; x < w; ++x) {
            i32 v = y * w + x;
            if (x > 0) adj[v].push_back(v - 1);
            if (x < w - 1) adj[v].push_back(v + 1);
            if (y > 0) adj[v].push_back(v - w);
            if (y < h - 1) adj[v].push_back(v + w);
        }
    for (i32 v = 0; v < n; ++v) g.row_offsets[v + 1] = g.row_offsets[v] + static_cast<i32>(adj[v].size());
    for (i32 v = 0; v < n; ++v)
        for (i32 t : adj[v]) g.col_indices.push_back(t);
    if (weighted) g.weights.assign(g.col_indices.size(), 1.0f);
    return g;
}

// ---------------------------------------------------------------------------
struct GraphEngine::Impl {
    i32 n = 0;
    std::int64_t m = 0;
    bool weighted = false;

    i32 *d_out_off = nullptr, *d_out_idx = nullptr;
    i32 *d_in_off = nullptr, *d_in_idx = nullptr;
    i32* d_out_deg = nullptr;
    float* d_w = nullptr;

    float *d_a = nullptr, *d_b = nullptr;
    double* d_scalar = nullptr;
    int* d_flag = nullptr;
};

GraphEngine::GraphEngine(const CsrGraph& g) : impl_(new Impl) {
    impl_->n = g.num_nodes();
    impl_->m = g.num_edges();
    impl_->weighted = !g.weights.empty();
    const i32 n = impl_->n;

    CsrGraph t = g.transpose();
    std::vector<i32> out_deg(n);
    for (i32 v = 0; v < n; ++v) out_deg[v] = g.row_offsets[v + 1] - g.row_offsets[v];

    auto up_i32 = [](i32** d, const std::vector<i32>& h) {
        CU_CHECK(cudaMalloc(d, sizeof(i32) * std::max<std::size_t>(h.size(), 1)));
        if (!h.empty())
            CU_CHECK(cudaMemcpy(*d, h.data(), sizeof(i32) * h.size(), cudaMemcpyHostToDevice));
    };
    up_i32(&impl_->d_out_off, g.row_offsets);
    up_i32(&impl_->d_out_idx, g.col_indices);
    up_i32(&impl_->d_in_off, t.row_offsets);
    up_i32(&impl_->d_in_idx, t.col_indices);
    up_i32(&impl_->d_out_deg, out_deg);

    if (impl_->weighted) {
        CU_CHECK(cudaMalloc(&impl_->d_w, sizeof(float) * g.weights.size()));
        CU_CHECK(cudaMemcpy(impl_->d_w, g.weights.data(), sizeof(float) * g.weights.size(),
                            cudaMemcpyHostToDevice));
    }
    CU_CHECK(cudaMalloc(&impl_->d_a, sizeof(float) * std::max(n, 1)));
    CU_CHECK(cudaMalloc(&impl_->d_b, sizeof(float) * std::max(n, 1)));
    CU_CHECK(cudaMalloc(&impl_->d_scalar, sizeof(double)));
    CU_CHECK(cudaMalloc(&impl_->d_flag, sizeof(int)));
}

GraphEngine::~GraphEngine() {
    if (!impl_) return;
    cudaFree(impl_->d_out_off); cudaFree(impl_->d_out_idx);
    cudaFree(impl_->d_in_off); cudaFree(impl_->d_in_idx);
    cudaFree(impl_->d_out_deg); cudaFree(impl_->d_w);
    cudaFree(impl_->d_a); cudaFree(impl_->d_b);
    cudaFree(impl_->d_scalar); cudaFree(impl_->d_flag);
    delete impl_;
}

PageRankResult GraphEngine::pagerank(float damping, int max_iter, float tol,
                                     Direction dir, Balance balance) {
    const i32 n = impl_->n;
    PageRankResult res;
    if (n == 0) return res;

    const int T = 256;
    const int Bn = (n + T - 1) / T;
    const int Bw = (n + (T / 32) - 1) / (T / 32);

    k_fill<<<Bn, T>>>(impl_->d_a, n, 1.0f / n);
    CU_CHECK_KERNEL();

    float* cur = impl_->d_a;
    float* nxt = impl_->d_b;
    int it = 0;
    double residual = 0.0;

    for (; it < max_iter; ++it) {
        // Dangling mass must be redistributed or total rank decays every pass.
        CU_CHECK(cudaMemset(impl_->d_scalar, 0, sizeof(double)));
        k_dangling_mass<<<64, T>>>(cur, impl_->d_out_deg, n, impl_->d_scalar);
        CU_CHECK_KERNEL();
        double dangling = 0.0;
        CU_CHECK(cudaMemcpy(&dangling, impl_->d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
        const float base = (1.0f - damping) / n + damping * static_cast<float>(dangling) / n;

        if (dir == Direction::Pull) {
            if (balance == Balance::WarpPerNode) {
                k_pagerank_pull_warp<<<Bw, T>>>(impl_->d_in_off, impl_->d_in_idx,
                                                impl_->d_out_deg, cur, nxt, n, damping, base);
            } else {
                k_pagerank_pull_thread<<<Bn, T>>>(impl_->d_in_off, impl_->d_in_idx,
                                                  impl_->d_out_deg, cur, nxt, n, damping, base);
            }
        } else {
            k_fill<<<Bn, T>>>(nxt, n, 0.0f);
            k_pagerank_push_warp<<<Bw, T>>>(impl_->d_out_off, impl_->d_out_idx, cur, nxt, n);
            k_finish_push<<<Bn, T>>>(nxt, n, damping, base);
        }
        CU_CHECK_KERNEL();

        CU_CHECK(cudaMemset(impl_->d_scalar, 0, sizeof(double)));
        k_l1_diff<<<64, T>>>(cur, nxt, n, impl_->d_scalar);
        CU_CHECK_KERNEL();
        CU_CHECK(cudaMemcpy(&residual, impl_->d_scalar, sizeof(double), cudaMemcpyDeviceToHost));

        std::swap(cur, nxt);
        if (residual < tol) { ++it; break; }
    }

    res.rank.resize(n);
    CU_CHECK(cudaMemcpy(res.rank.data(), cur, sizeof(float) * n, cudaMemcpyDeviceToHost));
    res.iterations = it;
    res.residual = residual;
    return res;
}

std::vector<float> GraphEngine::sssp(i32 source, int* iterations) {
    const i32 n = impl_->n;
    std::vector<float> out(n, kUnreachable);
    if (n == 0 || source < 0 || source >= n) return out;

    const int T = 256;
    const int Bn = (n + T - 1) / T;
    const int Bw = (n + (T / 32) - 1) / (T / 32);

    k_fill<<<Bn, T>>>(impl_->d_a, n, kUnreachable);
    CU_CHECK_KERNEL();
    float zero = 0.0f;
    CU_CHECK(cudaMemcpy(impl_->d_a + source, &zero, sizeof(float), cudaMemcpyHostToDevice));

    int it = 0;
    for (; it < n; ++it) {
        CU_CHECK(cudaMemset(impl_->d_flag, 0, sizeof(int)));
        k_relax_warp<<<Bw, T>>>(impl_->d_out_off, impl_->d_out_idx, impl_->d_w,
                                impl_->weighted, impl_->d_a, n, impl_->d_flag);
        CU_CHECK_KERNEL();
        int changed = 0;
        CU_CHECK(cudaMemcpy(&changed, impl_->d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (!changed) break;
    }
    if (iterations) *iterations = it;

    CU_CHECK(cudaMemcpy(out.data(), impl_->d_a, sizeof(float) * n, cudaMemcpyDeviceToHost));
    return out;
}

// ---------------------------------------------------------------------------
PageRankResult GraphEngine::pagerank_cpu(const CsrGraph& g, float damping,
                                         int max_iter, float tol) {
    const i32 n = g.num_nodes();
    PageRankResult res;
    if (n == 0) return res;

    CsrGraph t = g.transpose();
    std::vector<i32> out_deg(n);
    for (i32 v = 0; v < n; ++v) out_deg[v] = g.row_offsets[v + 1] - g.row_offsets[v];

    std::vector<float> cur(n, 1.0f / n), nxt(n, 0.0f);
    int it = 0;
    double residual = 0.0;
    for (; it < max_iter; ++it) {
        double dangling = 0.0;
        for (i32 v = 0; v < n; ++v) if (out_deg[v] == 0) dangling += cur[v];
        const float base = (1.0f - damping) / n + damping * static_cast<float>(dangling) / n;

        for (i32 v = 0; v < n; ++v) {
            float sum = 0.0f;
            for (i32 e = t.row_offsets[v]; e < t.row_offsets[v + 1]; ++e) {
                i32 u = t.col_indices[e];
                if (out_deg[u] > 0) sum += cur[u] / static_cast<float>(out_deg[u]);
            }
            nxt[v] = base + damping * sum;
        }
        residual = 0.0;
        for (i32 v = 0; v < n; ++v) residual += std::fabs(nxt[v] - cur[v]);
        cur.swap(nxt);
        if (residual < tol) { ++it; break; }
    }
    res.rank = cur;
    res.iterations = it;
    res.residual = residual;
    return res;
}

std::vector<float> GraphEngine::sssp_cpu(const CsrGraph& g, i32 source) {
    const i32 n = g.num_nodes();
    std::vector<float> dist(n, kUnreachable);
    if (n == 0 || source < 0 || source >= n) return dist;

    using Item = std::pair<float, i32>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    dist[source] = 0.0f;
    pq.push({0.0f, source});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        for (i32 e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
            i32 v = g.col_indices[e];
            float w = g.weights.empty() ? 1.0f : g.weights[e];
            if (d + w < dist[v]) {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

}  // namespace graph
