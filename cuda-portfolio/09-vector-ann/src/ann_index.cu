// IVF-Flat ANN index -- implementation.
//
// Distance kernels here are memory bound: one query vector is reused across
// thousands of database vectors, so the database stream IS the cost. Two things
// follow, and both are done below:
//
//   * load the database with float4 (128-bit per thread), the widest access
//     the memory system issues
//   * keep top-k in REGISTERS, never in global memory -- a global-memory heap
//     would add a round trip per candidate and dwarf the distance arithmetic

#include "ann_index.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace ann {
namespace {

constexpr int MAXK = IvfFlatIndex::kMaxK;
constexpr float kFar = 3.4e38f;

__device__ __forceinline__ void topk_insert(float* dist, int* idx, int k,
                                            float d, int id) {
    if (d >= dist[k - 1]) return;
    int m = k - 1;
    while (m > 0 && dist[m - 1] > d) {
        dist[m] = dist[m - 1];
        idx[m] = idx[m - 1];
        --m;
    }
    dist[m] = d;
    idx[m] = id;
}

// Squared L2 between a query held in registers and a database row, using
// float4 loads. dim must be a multiple of 4 (enforced host-side).
__device__ __forceinline__ float l2_f4(const float4* __restrict__ a,
                                       const float4* __restrict__ b, int d4) {
    float s = 0.0f;
    for (int i = 0; i < d4; ++i) {
        float4 x = a[i], y = b[i];
        float dx = x.x - y.x, dy = x.y - y.y, dz = x.z - y.z, dw = x.w - y.w;
        s += dx * dx + dy * dy + dz * dz + dw * dw;
    }
    return s;
}

// Cosine distance. Vectors are L2-normalised at insert time, so the dot product
// alone determines the ordering and no per-candidate norm is needed.
__device__ __forceinline__ float dot_f4(const float4* __restrict__ a,
                                        const float4* __restrict__ b, int d4) {
    float s = 0.0f;
    for (int i = 0; i < d4; ++i) {
        float4 x = a[i], y = b[i];
        s += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Assign each vector to its nearest centroid.
// ---------------------------------------------------------------------------
__global__ void k_assign(const float* __restrict__ vecs, int n, int dim,
                         const float* __restrict__ centroids, int nlist,
                         int metric_cosine, int* __restrict__ assign) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int d4 = dim / 4;
    const float4* v = reinterpret_cast<const float4*>(vecs + (size_t)i * dim);

    int best = 0;
    float bestd = kFar;
    for (int c = 0; c < nlist; ++c) {
        const float4* q = reinterpret_cast<const float4*>(centroids + (size_t)c * dim);
        float d = metric_cosine ? -dot_f4(v, q, d4) : l2_f4(v, q, d4);
        if (d < bestd) { bestd = d; best = c; }
    }
    assign[i] = best;
}

// ---------------------------------------------------------------------------
// Rank centroids for each query and keep the nprobe closest.
//
// Done in two passes through a scratch distance array rather than a register
// top-k. A register list is bounded by MAXK=32, and silently truncating nprobe
// to 32 is far worse than being slow: the caller asks for nprobe=256, gets 32
// lists scanned, and sees recall plateau at ~80% with no error anywhere.
// Selection here is O(nlist * nprobe) per query, which is nothing next to
// scanning the lists themselves.
// ---------------------------------------------------------------------------
__global__ void k_centroid_dists(const float* __restrict__ queries, int nq, int dim,
                                 const float* __restrict__ centroids, int nlist,
                                 int metric_cosine, float* __restrict__ dists) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nq * nlist) return;
    const int q = idx / nlist;
    const int c = idx - q * nlist;
    const int d4 = dim / 4;

    const float4* qv = reinterpret_cast<const float4*>(queries + (size_t)q * dim);
    const float4* cv = reinterpret_cast<const float4*>(centroids + (size_t)c * dim);
    dists[(size_t)q * nlist + c] = metric_cosine ? -dot_f4(qv, cv, d4) : l2_f4(qv, cv, d4);
}

__global__ void k_select_probes(float* __restrict__ dists, int nq, int nlist,
                                int nprobe, int* __restrict__ probe_out) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    float* row = dists + (size_t)q * nlist;

    for (int p = 0; p < nprobe; ++p) {
        int best = -1;
        float bestd = kFar;
        for (int c = 0; c < nlist; ++c) {
            if (row[c] < bestd) { bestd = row[c]; best = c; }
        }
        probe_out[(size_t)q * nprobe + p] = best;
        if (best >= 0) row[best] = kFar;   // consume it
    }
}

// ---------------------------------------------------------------------------
// Scan the probed lists. One WARP per query.
//
// Each lane keeps its own top-k in registers while striding through candidates,
// then the warp merges the 32 partial lists by repeatedly taking the global
// minimum. That merge costs k rounds of shuffle reduction -- trivial next to
// the distance work, and it never touches global memory.
// ---------------------------------------------------------------------------
// The whole WARP cooperates on ONE candidate at a time: lane i reads float4
// element i, i+32, i+64 ... of that vector, then a shuffle reduction sums the
// partials.
//
// The obvious alternative -- one lane per candidate, lanes striding over
// different vectors -- was measured first and is far slower. With 768 dims a
// row is 3072 bytes, so lane 0 and lane 1 would touch addresses 3072 bytes
// apart and every load would be its own transaction. Cooperating on one row
// keeps each warp load contiguous, which is the only way to approach peak
// bandwidth on a kernel that is pure streaming.
//
// Top-k then lives on lane 0 alone, which also removes the 32-way merge the
// per-lane version needed.
__device__ __forceinline__ float warp_reduce_add(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_distance(const float4* __restrict__ qv,
                                               const float4* __restrict__ dv,
                                               int d4, int lane, int metric_cosine) {
    float part = 0.0f;
    for (int i = lane; i < d4; i += 32) {
        float4 a = qv[i], b = dv[i];
        if (metric_cosine) {
            part += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        } else {
            float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z, dw = a.w - b.w;
            part += dx * dx + dy * dy + dz * dz + dw * dw;
        }
    }
    float total = warp_reduce_add(part);
    return metric_cosine ? (1.0f - total) : total;
}

__global__ void k_search_ivf(const float* __restrict__ db, int dim,
                             const int* __restrict__ list_start,
                             const int* __restrict__ list_count,
                             const int* __restrict__ list_ids,
                             const float* __restrict__ queries, int nq,
                             const int* __restrict__ probes, int nprobe,
                             int k, int metric_cosine,
                             int* __restrict__ out_ids, float* __restrict__ out_dist) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= nq) return;

    const int d4 = dim / 4;
    const float4* qv = reinterpret_cast<const float4*>(queries + (size_t)warp_id * dim);

    float bd[MAXK];
    int bi[MAXK];
    for (int i = 0; i < k; ++i) { bd[i] = kFar; bi[i] = -1; }

    for (int p = 0; p < nprobe; ++p) {
        const int cell = probes[(size_t)warp_id * nprobe + p];
        if (cell < 0) continue;
        const int begin = list_start[cell];
        const int cnt = list_count[cell];
        for (int j = 0; j < cnt; ++j) {
            const int id = list_ids[begin + j];
            const float4* dv = reinterpret_cast<const float4*>(db + (size_t)id * dim);
            float d = warp_distance(qv, dv, d4, lane, metric_cosine);
            if (lane == 0) topk_insert(bd, bi, k, d, id);
        }
    }
    if (lane == 0) {
        for (int r = 0; r < k; ++r) {
            out_ids[(size_t)warp_id * k + r] = bi[r];
            out_dist[(size_t)warp_id * k + r] = bd[r];
        }
    }
}

// Exhaustive scan, same warp-cooperative structure. The ground truth.
__global__ void k_search_brute(const float* __restrict__ db, int n, int dim,
                               const float* __restrict__ queries, int nq, int k,
                               int metric_cosine,
                               int* __restrict__ out_ids, float* __restrict__ out_dist) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= nq) return;

    const int d4 = dim / 4;
    const float4* qv = reinterpret_cast<const float4*>(queries + (size_t)warp_id * dim);

    float bd[MAXK];
    int bi[MAXK];
    for (int i = 0; i < k; ++i) { bd[i] = kFar; bi[i] = -1; }

    for (int j = 0; j < n; ++j) {
        const float4* dv = reinterpret_cast<const float4*>(db + (size_t)j * dim);
        float d = warp_distance(qv, dv, d4, lane, metric_cosine);
        if (lane == 0) topk_insert(bd, bi, k, d, j);
    }
    if (lane == 0) {
        for (int r = 0; r < k; ++r) {
            out_ids[(size_t)warp_id * k + r] = bi[r];
            out_dist[(size_t)warp_id * k + r] = bd[r];
        }
    }
}

void l2_normalize(std::vector<float>& v, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        float* p = &v[(std::size_t)i * dim];
        double s = 0.0;
        for (int d = 0; d < dim; ++d) s += double(p[d]) * p[d];
        float inv = (s > 0.0) ? float(1.0 / std::sqrt(s)) : 0.0f;
        for (int d = 0; d < dim; ++d) p[d] *= inv;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
struct IvfFlatIndex::Impl {
    int dim = 0, nlist = 0, n = 0;
    Metric metric = Metric::L2;
    bool trained = false;

    std::vector<float> h_centroids;
    std::vector<int> h_list_start, h_list_count;

    float* d_db = nullptr;
    float* d_centroids = nullptr;
    int* d_list_start = nullptr;
    int* d_list_count = nullptr;
    int* d_list_ids = nullptr;

    std::size_t bytes = 0;
    bool cosine() const { return metric == Metric::Cosine; }
};

IvfFlatIndex::IvfFlatIndex(int dim, int nlist, Metric metric) : impl_(new Impl) {
    if (dim <= 0 || dim % 4 != 0)
        throw std::invalid_argument("dim must be positive and a multiple of 4 "
                                    "(float4 loads)");
    if (nlist <= 0) throw std::invalid_argument("nlist must be positive");
    impl_->dim = dim;
    impl_->nlist = nlist;
    impl_->metric = metric;
}

IvfFlatIndex::~IvfFlatIndex() {
    if (!impl_) return;
    cudaFree(impl_->d_db);
    cudaFree(impl_->d_centroids);
    cudaFree(impl_->d_list_start);
    cudaFree(impl_->d_list_count);
    cudaFree(impl_->d_list_ids);
    delete impl_;
}

int IvfFlatIndex::dim() const { return impl_->dim; }
int IvfFlatIndex::nlist() const { return impl_->nlist; }
int IvfFlatIndex::size() const { return impl_->n; }
bool IvfFlatIndex::is_trained() const { return impl_->trained; }
std::size_t IvfFlatIndex::device_bytes() const { return impl_->bytes; }
std::vector<int> IvfFlatIndex::list_sizes() const { return impl_->h_list_count; }

void IvfFlatIndex::train(const std::vector<float>& vectors, int n, int iters,
                         unsigned seed) {
    const int dim = impl_->dim, nlist = impl_->nlist;
    if (n < nlist) throw std::invalid_argument("need at least nlist training vectors");
    if (vectors.size() != (std::size_t)n * dim)
        throw std::invalid_argument("training data size mismatch");

    std::vector<float> data = vectors;
    if (impl_->cosine()) l2_normalize(data, n, dim);

    // Lloyd's algorithm on the host. Training runs once and offline, so there
    // is nothing to gain from putting it on the GPU -- search is the hot path.
    std::mt19937 rng(seed);
    std::vector<int> pick(n);
    std::iota(pick.begin(), pick.end(), 0);
    std::shuffle(pick.begin(), pick.end(), rng);

    std::vector<float> cent((std::size_t)nlist * dim);
    for (int c = 0; c < nlist; ++c)
        std::copy_n(&data[(std::size_t)pick[c] * dim], dim, &cent[(std::size_t)c * dim]);

    std::vector<int> assign(n, 0);
    std::vector<double> acc((std::size_t)nlist * dim);
    std::vector<int> cnt(nlist);

    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < n; ++i) {
            const float* v = &data[(std::size_t)i * dim];
            int best = 0;
            double bestd = 1e300;
            for (int c = 0; c < nlist; ++c) {
                const float* p = &cent[(std::size_t)c * dim];
                double d = 0.0;
                if (impl_->cosine()) {
                    for (int t = 0; t < dim; ++t) d -= double(v[t]) * p[t];
                } else {
                    for (int t = 0; t < dim; ++t) { double e = double(v[t]) - p[t]; d += e * e; }
                }
                if (d < bestd) { bestd = d; best = c; }
            }
            assign[i] = best;
        }
        std::fill(acc.begin(), acc.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int i = 0; i < n; ++i) {
            const float* v = &data[(std::size_t)i * dim];
            double* a = &acc[(std::size_t)assign[i] * dim];
            for (int t = 0; t < dim; ++t) a[t] += v[t];
            cnt[assign[i]]++;
        }
        for (int c = 0; c < nlist; ++c) {
            if (cnt[c] == 0) {
                // An empty cell would be dead weight: reseed it on a random
                // point so every centroid keeps earning its place.
                int r = (int)(rng() % (unsigned)n);
                std::copy_n(&data[(std::size_t)r * dim], dim, &cent[(std::size_t)c * dim]);
                continue;
            }
            float* p = &cent[(std::size_t)c * dim];
            for (int t = 0; t < dim; ++t) p[t] = float(acc[(std::size_t)c * dim + t] / cnt[c]);
        }
    }
    if (impl_->cosine()) l2_normalize(cent, nlist, dim);

    impl_->h_centroids = std::move(cent);
    if (impl_->d_centroids) cudaFree(impl_->d_centroids);
    CU_CHECK(cudaMalloc(&impl_->d_centroids, impl_->h_centroids.size() * sizeof(float)));
    CU_CHECK(cudaMemcpy(impl_->d_centroids, impl_->h_centroids.data(),
                        impl_->h_centroids.size() * sizeof(float), cudaMemcpyHostToDevice));
    impl_->trained = true;
}

void IvfFlatIndex::add(const std::vector<float>& vectors, int n) {
    if (!impl_->trained) throw std::logic_error("index must be trained before add()");
    const int dim = impl_->dim, nlist = impl_->nlist;
    if (vectors.size() != (std::size_t)n * dim)
        throw std::invalid_argument("vector data size mismatch");

    std::vector<float> data = vectors;
    if (impl_->cosine()) l2_normalize(data, n, dim);

    if (impl_->d_db) cudaFree(impl_->d_db);
    CU_CHECK(cudaMalloc(&impl_->d_db, data.size() * sizeof(float)));
    CU_CHECK(cudaMemcpy(impl_->d_db, data.data(), data.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
    impl_->n = n;

    int* d_assign = nullptr;
    CU_CHECK(cudaMalloc(&d_assign, (std::size_t)n * sizeof(int)));
    const int T = 128;
    k_assign<<<(n + T - 1) / T, T>>>(impl_->d_db, n, dim, impl_->d_centroids, nlist,
                                     impl_->cosine() ? 1 : 0, d_assign);
    CU_CHECK_KERNEL();

    std::vector<int> assign(n);
    CU_CHECK(cudaMemcpy(assign.data(), d_assign, (std::size_t)n * sizeof(int),
                        cudaMemcpyDeviceToHost));
    cudaFree(d_assign);

    // Counting sort into contiguous inverted lists, so a cell scan reads a
    // single run of ids instead of chasing a linked structure.
    std::vector<int> count(nlist, 0), start(nlist, 0);
    for (int i = 0; i < n; ++i) count[assign[i]]++;
    int run = 0;
    for (int c = 0; c < nlist; ++c) { start[c] = run; run += count[c]; }

    std::vector<int> ids(n);
    std::vector<int> cursor = start;
    for (int i = 0; i < n; ++i) ids[cursor[assign[i]]++] = i;

    impl_->h_list_start = start;
    impl_->h_list_count = count;

    auto up = [](int** d, const std::vector<int>& h) {
        if (*d) cudaFree(*d);
        CU_CHECK(cudaMalloc(d, std::max<std::size_t>(h.size(), 1) * sizeof(int)));
        if (!h.empty())
            CU_CHECK(cudaMemcpy(*d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
    };
    up(&impl_->d_list_start, start);
    up(&impl_->d_list_count, count);
    up(&impl_->d_list_ids, ids);

    impl_->bytes = data.size() * sizeof(float) +
                   impl_->h_centroids.size() * sizeof(float) +
                   (std::size_t)n * sizeof(int) + (std::size_t)nlist * 2 * sizeof(int);
}

SearchResult IvfFlatIndex::search(const std::vector<float>& queries, int nq, int k,
                                  int nprobe, float* elapsed_ms) const {
    if (impl_->n == 0) throw std::logic_error("index is empty");
    if (k <= 0 || k > kMaxK) throw std::invalid_argument("k must be in [1,32]");
    if (nprobe <= 0 || nprobe > impl_->nlist)
        throw std::invalid_argument("nprobe must be in [1,nlist]");
    const int dim = impl_->dim;
    if (queries.size() != (std::size_t)nq * dim)
        throw std::invalid_argument("query size mismatch");

    std::vector<float> q = queries;
    if (impl_->cosine()) l2_normalize(q, nq, dim);

    float *d_q = nullptr, *d_dist = nullptr, *d_cdist = nullptr;
    int *d_probe = nullptr, *d_ids = nullptr;
    CU_CHECK(cudaMalloc(&d_q, q.size() * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_probe, (std::size_t)nq * nprobe * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_cdist, (std::size_t)nq * impl_->nlist * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_ids, (std::size_t)nq * k * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_dist, (std::size_t)nq * k * sizeof(float)));
    CU_CHECK(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int T = 128;
    cu::EventTimer t;
    t.start();
    const int ncd = nq * impl_->nlist;
    k_centroid_dists<<<(ncd + T - 1) / T, T>>>(d_q, nq, dim, impl_->d_centroids,
                                               impl_->nlist, impl_->cosine() ? 1 : 0,
                                               d_cdist);
    k_select_probes<<<(nq + T - 1) / T, T>>>(d_cdist, nq, impl_->nlist, nprobe, d_probe);
    const int wpb = T / 32;
    k_search_ivf<<<(nq + wpb - 1) / wpb, T>>>(impl_->d_db, dim, impl_->d_list_start,
                                              impl_->d_list_count, impl_->d_list_ids,
                                              d_q, nq, d_probe, nprobe, k,
                                              impl_->cosine() ? 1 : 0, d_ids, d_dist);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    SearchResult r;
    r.n_queries = nq;
    r.k = k;
    r.ids.resize((std::size_t)nq * k);
    r.distances.resize((std::size_t)nq * k);
    CU_CHECK(cudaMemcpy(r.ids.data(), d_ids, r.ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.distances.data(), d_dist, r.distances.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

    cudaFree(d_q); cudaFree(d_probe); cudaFree(d_cdist);
    cudaFree(d_ids); cudaFree(d_dist);
    return r;
}

SearchResult IvfFlatIndex::search_bruteforce(const std::vector<float>& queries, int nq,
                                             int k, float* elapsed_ms) const {
    if (impl_->n == 0) throw std::logic_error("index is empty");
    if (k <= 0 || k > kMaxK) throw std::invalid_argument("k must be in [1,32]");
    const int dim = impl_->dim;
    if (queries.size() != (std::size_t)nq * dim)
        throw std::invalid_argument("query size mismatch");

    std::vector<float> q = queries;
    if (impl_->cosine()) l2_normalize(q, nq, dim);

    float *d_q = nullptr, *d_dist = nullptr;
    int* d_ids = nullptr;
    CU_CHECK(cudaMalloc(&d_q, q.size() * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_ids, (std::size_t)nq * k * sizeof(int)));
    CU_CHECK(cudaMalloc(&d_dist, (std::size_t)nq * k * sizeof(float)));
    CU_CHECK(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));

    const int T = 128, wpb = T / 32;
    cu::EventTimer t;
    t.start();
    k_search_brute<<<(nq + wpb - 1) / wpb, T>>>(impl_->d_db, impl_->n, dim, d_q, nq, k,
                                                impl_->cosine() ? 1 : 0, d_ids, d_dist);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    SearchResult r;
    r.n_queries = nq;
    r.k = k;
    r.ids.resize((std::size_t)nq * k);
    r.distances.resize((std::size_t)nq * k);
    CU_CHECK(cudaMemcpy(r.ids.data(), d_ids, r.ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.distances.data(), d_dist, r.distances.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

    cudaFree(d_q); cudaFree(d_ids); cudaFree(d_dist);
    return r;
}

// ---------------------------------------------------------------------------
double recall_at_k(const SearchResult& approx, const SearchResult& exact) {
    if (approx.n_queries != exact.n_queries || approx.k != exact.k)
        throw std::invalid_argument("results must have matching shape");
    if (approx.n_queries == 0 || approx.k == 0) return 0.0;

    double total = 0.0;
    for (int q = 0; q < approx.n_queries; ++q) {
        std::unordered_set<int> truth;
        for (int j = 0; j < exact.k; ++j) {
            int id = exact.id_at(q, j);
            if (id >= 0) truth.insert(id);
        }
        if (truth.empty()) continue;
        int hit = 0;
        for (int j = 0; j < approx.k; ++j) {
            if (truth.count(approx.id_at(q, j))) ++hit;
        }
        total += double(hit) / truth.size();
    }
    return total / approx.n_queries;
}

}  // namespace ann
