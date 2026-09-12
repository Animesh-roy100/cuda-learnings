// Spatial indexing -- implementation.
//
// The pipeline exists to avoid the O(N^2) trap: at 2M points a brute-force
// neighbour search is 4x10^12 distance tests. A uniform grid reduces it to
// O(N * points-per-cell).
//
// Morton (Z-order) codes do two jobs at once. They ARE the cell id (bijective
// with cell coordinates), and sorting by them puts spatially-near points near
// each other in memory, so the 27-cell stencil reads land close together.

#include "spatial_index.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <stdexcept>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace spatial {
namespace {

using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr int GRID = SpatialIndex::kGrid;
constexpr int MAX_K = 32;

__host__ __device__ __forceinline__ u32 spread3(u32 v) {
    v &= 0x000003ffu;
    v = (v | (v << 16)) & 0x030000ffu;
    v = (v | (v << 8)) & 0x0300f00fu;
    v = (v | (v << 4)) & 0x030c30c3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}
__host__ __device__ __forceinline__ u32 morton_encode(u32 x, u32 y, u32 z) {
    return spread3(x) | (spread3(y) << 1) | (spread3(z) << 2);
}
__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__device__ __forceinline__ int cell_of(float v) {
    return clampi(static_cast<int>(v * GRID), 0, GRID - 1);
}

__global__ void k_morton(const float3* __restrict__ p, u32* code, u32* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    code[i] = morton_encode(cell_of(p[i].x), cell_of(p[i].y), cell_of(p[i].z));
    idx[i] = i;
}

__global__ void k_reorder(const float3* __restrict__ src, const u32* __restrict__ idx,
                          float3* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// A cell boundary is wherever the sorted code changes.
__global__ void k_cell_ranges(const u32* __restrict__ code, int n,
                              i32* __restrict__ start, i32* __restrict__ end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 c = code[i];
    if (i == 0 || code[i - 1] != c) start[c] = i;
    if (i == n - 1 || code[i + 1] != c) end[c] = i + 1;
}

__global__ void k_count_neighbours(const float3* __restrict__ p, int n,
                                   const i32* __restrict__ start, const i32* __restrict__ end,
                                   float eps2, i32* __restrict__ cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 a = p[i];
    int cx = cell_of(a.x), cy = cell_of(a.y), cz = cell_of(a.z);
    int c = 0;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
                u32 cell = morton_encode(nx, ny, nz);
                for (i32 j = start[cell]; j < end[cell]; ++j) {
                    float3 b = p[j];
                    float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
                    if (ddx * ddx + ddy * ddy + ddz * ddz <= eps2) ++c;
                }
            }
    cnt[i] = c;   // includes the point itself
}

__global__ void k_mark_core(const i32* cnt, unsigned char* is_core, int n, int min_pts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) is_core[i] = (cnt[i] >= min_pts) ? 1 : 0;
}

// Lock-free union-find, same atomicCAS discipline as the hash table.
__device__ __forceinline__ int find_root(i32* parent, int i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];      // path halving
        i = parent[i];
    }
    return i;
}
__device__ __forceinline__ void unite(i32* parent, int a, int b) {
    while (true) {
        a = find_root(parent, a);
        b = find_root(parent, b);
        if (a == b) return;
        if (a > b) { int t = a; a = b; b = t; }      // attach to the lower id
        if (atomicCAS(&parent[b], b, a) == b) return;
    }
}

__global__ void k_init_parent(i32* parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) parent[i] = i;
}

__global__ void k_union_cores(const float3* __restrict__ p, int n,
                              const i32* __restrict__ start, const i32* __restrict__ end,
                              const unsigned char* __restrict__ is_core,
                              float eps2, i32* __restrict__ parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !is_core[i]) return;
    float3 a = p[i];
    int cx = cell_of(a.x), cy = cell_of(a.y), cz = cell_of(a.z);
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
                u32 cell = morton_encode(nx, ny, nz);
                for (i32 j = start[cell]; j < end[cell]; ++j) {
                    if (j <= i || !is_core[j]) continue;     // visit each pair once
                    float3 b = p[j];
                    float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
                    if (ddx * ddx + ddy * ddy + ddz * ddz <= eps2) unite(parent, i, j);
                }
            }
}

__global__ void k_attach_border(const float3* __restrict__ p, int n,
                                const i32* __restrict__ start, const i32* __restrict__ end,
                                const unsigned char* __restrict__ is_core,
                                float eps2, i32* __restrict__ parent, i32* __restrict__ label) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (is_core[i]) { label[i] = find_root(parent, i); return; }

    float3 a = p[i];
    int cx = cell_of(a.x), cy = cell_of(a.y), cz = cell_of(a.z);
    int best = SpatialIndex::kNoise;
    for (int dz = -1; dz <= 1 && best < 0; ++dz)
        for (int dy = -1; dy <= 1 && best < 0; ++dy)
            for (int dx = -1; dx <= 1 && best < 0; ++dx) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
                u32 cell = morton_encode(nx, ny, nz);
                for (i32 j = start[cell]; j < end[cell]; ++j) {
                    if (!is_core[j]) continue;
                    float3 b = p[j];
                    float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
                    if (ddx * ddx + ddy * ddy + ddz * ddz <= eps2) {
                        best = find_root(parent, j);
                        break;
                    }
                }
            }
    label[i] = best;
}

__device__ __forceinline__ void knn_insert(float* best, int k, float d2) {
    if (d2 >= best[k - 1]) return;
    int m = k - 1;
    while (m > 0 && best[m - 1] > d2) { best[m] = best[m - 1]; --m; }
    best[m] = d2;
}

__global__ void k_knn_grid(const float3* __restrict__ p, int n,
                           const i32* __restrict__ start, const i32* __restrict__ end,
                           const i32* __restrict__ queries, int nq, int k,
                           float* __restrict__ out) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int qi = queries[q];
    float3 a = p[qi];
    int cx = cell_of(a.x), cy = cell_of(a.y), cz = cell_of(a.z);

    float best[MAX_K];
    for (int i = 0; i < k; ++i) best[i] = 3.4e38f;

    // Expand in shells until the k-th distance is inside the radius already
    // guaranteed covered. Stopping at a fixed radius would silently return
    // wrong neighbours in sparse regions.
    for (int r = 1; r <= GRID; ++r) {
        for (int dz = -r; dz <= r; ++dz)
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    if (r > 1 && abs(dx) < r && abs(dy) < r && abs(dz) < r) continue;  // shell only
                    int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
                    u32 cell = morton_encode(nx, ny, nz);
                    for (i32 j = start[cell]; j < end[cell]; ++j) {
                        if (j == qi) continue;
                        float3 b = p[j];
                        float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
                        knn_insert(best, k, ddx * ddx + ddy * ddy + ddz * ddz);
                    }
                }
        float safe = r * SpatialIndex::kCell;
        if (best[k - 1] < safe * safe) break;
        // Also stop once the shell has left the grid entirely.
        if (cx - r < 0 && cy - r < 0 && cz - r < 0 &&
            cx + r >= GRID && cy + r >= GRID && cz + r >= GRID) break;
    }
    for (int i = 0; i < k; ++i) out[q * k + i] = best[i];
}

__global__ void k_knn_brute(const float3* __restrict__ p, int n,
                            const i32* __restrict__ queries, int nq, int k,
                            float* __restrict__ out) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int qi = queries[q];
    float3 a = p[qi];
    float best[MAX_K];
    for (int i = 0; i < k; ++i) best[i] = 3.4e38f;
    for (int j = 0; j < n; ++j) {
        if (j == qi) continue;
        float3 b = p[j];
        float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
        knn_insert(best, k, ddx * ddx + ddy * ddy + ddz * ddz);
    }
    for (int i = 0; i < k; ++i) out[q * k + i] = best[i];
}

}  // namespace

// ---------------------------------------------------------------------------
struct SpatialIndex::Impl {
    int n = 0;
    u32 code_space = 0;
    float3* d_sorted = nullptr;
    u32* d_code = nullptr;
    u32* d_idx = nullptr;
    i32* d_start = nullptr;
    i32* d_end = nullptr;
    BuildStats stats;

    // Owns every resource, so a constructor that throws partway through
    // releases what it had already acquired. A class destructor never runs
    // for an object whose constructor threw. Every release is null-safe.
    ~Impl() {
        cudaFree(d_sorted);
        cudaFree(d_code);
        cudaFree(d_idx);
        cudaFree(d_start);
        cudaFree(d_end);
    }
};

SpatialIndex::SpatialIndex(const std::vector<Point3>& points) : impl_(new Impl) {
    float3* d_raw = nullptr;   // construction scratch, freed on every exit path
    try {
        impl_->n = static_cast<int>(points.size());
        impl_->code_space = morton_encode(GRID - 1, GRID - 1, GRID - 1) + 1;
        const int n = impl_->n;
        if (n == 0) return;

        CU_CHECK(cudaMalloc(&d_raw, sizeof(float3) * n));
        CU_CHECK(cudaMalloc(&impl_->d_sorted, sizeof(float3) * n));
        CU_CHECK(cudaMalloc(&impl_->d_code, sizeof(u32) * n));
        CU_CHECK(cudaMalloc(&impl_->d_idx, sizeof(u32) * n));
        CU_CHECK(cudaMalloc(&impl_->d_start, sizeof(i32) * impl_->code_space));
        CU_CHECK(cudaMalloc(&impl_->d_end, sizeof(i32) * impl_->code_space));
        CU_CHECK(cudaMemcpy(d_raw, points.data(), sizeof(float3) * n, cudaMemcpyHostToDevice));

        const int T = 256, B = (n + T - 1) / T;
        cu::EventTimer t;

        t.start();
        k_morton<<<B, T>>>(d_raw, impl_->d_code, impl_->d_idx, n);
        CU_CHECK_KERNEL();
        impl_->stats.morton_ms = t.stop();

        t.start();
        thrust::sort_by_key(thrust::device_ptr<u32>(impl_->d_code),
                            thrust::device_ptr<u32>(impl_->d_code + n),
                            thrust::device_ptr<u32>(impl_->d_idx));
        CU_CHECK(cudaDeviceSynchronize());
        impl_->stats.sort_ms = t.stop();

        t.start();
        k_reorder<<<B, T>>>(d_raw, impl_->d_idx, impl_->d_sorted, n);
        CU_CHECK(cudaMemset(impl_->d_start, 0, sizeof(i32) * impl_->code_space));
        CU_CHECK(cudaMemset(impl_->d_end, 0, sizeof(i32) * impl_->code_space));
        k_cell_ranges<<<B, T>>>(impl_->d_code, n, impl_->d_start, impl_->d_end);
        CU_CHECK_KERNEL();
        impl_->stats.ranges_ms = t.stop();

        cudaFree(d_raw);
    } catch (...) {
        cudaFree(d_raw);
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

SpatialIndex::~SpatialIndex() { delete impl_; }

BuildStats SpatialIndex::build_stats() const { return impl_->stats; }
i32 SpatialIndex::size() const { return impl_->n; }
u32 SpatialIndex::morton3(u32 x, u32 y, u32 z) { return morton_encode(x, y, z); }

std::vector<Point3> SpatialIndex::sorted_points() const {
    std::vector<Point3> out(impl_->n);
    if (impl_->n)
        CU_CHECK(cudaMemcpy(out.data(), impl_->d_sorted, sizeof(float3) * impl_->n,
                            cudaMemcpyDeviceToHost));
    return out;
}

std::vector<i32> SpatialIndex::permutation() const {
    std::vector<i32> out(impl_->n);
    if (impl_->n)
        CU_CHECK(cudaMemcpy(out.data(), impl_->d_idx, sizeof(i32) * impl_->n,
                            cudaMemcpyDeviceToHost));
    return out;
}

std::vector<float> SpatialIndex::knn(const std::vector<i32>& queries, int k,
                                     float* elapsed_ms) const {
    if (k <= 0 || k > MAX_K) throw std::invalid_argument("k must be in [1,32]");
    const int nq = static_cast<int>(queries.size());
    std::vector<float> out(static_cast<std::size_t>(nq) * k);
    if (nq == 0 || impl_->n == 0) return out;

    i32* d_q = nullptr;
    float* d_o = nullptr;
    CU_CHECK(cudaMalloc(&d_q, sizeof(i32) * nq));
    CU_CHECK(cudaMalloc(&d_o, sizeof(float) * nq * k));
    CU_CHECK(cudaMemcpy(d_q, queries.data(), sizeof(i32) * nq, cudaMemcpyHostToDevice));

    const int T = 256, B = (nq + T - 1) / T;
    cu::EventTimer t;
    t.start();
    k_knn_grid<<<B, T>>>(impl_->d_sorted, impl_->n, impl_->d_start, impl_->d_end,
                         d_q, nq, k, d_o);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    CU_CHECK(cudaMemcpy(out.data(), d_o, sizeof(float) * nq * k, cudaMemcpyDeviceToHost));
    cudaFree(d_q);
    cudaFree(d_o);
    return out;
}

std::vector<float> SpatialIndex::knn_bruteforce(const std::vector<i32>& queries, int k,
                                                float* elapsed_ms) const {
    if (k <= 0 || k > MAX_K) throw std::invalid_argument("k must be in [1,32]");
    const int nq = static_cast<int>(queries.size());
    std::vector<float> out(static_cast<std::size_t>(nq) * k);
    if (nq == 0 || impl_->n == 0) return out;

    i32* d_q = nullptr;
    float* d_o = nullptr;
    CU_CHECK(cudaMalloc(&d_q, sizeof(i32) * nq));
    CU_CHECK(cudaMalloc(&d_o, sizeof(float) * nq * k));
    CU_CHECK(cudaMemcpy(d_q, queries.data(), sizeof(i32) * nq, cudaMemcpyHostToDevice));

    const int T = 256, B = (nq + T - 1) / T;
    cu::EventTimer t;
    t.start();
    k_knn_brute<<<B, T>>>(impl_->d_sorted, impl_->n, d_q, nq, k, d_o);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    CU_CHECK(cudaMemcpy(out.data(), d_o, sizeof(float) * nq * k, cudaMemcpyDeviceToHost));
    cudaFree(d_q);
    cudaFree(d_o);
    return out;
}

DbscanResult SpatialIndex::dbscan(float eps, int min_pts) const {
    if (eps > kCell)
        throw std::invalid_argument("eps must be <= cell size; the 3x3x3 stencil "
                                    "only covers one cell of radius");
    DbscanResult res;
    const int n = impl_->n;
    if (n == 0) return res;

    i32 *d_cnt = nullptr, *d_parent = nullptr, *d_label = nullptr;
    unsigned char* d_core = nullptr;
    CU_CHECK(cudaMalloc(&d_cnt, sizeof(i32) * n));
    CU_CHECK(cudaMalloc(&d_parent, sizeof(i32) * n));
    CU_CHECK(cudaMalloc(&d_label, sizeof(i32) * n));
    CU_CHECK(cudaMalloc(&d_core, n));

    const int T = 256, B = (n + T - 1) / T;
    const float eps2 = eps * eps;
    cu::EventTimer t;

    t.start();
    k_count_neighbours<<<B, T>>>(impl_->d_sorted, n, impl_->d_start, impl_->d_end, eps2, d_cnt);
    CU_CHECK_KERNEL();
    res.count_ms = t.stop();

    t.start();
    k_mark_core<<<B, T>>>(d_cnt, d_core, n, min_pts);
    k_init_parent<<<B, T>>>(d_parent, n);
    k_union_cores<<<B, T>>>(impl_->d_sorted, n, impl_->d_start, impl_->d_end,
                            d_core, eps2, d_parent);
    CU_CHECK_KERNEL();
    res.union_ms = t.stop();

    t.start();
    k_attach_border<<<B, T>>>(impl_->d_sorted, n, impl_->d_start, impl_->d_end,
                              d_core, eps2, d_parent, d_label);
    CU_CHECK_KERNEL();
    res.border_ms = t.stop();

    std::vector<i32> sorted_labels(n);
    std::vector<unsigned char> core(n);
    CU_CHECK(cudaMemcpy(sorted_labels.data(), d_label, sizeof(i32) * n, cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(core.data(), d_core, n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        res.core_points += core[i];
        if (sorted_labels[i] == kNoise) ++res.noise_points;
    }

    // Everything above works in Morton order. Returning labels in that order
    // would be a silent trap: labels[i] would describe some other point than
    // the caller's points[i]. Scatter back through the permutation so the
    // result is indexed exactly like the input.
    std::vector<i32> perm(n);
    CU_CHECK(cudaMemcpy(perm.data(), impl_->d_idx, sizeof(i32) * n, cudaMemcpyDeviceToHost));
    res.labels.assign(n, kNoise);
    for (int i = 0; i < n; ++i) res.labels[perm[i]] = sorted_labels[i];

    // Compact root ids into dense cluster ids 0..C-1.
    std::vector<i32> roots;
    roots.reserve(1024);
    for (int i = 0; i < n; ++i)
        if (res.labels[i] != kNoise) roots.push_back(res.labels[i]);
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
    res.num_clusters = static_cast<i32>(roots.size());
    for (int i = 0; i < n; ++i) {
        if (res.labels[i] == kNoise) continue;
        res.labels[i] = static_cast<i32>(
            std::lower_bound(roots.begin(), roots.end(), res.labels[i]) - roots.begin());
    }

    cudaFree(d_cnt);
    cudaFree(d_parent);
    cudaFree(d_label);
    cudaFree(d_core);
    return res;
}

}  // namespace spatial
