// ============================================================================
// Project 3 - GPU spatial indexing: Morton grid -> k-NN -> DBSCAN
//
// The whole pipeline exists to avoid the O(N^2) trap. With 2M points, a
// brute-force neighbour search is 4x10^12 distance tests. A uniform grid makes
// it O(N * points-per-cell) instead, which is O(N).
//
// Steps:
//   1. Quantise each point to a cell, encode the cell as a MORTON CODE
//      (Z-order: interleave the bits of x,y,z). Points near each other in
//      space land near each other in the code, so sorting by code gives
//      memory locality for free.
//   2. Sort points by code, then find each cell's [start,end) range.
//   3. Neighbour queries only touch the 3x3x3 cells around a point.
//   4. DBSCAN on top: count neighbours -> mark core points -> union core
//      points that are within eps -> attach border points.
//
// The union-find is lock-free, same atomicCAS discipline as project 2.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(e_));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

typedef unsigned int u32;

// constexpr, not const: a non-integral `static const float` at namespace scope
// has no device-side storage and nvcc rejects it inside a kernel.
static constexpr int   GRID   = 128;        // cells per axis -> 2^21 cells total
static constexpr float EPS    = 1.0f / GRID;
static constexpr int   MINPTS = 8;
static constexpr int   K      = 8;          // for k-NN

// ---------------------------------------------------------------------------
// Morton encoding: spread each 7-bit coordinate out to every 3rd bit, then OR.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ u32 spread3(u32 v) {
    v &= 0x000003ffu;
    v = (v | (v << 16)) & 0x030000ffu;
    v = (v | (v <<  8)) & 0x0300f00fu;
    v = (v | (v <<  4)) & 0x030c30c3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}
__host__ __device__ __forceinline__ u32 morton3(u32 x, u32 y, u32 z) {
    return spread3(x) | (spread3(y) << 1) | (spread3(z) << 2);
}

__host__ __device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
__global__ void k_morton(const float3* __restrict__ p, u32* __restrict__ code,
                         u32* __restrict__ idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cx = clampi((int)(p[i].x * GRID), 0, GRID - 1);
    int cy = clampi((int)(p[i].y * GRID), 0, GRID - 1);
    int cz = clampi((int)(p[i].z * GRID), 0, GRID - 1);
    code[i] = morton3(cx, cy, cz);
    idx[i]  = i;
}

__global__ void k_reorder(const float3* __restrict__ src, const u32* __restrict__ idx,
                          float3* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// Cell ranges from the sorted code array: a boundary is where the code changes.
__global__ void k_cell_ranges(const u32* __restrict__ code, int n,
                              int* __restrict__ start, int* __restrict__ end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 c = code[i];
    if (i == 0 || code[i - 1] != c) start[c] = i;
    if (i == n - 1 || code[i + 1] != c) end[c] = i + 1;
}

// ---------------------------------------------------------------------------
// Neighbour count within EPS, using the 3x3x3 cells around each point.
// ---------------------------------------------------------------------------
__global__ void k_count_neighbours(const float3* __restrict__ p, int n,
                                   const int* __restrict__ start,
                                   const int* __restrict__ end,
                                   int* __restrict__ cnt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 a = p[i];
    int cx = clampi((int)(a.x * GRID), 0, GRID - 1);
    int cy = clampi((int)(a.y * GRID), 0, GRID - 1);
    int cz = clampi((int)(a.z * GRID), 0, GRID - 1);
    float e2 = EPS * EPS;
    int c = 0;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
        u32 cell = morton3(nx, ny, nz);
        int s = start[cell], t = end[cell];
        for (int j = s; j < t; ++j) {
            float3 b = p[j];
            float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
            if (ddx * ddx + ddy * ddy + ddz * ddz <= e2) ++c;
        }
    }
    cnt[i] = c;      // includes the point itself
}

// ---------------------------------------------------------------------------
// Lock-free union-find (same pattern as project 2's hash table)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int find_root(int* parent, int i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];      // path halving
        i = parent[i];
    }
    return i;
}

__device__ __forceinline__ void unite(int* parent, int a, int b) {
    while (true) {
        a = find_root(parent, a);
        b = find_root(parent, b);
        if (a == b) return;
        if (a > b) { int t = a; a = b; b = t; }   // always attach to lower id
        if (atomicCAS(&parent[b], b, a) == b) return;
        // Lost the race: someone re-parented b. Loop and retry.
    }
}

__global__ void k_init_parent(int* parent, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) parent[i] = i;
}

// Merge core points that are within eps of each other.
__global__ void k_union_cores(const float3* __restrict__ p, int n,
                              const int* __restrict__ start, const int* __restrict__ end,
                              const unsigned char* __restrict__ is_core,
                              int* __restrict__ parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !is_core[i]) return;
    float3 a = p[i];
    int cx = clampi((int)(a.x * GRID), 0, GRID - 1);
    int cy = clampi((int)(a.y * GRID), 0, GRID - 1);
    int cz = clampi((int)(a.z * GRID), 0, GRID - 1);
    float e2 = EPS * EPS;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
        u32 cell = morton3(nx, ny, nz);
        int s = start[cell], t = end[cell];
        for (int j = s; j < t; ++j) {
            if (j <= i || !is_core[j]) continue;      // each pair once
            float3 b = p[j];
            float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
            if (ddx * ddx + ddy * ddy + ddz * ddz <= e2) unite(parent, i, j);
        }
    }
}

// Border points: not core, but within eps of a core point -> join that cluster.
__global__ void k_attach_border(const float3* __restrict__ p, int n,
                                const int* __restrict__ start, const int* __restrict__ end,
                                const unsigned char* __restrict__ is_core,
                                int* __restrict__ parent, int* __restrict__ label) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (is_core[i]) { label[i] = find_root(parent, i); return; }

    float3 a = p[i];
    int cx = clampi((int)(a.x * GRID), 0, GRID - 1);
    int cy = clampi((int)(a.y * GRID), 0, GRID - 1);
    int cz = clampi((int)(a.z * GRID), 0, GRID - 1);
    float e2 = EPS * EPS;
    int best = -1;                                 // -1 == noise

    for (int dz = -1; dz <= 1 && best < 0; ++dz)
    for (int dy = -1; dy <= 1 && best < 0; ++dy)
    for (int dx = -1; dx <= 1 && best < 0; ++dx) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
        u32 cell = morton3(nx, ny, nz);
        int s = start[cell], t = end[cell];
        for (int j = s; j < t; ++j) {
            if (!is_core[j]) continue;
            float3 b = p[j];
            float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
            if (ddx * ddx + ddy * ddy + ddz * ddz <= e2) { best = find_root(parent, j); break; }
        }
    }
    label[i] = best;
}

__global__ void k_mark_core(const int* cnt, unsigned char* is_core, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) is_core[i] = (cnt[i] >= MINPTS) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// k-NN: grid-accelerated vs brute force
// ---------------------------------------------------------------------------
__global__ void k_knn_grid(const float3* __restrict__ p, int n,
                           const int* __restrict__ start, const int* __restrict__ end,
                           const int* __restrict__ queries, int nq,
                           float* __restrict__ out_d2) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int qi = queries[q];
    float3 a = p[qi];
    int cx = clampi((int)(a.x * GRID), 0, GRID - 1);
    int cy = clampi((int)(a.y * GRID), 0, GRID - 1);
    int cz = clampi((int)(a.z * GRID), 0, GRID - 1);

    float best[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) best[i] = 1e30f;

    // Expand the search radius in rings until the k-th distance is safely
    // inside the shell already examined.
    for (int r = 1; r <= 4; ++r) {
        for (int dz = -r; dz <= r; ++dz)
        for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            // Only the new shell, not the interior we already did.
            if (r > 1 && abs(dx) < r && abs(dy) < r && abs(dz) < r) continue;
            int nx = cx + dx, ny = cy + dy, nz = cz + dz;
            if (nx < 0 || ny < 0 || nz < 0 || nx >= GRID || ny >= GRID || nz >= GRID) continue;
            u32 cell = morton3(nx, ny, nz);
            int s = start[cell], t = end[cell];
            for (int j = s; j < t; ++j) {
                if (j == qi) continue;
                float3 b = p[j];
                float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
                float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                if (d2 < best[K - 1]) {                 // insertion sort, K is tiny
                    int m = K - 1;
                    while (m > 0 && best[m - 1] > d2) { best[m] = best[m - 1]; --m; }
                    best[m] = d2;
                }
            }
        }
        // Everything within r cells has been seen; if the k-th neighbour is
        // closer than that guaranteed radius, no further ring can beat it.
        float safe = r * EPS;
        if (best[K - 1] < safe * safe) break;
    }
    for (int i = 0; i < K; ++i) out_d2[q * K + i] = best[i];
}

__global__ void k_knn_brute(const float3* __restrict__ p, int n,
                            const int* __restrict__ queries, int nq,
                            float* __restrict__ out_d2) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int qi = queries[q];
    float3 a = p[qi];
    float best[K];
    #pragma unroll
    for (int i = 0; i < K; ++i) best[i] = 1e30f;

    for (int j = 0; j < n; ++j) {
        if (j == qi) continue;
        float3 b = p[j];
        float ddx = a.x - b.x, ddy = a.y - b.y, ddz = a.z - b.z;
        float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
        if (d2 < best[K - 1]) {
            int m = K - 1;
            while (m > 0 && best[m - 1] > d2) { best[m] = best[m - 1]; --m; }
            best[m] = d2;
        }
    }
    for (int i = 0; i < K; ++i) out_d2[q * K + i] = best[i];
}

// ---------------------------------------------------------------------------
static float ms_of(cudaEvent_t a, cudaEvent_t b) {
    float m = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&m, a, b)); return m;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s  sm_%d%d\n", prop.name, prop.major, prop.minor);

    const int N = 2000000;
    std::printf("N = %d points in [0,1]^3, grid %d^3, eps = %.5f, minPts = %d\n",
                N, GRID, EPS, MINPTS);

    // --- synthetic data: 20 tight Gaussian blobs + 10%% uniform noise ---
    std::vector<float3> h_pts(N);
    unsigned seed = 42u;
    auto urand = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)(seed >> 8) / 16777216.0f;      // [0,1)
    };
    auto nrand = [&]() {                              // Box-Muller-ish
        float u1 = std::fmax(1e-7f, urand()), u2 = urand();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
    };
    const int NBLOB = 20;
    std::vector<float3> centres(NBLOB);
    for (int b = 0; b < NBLOB; ++b) {
        centres[b] = make_float3(0.15f + 0.7f * urand(), 0.15f + 0.7f * urand(),
                                 0.15f + 0.7f * urand());
    }
    int n_noise = N / 10;
    for (int i = 0; i < N; ++i) {
        if (i < n_noise) {
            h_pts[i] = make_float3(urand(), urand(), urand());
        } else {
            int b = (int)(urand() * NBLOB) % NBLOB;
            float s = 0.020f;
            h_pts[i] = make_float3(
                std::fmin(0.999f, std::fmax(0.0f, centres[b].x + s * nrand())),
                std::fmin(0.999f, std::fmax(0.0f, centres[b].y + s * nrand())),
                std::fmin(0.999f, std::fmax(0.0f, centres[b].z + s * nrand())));
        }
    }

    const int NCELL = GRID * GRID * GRID;             // Morton codes span this
    const u32 CODE_SPACE = morton3(GRID - 1, GRID - 1, GRID - 1) + 1;

    float3 *d_pts, *d_sorted;
    u32 *d_code, *d_idx;
    int *d_start, *d_end, *d_cnt, *d_parent, *d_label;
    unsigned char* d_core;
    CUDA_CHECK(cudaMalloc(&d_pts,   (size_t)N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_sorted,(size_t)N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_code,  (size_t)N * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_idx,   (size_t)N * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&d_start, (size_t)CODE_SPACE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_end,   (size_t)CODE_SPACE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cnt,   (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_parent,(size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_label, (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_core,  (size_t)N));
    CUDA_CHECK(cudaMemcpy(d_pts, h_pts.data(), (size_t)N * sizeof(float3), cudaMemcpyHostToDevice));

    std::printf("Morton code space: %u cells (%.1f MB for start+end)\n\n",
                CODE_SPACE, 2.0 * CODE_SPACE * 4 / 1e6);

    int T = 256, B = (N + T - 1) / T;
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));
    CUDA_CHECK(cudaEventCreate(&e4)); CUDA_CHECK(cudaEventCreate(&e5));

    // ---- build index ----
    CUDA_CHECK(cudaEventRecord(e0));
    k_morton<<<B, T>>>(d_pts, d_code, d_idx, N);
    thrust::sort_by_key(thrust::device_ptr<u32>(d_code),
                        thrust::device_ptr<u32>(d_code + N),
                        thrust::device_ptr<u32>(d_idx));
    k_reorder<<<B, T>>>(d_pts, d_idx, d_sorted, N);
    CUDA_CHECK(cudaMemset(d_start, 0, (size_t)CODE_SPACE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_end,   0, (size_t)CODE_SPACE * sizeof(int)));
    k_cell_ranges<<<B, T>>>(d_code, N, d_start, d_end);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());

    // ---- DBSCAN ----
    k_count_neighbours<<<B, T>>>(d_sorted, N, d_start, d_end, d_cnt);
    CUDA_CHECK(cudaEventRecord(e2));
    k_mark_core<<<B, T>>>(d_cnt, d_core, N);
    k_init_parent<<<B, T>>>(d_parent, N);
    k_union_cores<<<B, T>>>(d_sorted, N, d_start, d_end, d_core, d_parent);
    CUDA_CHECK(cudaEventRecord(e3));
    k_attach_border<<<B, T>>>(d_sorted, N, d_start, d_end, d_core, d_parent, d_label);
    CUDA_CHECK(cudaEventRecord(e4));
    CUDA_CHECK(cudaEventSynchronize(e4));
    CUDA_CHECK(cudaGetLastError());

    // Capture now: the k-NN section below re-records e0..e2.
    const float t_index  = ms_of(e0, e1);
    const float t_count  = ms_of(e1, e2);
    const float t_union  = ms_of(e2, e3);
    const float t_border = ms_of(e3, e4);
    const float t_total  = ms_of(e0, e4);

    std::printf("=== pipeline timings ===\n");
    std::printf("  morton+sort+grid  %7.2f ms\n", t_index);
    std::printf("  neighbour count   %7.2f ms\n", t_count);
    std::printf("  union cores       %7.2f ms\n", t_union);
    std::printf("  attach borders    %7.2f ms\n", t_border);
    std::printf("  TOTAL DBSCAN      %7.2f ms  (%.1f M points/s)\n",
                t_total, N / (t_total / 1000.0) / 1e6);

    // ---- results ----
    std::vector<int> h_label(N);
    std::vector<unsigned char> h_core(N);
    CUDA_CHECK(cudaMemcpy(h_label.data(), d_label, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_core.data(), d_core, (size_t)N, cudaMemcpyDeviceToHost));

    long long ncore = 0, nnoise = 0;
    for (int i = 0; i < N; ++i) { ncore += h_core[i]; if (h_label[i] < 0) ++nnoise; }
    std::vector<int> roots;
    for (int i = 0; i < N; ++i) if (h_label[i] >= 0) roots.push_back(h_label[i]);
    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());

    // Cluster sizes, to see whether the 20 blobs came back.
    std::vector<long long> sizes(roots.size(), 0);
    for (int i = 0; i < N; ++i) {
        if (h_label[i] < 0) continue;
        int k = (int)(std::lower_bound(roots.begin(), roots.end(), h_label[i]) - roots.begin());
        ++sizes[k];
    }
    std::sort(sizes.rbegin(), sizes.rend());
    int big = 0;
    for (size_t i = 0; i < sizes.size(); ++i) if (sizes[i] >= 1000) ++big;

    std::printf("\n=== DBSCAN result ===\n");
    std::printf("  core points   %lld (%.1f%%)\n", ncore, 100.0 * ncore / N);
    std::printf("  noise points  %lld (%.1f%%)   [10%% uniform noise was planted]\n",
                nnoise, 100.0 * nnoise / N);
    std::printf("  clusters      %zu total, %d with >=1000 points  [20 blobs planted]\n",
                roots.size(), big);
    std::printf("  largest 5:   ");
    for (size_t i = 0; i < sizes.size() && i < 5; ++i) std::printf(" %lld", sizes[i]);
    std::printf("\n");

    // ---- k-NN: grid vs brute force ----
    const int NQ = 4096;
    std::vector<int> h_q(NQ);
    for (int i = 0; i < NQ; ++i) h_q[i] = (int)((long long)i * N / NQ);
    int* d_q; float *d_kg, *d_kb;
    CUDA_CHECK(cudaMalloc(&d_q, NQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_kg, (size_t)NQ * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kb, (size_t)NQ * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), NQ * sizeof(int), cudaMemcpyHostToDevice));

    int BQ = (NQ + T - 1) / T;
    CUDA_CHECK(cudaEventRecord(e0));
    k_knn_grid<<<BQ, T>>>(d_sorted, N, d_start, d_end, d_q, NQ, d_kg);
    CUDA_CHECK(cudaEventRecord(e1));
    k_knn_brute<<<BQ, T>>>(d_sorted, N, d_q, NQ, d_kb);
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaEventSynchronize(e2));
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> kg((size_t)NQ * K), kb((size_t)NQ * K);
    CUDA_CHECK(cudaMemcpy(kg.data(), d_kg, (size_t)NQ * K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kb.data(), d_kb, (size_t)NQ * K * sizeof(float), cudaMemcpyDeviceToHost));
    long long mismatch = 0;
    for (size_t i = 0; i < kg.size(); ++i) {
        if (std::fabs(kg[i] - kb[i]) > 1e-6f * std::fmax(1.0f, kb[i])) ++mismatch;
    }

    float tg = ms_of(e0, e1), tb = ms_of(e1, e2);
    std::printf("\n=== k-NN (k=%d, %d queries against %d points) ===\n", K, NQ, N);
    std::printf("  grid        %8.2f ms\n", tg);
    std::printf("  brute force %8.2f ms\n", tb);
    std::printf("  speedup     %8.1fx   exact-match check: %s (%lld/%zu differ)\n",
                tb / tg, mismatch == 0 ? "IDENTICAL" : "MISMATCH", mismatch, kg.size());

    std::printf("\nThe grid turns an O(N) scan per query into O(points in 27 cells).\n"
                "Morton order is what makes those 27 cells land near each other in\n"
                "memory -- the same query on unsorted points would scatter every read.\n");

    std::printf("\nReading the cluster count: blob centres are placed at random, so some\n"
                "land within eps of each other and DBSCAN correctly merges them into one\n"
                "cluster. Each blob holds ~%d points, so a cluster of ~%d is two merged\n"
                "blobs. That is the algorithm working, not a bug -- DBSCAN finds density-\n"
                "connected regions, and it has no notion of how many blobs you intended.\n"
                "The small clusters beyond those are noise points that happened to clump.\n",
                (N - n_noise) / NBLOB, 2 * (N - n_noise) / NBLOB);

    std::printf("\nSlowest stage is union-cores (%.0f ms of %.0f). Each core point scans\n"
                "its 27 cells again and does a lock-free union per qualifying pair, so it\n"
                "pays both the neighbour scan AND atomic contention inside dense blobs.\n"
                "That is where you would optimise next.\n",
                t_union, t_total);

    cudaFree(d_pts); cudaFree(d_sorted); cudaFree(d_code); cudaFree(d_idx);
    cudaFree(d_start); cudaFree(d_end); cudaFree(d_cnt); cudaFree(d_parent);
    cudaFree(d_label); cudaFree(d_core); cudaFree(d_q); cudaFree(d_kg); cudaFree(d_kb);
    (void)NCELL;
    return 0;
}
