// Memory layout and the advanced execution subsystems, each measured rather
// than assumed. Several of these features are documented as "supported on
// sm_75" while the thing that makes them worth using is not -- so every one
// reports what actually happened on this card.

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cu/check.hpp"
#include "cu/device.hpp"
#include "cu/timer.hpp"
#include "layout_advanced.h"

namespace cg = cooperative_groups;

namespace la {
namespace {

// A device buffer that frees itself. Every function here allocates a handful
// of arrays and can throw from CU_CHECK partway through.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_) CU_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void upload(const T* host) { CU_CHECK(cudaMemcpy(ptr_, host, bytes(), cudaMemcpyHostToDevice)); }
    void download(T* host) const { CU_CHECK(cudaMemcpy(host, ptr_, bytes(), cudaMemcpyDeviceToHost)); }
    void zero() { CU_CHECK(cudaMemset(ptr_, 0, bytes())); }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

// ===========================================================================
// 1. AoS vs SoA
// ===========================================================================
//
// Six fields, of which the kernel reads three. That is the case where layout
// decides the answer: with AoS the unread vx/vy/vz still travel from DRAM,
// because the memory system moves 32-byte sectors, not individual floats.
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void aos_kernel(const Particle* __restrict__ p, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Thread i touches bytes [24i, 24i+12). Consecutive threads are 24 bytes
    // apart, so the warp's 32 requests span 768 bytes in order to use 384.
    out[i] = p[i].x * p[i].x + p[i].y * p[i].y + p[i].z * p[i].z;
}

__global__ void soa_kernel(const float* __restrict__ x, const float* __restrict__ y,
                           const float* __restrict__ z, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Three separate streams, each perfectly sequential across the warp.
    out[i] = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
}

// ===========================================================================
// 2. Shared-memory bank conflicts
// ===========================================================================
//
// Shared memory is 32 banks of 4-byte words, addressed word % 32. A warp can
// service one word per bank per cycle; two lanes hitting different rows of the
// same bank are serialised.
constexpr int TILE = 32;

template <int WIDTH>
__global__ void shared_column_kernel(float* __restrict__ out, int iterations) {
    __shared__ float tile[TILE][WIDTH];
    const int tx = threadIdx.x;   // fast dimension: lanes of a warp vary tx
    const int ty = threadIdx.y;

    float acc = 0.0f;
    for (int it = 0; it < iterations; ++it) {
        tile[ty][tx] = float(tx * ty + it);
        __syncthreads();
        // Column-wise read. Address = tx * WIDTH + ty.
        //   WIDTH == 32: bank = (tx*32 + ty) % 32 = ty  -- identical for all 32
        //                lanes, so this one access takes 32 cycles.
        //   WIDTH == 33: bank = (tx*33 + ty) % 32 = (tx + ty) % 32 -- distinct
        //                for every lane, so it takes one.
        acc += tile[tx][ty];
        __syncthreads();
    }
    if (tx == 0 && ty == 0) out[blockIdx.x] = acc;
}

// ===========================================================================
// 3. Cooperative groups
// ===========================================================================

// Block-wide sum built from cg::reduce over 32-lane tiles.
//
// The leading barrier is not optional. This is called twice, and the second
// call would otherwise start writing warp_sums while slower threads from the
// first call were still reading it -- the exact shared-memory reuse race that
// made 07-montecarlo silently wrong.
__device__ float block_reduce_sum(cg::thread_block& block,
                                  cg::thread_block_tile<32>& tile,
                                  float* warp_sums, float value) {
    block.sync();
    const float tile_sum = cg::reduce(tile, value, cg::plus<float>());
    if (tile.thread_rank() == 0) warp_sums[tile.meta_group_rank()] = tile_sum;
    block.sync();

    float total = 0.0f;
    if (tile.meta_group_rank() == 0) {
        const float v = (tile.thread_rank() < int(tile.meta_group_size()))
                            ? warp_sums[tile.thread_rank()]
                            : 0.0f;
        total = cg::reduce(tile, v, cg::plus<float>());
    }
    return total;   // meaningful on thread 0 only
}

__global__ void coop_kernel(const float* __restrict__ in, int n,
                            float* __restrict__ partials,
                            float* __restrict__ demo_tile,
                            float* __restrict__ demo_block,
                            int* __restrict__ grid_size_out,
                            float* __restrict__ total_out) {
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    __shared__ float warp_sums[32];

    // A reduction with a known answer, so the numbers can be read off: every
    // thread contributes 1.0, so the tile sums to 32 and the block to
    // blockDim.x.
    const float one_tile = cg::reduce(tile, 1.0f, cg::plus<float>());
    const float one_block = block_reduce_sum(block, tile, warp_sums, 1.0f);
    if (blockIdx.x == 0 && block.thread_rank() == 0) {
        *demo_tile = one_tile;
        *demo_block = one_block;
    }

    // The real work: a grid-stride partial sum, reduced per block.
    float v = 0.0f;
    for (unsigned i = grid.thread_rank(); i < unsigned(n); i += grid.size()) v += in[i];
    const float block_total = block_reduce_sum(block, tile, warp_sums, v);
    if (block.thread_rank() == 0) partials[blockIdx.x] = block_total;

    // Every block's partial must be visible before any block reads them all.
    // __syncthreads() cannot express this: it synchronises one block. This is
    // the barrier that makes a single-kernel two-phase reduction legal.
    grid.sync();

    if (grid.thread_rank() == 0) {
        float t = 0.0f;
        for (unsigned b = 0; b < gridDim.x; ++b) t += partials[b];
        *total_out = t;
        *grid_size_out = int(grid.size());
    }
}

// ===========================================================================
// 4. Dynamic parallelism
// ===========================================================================
__global__ void dp_child(float* __restrict__ out, int base, int n, float scale) {
    const int i = threadIdx.x;
    if (i < n) out[base + i] = scale * float(i);
}

__global__ void dp_parent(float* __restrict__ out, int child_threads,
                          int* __restrict__ launches) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    // A launch from device code: no host round-trip. Under CUDA 12+ (CDP2) the
    // device-side cudaDeviceSynchronize() is gone, and the guarantee is that
    // the parent grid is not complete -- as the host observes it -- until every
    // child grid it launched has finished.
    dp_child<<<1, child_threads>>>(out, p * child_threads, child_threads, float(p + 1));
    if (cudaGetLastError() == cudaSuccess) atomicAdd(launches, 1);
}

// ===========================================================================
// 5. Tensor Cores (WMMA)
// ===========================================================================
using namespace nvcuda;

constexpr int WM = 16, WN = 16, WK = 16;

// One warp per 16x16 output tile. B is stored transposed so both kernels read
// it the same way and the comparison is about the instruction, not the layout.
__global__ void wmma_gemm(const __half* __restrict__ A, const __half* __restrict__ Bt,
                          float* __restrict__ C, int N) {
    const int warp_m = blockIdx.y * blockDim.y + threadIdx.y;
    const int warp_n = blockIdx.x;
    if (warp_m * WM >= N || warp_n * WN >= N) return;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < N; k += WK) {
        // A 16x16x16 multiply-accumulate as one instruction issued by the whole
        // warp. The fragment is not a normal array: its element ordering is
        // opaque and per-architecture, which is why it has to be loaded and
        // stored through the wmma API rather than indexed.
        wmma::load_matrix_sync(a_frag, A + warp_m * WM * N + k, N);
        wmma::load_matrix_sync(b_frag, Bt + warp_n * WN * N + k, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + warp_m * WM * N + warp_n * WN, c_frag, N, wmma::mem_row_major);
}

// Shared-memory tiled FP32 baseline. Not a naive triple loop: comparing a tuned
// instruction against an untuned kernel would measure the tuning.
__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ Bt,
                            float* __restrict__ C, int N) {
    // Padded to 17 for the reason section 2 demonstrates: Bs is read with the
    // lane-varying index first, which at width 16 would be a 16-way conflict.
    __shared__ float As[16][17];
    __shared__ float Bs[16][17];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * 16 + ty;
    const int col = blockIdx.x * 16 + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += 16) {
        As[ty][tx] = (row < N) ? A[row * N + k0 + tx] : 0.0f;
        Bs[tx][ty] = (col < N) ? Bt[col * N + k0 + ty] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

// The control for the WMMA comparison: the kernel above with exactly one thing
// changed -- the operands are fp16 in memory, widened to float on load. Same
// tiles, same barriers, same fp32 FMAs in the CUDA cores, half the bytes read.
// Whatever this gains over sgemm_tiled is bandwidth, not Tensor Cores.
__global__ void sgemm_tiled_half_in(const __half* __restrict__ A,
                                    const __half* __restrict__ Bt,
                                    float* __restrict__ C, int N) {
    __shared__ float As[16][17];
    __shared__ float Bs[16][17];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * 16 + ty;
    const int col = blockIdx.x * 16 + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += 16) {
        As[ty][tx] = (row < N) ? __half2float(A[row * N + k0 + tx]) : 0.0f;
        Bs[tx][ty] = (col < N) ? __half2float(Bt[col * N + k0 + ty]) : 0.0f;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

// ===========================================================================
// 6. Asynchronous shared-memory copy
// ===========================================================================
constexpr int ASYNC_TILE = 1024;

__global__ void copy_sync_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float s[ASYNC_TILE];
    cg::thread_block block = cg::this_thread_block();

    for (int base = blockIdx.x * ASYNC_TILE; base < n; base += gridDim.x * ASYNC_TILE) {
        const int count = min(ASYNC_TILE, n - base);
        // Every thread issues a load and then everyone waits at the barrier.
        // The load latency is exposed: nothing else can proceed meanwhile.
        for (int t = threadIdx.x; t < count; t += blockDim.x) s[t] = in[base + t];
        block.sync();
        for (int t = threadIdx.x; t < count; t += blockDim.x)
            out[base + t] = s[count - 1 - t] * 2.0f;
        block.sync();
    }
}

__global__ void copy_async_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float s[ASYNC_TILE];
    cg::thread_block block = cg::this_thread_block();

    for (int base = blockIdx.x * ASYNC_TILE; base < n; base += gridDim.x * ASYNC_TILE) {
        const int count = min(ASYNC_TILE, n - base);
        // On sm_80+ this becomes cp.async: DRAM writes straight into shared
        // memory without staging through registers, and the wait is a separate
        // instruction from the issue. On sm_75 it is correct but compiles to
        // the ordinary load-then-barrier above, so no speedup is expected here.
        cg::memcpy_async(block, s, in + base, sizeof(float) * count);
        cg::wait(block);
        for (int t = threadIdx.x; t < count; t += blockDim.x)
            out[base + t] = s[count - 1 - t] * 2.0f;
        block.sync();
    }
}

}  // namespace

// ---------------------------------------------------------------------------

LayoutResult compare_aos_soa(int particles) {
    LayoutResult r;
    if (particles <= 0) return r;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<Particle> h_aos(particles);
    std::vector<float> hx(particles), hy(particles), hz(particles);
    for (int i = 0; i < particles; ++i) {
        hx[i] = dist(rng);
        hy[i] = dist(rng);
        hz[i] = dist(rng);
        h_aos[i] = {hx[i], hy[i], hz[i], dist(rng), dist(rng), dist(rng)};
    }

    DeviceBuffer<Particle> d_aos(particles);
    DeviceBuffer<float> d_x(particles), d_y(particles), d_z(particles);
    DeviceBuffer<float> d_out_aos(particles), d_out_soa(particles);
    d_aos.upload(h_aos.data());
    d_x.upload(hx.data());
    d_y.upload(hy.data());
    d_z.upload(hz.data());

    const int threads = 256;
    const int blocks = (particles + threads - 1) / threads;

    auto aos = cu::benchmark([&] {
        aos_kernel<<<blocks, threads>>>(d_aos.get(), d_out_aos.get(), particles);
    });
    CU_CHECK_KERNEL();
    auto soa = cu::benchmark([&] {
        soa_kernel<<<blocks, threads>>>(d_x.get(), d_y.get(), d_z.get(), d_out_soa.get(), particles);
    });
    CU_CHECK_KERNEL();

    std::vector<float> out_aos(particles), out_soa(particles);
    d_out_aos.download(out_aos.data());
    d_out_soa.download(out_soa.data());
    r.results_match = std::equal(out_aos.begin(), out_aos.end(), out_soa.begin());

    r.aos_ms = aos.median_ms;
    r.soa_ms = soa.median_ms;
    // USEFUL bytes only: 3 input floats plus 1 output float per particle. Both
    // kernels do exactly this much work; AoS additionally drags 12 bytes of
    // vx/vy/vz across the bus that nothing reads, which is why its number lands
    // near half of SoA's rather than near peak.
    const double useful = double(particles) * (3 * sizeof(float) + sizeof(float));
    r.aos_gbps = useful / (r.aos_ms / 1000.0) / 1e9;
    r.soa_gbps = useful / (r.soa_ms / 1000.0) / 1e9;
    return r;
}

PaddingResult compare_shared_padding(int iterations) {
    PaddingResult r;
    if (iterations <= 0) return r;

    const int blocks = 1024;
    DeviceBuffer<float> d_a(blocks), d_b(blocks);
    dim3 block(TILE, TILE);

    auto un = cu::benchmark([&] {
        shared_column_kernel<32><<<blocks, block>>>(d_a.get(), iterations);
    });
    CU_CHECK_KERNEL();
    auto pad = cu::benchmark([&] {
        shared_column_kernel<33><<<blocks, block>>>(d_b.get(), iterations);
    });
    CU_CHECK_KERNEL();

    std::vector<float> a(blocks), b(blocks);
    d_a.download(a.data());
    d_b.download(b.data());
    r.results_match = std::equal(a.begin(), a.end(), b.begin());

    r.unpadded_ms = un.median_ms;
    r.padded_ms = pad.median_ms;
    r.extra_bytes = TILE * sizeof(float);   // one extra column, 32 rows
    return r;
}

CoopGroupsResult exercise_cooperative_groups(int elements) {
    CoopGroupsResult r;
    if (elements <= 0) return r;

    int supported = 0, device = 0;
    CU_CHECK(cudaGetDevice(&device));
    CU_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrCooperativeLaunch, device));
    if (!supported) {
        r.skip_reason = "device reports cudaDevAttrCooperativeLaunch = 0";
        return r;
    }
    r.cooperative_launch_supported = true;

    std::vector<float> h_in(elements, 1.0f);
    DeviceBuffer<float> d_in(elements);
    d_in.upload(h_in.data());

    const int threads = 256;

    // A grid-wide barrier can only be honoured if every block is resident at
    // once -- a block still waiting for a free SM would deadlock the ones
    // already parked at the barrier. So the grid size is not ours to pick: it
    // comes from the occupancy API.
    int blocks_per_sm = 0;
    CU_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, reinterpret_cast<const void*>(coop_kernel), threads, 0));
    int sm_count = 0;
    CU_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    int blocks = blocks_per_sm * sm_count;
    if (blocks < 1) blocks = 1;

    DeviceBuffer<float> d_partials(blocks), d_tile(1), d_block(1), d_total(1);
    DeviceBuffer<int> d_grid_size(1);
    d_partials.zero();
    d_tile.zero();
    d_block.zero();
    d_total.zero();
    d_grid_size.zero();

    float* p_in = d_in.get();
    float* p_part = d_partials.get();
    float* p_tile = d_tile.get();
    float* p_block = d_block.get();
    float* p_total = d_total.get();
    int* p_grid = d_grid_size.get();
    int n = elements;
    void* args[] = {&p_in, &n, &p_part, &p_tile, &p_block, &p_grid, &p_total};

    CU_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(coop_kernel),
                                         dim3(blocks), dim3(threads), args, 0, nullptr));
    CU_CHECK_KERNEL();

    float total = 0.0f;
    d_tile.download(&r.tiled_reduce);
    d_block.download(&r.block_reduce);
    d_total.download(&total);
    d_grid_size.download(&r.grid_size_seen);

    // If grid.sync() had not held, thread 0 would have summed partials that
    // some blocks had not written yet, and the total would fall short.
    r.grid_sync_ok = std::fabs(total - float(elements)) < 1.0f;
    return r;
}

DynamicParallelismResult exercise_dynamic_parallelism(int parents, int child_threads) {
    DynamicParallelismResult r;
    if (parents <= 0 || child_threads <= 0) return r;

    int device = 0, major = 0, minor = 0;
    CU_CHECK(cudaGetDevice(&device));
    CU_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CU_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major * 10 + minor < 35) {
        r.skip_reason = "dynamic parallelism needs compute capability 3.5+";
        return r;
    }
    r.supported = true;

    const int n = parents * child_threads;
    DeviceBuffer<float> d_out(n);
    DeviceBuffer<int> d_launches(1);
    d_out.zero();
    d_launches.zero();

    dp_parent<<<1, parents>>>(d_out.get(), child_threads, d_launches.get());
    CU_CHECK_KERNEL();

    r.output.resize(n);
    d_out.download(r.output.data());
    d_launches.download(&r.child_launches);
    return r;
}

WmmaResult compare_wmma_vs_fp32(int matrix_dim) {
    WmmaResult r;
    if (matrix_dim <= 0 || matrix_dim % 16 != 0) {
        r.skip_reason = "matrix_dim must be a positive multiple of 16";
        return r;
    }

    int device = 0, major = 0, minor = 0;
    CU_CHECK(cudaGetDevice(&device));
    CU_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CU_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    if (major * 10 + minor < 70) {
        r.skip_reason = "WMMA requires compute capability 7.0+";
        return r;
    }

    const int N = matrix_dim;
    r.matrix_dim = N;
    r.ran = true;

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hA(std::size_t(N) * N), hBt(std::size_t(N) * N);
    std::vector<__half> hA16(hA.size()), hBt16(hBt.size());
    for (std::size_t i = 0; i < hA.size(); ++i) {
        hA[i] = dist(rng);
        hBt[i] = dist(rng);
        hA16[i] = __float2half(hA[i]);
        hBt16[i] = __float2half(hBt[i]);
    }

    DeviceBuffer<__half> dA16(hA16.size()), dBt16(hBt16.size());
    DeviceBuffer<float> dA(hA.size()), dBt(hBt.size());
    DeviceBuffer<float> dC_wmma(hA.size()), dC_fp32(hA.size()), dC_half(hA.size());
    dA16.upload(hA16.data());
    dBt16.upload(hBt16.data());
    dA.upload(hA.data());
    dBt.upload(hBt.data());

    const int tiles = N / 16;
    dim3 wmma_block(32, 4);
    dim3 wmma_grid(tiles, (tiles + 3) / 4);
    auto w = cu::benchmark([&] {
        wmma_gemm<<<wmma_grid, wmma_block>>>(dA16.get(), dBt16.get(), dC_wmma.get(), N);
    }, 10, 3);
    CU_CHECK_KERNEL();

    dim3 fp32_block(16, 16);
    dim3 fp32_grid(tiles, tiles);
    auto f = cu::benchmark([&] {
        sgemm_tiled<<<fp32_grid, fp32_block>>>(dA.get(), dBt.get(), dC_fp32.get(), N);
    }, 10, 3);
    CU_CHECK_KERNEL();

    auto h = cu::benchmark([&] {
        sgemm_tiled_half_in<<<fp32_grid, fp32_block>>>(dA16.get(), dBt16.get(), dC_half.get(), N);
    }, 10, 3);
    CU_CHECK_KERNEL();

    r.wmma_ms = w.median_ms;
    r.fp32_ms = f.median_ms;
    r.fp16in_ms = h.median_ms;

    std::vector<float> cw(hA.size()), cf(hA.size());
    dC_wmma.download(cw.data());
    dC_fp32.download(cf.data());
    double worst = 0.0;
    for (std::size_t i = 0; i < cw.size(); ++i)
        worst = std::max(worst, double(std::fabs(cw[i] - cf[i])));
    r.max_abs_error = worst;
    return r;
}

AsyncCopyResult compare_async_copy(int elements) {
    AsyncCopyResult r;
    if (elements <= 0) return r;

    int device = 0, major = 0, minor = 0;
    CU_CHECK(cudaGetDevice(&device));
    CU_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CU_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    r.compute_capability = major * 10 + minor;
    r.hardware_accelerated = r.compute_capability >= 80;

    std::vector<float> h_in(elements);
    for (int i = 0; i < elements; ++i) h_in[i] = float(i % 1000) * 0.5f;

    DeviceBuffer<float> d_in(elements), d_sync(elements), d_async(elements);
    d_in.upload(h_in.data());
    d_sync.zero();
    d_async.zero();

    const int threads = 256;
    const int blocks = std::min(1024, (elements + ASYNC_TILE - 1) / ASYNC_TILE);

    auto s = cu::benchmark([&] {
        copy_sync_kernel<<<blocks, threads>>>(d_in.get(), d_sync.get(), elements);
    });
    CU_CHECK_KERNEL();
    auto a = cu::benchmark([&] {
        copy_async_kernel<<<blocks, threads>>>(d_in.get(), d_async.get(), elements);
    });
    CU_CHECK_KERNEL();

    r.sync_ms = s.median_ms;
    r.async_ms = a.median_ms;

    std::vector<float> os(elements), oa(elements);
    d_sync.download(os.data());
    d_async.download(oa.data());
    r.results_match = std::equal(os.begin(), os.end(), oa.begin());
    return r;
}

}  // namespace la
