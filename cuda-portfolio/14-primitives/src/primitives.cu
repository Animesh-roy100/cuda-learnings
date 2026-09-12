// CUDA primitives, measured in isolation.

#include "primitives.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace prim {
namespace {

constexpr int BANKS = 32;
constexpr int SMEM_FLOATS = BANKS * BANKS;   // 1024 floats = 4 KB

// A deliberately tiny kernel. The arithmetic is irrelevant; the point is that
// it finishes almost instantly, so per-launch overhead dominates and becomes
// measurable.
__global__ void k_tiny(float* __restrict__ d, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = fmaf(d[i], 1.000001f, 1e-7f);
}

// Touch-everything kernel for the memory-mode comparison.
__global__ void k_touch(float* __restrict__ d, std::size_t n, float k) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) d[i] = d[i] * k + 1.0f;
}

__global__ void k_sum(const float* __restrict__ d, std::size_t n,
                      double* __restrict__ out) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
    double s = 0.0;
    for (; i < n; i += stride) s += d[i];

    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    __shared__ double w[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) w[wid] = s;
    __syncthreads();
    if (wid == 0) {
        const int nw = (blockDim.x + 31) / 32;
        double v = (lane < nw) ? w[lane] : 0.0;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// Bank conflicts.
//
// Lane L reads s[(L * STRIDE + it) % 1024], so lanes are STRIDE apart and land
// on bank (L * STRIDE + it) % 32.
//   STRIDE = 1  -> 32 distinct banks, no conflict
//   STRIDE = 2  -> 16 distinct banks, 2-way
//   STRIDE = 32 -> one bank for the whole warp, 32-way
//
// The `+ it` keeps the index changing so the compiler cannot hoist the load out
// of the loop, while leaving the LANE-TO-LANE stride -- and therefore the
// conflict degree -- untouched.
// ---------------------------------------------------------------------------
template <int STRIDE>
__global__ void k_bank(float* __restrict__ out, int iterations) {
    __shared__ float s[SMEM_FLOATS];
    const int tid = threadIdx.x;
    for (int i = tid; i < SMEM_FLOATS; i += blockDim.x) s[i] = float(i) * 0.5f;
    __syncthreads();

    float acc = 0.0f;
    for (int it = 0; it < iterations; ++it) {
        acc += s[(tid * STRIDE + it) & (SMEM_FLOATS - 1)];
    }
    // Never true; exists so the optimiser cannot delete the loop.
    if (acc == -12345.0f) out[tid] = acc;
}

__global__ void k_activemask(unsigned* __restrict__ masks, int* __restrict__ branches) {
    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    // Deliberate divergence: the two halves of the warp take different paths,
    // and __activemask reports which lanes are live at that point.
    if ((lane & 1) == 0) {
        masks[lane] = __activemask();
        branches[lane] = 0;
    } else {
        masks[lane] = __activemask();
        branches[lane] = 1;
    }
}

int grid_for(std::size_t n, int block) {
    const std::size_t g = (n + block - 1) / block;
    return (int)std::min<std::size_t>(g, 65535);
}

// CUDA 13 changed cudaMemAdvise and cudaMemPrefetchAsync to take a
// cudaMemLocation STRUCT where CUDA 12 took a plain device ordinal. Code
// written against either signature fails to compile on the other, so both
// spellings live here behind a version check -- which is what lets this build
// on CUDA 13 locally and CUDA 12 on Colab from the same source.
inline cudaError_t mem_advise(const void* p, std::size_t bytes,
                              cudaMemoryAdvise advice, int device) {
#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    if (device == cudaCpuDeviceId) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
    }
    return cudaMemAdvise(p, bytes, advice, loc);
#else
    return cudaMemAdvise(p, bytes, advice, device);
#endif
}

inline cudaError_t mem_prefetch(const void* p, std::size_t bytes, int device,
                                cudaStream_t stream) {
#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;
    return cudaMemPrefetchAsync(p, bytes, loc, 0u, stream);
#else
    return cudaMemPrefetchAsync(p, bytes, device, stream);
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
const char* to_string(MemoryMode m) {
    switch (m) {
        case MemoryMode::ExplicitCopy: return "explicit copy";
        case MemoryMode::ManagedNaive: return "managed (naive)";
        case MemoryMode::ManagedAdvised: return "managed + advise";
        case MemoryMode::ManagedPrefetched: return "managed + prefetch";
    }
    return "?";
}

ManagedMemoryCaps query_managed_caps() {
    int dev = 0;
    CU_CHECK(cudaGetDevice(&dev));
    ManagedMemoryCaps c;
    int v = 0;
    CU_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrManagedMemory, dev));
    c.managed_memory = v != 0;
    CU_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentManagedAccess, dev));
    c.concurrent_managed_access = v != 0;
    CU_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccess, dev));
    c.pageable_memory_access = v != 0;
    CU_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrDirectManagedMemAccessFromHost, dev));
    c.direct_managed_from_host = v != 0;
    return c;
}

// ---------------------------------------------------------------------------
GraphComparison compare_graph_vs_stream(int chain_length, int iterations, int elements) {
    if (chain_length <= 0 || iterations <= 0 || elements <= 0)
        throw std::invalid_argument("chain_length, iterations and elements must be positive");

    GraphComparison r;
    r.chain_length = chain_length;
    r.iterations = iterations;

    const int T = 256;
    const int B = (elements + T - 1) / T;

    float *d_a = nullptr, *d_b = nullptr;
    CU_CHECK(cudaMalloc(&d_a, (std::size_t)elements * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_b, (std::size_t)elements * sizeof(float)));

    std::vector<float> init((std::size_t)elements, 1.0f);
    auto reset = [&](float* p) {
        CU_CHECK(cudaMemcpy(p, init.data(), init.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
    };

    cudaStream_t stream;
    CU_CHECK(cudaStreamCreate(&stream));
    cu::EventTimer timer;

    // --- stream path: every kernel launched individually, every iteration ---
    reset(d_a);
    for (int i = 0; i < chain_length; ++i) k_tiny<<<B, T, 0, stream>>>(d_a, elements);
    CU_CHECK(cudaStreamSynchronize(stream));   // warm up

    reset(d_a);
    timer.start(stream);
    for (int it = 0; it < iterations; ++it)
        for (int i = 0; i < chain_length; ++i) k_tiny<<<B, T, 0, stream>>>(d_a, elements);
    r.stream_ms = timer.stop(stream);
    CU_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> from_stream((std::size_t)elements);
    CU_CHECK(cudaMemcpy(from_stream.data(), d_a, from_stream.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

    // --- graph path: capture the same chain once, then replay it ---
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;

    cu::EventTimer cap;
    cap.start(stream);
    CU_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < chain_length; ++i) k_tiny<<<B, T, 0, stream>>>(d_b, elements);
    CU_CHECK(cudaStreamEndCapture(stream, &graph));
    CU_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    r.capture_ms = cap.stop(stream);

    reset(d_b);
    CU_CHECK(cudaGraphLaunch(exec, stream));   // warm up
    CU_CHECK(cudaStreamSynchronize(stream));

    reset(d_b);
    timer.start(stream);
    for (int it = 0; it < iterations; ++it) CU_CHECK(cudaGraphLaunch(exec, stream));
    r.graph_ms = timer.stop(stream);
    CU_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> from_graph((std::size_t)elements);
    CU_CHECK(cudaMemcpy(from_graph.data(), d_b, from_graph.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

    // A graph must reproduce the stream result exactly; it changes WHEN work is
    // submitted, never what the work is.
    r.results_match = (from_stream == from_graph);

    CU_CHECK(cudaGraphExecDestroy(exec));
    CU_CHECK(cudaGraphDestroy(graph));
    CU_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_a);
    cudaFree(d_b);
    return r;
}

// ---------------------------------------------------------------------------
std::vector<MemoryResult> compare_memory_modes(std::size_t elements, int passes) {
    if (elements == 0 || passes <= 0)
        throw std::invalid_argument("elements and passes must be positive");

    const auto caps = query_managed_caps();
    const int T = 256;
    const int B = grid_for(elements, T);
    const std::size_t bytes = elements * sizeof(float);

    int dev = 0;
    CU_CHECK(cudaGetDevice(&dev));

    double* d_sum = nullptr;
    CU_CHECK(cudaMalloc(&d_sum, sizeof(double)));

    auto checksum_of = [&](float* p) {
        CU_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
        k_sum<<<B, T>>>(p, elements, d_sum);
        CU_CHECK_KERNEL();
        double h = 0.0;
        CU_CHECK(cudaMemcpy(&h, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    };

    std::vector<MemoryResult> out;
    const MemoryMode modes[] = {MemoryMode::ExplicitCopy, MemoryMode::ManagedNaive,
                                MemoryMode::ManagedAdvised, MemoryMode::ManagedPrefetched};

    for (MemoryMode mode : modes) {
        MemoryResult r;
        r.mode = mode;

        if (mode != MemoryMode::ExplicitCopy && !caps.managed_memory) {
            r.supported = false;
            r.skip_reason = "device reports no managed-memory support";
            out.push_back(r);
            continue;
        }
        // Both tuning APIs are gated on concurrentManagedAccess, and calling
        // either without it returns cudaErrorInvalidDevice. That error is not
        // sticky, but left unread it is picked up by the next kernel check and
        // misreported as a launch failure. So they are skipped up front rather
        // than attempted.
        if (mode == MemoryMode::ManagedAdvised && !caps.advise_supported()) {
            r.supported = false;
            r.skip_reason = "cudaMemAdvise needs concurrentManagedAccess, which "
                            "is 0 here (Windows WDDM)";
            out.push_back(r);
            continue;
        }
        if (mode == MemoryMode::ManagedPrefetched && !caps.prefetch_supported()) {
            r.supported = false;
            r.skip_reason = "cudaMemPrefetchAsync needs concurrentManagedAccess, "
                            "which is 0 here (Windows WDDM)";
            out.push_back(r);
            continue;
        }

        // Timed on the HOST clock, not CUDA events. This comparison deliberately
        // includes host-side work -- producing the input and consuming the
        // output -- because with managed memory that is exactly where page
        // migration is paid, and a GPU-only timer would hide the cost it is
        // meant to expose.
        //
        // Both modes model one workload: host produces the buffer, GPU
        // transforms it, host consumes the result. Anything less symmetric
        // measures the harness rather than the memory mode.
        using clock = std::chrono::steady_clock;
        double host_sink = 0.0;

        if (mode == MemoryMode::ExplicitCopy) {
            std::vector<float> host(elements, 1.0f);
            float* d = nullptr;
            CU_CHECK(cudaMalloc(&d, bytes));

            CU_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));
            k_touch<<<B, T>>>(d, elements, 1.0f);
            CU_CHECK_KERNEL();

            const auto t0 = clock::now();
            for (int p = 0; p < passes; ++p) {
                for (std::size_t i = 0; i < elements; ++i) host[i] = 1.0f;   // produce
                CU_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));
                k_touch<<<B, T>>>(d, elements, 1.0f);
                CU_CHECK(cudaMemcpy(host.data(), d, bytes, cudaMemcpyDeviceToHost));
                for (std::size_t i = 0; i < elements; ++i) host_sink += host[i];  // consume
            }
            CU_CHECK(cudaDeviceSynchronize());
            r.ms = float(std::chrono::duration<double, std::milli>(clock::now() - t0).count()
                         / passes);

            // Checksum a DETERMINISTIC state, not whatever the timing loop
            // happened to leave behind. `host` has been read back once per
            // pass, so its contents depend on the pass count -- checksumming
            // that compares the loop trip count, not the memory mode.
            std::fill(host.begin(), host.end(), 1.0f);
            CU_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));
            k_touch<<<B, T>>>(d, elements, 1.0f);
            CU_CHECK_KERNEL();
            r.checksum = checksum_of(d);
            cudaFree(d);
        } else {
            float* m = nullptr;
            CU_CHECK(cudaMallocManaged(&m, bytes));
            for (std::size_t i = 0; i < elements; ++i) m[i] = 1.0f;

            if (mode == MemoryMode::ManagedAdvised) {
                // Tell the driver where the data belongs so it stops guessing.
                CU_CHECK(mem_advise(m, bytes, cudaMemAdviseSetPreferredLocation, dev));
                CU_CHECK(mem_advise(m, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
            }

            k_touch<<<B, T>>>(m, elements, 1.0f);
            CU_CHECK_KERNEL();
            for (std::size_t i = 0; i < elements; ++i) host_sink += m[i];   // warm

            const auto t0 = clock::now();
            for (int p = 0; p < passes; ++p) {
                for (std::size_t i = 0; i < elements; ++i) m[i] = 1.0f;      // produce
                if (mode == MemoryMode::ManagedPrefetched)
                    CU_CHECK(mem_prefetch(m, bytes, dev, nullptr));
                k_touch<<<B, T>>>(m, elements, 1.0f);
                CU_CHECK(cudaDeviceSynchronize());
                // Reading on the host is what forces migration BACK, and is the
                // entire reason managed memory can be slow. An earlier version
                // wrote `(void)m[i]` here, which the compiler elides outright --
                // no read, no migration, and managed appeared 21x faster than
                // explicit copies. Accumulating into a value that is observed
                // later is what makes the read real.
                for (std::size_t i = 0; i < elements; ++i) host_sink += m[i];  // consume
            }
            r.ms = float(std::chrono::duration<double, std::milli>(clock::now() - t0).count()
                         / passes);

            // Same deterministic state as the explicit path: reset, one touch,
            // then checksum. Both must land on exactly 2.0 per element.
            for (std::size_t i = 0; i < elements; ++i) m[i] = 1.0f;
            CU_CHECK(cudaDeviceSynchronize());
            k_touch<<<B, T>>>(m, elements, 1.0f);
            CU_CHECK_KERNEL();
            r.checksum = checksum_of(m);
            cudaFree(m);
        }
        // Observe host_sink so the consume loops cannot be optimised away.
        if (host_sink == -1.0) r.skip_reason = "unreachable";
        out.push_back(r);
    }
    cudaFree(d_sum);
    return out;
}

// ---------------------------------------------------------------------------
std::vector<BankResult> measure_bank_conflicts(int iterations) {
    if (iterations <= 0) throw std::invalid_argument("iterations must be positive");

    float* d_out = nullptr;
    CU_CHECK(cudaMalloc(&d_out, 256 * sizeof(float)));

    // One warp per block isolates the conflict: with more warps the scheduler
    // hides the serialisation behind other warps and the effect disappears.
    const int T = 32;
    cu::EventTimer t;

    auto run = [&](auto kernel) {
        kernel<<<1, T>>>(d_out, iterations);   // warm
        CU_CHECK_KERNEL();
        float best = 1e30f;
        for (int i = 0; i < 10; ++i) {
            t.start();
            kernel<<<1, T>>>(d_out, iterations);
            CU_CHECK_KERNEL();
            best = std::min(best, t.stop());
        }
        return best;
    };

    std::vector<BankResult> out;
    const float base = run(k_bank<1>);
    auto add = [&](int stride, int ways, float ms) {
        BankResult r;
        r.stride = stride;
        r.expected_way_conflict = ways;
        r.ms = ms;
        r.slowdown_vs_stride1 = base > 0 ? ms / base : 0.0;
        out.push_back(r);
    };
    add(1, 1, base);
    add(2, 2, run(k_bank<2>));
    add(4, 4, run(k_bank<4>));
    add(8, 8, run(k_bank<8>));
    add(16, 16, run(k_bank<16>));
    add(32, 32, run(k_bank<32>));

    cudaFree(d_out);
    return out;
}

// ---------------------------------------------------------------------------
int suggested_block_size() {
    int min_grid = 0, block = 0;
    CU_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, k_touch, 0, 0));
    return block;
}

std::vector<OccupancyReport> analyze_occupancy(std::size_t elements) {
    cudaDeviceProp prop{};
    CU_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    float* d = nullptr;
    CU_CHECK(cudaMalloc(&d, elements * sizeof(float)));
    CU_CHECK(cudaMemset(d, 0, elements * sizeof(float)));

    std::vector<OccupancyReport> out;
    cu::EventTimer t;

    for (int block : {32, 64, 128, 256, 512, 1024}) {
        OccupancyReport r;
        r.block_size = block;
        r.max_warps_per_sm = max_warps;

        CU_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &r.active_blocks_per_sm, k_touch, block, 0));
        r.active_warps_per_sm = r.active_blocks_per_sm * (block / prop.warpSize);
        r.occupancy = double(r.active_warps_per_sm) / max_warps;

        const int grid = grid_for(elements, block);
        k_touch<<<grid, block>>>(d, elements, 1.0f);
        CU_CHECK_KERNEL();
        float best = 1e30f;
        for (int i = 0; i < 5; ++i) {
            t.start();
            k_touch<<<grid, block>>>(d, elements, 1.0f);
            CU_CHECK_KERNEL();
            best = std::min(best, t.stop());
        }
        r.measured_ms = best;
        out.push_back(r);
    }
    cudaFree(d);
    return out;
}

// ---------------------------------------------------------------------------
std::vector<ActiveMaskSample> sample_activemask() {
    unsigned* d_mask = nullptr;
    int* d_branch = nullptr;
    CU_CHECK(cudaMalloc(&d_mask, 32 * sizeof(unsigned)));
    CU_CHECK(cudaMalloc(&d_branch, 32 * sizeof(int)));
    CU_CHECK(cudaMemset(d_mask, 0, 32 * sizeof(unsigned)));
    CU_CHECK(cudaMemset(d_branch, 0, 32 * sizeof(int)));

    k_activemask<<<1, 32>>>(d_mask, d_branch);
    CU_CHECK_KERNEL();

    std::vector<unsigned> masks(32);
    std::vector<int> branches(32);
    CU_CHECK(cudaMemcpy(masks.data(), d_mask, 32 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(branches.data(), d_branch, 32 * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<ActiveMaskSample> out(32);
    for (int i = 0; i < 32; ++i) {
        out[i].lane = i;
        out[i].mask = masks[i];
        out[i].branch = branches[i];
        out[i].popcount = 0;
        for (int b = 0; b < 32; ++b)
            if (masks[i] & (1u << b)) ++out[i].popcount;
    }
    cudaFree(d_mask);
    cudaFree(d_branch);
    return out;
}

}  // namespace prim
