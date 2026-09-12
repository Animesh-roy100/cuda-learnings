#pragma once
//
// CUDA primitives, measured rather than described.
//
// The other thirteen projects each solve a problem and use whatever primitives
// that problem needs. This one inverts that: each section isolates ONE
// mechanism and measures what it actually buys, so the number is attributable
// to the mechanism and nothing else.
//
// Four things are covered, each of which is easy to assert and harder to
// demonstrate:
//   * CUDA Graphs            -- launch overhead, on a chain of tiny kernels
//   * Unified Memory         -- against explicit copies, with advice and prefetch
//   * Shared memory banks    -- a real conflict, swept from 1-way to 32-way
//   * Occupancy              -- the API, and why maximum occupancy is not the goal
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace prim {

// ---------------------------------------------------------------------------
// What this device actually supports. Several Unified Memory features are
// unavailable under the Windows WDDM driver model, and the right response is
// to detect that rather than to produce a misleading measurement.
// ---------------------------------------------------------------------------
struct ManagedMemoryCaps {
    bool managed_memory = false;              // cudaMallocManaged works at all
    bool concurrent_managed_access = false;   // GPU may touch managed pages while host does
    bool pageable_memory_access = false;      // kernels may dereference plain malloc'd pointers
    bool direct_managed_from_host = false;

    // BOTH managed-memory tuning APIs are gated on concurrent access, not just
    // prefetch. Measured: with concurrentManagedAccess == 0, cudaMemAdvise
    // returns cudaErrorInvalidDevice exactly as cudaMemPrefetchAsync does.
    //
    // Gate them up front rather than calling "to see what happens". The error
    // is NOT sticky -- measured, the context stays fully usable -- but a failed
    // call leaves its error recorded, and the next CU_CHECK_KERNEL reads that
    // record and reports a healthy kernel as failed. (This was originally
    // misdiagnosed here as context poisoning; see error-paths/.)
    bool advise_supported() const { return concurrent_managed_access; }
    bool prefetch_supported() const { return concurrent_managed_access; }
};
ManagedMemoryCaps query_managed_caps();

// ---------------------------------------------------------------------------
// CUDA Graphs
//
// Every kernel launch costs a few microseconds of CPU-side driver work. That is
// irrelevant for one big kernel and dominant for a hundred small ones -- which
// is exactly the shape of transformer decode, or any iterative solver. A graph
// records the whole DAG once and replays it with a single call.
// ---------------------------------------------------------------------------
struct GraphComparison {
    int chain_length = 0;      // kernels per iteration
    int iterations = 0;
    float stream_ms = 0.0f;    // launched individually, every iteration
    float graph_ms = 0.0f;     // captured once, replayed
    float capture_ms = 0.0f;   // one-off cost of building the graph
    bool results_match = false;

    double speedup() const { return graph_ms > 0 ? stream_ms / graph_ms : 0.0; }
    double us_saved_per_launch() const {
        const double n = double(chain_length) * iterations;
        return n > 0 ? (stream_ms - graph_ms) * 1000.0 / n : 0.0;
    }
};
GraphComparison compare_graph_vs_stream(int chain_length, int iterations,
                                        int elements = 1 << 14);

// ---------------------------------------------------------------------------
// Unified Memory
// ---------------------------------------------------------------------------
enum class MemoryMode {
    ExplicitCopy,       // cudaMalloc + cudaMemcpy, the baseline
    ManagedNaive,       // cudaMallocManaged, migration left to the driver
    ManagedAdvised,     // + cudaMemAdvise(PreferredLocation / ReadMostly)
    ManagedPrefetched,  // + cudaMemPrefetchAsync before the kernel
};
const char* to_string(MemoryMode m);

struct MemoryResult {
    MemoryMode mode{};
    bool supported = true;       // false when the platform lacks the feature
    std::string skip_reason;
    float ms = 0.0f;
    double checksum = 0.0;       // every mode must agree
};
std::vector<MemoryResult> compare_memory_modes(std::size_t elements, int passes);

// ---------------------------------------------------------------------------
// Shared memory bank conflicts
//
// Shared memory is 32 banks of 4 bytes. Lanes hitting distinct banks are
// serviced in one cycle; lanes hitting the same bank are serialised. A stride
// of 32 floats maps every lane of a warp onto bank 0 -- the worst case, and a
// 32-way conflict.
// ---------------------------------------------------------------------------
struct BankResult {
    int stride = 1;
    int expected_way_conflict = 1;   // lanes landing on the same bank
    float ms = 0.0f;
    double slowdown_vs_stride1 = 1.0;
};
std::vector<BankResult> measure_bank_conflicts(int iterations = 2000);

// ---------------------------------------------------------------------------
// Occupancy
//
// cudaOccupancyMaxActiveBlocksPerMultiprocessor answers "how many blocks of
// this size fit on an SM given this kernel's registers and shared memory".
// Higher is not automatically better, which the benchmark demonstrates.
// ---------------------------------------------------------------------------
struct OccupancyReport {
    int block_size = 0;
    int active_blocks_per_sm = 0;
    int active_warps_per_sm = 0;
    int max_warps_per_sm = 0;
    double occupancy = 0.0;          // 0..1
    float measured_ms = 0.0f;        // same work, this block size
};
std::vector<OccupancyReport> analyze_occupancy(std::size_t elements);

// Block size CUDA itself suggests for the benchmark kernel.
int suggested_block_size();

// ---------------------------------------------------------------------------
// __activemask: which lanes of a warp are live at this instruction. Useful for
// reasoning about divergence, and the one warp primitive the other projects
// never needed.
// ---------------------------------------------------------------------------
struct ActiveMaskSample {
    int lane = 0;
    unsigned mask = 0;
    int popcount = 0;
    int branch = 0;   // 0 = even lanes, 1 = odd lanes
};
std::vector<ActiveMaskSample> sample_activemask();

}  // namespace prim
