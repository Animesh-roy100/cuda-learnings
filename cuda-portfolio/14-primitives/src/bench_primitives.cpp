// Each CUDA primitive, measured in isolation.

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "cu/device.hpp"
#include "primitives.h"

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("Each section isolates ONE mechanism, so the number is\n");
    std::printf("attributable to that mechanism and nothing else.\n");

    // ---------------------------------------------------------------- graphs
    std::printf("\n=== 1. CUDA Graphs: launch overhead ===\n");
    std::printf("  %-10s %-10s %12s %12s %10s %14s\n",
                "kernels", "iters", "stream ms", "graph ms", "speedup", "us/launch saved");
    for (auto cfg : {std::pair<int,int>{1, 200}, {10, 200}, {50, 200},
                     {100, 200}, {200, 100}}) {
        auto r = prim::compare_graph_vs_stream(cfg.first, cfg.second, 4096);
        std::printf("  %-10d %-10d %12.2f %12.2f %9.2fx %14.2f  %s\n",
                    r.chain_length, r.iterations, r.stream_ms, r.graph_ms,
                    r.speedup(), r.us_saved_per_launch(),
                    r.results_match ? "identical" : "MISMATCH");
    }
    std::printf("\n  Every kernel launch costs a few microseconds of driver work on\n");
    std::printf("  the CPU. Irrelevant for one big kernel, dominant for a hundred\n");
    std::printf("  tiny ones -- which is the shape of transformer decode and of any\n");
    std::printf("  iterative solver. Read the top row as the control: with a single\n");
    std::printf("  kernel there is nothing to amortise and graphs should NOT help.\n");

    // ------------------------------------------------------- unified memory
    std::printf("\n=== 2. Unified Memory vs explicit copies ===\n");
    const auto caps = prim::query_managed_caps();
    std::printf("  device capabilities:\n");
    std::printf("    managedMemory             %d\n", caps.managed_memory);
    std::printf("    concurrentManagedAccess   %d\n", caps.concurrent_managed_access);
    std::printf("    pageableMemoryAccess      %d\n", caps.pageable_memory_access);
    std::printf("    directManagedMemFromHost  %d\n\n", caps.direct_managed_from_host);

    auto mem = prim::compare_memory_modes(1 << 22, 5);   // 16 MB
    std::printf("  %-22s %12s %18s\n", "mode", "ms/pass", "checksum");
    for (const auto& r : mem) {
        if (!r.supported) {
            std::printf("  %-22s %12s  SKIPPED: %s\n",
                        prim::to_string(r.mode), "-", r.skip_reason.c_str());
        } else {
            std::printf("  %-22s %12.3f %18.1f\n",
                        prim::to_string(r.mode), r.ms, r.checksum);
        }
    }
    if (!caps.prefetch_supported()) {
        std::printf("\n  cudaMemPrefetchAsync REQUIRES concurrentManagedAccess, which is\n");
        std::printf("  0 under the Windows WDDM driver model. The call returns an error\n");
        std::printf("  rather than quietly doing nothing, so the row is skipped instead\n");
        std::printf("  of reporting a timing from a call that failed. On Linux the same\n");
        std::printf("  hardware reports 1 and the row runs -- the limit is the driver\n");
        std::printf("  model, not the GPU.\n");
    }

    // -------------------------------------------------------- bank conflicts
    std::printf("\n=== 3. Shared memory bank conflicts ===\n");
    std::printf("  32 banks x 4 bytes. Lane L reads s[L*stride + it], so lanes land\n");
    std::printf("  on bank (L*stride) %% 32. One warp per block, to stop the\n");
    std::printf("  scheduler hiding the serialisation behind other warps.\n\n");
    std::printf("  %-8s %-14s %12s %12s\n", "stride", "expected", "ms", "vs stride-1");
    for (const auto& b : prim::measure_bank_conflicts(20000)) {
        char ways[16];
        std::snprintf(ways, sizeof(ways), "%d-way", b.expected_way_conflict);
        std::printf("  %-8d %-14s %12.3f %11.2fx\n",
                    b.stride, ways, b.ms, b.slowdown_vs_stride1);
    }

    // ------------------------------------------------------------- occupancy
    std::printf("\n=== 4. Occupancy ===\n");
    std::printf("  cudaOccupancyMaxPotentialBlockSize suggests: %d threads\n\n",
                prim::suggested_block_size());
    std::printf("  %-8s %10s %12s %12s %12s\n",
                "block", "blocks/SM", "warps/SM", "occupancy", "measured ms");
    for (const auto& o : prim::analyze_occupancy(1 << 24)) {
        std::printf("  %-8d %10d %12d %11.0f%% %12.3f\n",
                    o.block_size, o.active_blocks_per_sm, o.active_warps_per_sm,
                    100.0 * o.occupancy, o.measured_ms);
    }
    std::printf("\n  Occupancy is latency-hiding CAPACITY, not speed, and this table\n");
    std::printf("  shows why the distinction matters. Block sizes 64 through 1024 all\n");
    std::printf("  reach 100%% occupancy, yet their runtimes differ by ~25%%. Once a\n");
    std::printf("  memory-bound kernel has enough warps to cover DRAM latency, more\n");
    std::printf("  warps buy nothing, and very large blocks start to cost: coarser\n");
    std::printf("  scheduling granularity and a longer tail when the last block\n");
    std::printf("  drains.\n\n");
    std::printf("  Note too that cudaOccupancyMaxPotentialBlockSize suggests 1024 --\n");
    std::printf("  the SLOWEST row here. It optimises occupancy, which is what it\n");
    std::printf("  claims to do; it does not promise the fastest configuration.\n");
    std::printf("  Treat it as a starting point to measure from, not an answer.\n");

    // ------------------------------------------------------------ activemask
    std::printf("\n=== 5. __activemask under divergence ===\n");
    auto masks = prim::sample_activemask();
    std::printf("  A warp split by (lane & 1). Each lane reports which lanes are\n");
    std::printf("  live alongside it at that instruction.\n\n");
    for (int i : {0, 1, 2, 3, 30, 31}) {
        const auto& m = masks[i];
        std::printf("  lane %-3d branch %d  mask 0x%08x  (%d lanes active)\n",
                    m.lane, m.branch, m.mask, m.popcount);
    }
    std::printf("\n  Even lanes see 0x55555555, odd lanes 0xAAAAAAAA -- 16 active each.\n");
    std::printf("  That is branch divergence made visible: the hardware runs the two\n");
    std::printf("  halves in sequence, and each half sees only itself.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
