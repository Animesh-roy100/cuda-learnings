// Layout and the advanced subsystems, with their measured results printed.

#include <cstdio>
#include <stdexcept>
#include <vector>

#include "cu/device.hpp"
#include "layout_advanced.h"

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("Every number below is measured on this card, not quoted from a\n");
    std::printf("guide. One of the six features turns out not to help here, and one\n");
    std::printf("helps more than the spec sheet suggests it should -- which is\n");
    std::printf("exactly why they are worth measuring rather than assuming.\n");

    // ------------------------------------------------------------ AoS vs SoA
    std::printf("\n=== 1. Array of Structures vs Structure of Arrays ===\n");
    const int particles = 1 << 22;   // 4M
    auto L = la::compare_aos_soa(particles);
    std::printf("  %d particles, 6 floats each, kernel reads 3 of them\n", particles);
    std::printf("  %-28s %9s %12s\n", "layout", "ms", "useful GB/s");
    std::printf("  %-28s %9.3f %12.1f\n", "AoS  {x,y,z,vx,vy,vz}[]", L.aos_ms, L.aos_gbps);
    std::printf("  %-28s %9.3f %12.1f\n", "SoA  x[],y[],z[],vx[]...", L.soa_ms, L.soa_gbps);
    std::printf("  speedup %.2fx   identical results: %s\n",
                L.speedup(), L.results_match ? "yes" : "NO");
    std::printf("\n  Same arithmetic, same answers, different arrangement. AoS moves\n");
    std::printf("  24 bytes per particle to use 12: the velocities sit inside the\n");
    std::printf("  same 32-byte sectors the positions do, so DRAM delivers them\n");
    std::printf("  whether or not the kernel looks at them. %.0f%% of peak is\n",
                100.0 * L.soa_gbps / dev.peak_bandwidth_gbps());
    std::printf("  reachable with SoA; AoS cannot get there at any occupancy.\n");

    // ------------------------------------------------------- shared padding
    std::printf("\n=== 2. Shared-memory bank conflicts ===\n");
    auto P = la::compare_shared_padding(64);
    std::printf("  32x32 float tile, read column-wise by a 32x32 block\n");
    std::printf("  %-28s %9.3f ms\n", "tile[32][32]  (conflicting)", P.unpadded_ms);
    std::printf("  %-28s %9.3f ms\n", "tile[32][33]  (padded)", P.padded_ms);
    std::printf("  speedup %.2fx for %zu extra bytes   identical results: %s\n",
                P.speedup(), P.extra_bytes, P.results_match ? "yes" : "NO");
    std::printf("\n  Shared memory is 32 banks of 4-byte words. At width 32 the whole\n");
    std::printf("  warp asks for bank ty and the access serialises 32 ways; at width\n");
    std::printf("  33 each row is shifted one bank and the lanes spread across all\n");
    std::printf("  32. One character in a declaration.\n");

    // ------------------------------------------------------ cooperative groups
    std::printf("\n=== 3. Cooperative groups and the grid-wide barrier ===\n");
    const int elements = 1 << 20;
    auto C = la::exercise_cooperative_groups(elements);
    if (!C.cooperative_launch_supported) {
        std::printf("  SKIPPED: %s\n", C.skip_reason.c_str());
    } else {
        std::printf("  cg::reduce over a 32-lane tile, every thread giving 1.0 -> %.0f\n",
                    C.tiled_reduce);
        std::printf("  cg::reduce over the whole 256-thread block            -> %.0f\n",
                    C.block_reduce);
        std::printf("  grid.size() as seen from inside the kernel            -> %d\n",
                    C.grid_size_seen);
        std::printf("  grid.sync() held across %d elements                -> %s\n",
                    elements, C.grid_sync_ok ? "yes" : "NO");
        std::printf("\n  cg::reduce compiles to the same shuffle tree written by hand in\n");
        std::printf("  15-warp-primitives, but a tile is a first-class object: it can\n");
        std::printf("  be passed to a function, so a reduction stops being a macro.\n");
        std::printf("\n  grid.sync() is the one thing __syncthreads() cannot do. It is\n");
        std::printf("  why the grid size above is not a free choice -- every block must\n");
        std::printf("  be resident simultaneously or the barrier deadlocks, so the size\n");
        std::printf("  comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor.\n");
    }

    // ---------------------------------------------------- dynamic parallelism
    std::printf("\n=== 4. Dynamic parallelism: kernels launching kernels ===\n");
    const int parents = 8, child_threads = 32;
    auto D = la::exercise_dynamic_parallelism(parents, child_threads);
    if (!D.supported) {
        std::printf("  SKIPPED: %s\n", D.skip_reason.c_str());
    } else {
        std::printf("  %d parent threads each launched a child of %d threads\n",
                    parents, child_threads);
        std::printf("  child launches reported: %d\n", D.child_launches);
        std::printf("  out[p*%d + i] should be (p+1)*i:\n", child_threads);
        for (int p = 0; p < 3; ++p)
            std::printf("    parent %d -> %.0f %.0f %.0f %.0f ...\n", p,
                        D.output[p * child_threads + 0], D.output[p * child_threads + 1],
                        D.output[p * child_threads + 2], D.output[p * child_threads + 3]);
        std::printf("\n  The host issued ONE launch. The other %d came from device code.\n",
                    D.child_launches);
        std::printf("  Under CUDA 12+ the device-side cudaDeviceSynchronize() no longer\n");
        std::printf("  exists; the parent grid simply is not complete, as the host sees\n");
        std::printf("  it, until its children are. Costs -rdc=true and cudadevrt.\n");
    }

    // --------------------------------------------------------- tensor cores
    std::printf("\n=== 5. Tensor Cores (WMMA) vs tiled FP32 ===\n");
    auto W = la::compare_wmma_vs_fp32(512);
    if (!W.ran) {
        std::printf("  SKIPPED: %s\n", W.skip_reason.c_str());
    } else {
        std::printf("  %dx%d matmul, 16x16x16 fragments. Three kernels, one variable\n",
                    W.matrix_dim, W.matrix_dim);
        std::printf("  changed at a time, so the speedup can be attributed:\n\n");
        std::printf("  %-34s %9s %10s\n", "kernel", "ms", "vs above");
        std::printf("  %-34s %9.3f %10s\n", "fp32 operands, fp32 math", W.fp32_ms, "--");
        std::printf("  %-34s %9.3f %9.2fx\n", "fp16 operands, fp32 math",
                    W.fp16in_ms, W.bandwidth_speedup());
        std::printf("  %-34s %9.3f %9.2fx\n", "fp16 operands, wmma mma_sync",
                    W.wmma_ms, W.instruction_speedup());
        std::printf("  %-34s %9s %9.2fx\n", "end to end", "", W.speedup());
        std::printf("  max |wmma - fp32| = %.4f\n", W.max_abs_error);
        std::printf("\n  NVIDIA lists the GTX 16-series as shipping WITHOUT Tensor Cores,\n");
        std::printf("  so the expectation going in was that WMMA would be correct here\n");
        std::printf("  but not faster. The middle row is what makes the answer readable:\n");
        std::printf("  narrowing the operands to fp16 is worth %.2fx on its own, and\n",
                    W.bandwidth_speedup());
        std::printf("  mma_sync is worth a further %.2fx on top of that. The two effects\n",
                    W.instruction_speedup());
        std::printf("  are separable only because the middle kernel is the fp32 one with\n");
        std::printf("  exactly one thing changed.\n");
        if (W.bandwidth_speedup() < 1.1) {
            std::printf("\n  The middle row came out flat, which is itself the finding: at\n");
            std::printf("  this size every byte is reused %d times, so the kernel is\n", W.matrix_dim);
            std::printf("  compute-bound and halving the operand width buys nothing. That\n");
            std::printf("  leaves the whole %.2fx sitting on mma_sync -- so whatever the\n",
                        W.instruction_speedup());
            std::printf("  spec sheet says about Tensor Cores, this chip is retiring HMMA\n");
            std::printf("  meaningfully faster than it retires FP32 FMA.\n");
        }
        std::printf("\n  The error term is fp16 input precision, not a bug: 11 mantissa\n");
        std::printf("  bits accumulated over %d products. Note it is nonzero -- if this\n", W.matrix_dim);
        std::printf("  read 0.0000 it would mean the fp16 path never actually ran.\n");
    }

    // ---------------------------------------------------------- async copy
    std::printf("\n=== 6. Asynchronous global -> shared copy ===\n");
    auto A = la::compare_async_copy(1 << 22);
    std::printf("  %-28s %9.3f ms\n", "load + __syncthreads", A.sync_ms);
    std::printf("  %-28s %9.3f ms\n", "cg::memcpy_async + wait", A.async_ms);
    std::printf("  speedup %.2fx   identical results: %s\n",
                A.speedup(), A.results_match ? "yes" : "NO");
    std::printf("  hardware cp.async: %s (sm_%d, needs sm_80+)\n",
                A.hardware_accelerated ? "yes" : "no", A.compute_capability);
    if (!A.hardware_accelerated) {
        std::printf("\n  No speedup, and none was expected. cg::memcpy_async compiles from\n");
        std::printf("  sm_70, but the cp.async instruction that lets DRAM write straight\n");
        std::printf("  into shared memory arrived with Ampere. Here it falls back to the\n");
        std::printf("  ordinary load-then-barrier, which is why this measurement matters:\n");
        std::printf("  the API being available is not the same as the hardware being\n");
        std::printf("  there, and only the clock can tell the two apart.\n");
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
