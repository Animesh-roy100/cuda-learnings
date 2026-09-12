// 01 - Know your hardware.
// Every optimization decision you make later depends on these numbers.
// Run this first and keep the output somewhere you can look at it.

#include <cstdio>
#include "../common/cuda_check.h"

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::printf("CUDA devices found: %d\n\n", count);

    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));

        std::printf("Device %d: %s\n", d, p.name);
        std::printf("  Compute capability      : %d.%d  (use -arch=sm_%d%d)\n",
                    p.major, p.minor, p.major, p.minor);
        std::printf("  Global memory           : %.2f GB\n",
                    p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        std::printf("  SM count                : %d\n", p.multiProcessorCount);
        std::printf("  Warp size               : %d\n", p.warpSize);
        std::printf("  Max threads / block     : %d\n", p.maxThreadsPerBlock);
        std::printf("  Max threads / SM        : %d\n", p.maxThreadsPerMultiProcessor);
        std::printf("  Shared memory / block   : %zu KB\n", p.sharedMemPerBlock / 1024);
        std::printf("  Shared memory / SM      : %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
        std::printf("  Registers / block       : %d\n", p.regsPerBlock);
        std::printf("  L2 cache                : %d KB\n", p.l2CacheSize / 1024);
        std::printf("  Memory bus width        : %d bits\n", p.memoryBusWidth);
        // CUDA 13 removed memoryClockRate/clockRate from cudaDeviceProp;
        // the attribute API is now the supported way to read them.
        int memClockKHz = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, d));
        std::printf("  Memory clock            : %.2f GHz\n", memClockKHz / 1.0e6);

        // Peak bandwidth = clock * bus width * 2 (DDR) -- this is the number
        // most simple kernels are actually limited by, not FLOPS.
        double bw = 2.0 * memClockKHz * (p.memoryBusWidth / 8) / 1.0e6;
        std::printf("  Peak bandwidth          : %.1f GB/s   <-- your real ceiling\n", bw);

        std::printf("  Max grid dimensions     : (%d, %d, %d)\n",
                    p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        std::printf("  Max block dimensions    : (%d, %d, %d)\n",
                    p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        std::printf("  Concurrent kernels      : %s\n", p.concurrentKernels ? "yes" : "no");
        std::printf("  Async engines (copy)    : %d\n", p.asyncEngineCount);
        std::printf("  Unified addressing      : %s\n", p.unifiedAddressing ? "yes" : "no");
        std::printf("\n");
    }
    return 0;
}
