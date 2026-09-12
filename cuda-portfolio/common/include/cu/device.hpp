#pragma once
//
// Device capabilities. Peak bandwidth in particular is the number every
// memory-bound kernel in this repo is judged against -- without it, "180 GB/s"
// is a number with no meaning.
//
#include <cuda_runtime.h>

#include <cstdio>
#include <string>

#include "cu/check.hpp"

namespace cu {

struct DeviceInfo {
    std::string name;
    int major = 0, minor = 0;
    int sm_count = 0;
    int warp_size = 0;
    size_t total_mem = 0;
    size_t shared_per_block = 0;
    int l2_bytes = 0;
    int bus_width_bits = 0;
    int mem_clock_khz = 0;
    int async_engines = 0;
    bool can_map_host = false;

    // DDR transfers on both clock edges, hence the factor of 2.
    double peak_bandwidth_gbps() const {
        return 2.0 * mem_clock_khz * (bus_width_bits / 8) / 1.0e6;
    }
    int arch() const { return major * 10 + minor; }
    bool supports_dp4a() const { return arch() >= 61; }
};

inline DeviceInfo query_device(int ordinal = 0) {
    cudaDeviceProp p{};
    CU_CHECK(cudaGetDeviceProperties(&p, ordinal));

    DeviceInfo d;
    d.name = p.name;
    d.major = p.major;
    d.minor = p.minor;
    d.sm_count = p.multiProcessorCount;
    d.warp_size = p.warpSize;
    d.total_mem = p.totalGlobalMem;
    d.shared_per_block = p.sharedMemPerBlock;
    d.l2_bytes = p.l2CacheSize;
    d.bus_width_bits = p.memoryBusWidth;
    d.async_engines = p.asyncEngineCount;
    d.can_map_host = p.canMapHostMemory != 0;

    // CUDA 13 removed memoryClockRate/clockRate from cudaDeviceProp; the
    // attribute API is the supported replacement.
    CU_CHECK(cudaDeviceGetAttribute(&d.mem_clock_khz, cudaDevAttrMemoryClockRate, ordinal));
    return d;
}

inline void print_banner(const DeviceInfo& d) {
    std::printf("%s  sm_%d%d  %d SMs  %.2f GB  peak %.1f GB/s\n",
                d.name.c_str(), d.major, d.minor, d.sm_count,
                d.total_mem / (1024.0 * 1024.0 * 1024.0), d.peak_bandwidth_gbps());
}

// Fraction of peak achieved -- the only honest score for a memory-bound kernel.
inline double bandwidth_efficiency(const DeviceInfo& d, double bytes, double ms) {
    double achieved = bytes / (ms / 1000.0) / 1e9;
    return achieved / d.peak_bandwidth_gbps();
}

}  // namespace cu
