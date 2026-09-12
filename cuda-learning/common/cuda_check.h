#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap EVERY CUDA API call in this. CUDA fails silently by default --
// it returns error codes instead of throwing, so an unchecked failure
// shows up later as garbage results, not as a crash.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s (%s)\n",              \
                         __FILE__, __LINE__,                                 \
                         cudaGetErrorName(err_),                             \
                         cudaGetErrorString(err_));                          \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Kernel launches don't return an error code. You have to ask for it
// separately: once for launch-time errors (bad config), once after a
// sync for runtime errors (out-of-bounds access inside the kernel).
#define CUDA_CHECK_KERNEL()                                                  \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)
