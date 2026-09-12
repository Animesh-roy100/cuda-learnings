// 02 - The CUDA execution model, in one screen.
//
// The whole mental model: you launch a GRID of BLOCKS of THREADS.
// Every thread runs the same code, but gets different index values.
// That's it. Everything else is detail.

#include <cstdio>
#include "../common/cuda_check.h"

// __global__ means: callable from host, runs on device.
__global__ void hello() {
    // Each thread computes its own unique global ID from the built-ins.
    // This line is the single most important pattern in all of CUDA.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Threads are grouped into warps of 32 that execute in lockstep.
    int warp = threadIdx.x / warpSize;

    std::printf("block %d, thread %2d -> global id %2d (warp %d of this block)\n",
                blockIdx.x, threadIdx.x, tid, warp);
}

int main() {
    std::printf("Launching <<<2, 8>>> = 2 blocks x 8 threads = 16 threads\n\n");

    // <<<gridDim, blockDim>>> is the launch configuration.
    hello<<<2, 8>>>();

    // The launch is ASYNCHRONOUS. Without this sync, main() could exit
    // before the GPU finishes and you'd see no output at all.
    CUDA_CHECK_KERNEL();

    std::printf("\nNote the ordering: blocks and warps complete in whatever\n"
                "order the scheduler picks. Never rely on it.\n");
    return 0;
}
