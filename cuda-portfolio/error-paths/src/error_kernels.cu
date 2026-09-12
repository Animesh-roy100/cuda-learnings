// Kernels that fail on purpose, behind a CUDA-free interface so the test suite
// stays plain C++.

#include "error_kernels.h"

#include "cu/check.hpp"

namespace ep {
namespace {

__global__ void fill_index(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i * 3;
}

__global__ void write_to(int* p) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *p = 1;
}

}  // namespace

bool launch_and_verify(int n) {
    int* d = nullptr;
    CU_CHECK(cudaMalloc(&d, sizeof(int) * n));
    const int t = 256;
    fill_index<<<(n + t - 1) / t, t>>>(d, n);
    try {
        CU_CHECK_KERNEL();
    } catch (...) {
        cudaFree(d);
        throw;
    }
    std::vector<int> h(n);
    const cudaError_t copy = cudaMemcpy(h.data(), d, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaFree(d);
    cu::check(copy, __FILE__, __LINE__, "cudaMemcpy");
    for (int i = 0; i < n; ++i)
        if (h[i] != i * 3) return false;
    return true;
}

void launch_with_oversized_block() {
    int* d = nullptr;
    CU_CHECK(cudaMalloc(&d, sizeof(int) * 4096));
    // 4096 threads per block exceeds the 1024 maximum on every CUDA device.
    fill_index<<<1, 4096>>>(d, 4096);
    const cudaError_t launch = cudaGetLastError();
    cudaFree(d);
    cu::check(launch, __FILE__, __LINE__, "kernel launch");
}

void write_to_illegal_address() {
    // Low addresses are never mapped into the GPU's virtual address space, so
    // this is a guaranteed fault rather than a quiet out-of-bounds write that
    // happens to land in someone else's allocation.
    write_to<<<1, 1>>>(reinterpret_cast<int*>(0x40));
    CU_CHECK_KERNEL();
}

}  // namespace ep
