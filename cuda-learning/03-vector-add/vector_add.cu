// 03 - The canonical first real kernel, plus the lesson that matters:
// this kernel is MEMORY BOUND. It does 1 add per 12 bytes moved.
// Compare the measured bandwidth to the peak from 01-device-query.

#include <cstdio>
#include <vector>
#include "../common/cuda_check.h"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // The bounds check is NOT optional. n is rarely a multiple of blockDim,
    // so the last block always has threads with i >= n.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 24;                  // ~16.7M elements
    const size_t bytes = N * sizeof(float);
    std::printf("N = %d elements, %.1f MB per array\n", N, bytes / (1024.0 * 1024.0));

    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N, 0.0f);

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;                        // 8 warps; a good default
    const int blocks  = (N + threads - 1) / threads; // ceiling division idiom
    std::printf("Launching <<<%d, %d>>>\n", blocks, threads);

    // CUDA events are the correct way to time GPU work. Host-side clocks
    // measure the wrong thing because launches are asynchronous.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);  // warm-up
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify. Always verify -- a fast wrong answer is worthless.
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 3.0f) { ++errors; }
    }
    std::printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");

    // 3 arrays touched (read a, read b, write c) = 12 bytes per element.
    double gb = 3.0 * bytes / 1.0e9;
    std::printf("Kernel time: %.3f ms\n", ms);
    std::printf("Effective bandwidth: %.1f GB/s\n", gb / (ms / 1000.0));
    std::printf("\nCompare that to the peak bandwidth from 01-device-query.\n"
                "Getting 70-80%% of peak means you are done optimizing this kernel.\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
