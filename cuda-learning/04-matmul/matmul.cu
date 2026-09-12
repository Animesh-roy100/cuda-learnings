// 04 - Naive vs tiled matrix multiply.
//
// This is THE lesson of CUDA optimization. Both kernels do exactly the same
// arithmetic. The tiled one is several times faster purely because it moves
// less data. Run it, then read the comments.

#include <cstdio>
#include <vector>
#include <cmath>
#include "../common/cuda_check.h"

#define TILE 16   // 16x16 = 256 threads per block

// ---------------------------------------------------------------------------
// Naive: every thread reads its full row of A and column of B from global
// memory. For an NxN matmul that is 2*N reads per output element, and
// neighbouring threads re-read the exact same values over and over.
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Tiled: the block cooperatively stages a TILE x TILE chunk of A and of B into
// shared memory (on-chip, ~100x lower latency than global), then every thread
// in the block reuses those staged values TILE times.
// Global memory traffic drops by a factor of TILE.
// ---------------------------------------------------------------------------
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        // Stage one tile of each matrix. Guard against the ragged edge.
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        sA[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        sB[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        // Barrier 1: nobody computes until the whole tile is loaded.
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        // Barrier 2: nobody overwrites the tile until everyone is done reading.
        // Forgetting THIS one is the classic CUDA race condition.
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

float time_kernel(void (*launch)(const float*, const float*, float*, int),
                  const float* dA, const float* dB, float* dC, int N,
                  dim3 grid, dim3 block) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch<<<grid, block>>>(dA, dB, dC, N);   // warm-up
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaEventRecord(start));
    launch<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

int main() {
    const int N = 1024;
    const size_t bytes = (size_t)N * N * sizeof(float);
    std::printf("Matrix multiply: %d x %d\n\n", N, N);

    std::vector<float> hA(N * N), hB(N * N), hC(N * N);
    for (int i = 0; i < N * N; ++i) {
        hA[i] = (float)((i % 13) - 6) * 0.1f;
        hB[i] = (float)((i % 7) - 3) * 0.1f;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    float t_naive = time_kernel(matmul_naive, dA, dB, dC, N, grid, block);
    std::vector<float> ref(N * N);
    CUDA_CHECK(cudaMemcpy(ref.data(), dC, bytes, cudaMemcpyDeviceToHost));

    float t_tiled = time_kernel(matmul_tiled, dA, dB, dC, N, grid, block);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    double maxdiff = 0.0;
    for (int i = 0; i < N * N; ++i) {
        maxdiff = std::fmax(maxdiff, std::fabs(ref[i] - hC[i]));
    }

    // 2*N^3 floating point ops (one multiply + one add per inner iteration).
    double gflop = 2.0 * N * N * N / 1.0e9;
    std::printf("  naive : %8.3f ms   %7.1f GFLOP/s\n", t_naive, gflop / (t_naive / 1000.0));
    std::printf("  tiled : %8.3f ms   %7.1f GFLOP/s\n", t_tiled, gflop / (t_tiled / 1000.0));
    std::printf("  speedup: %.2fx\n", t_naive / t_tiled);
    std::printf("  max difference between the two: %.6f  (%s)\n",
                maxdiff, maxdiff < 1e-3 ? "same answer, as expected" : "MISMATCH");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
