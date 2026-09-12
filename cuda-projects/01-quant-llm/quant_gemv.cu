// ============================================================================
// Project 1 - Quantized LLM inference: the kernel that actually matters.
//
// During single-token decode (batch=1), a transformer does GEMV, not GEMM:
// one activation vector against every weight matrix. Arithmetic intensity is
// ~2 FLOP per weight byte, so you are STREAMING WEIGHTS and nothing else.
// Decode speed = weight bytes / memory bandwidth. That is the whole game.
//
// Which is why quantization is the single biggest inference win:
//   FP32 weights -> 4 bytes each
//   INT8         -> 1 byte      = 4x less traffic  = ~4x faster
//   INT4         -> 0.5 bytes   = 8x less traffic  = ~8x faster
//
// The GTX 1650 has no Tensor Cores, so we use __dp4a: one instruction that
// does a 4-way INT8 dot product accumulating into INT32. Available since
// sm_61; your sm_75 has it.
//
// Layout is llama.cpp Q4_0-style: groups of 32 weights share one FP32 scale,
// symmetric, stored as packed nibbles.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(e_));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

static const int QGROUP = 32;   // weights per quantization group

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Extract nibble i (0..7) from a packed uint32 and recentre to [-8, 7].
__device__ __forceinline__ int nib(unsigned int p, int i) {
    return (int)((p >> (4 * i)) & 0xfu) - 8;
}

__device__ __forceinline__ int pack4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}

__device__ __forceinline__ float warp_reduce(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Baseline: FP32 GEMV. One warp per output row.
// ---------------------------------------------------------------------------
__global__ void gemv_fp32(const float* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ y, int M, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;
    int lane = threadIdx.x & 31;

    const float4* w4 = reinterpret_cast<const float4*>(W + (size_t)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int K4 = K / 4;

    float acc = 0.0f;
    // float4 loads give each thread a 128-bit access, which is what DRAM likes.
    for (int i = lane; i < K4; i += 32) {
        float4 a = w4[i], b = x4[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// W8A8: INT8 weights, INT8 activations, __dp4a accumulation.
// Per-group scales on the weights, one global scale on the activations.
// ---------------------------------------------------------------------------
__global__ void gemv_q8(const signed char* __restrict__ Wq, const float* __restrict__ Ws,
                        const signed char* __restrict__ xq, float xs,
                        float* __restrict__ y, int M, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;
    int lane = threadIdx.x & 31;

    int ngroups = K / QGROUP;
    const signed char* w = Wq + (size_t)row * K;
    const float* ws = Ws + (size_t)row * ngroups;

    float acc = 0.0f;
    for (int g = lane; g < ngroups; g += 32) {
        // 32 int8 weights = 32 bytes = two 16-byte loads.
        const int4* wp = reinterpret_cast<const int4*>(w + g * QGROUP);
        const int4* xp = reinterpret_cast<const int4*>(xq + g * QGROUP);
        int4 w0 = wp[0], w1 = wp[1];
        int4 x0 = xp[0], x1 = xp[1];

        int s = 0;
        s = __dp4a(w0.x, x0.x, s);  s = __dp4a(w0.y, x0.y, s);
        s = __dp4a(w0.z, x0.z, s);  s = __dp4a(w0.w, x0.w, s);
        s = __dp4a(w1.x, x1.x, s);  s = __dp4a(w1.y, x1.y, s);
        s = __dp4a(w1.z, x1.z, s);  s = __dp4a(w1.w, x1.w, s);

        acc += (float)s * ws[g];
    }
    acc = warp_reduce(acc) * xs;
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// W4A8: the real target. INT4 packed weights, INT8 activations.
// 32 weights arrive as ONE 16-byte load, unpack to 8 int32 lanes of 4x int8,
// then 8 __dp4a instructions.
// ---------------------------------------------------------------------------
__global__ void gemv_q4(const unsigned char* __restrict__ Wq, const float* __restrict__ Ws,
                        const signed char* __restrict__ xq, float xs,
                        float* __restrict__ y, int M, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= M) return;
    int lane = threadIdx.x & 31;

    int ngroups = K / QGROUP;
    const unsigned char* w = Wq + (size_t)row * (K / 2);   // 2 weights per byte
    const float* ws = Ws + (size_t)row * ngroups;

    float acc = 0.0f;
    for (int g = lane; g < ngroups; g += 32) {
        // 32 nibble-packed weights = 16 bytes = one 128-bit load.
        uint4 p = *reinterpret_cast<const uint4*>(w + g * (QGROUP / 2));
        const int4* xp = reinterpret_cast<const int4*>(xq + g * QGROUP);
        int4 x0 = xp[0], x1 = xp[1];

        unsigned int q[4] = { p.x, p.y, p.z, p.w };
        int xv[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };

        int s = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int lo = pack4(nib(q[j], 0), nib(q[j], 1), nib(q[j], 2), nib(q[j], 3));
            int hi = pack4(nib(q[j], 4), nib(q[j], 5), nib(q[j], 6), nib(q[j], 7));
            s = __dp4a(lo, xv[2 * j],     s);
            s = __dp4a(hi, xv[2 * j + 1], s);
        }
        acc += (float)s * ws[g];
    }
    acc = warp_reduce(acc) * xs;
    if (lane == 0) y[row] = acc;
}

// ---------------------------------------------------------------------------
// Host-side quantization (symmetric, per group of 32)
// ---------------------------------------------------------------------------
static void quantize_rows_int8(const std::vector<float>& W, int M, int K,
                               std::vector<signed char>& q, std::vector<float>& scales) {
    int ng = K / QGROUP;
    q.resize((size_t)M * K);
    scales.resize((size_t)M * ng);
    for (int m = 0; m < M; ++m) {
        for (int g = 0; g < ng; ++g) {
            const float* src = &W[(size_t)m * K + g * QGROUP];
            float amax = 0.0f;
            for (int i = 0; i < QGROUP; ++i) amax = std::fmax(amax, std::fabs(src[i]));
            float sc = amax / 127.0f;
            if (sc == 0.0f) sc = 1e-12f;
            scales[(size_t)m * ng + g] = sc;
            for (int i = 0; i < QGROUP; ++i) {
                int v = (int)std::lrintf(src[i] / sc);
                q[(size_t)m * K + g * QGROUP + i] =
                    (signed char)std::max(-127, std::min(127, v));
            }
        }
    }
}

static void quantize_rows_int4(const std::vector<float>& W, int M, int K,
                               std::vector<unsigned char>& q, std::vector<float>& scales) {
    int ng = K / QGROUP;
    q.assign((size_t)M * (K / 2), 0);
    scales.resize((size_t)M * ng);
    for (int m = 0; m < M; ++m) {
        for (int g = 0; g < ng; ++g) {
            const float* src = &W[(size_t)m * K + g * QGROUP];
            float amax = 0.0f;
            for (int i = 0; i < QGROUP; ++i) amax = std::fmax(amax, std::fabs(src[i]));
            // Symmetric INT4 range is [-8, 7]; divide by 7 so nothing clips high.
            float sc = amax / 7.0f;
            if (sc == 0.0f) sc = 1e-12f;
            scales[(size_t)m * ng + g] = sc;
            for (int i = 0; i < QGROUP; ++i) {
                int v = (int)std::lrintf(src[i] / sc);
                v = std::max(-8, std::min(7, v));
                unsigned int nibble = (unsigned int)(v + 8) & 0xf;   // stored biased
                size_t byte = (size_t)m * (K / 2) + (size_t)(g * QGROUP + i) / 2;
                if (i % 2 == 0) q[byte] |= (unsigned char)nibble;
                else            q[byte] |= (unsigned char)(nibble << 4);
            }
        }
    }
}

// CPU model of the EXACT integer math the kernel does. This lets us prove the
// kernel is correct, separately from the question of how lossy INT4 is.
static void cpu_ref_q4(const std::vector<unsigned char>& q, const std::vector<float>& ws,
                       const std::vector<signed char>& xq, float xs,
                       std::vector<float>& y, int M, int K) {
    int ng = K / QGROUP;
    y.assign(M, 0.0f);
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;
        for (int g = 0; g < ng; ++g) {
            int s = 0;
            for (int i = 0; i < QGROUP; ++i) {
                size_t byte = (size_t)m * (K / 2) + (size_t)(g * QGROUP + i) / 2;
                int nibble = (i % 2 == 0) ? (q[byte] & 0xf) : (q[byte] >> 4);
                s += (nibble - 8) * (int)xq[g * QGROUP + i];
            }
            acc += (float)s * ws[(size_t)m * ng + g];
        }
        y[m] = acc * xs;
    }
}

// ---------------------------------------------------------------------------
struct Bench { float ms; double gbs; };

template <typename F>
static Bench timed(F launch, double bytes) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) launch();            // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());
    const int iters = 50;
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    ms /= iters;
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    Bench r; r.ms = ms; r.gbs = bytes / (ms / 1000.0) / 1e9;
    return r;
}

static void run_shape(const char* name, int M, int K, bool verify) {
    std::printf("\n--- %s  M=%d K=%d ---\n", name, M, K);
    int ng = K / QGROUP;

    std::vector<float> W((size_t)M * K), x(K);
    unsigned seed = 1234u;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed >> 8) / 8388608.0f - 1.0f);   // ~[-1,1]
    };
    for (size_t i = 0; i < W.size(); ++i) W[i] = rnd() * 0.05f;   // weight-like
    for (int i = 0; i < K; ++i) x[i] = rnd();

    std::vector<signed char> Wq8; std::vector<float> Ws8;
    quantize_rows_int8(W, M, K, Wq8, Ws8);
    std::vector<unsigned char> Wq4; std::vector<float> Ws4;
    quantize_rows_int4(W, M, K, Wq4, Ws4);

    float amax = 0.0f;
    for (int i = 0; i < K; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    float xs = amax / 127.0f;
    std::vector<signed char> xq(K);
    for (int i = 0; i < K; ++i) {
        xq[i] = (signed char)std::max(-127, std::min(127, (int)std::lrintf(x[i] / xs)));
    }

    float *dW, *dx, *dy, *dWs8, *dWs4;
    signed char *dWq8, *dxq;
    unsigned char* dWq4;
    CUDA_CHECK(cudaMalloc(&dW,   (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dx,   (size_t)K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy,   (size_t)M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWq8, (size_t)M * K));
    CUDA_CHECK(cudaMalloc(&dWs8, (size_t)M * ng * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWq4, (size_t)M * (K / 2)));
    CUDA_CHECK(cudaMalloc(&dWs4, (size_t)M * ng * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dxq,  (size_t)K));

    CUDA_CHECK(cudaMemcpy(dW,   W.data(),   (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx,   x.data(),   (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq8, Wq8.data(), (size_t)M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWs8, Ws8.data(), (size_t)M * ng * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq4, Wq4.data(), (size_t)M * (K / 2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWs4, Ws4.data(), (size_t)M * ng * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dxq,  xq.data(),  (size_t)K, cudaMemcpyHostToDevice));

    const int WPB = 8;                      // warps per block
    dim3 blk(WPB * 32);
    dim3 grd((M + WPB - 1) / WPB);

    std::vector<float> y32(M), y8(M), y4(M);

    Bench b32 = timed([&]{ gemv_fp32<<<grd, blk>>>(dW, dx, dy, M, K); },
                      (double)M * K * 4.0);
    CUDA_CHECK(cudaMemcpy(y32.data(), dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost));

    Bench b8 = timed([&]{ gemv_q8<<<grd, blk>>>(dWq8, dWs8, dxq, xs, dy, M, K); },
                     (double)M * K * 1.0);
    CUDA_CHECK(cudaMemcpy(y8.data(), dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost));

    Bench b4 = timed([&]{ gemv_q4<<<grd, blk>>>(dWq4, dWs4, dxq, xs, dy, M, K); },
                     (double)M * K * 0.5);
    CUDA_CHECK(cudaMemcpy(y4.data(), dy, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaGetLastError());

    // Accuracy vs the FP32 result. This is quantization loss, and is expected.
    auto rel_err = [&](const std::vector<float>& a) {
        double num = 0, den = 0;
        for (int i = 0; i < M; ++i) {
            double d = a[i] - y32[i];
            num += d * d;
            den += (double)y32[i] * y32[i];
        }
        return std::sqrt(num / std::max(den, 1e-30));
    };

    std::printf("  %-4s %8.3f ms  %7.1f GB/s   (baseline)\n", "fp32", b32.ms, b32.gbs);
    std::printf("  %-4s %8.3f ms  %7.1f GB/s   %.2fx faster, rel.err %.4f\n",
                "int8", b8.ms, b8.gbs, b32.ms / b8.ms, rel_err(y8));
    std::printf("  %-4s %8.3f ms  %7.1f GB/s   %.2fx faster, rel.err %.4f\n",
                "int4", b4.ms, b4.gbs, b32.ms / b4.ms, rel_err(y4));

    if (verify) {
        // Prove the INT4 kernel implements the intended integer math.
        //
        // Compare with an L2-relative norm, NOT per-element relative error.
        // The GPU sums groups in warp-strided order and the CPU sums them
        // sequentially; with signed weights the dot product nearly cancels,
        // so a tiny absolute difference looks huge relative to one small
        // element. ||err|| / ||ref|| is the honest measure here.
        std::vector<float> ref;
        cpu_ref_q4(Wq4, Ws4, xq, xs, ref, M, K);
        double num = 0, den = 0, maxabs = 0;
        for (int i = 0; i < M; ++i) {
            double d = (double)ref[i] - y4[i];
            num += d * d;
            den += (double)ref[i] * ref[i];
            maxabs = std::fmax(maxabs, std::fabs(d));
        }
        double l2 = std::sqrt(num / std::max(den, 1e-30));
        std::printf("  kernel vs CPU model: L2 rel %.2e, max abs %.2e -> %s\n",
                    l2, maxabs, l2 < 1e-5 ? "KERNEL CORRECT" : "MISMATCH");
    }

    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    cudaFree(dWq8); cudaFree(dWs8); cudaFree(dWq4); cudaFree(dWs4); cudaFree(dxq);
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    int mc = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mc, cudaDevAttrMemoryClockRate, 0));
    double peak = 2.0 * mc * (p.memoryBusWidth / 8) / 1.0e6;
    std::printf("%s  sm_%d%d  peak bandwidth %.1f GB/s\n", p.name, p.major, p.minor, peak);
    std::printf("Decode-time GEMV is memory bound: speed = weight bytes / bandwidth.\n");

    // Llama 3.2 1B shapes (hidden 2048, intermediate 8192, GQA with 8 kv heads).
    run_shape("attn q_proj ", 2048, 2048, true);
    run_shape("mlp gate/up ", 8192, 2048, true);
    run_shape("mlp down    ", 2048, 8192, true);

    // What that means for a whole model.
    const long long H = 2048, I = 8192, L = 16, KVD = 512, V = 128256;
    long long per_layer = H * H            // q_proj
                        + H * KVD * 2      // k_proj + v_proj (GQA: 8 kv heads x 64)
                        + H * H            // o_proj
                        + H * I * 2        // gate + up
                        + I * H;           // down
    long long total = per_layer * L + V * H;   // + lm_head

    std::printf("\n=== Llama 3.2 1B decode projection ===\n");
    std::printf("Total weights: %.2f B parameters\n", total / 1e9);
    struct Fmt { const char* n; double bpw; };
    Fmt fmt[] = { { "fp32", 4.0 }, { "fp16", 2.0 }, { "int8", 1.0 }, { "int4", 0.5625 } };
    for (int i = 0; i < 4; ++i) {
        double bytes = (double)total * fmt[i].bpw;
        double gb = bytes / 1e9;
        double tps = peak * 1e9 / bytes;        // one full weight sweep per token
        std::printf("  %-4s %6.2f GB  %-16s -> %6.1f tok/s at peak, ~%.0f realistic\n",
                    fmt[i].n, gb,
                    gb > 4.0 ? "EXCEEDS 4GB VRAM" : "fits in 4GB",
                    tps, tps * 0.75);
    }
    std::printf("\nint4 (4 bits + a 32-wide scale = 4.5 bits/weight) is what makes a 1B\n"
                "model genuinely fast here: the only format leaving room for the KV cache.\n");
    return 0;
}
