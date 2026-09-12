#pragma once
//
// Kernel launchers. Pure C++20 signatures -- no CUDA syntax, so tests and host
// pipeline code never need nvcc.
//
#include <cstdint>
#include <vector>

namespace llm {

// Q4_0, the llama.cpp layout: 32 weights share one scale, stored as signed
// nibbles biased by +8. 4.5 bits per weight all-in.
//
// This is the format that makes a 1B model fast on a 4 GB card: decode is
// memory bound, so time is (weight bytes / bandwidth), and INT4 moves 8x fewer
// bytes than FP32.
struct Q4Matrix {
    int rows = 0;
    int cols = 0;                          // must be a multiple of 32
    std::vector<std::uint8_t> qs;          // rows * cols/2  packed nibbles
    std::vector<float> scales;             // rows * cols/32 per-group scales

    std::size_t weight_bytes() const { return qs.size() + scales.size() * sizeof(float); }
    double bits_per_weight() const {
        return rows && cols ? 8.0 * weight_bytes() / (double(rows) * cols) : 0.0;
    }
};

Q4Matrix quantize_q4(const std::vector<float>& w, int rows, int cols);
std::vector<float> dequantize_q4(const Q4Matrix& m);

// y = W x, W quantized to Q4_0, activations quantized to INT8 on the fly and
// accumulated with __dp4a (4-way INT8 dot product, sm_61+).
std::vector<float> gemv_q4(const Q4Matrix& w, const std::vector<float>& x,
                           float* elapsed_ms = nullptr);

// FP32 reference GEMV, same launch shape -- the baseline to beat.
std::vector<float> gemv_f32(const std::vector<float>& w, int rows, int cols,
                            const std::vector<float>& x, float* elapsed_ms = nullptr);

// CPU model of the EXACT integer arithmetic gemv_q4 performs, including the
// INT8 activation quantization.
//
// This separates two things that are easy to conflate: whether the KERNEL is
// correct, and how lossy W4A8 is. Comparing the kernel against dequantized
// FP32 measures the latter and says nothing useful about the former -- it
// fails when quantization is working exactly as designed. Comparing against
// this model isolates kernel correctness, and should match to float epsilon.
std::vector<float> gemv_q4_cpu(const Q4Matrix& w, const std::vector<float>& x);

// RMSNorm: x * weight / sqrt(mean(x^2) + eps). No mean subtraction, unlike
// LayerNorm -- that is the whole difference, and it saves a pass.
std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& weight,
                           float eps);

// Fused RMSNorm + rotary position embedding.
//
// Unfused, these are two kernels and therefore two full round trips to device
// memory for the same vector. At decode time the vector is small and the work
// is trivial, so those round trips ARE the cost. Fusing removes one entirely.
// Input is laid out [n_heads][head_dim]; head_dim must be even.
std::vector<float> rmsnorm_rope(const std::vector<float>& x,
                                const std::vector<float>& weight, float eps,
                                int n_heads, int head_dim, int pos, float theta,
                                float* elapsed_ms = nullptr);

// Same maths on the CPU, for tests to check against.
std::vector<float> rmsnorm_cpu(const std::vector<float>& x,
                               const std::vector<float>& weight, float eps);
std::vector<float> rmsnorm_rope_cpu(const std::vector<float>& x,
                                    const std::vector<float>& weight, float eps,
                                    int n_heads, int head_dim, int pos, float theta);

// Unfused device path, so the benchmark can price the fusion.
std::vector<float> rmsnorm_then_rope_unfused(const std::vector<float>& x,
                                             const std::vector<float>& weight, float eps,
                                             int n_heads, int head_dim, int pos,
                                             float theta, float* elapsed_ms = nullptr);

void softmax_cpu(std::vector<float>& v);

}  // namespace llm
