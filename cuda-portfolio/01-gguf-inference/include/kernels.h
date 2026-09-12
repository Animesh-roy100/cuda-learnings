#pragma once
//
// Kernel launchers. Pure C++20 signatures -- no CUDA syntax, so tests and host
// pipeline code never need nvcc.
//
// Contract shared by every function here (see NUMERICS.md for the numbers):
//
//   - Arguments are validated before anything touches the device. A bad shape,
//     size, epsilon, theta or position is std::invalid_argument and no memory
//     is allocated and no kernel launched. Byte counts are computed with
//     checked size_t arithmetic.
//   - A request the DEVICE cannot run -- a launch past its grid or block
//     limits, __dp4a forced on a GPU without it -- is llm::DeviceUnsupported,
//     also before any launch.
//   - These are test and benchmark entry points: each call allocates its own
//     device buffers. The token loop in 19-llm-engine allocates a workspace
//     once instead, which is what a generation path must do.
//   - Thread safety: functions are re-entrant; calls on different threads use
//     independent buffers and the device's default stream.
//
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

class GgufFile;
struct GgufTensor;

// The device cannot execute a valid request: missing instruction, launch past
// a device limit. Distinct from invalid_argument, which is the caller's error.
class DeviceUnsupported : public std::runtime_error {
public:
    explicit DeviceUnsupported(const std::string& what) : std::runtime_error(what) {}
};

// Q4 with the llama.cpp nibble arithmetic: 32 weights share one scale, stored
// as signed nibbles biased by +8.
//
// This is an INTERNAL layout, not the on-disk GGUF block: scales are FP32
// (5.0 bits per weight, not 4.5) and nibbles are interleaved (weight 2b in the
// low nibble of byte b, 2b+1 in the high), which is the order the __dp4a kernel
// consumes. Native GGUF Q4_0 blocks -- FP16 scale, first 16 weights in the low
// nibbles, last 16 in the high -- must go through q4_from_gguf_q4_0, never be
// copied in as if they were this.
//
// Decode is memory bound, so time is (weight bytes / bandwidth), and INT4
// moves 8x fewer bytes than FP32.
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

// rows > 0, cols > 0, cols % 32 == 0, qs.size() == rows*cols/2,
// scales.size() == rows*cols/32, every scale finite. Throws invalid_argument.
void validate(const Q4Matrix& m);

// Weights must be finite; NaN or infinity is rejected rather than quantized.
Q4Matrix quantize_q4(const std::vector<float>& w, int rows, int cols);
std::vector<float> dequantize_q4(const Q4Matrix& m);

// Import native GGUF Q4_0: `bytes` must be exactly rows * (cols/32) * 18.
// FP16 scales are decoded; a non-finite scale is rejected.
Q4Matrix q4_from_gguf_q4_0(const std::uint8_t* blocks, std::size_t bytes, int rows, int cols);
// Same, from a parsed file: the tensor must be Q4_0 and two-dimensional.
Q4Matrix q4_from_gguf(const GgufFile& file, const GgufTensor& tensor);

// Which integer inner loop gemv_q4 runs.
enum class Q4Kernel {
    Auto,       // Dp4a where the device has it (sm_61+), otherwise Portable
    Dp4a,       // 4-way INT8 dot products; DeviceUnsupported below sm_61
    Portable,   // plain integer multiply-adds; any device
};
bool device_supports_dp4a();

// y = W x, W quantized to Q4, activations quantized to INT8 on the fly.
// Activations must be finite. Both kernels compute the same integers in the
// same order, so their outputs are bitwise identical.
std::vector<float> gemv_q4(const Q4Matrix& w, const std::vector<float>& x,
                           float* elapsed_ms = nullptr, Q4Kernel kernel = Q4Kernel::Auto);

// FP32 reference GEMV, same launch shape -- the baseline to beat.
// rows > 0, cols > 0, cols % 4 == 0, w.size() == rows*cols, x.size() == cols.
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
// x non-empty, weight.size() == x.size(), eps finite and >= 0.
std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& weight,
                           float eps);

// Fused RMSNorm + rotary position embedding (the original GPT-NeoX/Llama
// "rotate half" variant, no frequency scaling).
//
// Unfused, these are two kernels and therefore two full round trips to device
// memory for the same vector. Fusing removes one -- but the fused kernel
// stages the whole vector in one block's shared memory, which is a per-block
// device limit (48 KB on sm_75). Vectors past it run the unfused kernels,
// which use global memory, and produce the same result; `fused` reports which
// ran.
//
// Input is laid out [n_heads][head_dim]. n_heads > 0, head_dim > 0 and even,
// n_heads*head_dim == x.size() == weight.size(), eps finite >= 0, theta finite
// > 0, 0 <= pos <= 2^24 (positions are exact in FP32 up to there).
std::vector<float> rmsnorm_rope(const std::vector<float>& x,
                                const std::vector<float>& weight, float eps,
                                int n_heads, int head_dim, int pos, float theta,
                                float* elapsed_ms = nullptr, bool* fused = nullptr);

// Largest vector the fused kernel can stage on the current device.
std::size_t rmsnorm_rope_fused_capacity();

// Same maths on the CPU, for tests to check against. Same validation.
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

// Numerically stable (max-subtracted, double accumulation). Any NaN makes every
// output NaN. +inf entries share the probability equally and finite entries get
// 0. All -inf is invalid_argument: there is no distribution to return.
void softmax_cpu(std::vector<float>& v);

}  // namespace llm
