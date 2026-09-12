#pragma once
//
// Q4_0 kernels behind a plain-C++ interface: raw pointers and a stream, no
// torch headers and no CUDA syntax. nvcc never sees torch's headers and the
// binding never sees a kernel, which is what keeps this extension building on
// both MSVC and gcc.
//
// Weight layout (the same as 19-llm-engine's device weights):
//   nibbles  uint8  [out_features, in_features / 2]   interleaved: weight e of
//            a row at byte e/2, shift 4*(e&1), stored with a +8 bias
//   scales   float32 [out_features, in_features / 32] one per group of 32
//
#include <cstdint>

namespace q4 {

// y[b][r] = sum over groups g: scale[r][g] * sum_j (nib[r][g,j] - 8) * x[b][g,j]
// batch x in_features activations in, batch x out_features out. All pointers
// are device pointers; `stream` is the CUDA stream to launch on.
void gemv_float(const std::uint8_t* nibbles, const float* scales, const float* x, float* y,
                int batch, int out_features, int in_features, void* stream);

// Activations quantized to int8 per group of 32 on the device, then multiplied
// with __dp4a (W4A8). The scratch buffers must hold batch*in_features int8 and
// batch*(in_features/32) floats.
void gemv_int8(const std::uint8_t* nibbles, const float* scales, const float* x, float* y,
               signed char* xq_scratch, float* xs_scratch, int batch, int out_features,
               int in_features, void* stream);

// Dense float32 [out_features, in_features] from the packed form.
void dequantize(const std::uint8_t* nibbles, const float* scales, float* w, int out_features,
                int in_features, void* stream);

// Surfaces the last kernel-launch error, if any, as a message; empty if none.
const char* last_error();

}  // namespace q4
