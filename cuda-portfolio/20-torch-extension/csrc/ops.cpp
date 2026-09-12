// PyTorch binding for the Q4_0 kernels. Compiled by the host compiler only.
//
// Every argument is validated here, before a kernel sees it: device, dtype,
// contiguity, shapes, and the grid limits a launch can express. A kernel that
// receives a CPU pointer does not fail cleanly -- it faults the context -- so
// the checks are not optional politeness.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <string>

#include "q4_kernels.h"

namespace {

constexpr std::int64_t kMaxGridDim = 65535;   // CUDA grid y/z dimension limit

void check_weights(const torch::Tensor& nibbles, const torch::Tensor& scales) {
    TORCH_CHECK(nibbles.is_cuda() && scales.is_cuda(), "q4: weights must be CUDA tensors");
    TORCH_CHECK(nibbles.dtype() == torch::kUInt8, "q4: nibbles must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kFloat32, "q4: scales must be float32");
    TORCH_CHECK(nibbles.dim() == 2 && scales.dim() == 2, "q4: nibbles and scales must be 2-D");
    TORCH_CHECK(nibbles.is_contiguous() && scales.is_contiguous(), "q4: weights must be contiguous");
    TORCH_CHECK(nibbles.size(0) == scales.size(0), "q4: nibbles and scales disagree on out_features");
    TORCH_CHECK(nibbles.size(1) == scales.size(1) * 16,
                "q4: nibbles must hold 16 bytes per scale (32 weights per group)");
    TORCH_CHECK(nibbles.get_device() == scales.get_device(), "q4: weights on different devices");
    TORCH_CHECK(nibbles.size(0) <= kMaxGridDim, "q4: out_features above 65535 is not supported");
}

void check_raised() {
    const char* err = q4::last_error();
    TORCH_CHECK(err[0] == '\0', "q4: ", err);
}

// x: [..., in_features] -> [batch, in_features], contiguous float32 on the
// weights' device.
torch::Tensor flatten_input(const torch::Tensor& x, const torch::Tensor& nibbles) {
    TORCH_CHECK(x.is_cuda(), "q4: input must be a CUDA tensor");
    TORCH_CHECK(x.get_device() == nibbles.get_device(), "q4: input and weights on different devices");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "q4: input must be float32");
    const std::int64_t in_features = nibbles.size(1) * 2;
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) == in_features, "q4: input's last dimension must be ",
                in_features);
    return x.reshape({-1, in_features}).contiguous();
}

torch::Tensor gemv(const torch::Tensor& x, const torch::Tensor& nibbles,
                   const torch::Tensor& scales, bool int8) {
    check_weights(nibbles, scales);
    const c10::cuda::CUDAGuard guard(nibbles.device());
    auto flat = flatten_input(x, nibbles);
    const std::int64_t batch = flat.size(0);
    const int out_features = int(nibbles.size(0));
    const int in_features = int(nibbles.size(1) * 2);

    auto y = torch::empty({batch, out_features}, flat.options());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    torch::Tensor xq, xs;
    if (int8) {
        xq = torch::empty({batch, in_features}, flat.options().dtype(torch::kInt8));
        xs = torch::empty({batch, in_features / 32}, flat.options());
    }
    // A launch grid's batch dimension tops out at 65535, so larger batches are
    // launched in chunks.
    for (std::int64_t b0 = 0; b0 < batch; b0 += kMaxGridDim) {
        const int nb = int(std::min(kMaxGridDim, batch - b0));
        const float* xp = flat.data_ptr<float>() + b0 * in_features;
        float* yp = y.data_ptr<float>() + b0 * out_features;
        if (int8) {
            q4::gemv_int8(nibbles.data_ptr<std::uint8_t>(), scales.data_ptr<float>(), xp, yp,
                          xq.data_ptr<std::int8_t>() + 0, xs.data_ptr<float>(), nb, out_features,
                          in_features, stream);
        } else {
            q4::gemv_float(nibbles.data_ptr<std::uint8_t>(), scales.data_ptr<float>(), xp, yp, nb,
                           out_features, in_features, stream);
        }
        check_raised();
    }
    auto sizes = x.sizes().vec();
    sizes.back() = out_features;
    return y.reshape(sizes);
}

torch::Tensor gemv_float(const torch::Tensor& x, const torch::Tensor& nibbles,
                         const torch::Tensor& scales) {
    return gemv(x, nibbles, scales, false);
}

torch::Tensor gemv_int8(const torch::Tensor& x, const torch::Tensor& nibbles,
                        const torch::Tensor& scales) {
    return gemv(x, nibbles, scales, true);
}

torch::Tensor dequantize(const torch::Tensor& nibbles, const torch::Tensor& scales) {
    check_weights(nibbles, scales);
    const c10::cuda::CUDAGuard guard(nibbles.device());
    const int out_features = int(nibbles.size(0));
    const int in_features = int(nibbles.size(1) * 2);
    auto w = torch::empty({out_features, in_features}, scales.options());
    q4::dequantize(nibbles.data_ptr<std::uint8_t>(), scales.data_ptr<float>(), w.data_ptr<float>(),
                   out_features, in_features, c10::cuda::getCurrentCUDAStream().stream());
    check_raised();
    return w;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Q4_0 quantized linear kernels from the cuda-learnings portfolio";
    m.def("gemv_float", &gemv_float, "Q4_0 weights x float32 activations (W4A16)");
    m.def("gemv_int8", &gemv_int8, "Q4_0 weights x per-group int8 activations via __dp4a (W4A8)");
    m.def("dequantize", &dequantize, "Q4_0 packed weights to dense float32");
}
