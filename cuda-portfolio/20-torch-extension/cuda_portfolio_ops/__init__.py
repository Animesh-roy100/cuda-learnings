"""Q4_0 quantized linear layers backed by hand-written CUDA kernels.

    import torch
    from cuda_portfolio_ops import Q4Linear

    layer = torch.nn.Linear(2048, 5632, bias=False).cuda()
    q = Q4Linear.from_float(layer)            # 4.5 bits per weight
    y = q(torch.randn(16, 2048, device="cuda"))

The kernels are the W4A8 and W4A16 GEMVs of the portfolio's 19-llm-engine,
batched. Weights are frozen; gradients flow to the input (and bias), which is
the shape adapter fine-tuning over a quantized base model needs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import _C

__all__ = ["quantize_q4_0", "dequantize_q4_0", "q4_linear", "Q4Linear"]

GROUP = 32


def quantize_q4_0(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [out, in] float matrix to Q4_0 (ggml's scheme).

    Per group of 32 weights, the scale is the signed value of largest magnitude
    divided by -8, so that value lands exactly on code 0 and the rest round into
    0..15. Returns (nibbles uint8 [out, in/2], scales float32 [out, in/32]) on
    the weight's device, nibbles packed interleaved as the kernels read them.
    """
    if weight.dim() != 2:
        raise ValueError("quantize_q4_0: weight must be 2-D [out_features, in_features]")
    out_features, in_features = weight.shape
    if in_features % GROUP:
        raise ValueError(f"quantize_q4_0: in_features ({in_features}) must be a multiple of {GROUP}")

    w = weight.detach().to(torch.float32).reshape(out_features, in_features // GROUP, GROUP)
    idx = w.abs().argmax(dim=-1, keepdim=True)
    extreme = torch.gather(w, -1, idx).squeeze(-1)           # signed max-magnitude value
    scale = extreme / -8.0
    inv = torch.where(scale != 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.clamp(torch.floor(w * inv.unsqueeze(-1) + 8.5), 0, 15).to(torch.uint8)

    q = q.reshape(out_features, in_features)
    nibbles = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return nibbles, scale.contiguous()


def dequantize_q4_0(nibbles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dense float32 [out, in] from Q4_0, computed by a CUDA kernel."""
    return _C.dequantize(nibbles, scales)


class _Q4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nibbles, scales, bias, int8: bool):
        ctx.save_for_backward(nibbles, scales)
        ctx.has_bias = bias is not None
        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        x32 = x.to(torch.float32)
        y = _C.gemv_int8(x32, nibbles, scales) if int8 else _C.gemv_float(x32, nibbles, scales)
        if bias is not None:
            y = y + bias.to(torch.float32)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        nibbles, scales = ctx.saved_tensors
        grad_x = grad_bias = None
        g32 = grad_out.to(torch.float32)
        if ctx.needs_input_grad[0]:
            # dL/dx = dL/dy W. The exact dequantized matrix, not the int8 path:
            # a lossy backward would bias every update made through it.
            w = _C.dequantize(nibbles, scales)
            grad_x = (g32 @ w).to(ctx.input_dtype).reshape(ctx.input_shape)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = g32.reshape(-1, g32.shape[-1]).sum(0)
        return grad_x, None, None, grad_bias, None


def q4_linear(
    x: torch.Tensor,
    nibbles: torch.Tensor,
    scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    int8_activations: bool = True,
) -> torch.Tensor:
    """y = x W^T + b with W in Q4_0. Accepts any leading dimensions."""
    return _Q4LinearFn.apply(x, nibbles, scales, bias, int8_activations)


class Q4Linear(torch.nn.Module):
    """A frozen Q4_0 linear layer. Construct with from_float()."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 int8_activations: bool = True, device=None):
        super().__init__()
        if in_features % GROUP:
            raise ValueError(f"in_features ({in_features}) must be a multiple of {GROUP}")
        self.in_features = in_features
        self.out_features = out_features
        self.int8_activations = int8_activations
        self.register_buffer("nibbles", torch.zeros(out_features, in_features // 2,
                                                    dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.zeros(out_features, in_features // GROUP,
                                                   dtype=torch.float32, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device)) if bias else None

    @classmethod
    def from_float(cls, linear: torch.nn.Linear, int8_activations: bool = True) -> "Q4Linear":
        q = cls(linear.in_features, linear.out_features, linear.bias is not None,
                int8_activations, device=linear.weight.device)
        nib, sc = quantize_q4_0(linear.weight)
        q.nibbles.copy_(nib)
        q.scales.copy_(sc)
        if linear.bias is not None:
            q.bias.data.copy_(linear.bias.detach().to(torch.float32))
        return q

    def weight_bytes(self) -> int:
        return self.nibbles.numel() + self.scales.numel() * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return q4_linear(x, self.nibbles, self.scales, self.bias, self.int8_activations)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, int8_activations={self.int8_activations}")
