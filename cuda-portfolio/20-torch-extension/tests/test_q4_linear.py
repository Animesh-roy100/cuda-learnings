"""Tests for the Q4_0 extension.

Every kernel output is checked against PyTorch itself -- F.linear on the
dequantized matrix, in float64 -- and every gradient against autograd through
that same expression. Nothing here trusts the extension to check itself.
"""

import pytest
import torch
import torch.nn.functional as F

import cuda_portfolio_ops as ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = "cuda"


def rel_err(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def make_layer(out_f=256, in_f=512, bias=True, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Linear(in_f, out_f, bias=bias).to(DEV)


# ----------------------------------------------------------------- quantizer

def test_quantize_round_trips_exactly_representable_weights():
    # Groups whose values are integer multiples of their scale lose nothing:
    # scale = -max/8 and codes -8..7 times the scale.
    codes = torch.arange(32, device=DEV, dtype=torch.float32) % 16 - 8     # -8..7
    w = torch.stack([codes * s for s in (0.5, -0.25, 2.0, 1.0)]).repeat(1, 2)   # [4, 64]
    nib, sc = ops.quantize_q4_0(w)
    assert torch.equal(ops.dequantize_q4_0(nib, sc), w)


def test_quantization_error_is_half_a_step_except_on_the_clipped_side():
    # Q4_0 codes run -8..7 and the scale puts the most extreme value on -8. A
    # value of the opposite sign can reach +8 steps but is clipped to +7, so
    # the bound is one step there and half a step everywhere else.
    layer = make_layer(64, 128)
    nib, sc = ops.quantize_q4_0(layer.weight)
    w = ops.dequantize_q4_0(nib, sc)
    step = sc.abs().repeat_interleave(32, dim=1)
    err = (w - layer.weight).abs()
    ratio = layer.weight / sc.repeat_interleave(32, dim=1)      # in steps, extreme at -8
    clipped = ratio > 7.5
    assert torch.all(err[~clipped] <= step[~clipped] * 0.5 + 1e-6)
    assert torch.all(err[clipped] <= step[clipped] * 1.0 + 1e-6)


def test_gpu_dequantize_matches_the_packing_it_undoes():
    # Independent unpacking in PyTorch: interleaved nibbles, +8 bias.
    layer = make_layer(8, 64)
    nib, sc = ops.quantize_q4_0(layer.weight)
    lo, hi = (nib & 0xF).to(torch.int16), (nib >> 4).to(torch.int16)
    q = torch.stack([lo, hi], dim=-1).reshape(8, 64) - 8
    ref = q.to(torch.float32) * sc.repeat_interleave(32, dim=1)
    assert torch.equal(ops.dequantize_q4_0(nib, sc), ref)


# ----------------------------------------------------------------- forward

@pytest.mark.parametrize("shape", [(1, 512), (17, 512), (3, 5, 512)])
def test_float_kernel_matches_pytorch_linear(shape):
    layer = make_layer()
    q = ops.Q4Linear.from_float(layer, int8_activations=False)
    x = torch.randn(*shape, device=DEV)
    ref = F.linear(x.double(), ops.dequantize_q4_0(q.nibbles, q.scales).double(), q.bias.double())
    y = q(x)
    assert y.shape == (*shape[:-1], 256)
    assert rel_err(y, ref) < 1e-5


def test_int8_kernel_is_close_to_pytorch_linear():
    layer = make_layer()
    q = ops.Q4Linear.from_float(layer, int8_activations=True)
    x = torch.randn(32, 512, device=DEV)
    ref = F.linear(x.double(), ops.dequantize_q4_0(q.nibbles, q.scales).double(), q.bias.double())
    # Per-group int8 activations: lossy, but within a few percent.
    assert rel_err(q(x), ref) < 0.03


def test_quantized_layer_tracks_the_original_float_layer():
    layer = make_layer(1024, 1024, bias=False)
    x = torch.randn(8, 1024, device=DEV)
    for int8 in (False, True):
        q = ops.Q4Linear.from_float(layer, int8_activations=int8)
        assert rel_err(q(x), layer(x)) < 0.25, "4-bit weights should not change the output wholesale"


def test_batches_beyond_one_launch_grid_dimension():
    # The binding launches in chunks of 65535 batch elements.
    layer = make_layer(32, 64, bias=False)
    q = ops.Q4Linear.from_float(layer, int8_activations=False)
    x = torch.randn(70000, 64, device=DEV)
    ref = F.linear(x, ops.dequantize_q4_0(q.nibbles, q.scales))
    assert rel_err(q(x), ref) < 1e-5


def test_preserves_half_precision_inputs():
    layer = make_layer()
    q = ops.Q4Linear.from_float(layer)
    y = q(torch.randn(4, 512, device=DEV, dtype=torch.float16))
    assert y.dtype == torch.float16


# ---------------------------------------------------------------- backward

def test_input_gradient_matches_autograd_through_the_dequantized_matrix():
    layer = make_layer()
    q = ops.Q4Linear.from_float(layer, int8_activations=True)
    x = torch.randn(6, 512, device=DEV, requires_grad=True)
    upstream = torch.randn(6, 256, device=DEV)
    (q(x) * upstream).sum().backward()

    w = ops.dequantize_q4_0(q.nibbles, q.scales).double()
    x_ref = x.detach().double().requires_grad_(True)
    b_ref = q.bias.detach().double().requires_grad_(True)
    (F.linear(x_ref, w, b_ref) * upstream.double()).sum().backward()

    assert rel_err(x.grad, x_ref.grad) < 1e-6
    assert rel_err(q.bias.grad, b_ref.grad) < 1e-6


def test_weights_are_frozen():
    q = ops.Q4Linear.from_float(make_layer())
    assert [n for n, p in q.named_parameters()] == ["bias"]
    assert not q.nibbles.requires_grad and not q.scales.requires_grad


# ------------------------------------------------------------------ memory

def test_weights_take_5_bits_each_at_runtime():
    # GGUF stores Q4_0 scales as fp16: 4.5 bits per weight on disk. The runtime
    # keeps them as float32 so no kernel decodes fp16, at 4 + 32/32 = 5 bits --
    # still 3.2x smaller than fp16 weights.
    q = ops.Q4Linear.from_float(make_layer(2048, 5632, bias=False))
    bits = q.weight_bytes() * 8 / (2048 * 5632)
    assert abs(bits - 5.0) < 1e-9


# ------------------------------------------------------------------ errors

def test_rejects_bad_inputs_before_any_kernel_runs():
    q = ops.Q4Linear.from_float(make_layer())
    with pytest.raises(RuntimeError, match="CUDA"):
        q(torch.randn(2, 512))                                   # CPU tensor
    with pytest.raises(RuntimeError, match="last dimension"):
        q(torch.randn(2, 500, device=DEV))
    with pytest.raises(RuntimeError, match="float32"):
        ops._C.gemv_float(torch.randn(2, 512, device=DEV, dtype=torch.float64),
                          q.nibbles, q.scales)
    with pytest.raises(ValueError, match="multiple of 32"):
        ops.quantize_q4_0(torch.randn(4, 100, device=DEV))
    # And the context is still healthy afterwards.
    assert torch.isfinite(q(torch.randn(2, 512, device=DEV))).all()
