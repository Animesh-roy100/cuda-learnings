# 20. A PyTorch extension: Q4_0 linear layers (`cuda_portfolio_ops`)

19-llm-engine's quantized GEMV, packaged as a PyTorch op with autograd:

```python
import torch
from cuda_portfolio_ops import Q4Linear

layer = torch.nn.Linear(2048, 5632, bias=False).cuda()
q = Q4Linear.from_float(layer)                  # frozen 4-bit weights
y = q(torch.randn(16, 2048, device="cuda"))     # any leading dimensions
y.sum().backward()                              # gradients reach the input
```

## Build

```bash
pip install --no-build-isolation ./20-torch-extension
pytest 20-torch-extension/tests
python 20-torch-extension/bench.py
```

`--no-build-isolation` compiles against the torch you already have. The CUDA
toolkit's **major** version must match torch's - torch refuses to build
otherwise. On Windows, run from a shell with `cl.exe` on PATH (after
`vcvars64.bat`). Measured here with torch 2.14.0+cu132 against a CUDA 13.4
toolkit, which builds with a minor-version warning. On Colab, torch and a
matching toolkit are preinstalled.

## Design

- **Kernels and bindings are separate translation units.** `q4_kernels.cu`
  includes no torch header and `ops.cpp` launches no kernel. That is not
  tidiness: torch 2.14's headers require C++20, and the split lets nvcc stay on
  C++17 while the binding compiles as C++20.
- **Every argument is checked before a kernel sees it** - device, dtype,
  contiguity, shape. A kernel handed a CPU pointer does not raise; it faults
  the context. The tests confirm the context is healthy after each rejection.
- **Launches use PyTorch's current CUDA stream**, so the op composes with
  torch's own asynchrony instead of forcing a synchronize.
- **Batches beyond 65,535** are launched in chunks: the batch occupies a grid
  dimension, and grid y is capped at 65,535.
- **Backward uses the exact dequantized matrix**, not the int8 path. A lossy
  backward would bias every update made through it.

## Tests

Every output is checked against PyTorch itself - `F.linear` on the dequantized
matrix, in float64 - and every gradient against autograd through the same
expression. 14 tests: exact round trips of representable weights, the
quantization error bound, independent unpacking, 1-D to 3-D inputs, batches past
one grid dimension, fp16 inputs, input and bias gradients, frozen weights,
memory, and input validation.

Two of the first tests were wrong, and the corrections are facts about Q4_0:

- **The error bound is not half a step everywhere.** Codes run -8..7 and the
  scale puts the most extreme value on -8, so a value of the opposite sign can
  reach +8 steps and is clipped to +7: up to one full step of error on that side.
- **It is 5 bits per weight at runtime, not 4.5.** GGUF stores the scales as
  fp16; the kernels read float32 so no kernel decodes fp16.

## Results (GTX 1650, TinyLlama layer shapes)

| layer, batch | fp32 `F.linear` | fp16 `F.linear` | dequant + linear | **Q4 int8** |
|---|---|---|---|---|
| 2048->2048, 1 | 0.158 ms | 0.125 ms | 1.113 ms | **0.126 ms** |
| 2048->5632, 1 | 0.316 ms | 0.218 ms | 2.445 ms | **0.126 ms** |
| 2048->5632, 16 | 1.033 ms | 5.101 ms | 3.070 ms | **0.717 ms** |
| 2048->5632, 128 | **1.745 ms** | 9.866 ms | 3.710 ms | 6.089 ms |

Weights: 44.0 MB fp32, 22.0 MB fp16, **6.9 MB** Q4_0.

**The kernel wins where decoding happens**: at batch 1 to 16 it is 1.4-2.5x
faster than PyTorch's fp32 linear with 6.4x less weight memory. **At batch 128
cuBLAS wins by 3.5x**: a batched GEMM amortizes its weight reads across the
batch, while this GEMV reads the weights again for every batch element - so it
scales linearly and cuBLAS does not. The crossover sits between 16 and 128.

`fp16 F.linear` is slower than fp32 at every batch above 1 on this card - the
same cuBLAS behaviour 17-tc-gemm measured. The Q4 float path (not shown in the
table; 6-128 ms) exists as the exact reference, not as a fast path.

## Not built

- No kernel for batched prefill: at batch 128 the right answer on this card is
  to dequantize once and let cuBLAS run the GEMM.
- Not in CI: building it needs a CUDA-enabled torch wheel of several gigabytes
  per run.
