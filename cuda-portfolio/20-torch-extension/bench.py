"""Q4_0 linear layers against PyTorch's own linear, at TinyLlama's layer shapes.

    python bench.py

Every configuration runs the same weights. Timed with torch.cuda.synchronize
around each call after warm-up; each number is the median of the runs.
"""

import statistics
import time

import torch
import torch.nn.functional as F

import cuda_portfolio_ops as ops

DEV = "cuda"


def median_ms(fn, runs=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


def main():
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}")
    shapes = [("attn q/o  2048->2048", 2048, 2048),
              ("ffn up    2048->5632", 5632, 2048),
              ("ffn down  5632->2048", 2048, 5632)]
    batches = [1, 16, 128]

    for label, out_f, in_f in shapes:
        torch.manual_seed(0)
        w32 = torch.randn(out_f, in_f, device=DEV) * 0.02
        w16 = w32.half()
        nib, sc = ops.quantize_q4_0(w32)
        mb = lambda t: t.numel() * t.element_size() / 2**20
        print(f"\n=== {label} ===  weights: fp32 {mb(w32):.1f} MB, fp16 {mb(w16):.1f} MB, "
              f"q4_0 {mb(nib) + mb(sc):.1f} MB")
        print(f"  {'batch':>5} {'fp32 F.linear':>14} {'fp16 F.linear':>14} {'dequant+linear':>15} "
              f"{'q4 float':>9} {'q4 int8':>8}")
        for b in batches:
            x32 = torch.randn(b, in_f, device=DEV)
            x16 = x32.half()
            t = {
                "fp32": median_ms(lambda: F.linear(x32, w32)),
                "fp16": median_ms(lambda: F.linear(x16, w16)),
                "deq": median_ms(lambda: F.linear(x32, ops.dequantize_q4_0(nib, sc))),
                "q4f": median_ms(lambda: ops.q4_linear(x32, nib, sc, int8_activations=False)),
                "q4i": median_ms(lambda: ops.q4_linear(x32, nib, sc, int8_activations=True)),
            }
            print(f"  {b:>5} {t['fp32']:>11.3f} ms {t['fp16']:>11.3f} ms {t['deq']:>12.3f} ms "
                  f"{t['q4f']:>6.3f} ms {t['q4i']:>5.3f} ms")


if __name__ == "__main__":
    main()
