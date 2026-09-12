// Attention: materialized (cuBLAS) vs fused online-softmax kernels.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "cu/device.hpp"
#include "flash_attention.h"

namespace {

std::vector<float> random_tensor(const fa::Shape& s, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> t(std::size_t(s.heads) * s.seq * s.head_dim);
    for (auto& x : t) x = d(rng);
    return t;
}

double rel_frobenius(const std::vector<float>& x, const std::vector<float>& ref) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = double(x[i]) - ref[i];
        num += d * d;
        den += double(ref[i]) * ref[i];
    }
    return std::sqrt(num / den);
}

double mb(std::size_t b) { return double(b) / (1024.0 * 1024.0); }

}  // namespace

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("Scaled dot-product attention, head_dim 64. Naive materializes the\n");
    std::printf("[heads][seq][seq] score matrix with cuBLAS; the fused kernels never do.\n");

    // ------------------------------------------------------------ tile sweep
    {
        fa::Shape s{8, 2048, 64, false};
        fa::Attention att(s);
        att.set_qkv(random_tensor(s, 1), random_tensor(s, 2), random_tensor(s, 3));
        const float global = att.time(fa::Kernel::FusedGlobal, 0, 5, 2);
        std::printf("\n=== Tile sweep: 8 heads x 2048 tokens ===\n");
        std::printf("  %-22s %11s %9s %12s\n", "kernel", "blocks/SM", "ms", "vs global");
        std::printf("  %-22s %11d %9.2f %11.2fx\n", "fused, from global",
                    att.blocks_per_sm(fa::Kernel::FusedGlobal), global, 1.0);
        for (int t : fa::tile_sizes()) {
            const float ms = att.time(fa::Kernel::FusedTiled, t, 5, 2);
            char label[32];
            std::snprintf(label, sizeof label, "fused, tile %d", t);
            std::printf("  %-22s %11d %9.2f %11.2fx\n", label,
                        att.blocks_per_sm(fa::Kernel::FusedTiled, t), ms, global / ms);
        }
    }

    // --------------------------------------------------------- length sweep
    int best_tile = 32;
    {
        fa::Shape s{8, 2048, 64, false};
        fa::Attention att(s);
        att.set_qkv(random_tensor(s, 1), random_tensor(s, 2), random_tensor(s, 3));
        float best = 1e30f;
        for (int t : fa::tile_sizes()) {
            const float ms = att.time(fa::Kernel::FusedTiled, t, 3, 1);
            if (ms < best) best = ms, best_tile = t;
        }
    }

    std::printf("\n=== Sequence length: 8 heads, bidirectional (tiled uses tile %d) ===\n",
                best_tile);
    std::printf("  %6s  %-28s %10s %11s %10s %9s\n", "seq", "kernel", "ms", "device MB",
                "vs naive", "error");
    for (int n : {512, 1024, 2048, 4096, 8192}) {
        fa::Shape s{8, n, 64, false};
        fa::Attention att(s);
        const auto q = random_tensor(s, 11), k = random_tensor(s, 12), v = random_tensor(s, 13);
        att.set_qkv(q, k, v);
        const int iters = n >= 4096 ? 3 : 5;
        const int warm = n >= 4096 ? 1 : 2;

        // The naive kernel's accuracy is the reference at sizes where an FP64
        // host product would take minutes.
        const auto ref = att.forward(fa::Kernel::NaiveCublas);
        const float naive = att.time(fa::Kernel::NaiveCublas, 0, iters, warm);
        for (auto kern : {fa::Kernel::NaiveCublas, fa::Kernel::FusedGlobal, fa::Kernel::FusedTiled}) {
            const float ms = kern == fa::Kernel::NaiveCublas ? naive
                                                             : att.time(kern, best_tile, iters, warm);
            const double err = kern == fa::Kernel::NaiveCublas
                                   ? 0.0
                                   : rel_frobenius(att.forward(kern, best_tile), ref);
            std::printf("  %6d  %-28s %10.2f %11.1f %9.2fx %9.1e\n", n, fa::to_string(kern), ms,
                        mb(att.device_bytes(kern)), naive / ms, err);
        }
    }

    // ---------------------------------------------------------------- causal
    std::printf("\n=== Causal masking, 8 heads x 4096 tokens ===\n");
    for (bool causal : {false, true}) {
        fa::Shape s{8, 4096, 64, causal};
        fa::Attention att(s);
        att.set_qkv(random_tensor(s, 21), random_tensor(s, 22), random_tensor(s, 23));
        std::printf("  %-14s", causal ? "causal" : "bidirectional");
        for (auto kern : {fa::Kernel::NaiveCublas, fa::Kernel::FusedGlobal, fa::Kernel::FusedTiled})
            std::printf("   %s %.1f ms", kern == fa::Kernel::NaiveCublas
                                             ? "naive"
                                             : (kern == fa::Kernel::FusedGlobal ? "global" : "tiled"),
                        att.time(kern, best_tile, 3, 1));
        std::printf("\n");
    }
    std::printf("  A causal query skips the keys after it, so the fused kernels do about\n");
    std::printf("  half the work. The naive kernel still multiplies the full N x N matrix\n");
    std::printf("  and only zeroes the masked half during the softmax.\n");

    // --------------------------------------------------- beyond VRAM (naive)
    {
        fa::Shape s{16, 8192, 64, false};
        fa::Attention att(s);
        att.set_qkv(random_tensor(s, 31), random_tensor(s, 32), random_tensor(s, 33));
        std::printf("\n=== 16 heads x 8192 tokens: a score matrix larger than the card ===\n");
        std::printf("  naive needs %.0f MB of device memory on a %.0f MB card\n",
                    mb(att.device_bytes(fa::Kernel::NaiveCublas)),
                    double(dev.total_mem) / (1024.0 * 1024.0));
        const float tiled = att.time(fa::Kernel::FusedTiled, best_tile, 2, 1);
        std::printf("  fused tiled: %.1f ms using %.0f MB\n", tiled,
                    mb(att.device_bytes(fa::Kernel::FusedTiled)));
        try {
            const float naive = att.time(fa::Kernel::NaiveCublas, 0, 1, 0);
            std::printf("  naive:       %.1f ms -- it did not fail. On Windows the driver's\n", naive);
            std::printf("  sysmem fallback moved the score matrix into system RAM, so the\n");
            std::printf("  cost of exceeding VRAM appears as time (%.1fx the fused kernel)\n",
                        naive / tiled);
            std::printf("  instead of as an out-of-memory error.\n");
        } catch (const std::exception& e) {
            std::printf("  naive:       failed -- %s\n", e.what());
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
