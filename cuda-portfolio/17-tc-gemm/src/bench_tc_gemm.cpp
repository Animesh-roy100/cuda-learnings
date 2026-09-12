// Tiled tensor-core GEMM against cuBLAS.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "cu/device.hpp"
#include "tc_gemm.h"

namespace {

std::vector<float> random_matrix(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> m(std::size_t(n) * n);
    for (auto& v : m) v = d(rng);
    return m;
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

// The cuBLAS path each hand-written path is judged against: same operand
// precision, and for fp16 whichever of cuBLAS's two fp16 configurations is
// faster -- judging against the slower one would flatter the hand-written code.
tc::Path cublas_peer(tc::Path p, const std::map<tc::Path, float>& ms) {
    if (tc::is_int8(p)) return tc::Path::CublasInt8;
    if (tc::is_fp16(p))
        return ms.at(tc::Path::CublasFp16) <= ms.at(tc::Path::CublasFp16Hgemm)
                   ? tc::Path::CublasFp16
                   : tc::Path::CublasFp16Hgemm;
    return tc::Path::CublasFp32;
}

}  // namespace

int main() try {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("C = A*B, square, row-major. GFLOPS = billions of multiply-adds per\n");
    std::printf("second (N^3 per product). Error is relative Frobenius vs cuBLAS fp32.\n");

    for (int n : {512, 2048, 4096}) {
        const auto a = random_matrix(n, 11);
        const auto b = random_matrix(n, 12);
        tc::Gemm g(n);
        g.set_inputs(a, b);
        const auto ref = g.multiply(tc::Path::CublasFp32);

        // Fewer repetitions at sizes where one tiled product takes seconds.
        const int iters = n >= 4096 ? 3 : 5;
        const int warm = n >= 4096 ? 1 : 2;

        std::map<tc::Path, float> ms;
        for (auto p : tc::all_paths()) ms[p] = g.time(p, iters, warm);

        std::printf("\n=== N = %d ===\n", n);
        std::printf("  %-28s %10s %9s %13s %11s\n", "path", "ms", "GFLOPS", "% of cuBLAS", "error");
        for (auto p : tc::all_paths()) {
            const double err = (p == tc::Path::CublasFp32) ? 0.0 : rel_frobenius(g.multiply(p), ref);
            const double pct = tc::is_cublas(p) ? 100.0 : 100.0 * ms[cublas_peer(p, ms)] / ms[p];
            std::printf("  %-28s %10.2f %9.1f %12.1f%% %11.2e\n", tc::to_string(p), ms[p],
                        tc::Gemm::gflops(n, ms[p]), pct, err);
        }

        auto ratio = [&](tc::Path slow, tc::Path fast) { return ms[slow] / ms[fast]; };
        std::printf("\n  controlled comparisons (>1 means the second path is faster):\n");
        std::printf("    staging by element vs from global   fp16 %5.2fx   int8 %5.2fx\n",
                    ratio(tc::Path::Fp16WmmaGlobal, tc::Path::Fp16WmmaStaged),
                    ratio(tc::Path::Int8WmmaGlobal, tc::Path::Int8WmmaStaged));
        std::printf("    staging by word vs by element        fp16 %5.2fx   int8 %5.2fx\n",
                    ratio(tc::Path::Fp16WmmaStaged, tc::Path::Fp16WmmaStagedWord),
                    ratio(tc::Path::Int8WmmaStaged, tc::Path::Int8WmmaStagedWord));
        std::printf("    staging by word vs from global       fp16 %5.2fx   int8 %5.2fx\n",
                    ratio(tc::Path::Fp16WmmaGlobal, tc::Path::Fp16WmmaStagedWord),
                    ratio(tc::Path::Int8WmmaGlobal, tc::Path::Int8WmmaStagedWord));
        std::printf("    pipelined vs staged by word          fp16 %5.2fx   int8 %5.2fx\n",
                    ratio(tc::Path::Fp16WmmaStagedWord, tc::Path::Fp16WmmaPipelined),
                    ratio(tc::Path::Int8WmmaStagedWord, tc::Path::Int8WmmaPipelined));
        std::printf("    pipelined vs from global             fp16 %5.2fx   int8 %5.2fx\n",
                    ratio(tc::Path::Fp16WmmaGlobal, tc::Path::Fp16WmmaPipelined),
                    ratio(tc::Path::Int8WmmaGlobal, tc::Path::Int8WmmaPipelined));
        std::printf("    int8 tensor cores vs int8 __dp4a     %5.2fx  (same data, from global)\n",
                    ratio(tc::Path::Int8Dp4a, tc::Path::Int8WmmaGlobal));
        std::printf("    fp16 wmma vs fp32 tiled              %5.2fx  (both hand-written)\n",
                    ratio(tc::Path::Fp32Tiled, tc::Path::Fp16WmmaGlobal));
        std::printf("    cuBLAS fp16: 16F vs 32F compute      %5.2fx\n",
                    ratio(tc::Path::CublasFp16, tc::Path::CublasFp16Hgemm));
        std::printf("    cuBLAS: int8 vs fp32                 %5.2fx\n",
                    ratio(tc::Path::CublasFp32, tc::Path::CublasInt8));
    }

    {
        // The band sweep: occupancy against instruction-level parallelism.
        const int n = 2048;
        tc::Gemm g(n);
        g.set_inputs(random_matrix(n, 11), random_matrix(n, 12));
        const float global_fp16 = g.time(tc::Path::Fp16WmmaGlobal, 5, 2);
        const float global_int8 = g.time(tc::Path::Int8WmmaGlobal, 5, 2);
        std::printf("\n=== Band sweep at N = %d: occupancy vs work per warp ===\n", n);
        std::printf("  Loading from global, for reference: fp16 %.2f ms, int8 %.2f ms\n",
                    global_fp16, global_int8);
        for (bool int8 : {false, true})
            for (auto s : {tc::Staging::ByElement, tc::Staging::ByWord, tc::Staging::Pipelined}) {
                std::printf("\n  %s, %s\n", int8 ? "int8" : "fp16", tc::to_string(s));
                std::printf("  %6s %13s %9s %9s %12s\n", "band", "compute warps", "ms",
                            "GFLOPS", "vs global");
                for (const auto& p : g.sweep_band(s, int8, 5, 2))
                    std::printf("  %6d %13d %9.2f %9.1f %11.2fx\n", p.band,
                                p.compute_warps_per_sm, p.ms, tc::Gemm::gflops(n, p.ms),
                                (int8 ? global_int8 : global_fp16) / p.ms);
            }
    }

    std::printf("\n=== Occupancy: compute warps resident per SM ===\n");
    std::printf("  %-28s %8s %13s %15s\n", "path", "threads", "blocks / SM", "compute warps");
    for (const auto& o : tc::occupancy())
        std::printf("  %-28s %8d %13d %15d\n", tc::to_string(o.path), o.threads_per_block,
                    o.blocks_per_sm, o.compute_warps_per_sm());

    auto u4 = tc::check_u4_fragment();
    std::printf("\n=== Experimental 4-bit fragment (8x8x32, u4) ===\n");
    std::printf("  %d of %d products wrong against a host reference\n", u4.wrong, u4.total);
    std::printf("  The sub-byte WMMA types exist on sm_73+ and were deprecated after\n");
    std::printf("  Turing: kept here as a correctness check, not as a path to build on.\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
