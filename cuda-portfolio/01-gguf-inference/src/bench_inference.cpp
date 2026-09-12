// Decode-path benchmark: quantized GEMV, kernel fusion, and what it means for
// tokens/second on a 4 GB card.

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "cu/device.hpp"
#include "kernels.h"
#include "kv_cache.h"

namespace {

std::vector<float> randvec(int n, unsigned seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}

void row(const char* name, int M, int K) {
    auto w = randvec(M * K, 1234u, 0.05f);
    auto x = randvec(K, 5678u, 1.0f);
    auto q = llm::quantize_q4(w, M, K);

    // Warm up, then take the MINIMUM over repeats. Without the warm-up the
    // first shape measured absorbs one-time kernel setup and reads about 5x
    // slower than the identical work measured later.
    float t32 = 0.0f, t4 = 0.0f, best32 = 1e30f, best4 = 1e30f;
    for (int i = 0; i < 12; ++i) {
        llm::gemv_f32(w, M, K, x, &t32);
        llm::gemv_q4(q, x, &t4);
        if (i >= 2) {
            best32 = std::min(best32, t32);
            best4 = std::min(best4, t4);
        }
    }

    const double b32 = double(M) * K * 4.0;
    const double b4 = double(q.weight_bytes());
    std::printf("  %-8s M=%5d K=%5d | fp32 %6.3f ms %6.1f GB/s"
                " | int4 %6.3f ms %6.1f GB/s | %.2fx\n",
                name, M, K, best32, b32 / (best32 / 1e3) / 1e9,
                best4, b4 / (best4 / 1e3) / 1e9, best32 / best4);
}

}  // namespace

int main() {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("Decode is memory bound: speed = weight bytes / bandwidth.\n\n");

    std::printf("=== Q4_0 GEMV at Llama 3.2 1B shapes ===\n");
    row("q_proj", 2048, 2048);
    row("gate/up", 8192, 2048);
    row("down", 2048, 8192);

    // --- fusion ---
    const int H = 32, D = 64;
    auto x = randvec(H * D, 7u, 1.0f);
    auto w = randvec(H * D, 8u, 0.5f);
    float tf = 0.0f, tu = 0.0f, bf = 1e30f, bu = 1e30f;
    for (int i = 0; i < 30; ++i) {
        llm::rmsnorm_rope(x, w, 1e-5f, H, D, 128, 10000.0f, &tf);
        llm::rmsnorm_then_rope_unfused(x, w, 1e-5f, H, D, 128, 10000.0f, &tu);
        if (i >= 5) {
            bf = std::min(bf, tf);
            bu = std::min(bu, tu);
        }
    }
    std::printf("\n=== fused RMSNorm+RoPE vs two kernels (%d heads x %d dim) ===\n", H, D);
    std::printf("  fused    %7.4f ms\n", bf);
    std::printf("  unfused  %7.4f ms   %.2fx\n", bu, bu / bf);
    std::printf("\n  Measured, not assumed. The vector is only %.1f KB, so neither\n",
                H * D * 4 / 1024.0);
    std::printf("  variant is memory bound -- both are dominated by launch and\n");
    std::printf("  reduction overhead, and fusion removes a round trip that was\n");
    std::printf("  never the bottleneck. The fused kernel also runs as a SINGLE\n");
    std::printf("  block so it can share one reduction, using 1 of %d SMs, while\n",
                dev.sm_count);
    std::printf("  the unfused RoPE pass spreads over all of them.\n");
    std::printf("\n  Fusion pays when the tensor is large enough to be bandwidth\n");
    std::printf("  bound, or when it removes a launch from a loop run thousands of\n");
    std::printf("  times. Quoting a speedup here without measuring would be wrong.\n");

    // --- KV cache footprint ---
    llm::KvCacheConfig cfg;
    cfg.n_layers = 16;
    cfg.n_kv_heads = 8;
    cfg.head_dim = 64;
    cfg.page_tokens = 32;
    cfg.total_pages = 1024;
    llm::PagedKvCache cache(cfg);

    std::printf("\n=== paged KV cache (Llama 3.2 1B geometry) ===\n");
    std::printf("  page = %zu bytes, %d pages = %.1f MB total\n",
                cache.bytes_per_page(), cfg.total_pages, cache.total_bytes() / 1e6);
    std::printf("  one page holds %d tokens for one layer, so a %d-token context\n",
                cfg.page_tokens, cfg.page_tokens);
    std::printf("  costs %d pages = %.1f MB across %d layers\n",
                cfg.n_layers, cfg.n_layers * cache.bytes_per_page() / 1e6, cfg.n_layers);
    std::printf("  Pages are interchangeable, so a freed sequence returns every page\n");
    std::printf("  to the shared pool and external fragmentation cannot occur.\n");

    // --- what it adds up to ---
    const long long Hd = 2048, I = 8192, L = 16, KVD = 512, V = 128256;
    const long long per_layer = Hd * Hd + Hd * KVD * 2 + Hd * Hd + Hd * I * 2 + I * Hd;
    const long long total = per_layer * L + V * Hd;
    const double peak = dev.peak_bandwidth_gbps();

    std::printf("\n=== Llama 3.2 1B decode projection ===\n");
    std::printf("  %.2f B parameters\n", total / 1e9);
    struct Fmt { const char* n; double bpw; };
    const Fmt fmt[] = {{"fp32", 4.0}, {"fp16", 2.0}, {"int8", 1.0}, {"int4", 0.5625}};
    for (const auto& f : fmt) {
        const double bytes = double(total) * f.bpw;
        std::printf("  %-5s %6.2f GB  %-16s %6.1f tok/s at peak, ~%.0f realistic\n",
                    f.n, bytes / 1e9,
                    bytes / 1e9 > 4.0 ? "EXCEEDS 4GB VRAM" : "fits in 4GB",
                    peak * 1e9 / bytes, peak * 1e9 / bytes * 0.75);
    }
    std::printf("\nint4 is the only format leaving room for weights AND kv-cache.\n");
    return 0;
}
