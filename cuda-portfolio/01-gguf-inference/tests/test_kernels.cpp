// Kernel tests: Q4_0 quantization, __dp4a GEMV, RMSNorm, fused RMSNorm+RoPE.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "kernels.h"
#include "kv_cache.h"

using llm::KvCacheConfig;
using llm::PagedKvCache;
using llm::Q4Matrix;

namespace {

std::vector<float> randvec(int n, unsigned seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}

// Relative L2 error. Per-element relative error is meaningless for dot
// products: signed terms nearly cancel, so a tiny absolute error looks huge
// next to one small output element.
double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - b[i];
        num += d * d;
        den += double(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

TEST(Quant, RoundTripIsWithinQuantizationStep) {
    const int R = 16, C = 128;
    auto w = randvec(R * C, 1, 0.05f);
    auto q = llm::quantize_q4(w, R, C);
    auto d = llm::dequantize_q4(q);

    ASSERT_EQ(d.size(), w.size());
    // Each group of 32 shares a scale = amax/7, so error is bounded by half a
    // step: amax/14.
    for (int r = 0; r < R; ++r)
        for (int g = 0; g < C / 32; ++g) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i)
                amax = std::fmax(amax, std::fabs(w[r * C + g * 32 + i]));
            for (int i = 0; i < 32; ++i) {
                std::size_t k = r * C + g * 32 + i;
                EXPECT_LE(std::fabs(d[k] - w[k]), amax / 14.0f + 1e-6f);
            }
        }
}

TEST(Quant, BitsPerWeightIsFourPointFive) {
    auto w = randvec(64 * 256, 2, 0.05f);
    auto q = llm::quantize_q4(w, 64, 256);
    // 32 nibbles (16 bytes) + one fp32 scale (4 bytes) = 20 bytes / 32 weights.
    // Real GGUF Q4_0 stores the scale as fp16 for 4.5 bits; this uses fp32.
    EXPECT_NEAR(q.bits_per_weight(), 5.0, 1e-9);
}

TEST(Quant, RejectsBadShape) {
    auto w = randvec(100, 3);
    EXPECT_THROW(llm::quantize_q4(w, 10, 10), std::invalid_argument);   // 10 % 32 != 0
    EXPECT_THROW(llm::quantize_q4(w, 3, 32), std::invalid_argument);    // size mismatch
}

TEST(Quant, ZeroRowQuantizesToZeroWithoutNaN) {
    std::vector<float> w(64, 0.0f);
    auto q = llm::quantize_q4(w, 2, 32);
    auto d = llm::dequantize_q4(q);
    for (float v : d) {
        EXPECT_FALSE(std::isnan(v));
        EXPECT_FLOAT_EQ(v, 0.0f);
    }
}

// KERNEL CORRECTNESS. Compared against a CPU model of the identical W4A8
// integer arithmetic, so the only remaining difference is float accumulation
// order. Anything above epsilon here is a real packing or __dp4a bug.
TEST(GemvQ4, KernelMatchesCpuModelExactly) {
    const int M = 512, K = 1024;
    auto w = randvec(M * K, 4, 0.05f);
    auto x = randvec(K, 5);
    auto q = llm::quantize_q4(w, M, K);

    auto gpu = llm::gemv_q4(q, x);
    auto cpu = llm::gemv_q4_cpu(q, x);

    ASSERT_EQ(gpu.size(), cpu.size());
    EXPECT_LT(rel_l2(gpu, cpu), 1e-5) << "kernel disagrees with its own integer model";
}

// QUANTIZATION LOSS, a different question entirely. Weights are already INT4
// in both paths here, so this isolates the INT8 ACTIVATION quantization.
// A per-tensor INT8 scale over a Gaussian activation vector loses a few tenths
// of a percent; measured ~0.6% at K=1024. This bound describes the format, not
// the implementation, and tightening it would only make the suite flaky.
TEST(GemvQ4, ActivationQuantizationLossIsBounded) {
    const int M = 512, K = 1024;
    auto w = randvec(M * K, 4, 0.05f);
    auto x = randvec(K, 5);

    auto q = llm::quantize_q4(w, M, K);
    auto deq = llm::dequantize_q4(q);

    auto w4a8 = llm::gemv_q4(q, x);          // INT8 activations
    auto w4a32 = llm::gemv_f32(deq, M, K, x);  // FP32 activations, same weights

    EXPECT_LT(rel_l2(w4a8, w4a32), 2e-2) << "INT8 activation loss";
}

TEST(GemvQ4, AgreesWithFp32WithinQuantizationError) {
    const int M = 256, K = 512;
    auto w = randvec(M * K, 6, 0.05f);
    auto x = randvec(K, 7);
    auto q = llm::quantize_q4(w, M, K);

    auto qy = llm::gemv_q4(q, x);
    auto fy = llm::gemv_f32(w, M, K, x);
    // INT4 weights: a few percent is the expected, irreducible loss.
    EXPECT_LT(rel_l2(qy, fy), 0.15);
}

TEST(GemvQ4, RejectsMismatchedInput) {
    auto w = randvec(32 * 64, 8, 0.05f);
    auto q = llm::quantize_q4(w, 32, 64);
    EXPECT_THROW(llm::gemv_q4(q, randvec(63, 9)), std::invalid_argument);
}

TEST(RmsNorm, MatchesCpu) {
    auto x = randvec(2048, 10);
    auto w = randvec(2048, 11, 0.5f);
    auto gpu = llm::rmsnorm(x, w, 1e-5f);
    auto cpu = llm::rmsnorm_cpu(x, w, 1e-5f);
    EXPECT_LT(rel_l2(gpu, cpu), 1e-5);
}

TEST(RmsNorm, UnitWeightPreservesRmsOfOne) {
    auto x = randvec(1024, 12);
    std::vector<float> w(1024, 1.0f);
    auto y = llm::rmsnorm(x, w, 0.0f);
    double ss = 0.0;
    for (float v : y) ss += double(v) * v;
    EXPECT_NEAR(std::sqrt(ss / y.size()), 1.0, 1e-4);
}

TEST(Rope, FusedMatchesCpu) {
    const int H = 8, D = 64;
    auto x = randvec(H * D, 13);
    auto w = randvec(H * D, 14, 0.5f);
    for (int pos : {0, 1, 7, 128, 1024}) {
        auto gpu = llm::rmsnorm_rope(x, w, 1e-5f, H, D, pos, 10000.0f);
        auto cpu = llm::rmsnorm_rope_cpu(x, w, 1e-5f, H, D, pos, 10000.0f);
        EXPECT_LT(rel_l2(gpu, cpu), 1e-5) << "pos=" << pos;
    }
}

TEST(Rope, FusedMatchesUnfused) {
    const int H = 8, D = 64;
    auto x = randvec(H * D, 15);
    auto w = randvec(H * D, 16, 0.5f);
    auto fused = llm::rmsnorm_rope(x, w, 1e-5f, H, D, 42, 10000.0f);
    auto unfused = llm::rmsnorm_then_rope_unfused(x, w, 1e-5f, H, D, 42, 10000.0f);
    EXPECT_LT(rel_l2(fused, unfused), 1e-5) << "fusion must not change the answer";
}

TEST(Rope, AtPositionZeroIsIdentityRotation) {
    const int H = 4, D = 32;
    auto x = randvec(H * D, 17);
    std::vector<float> w(H * D, 1.0f);
    auto roped = llm::rmsnorm_rope(x, w, 1e-5f, H, D, 0, 10000.0f);
    auto plain = llm::rmsnorm(x, w, 1e-5f);
    // angle = 0 -> cos=1, sin=0 -> no rotation at all.
    EXPECT_LT(rel_l2(roped, plain), 1e-6);
}

TEST(Rope, PreservesNormWithinEachHead) {
    const int H = 4, D = 32;
    auto x = randvec(H * D, 18);
    std::vector<float> w(H * D, 1.0f);
    auto before = llm::rmsnorm(x, w, 1e-5f);
    auto after = llm::rmsnorm_rope(x, w, 1e-5f, H, D, 77, 10000.0f);
    // A rotation is orthogonal, so per-head magnitude must be unchanged.
    for (int h = 0; h < H; ++h) {
        double a = 0, b = 0;
        for (int i = 0; i < D; ++i) {
            a += double(before[h * D + i]) * before[h * D + i];
            b += double(after[h * D + i]) * after[h * D + i];
        }
        EXPECT_NEAR(std::sqrt(a), std::sqrt(b), 1e-3) << "head " << h;
    }
}

TEST(Rope, RejectsBadShape) {
    auto x = randvec(100, 19);
    auto w = randvec(100, 20);
    EXPECT_THROW(llm::rmsnorm_rope(x, w, 1e-5f, 3, 32, 0, 10000.0f), std::invalid_argument);
    auto x2 = randvec(4 * 33, 21);
    auto w2 = randvec(4 * 33, 22);
    EXPECT_THROW(llm::rmsnorm_rope(x2, w2, 1e-5f, 4, 33, 0, 10000.0f), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Paged KV cache
// ---------------------------------------------------------------------------
namespace {
KvCacheConfig small_cfg() {
    KvCacheConfig c;
    c.n_layers = 2;
    c.n_kv_heads = 2;
    c.head_dim = 4;
    c.page_tokens = 4;
    c.total_pages = 16;
    return c;
}
}  // namespace

TEST(KvCache, StartsEmpty) {
    PagedKvCache c(small_cfg());
    EXPECT_EQ(c.pages_in_use(), 0);
    EXPECT_EQ(c.free_pages(), 16);
    EXPECT_EQ(c.sequence_count(), 0);
    EXPECT_EQ(c.bytes_per_page(), 4u * 2 * 4 * 2 * sizeof(float));
}

TEST(KvCache, AppendAndGatherRoundTrip) {
    auto cfg = small_cfg();
    PagedKvCache c(cfg);
    int s = c.create_sequence();

    const int vals = cfg.n_kv_heads * cfg.head_dim;
    std::vector<std::vector<float>> keys;
    for (int t = 0; t < 10; ++t) {
        std::vector<float> k(vals), v(vals);
        for (int i = 0; i < vals; ++i) {
            k[i] = float(t * 100 + i);
            v[i] = float(-(t * 100 + i));
        }
        keys.push_back(k);
        for (int l = 0; l < cfg.n_layers; ++l) c.append(s, l, k, v);
    }

    EXPECT_EQ(c.length(s), 10);
    auto gk = c.gather_keys(s, 0);
    ASSERT_EQ(gk.size(), static_cast<std::size_t>(10 * vals));
    for (int t = 0; t < 10; ++t)
        for (int i = 0; i < vals; ++i)
            EXPECT_FLOAT_EQ(gk[t * vals + i], keys[t][i]) << "t=" << t << " i=" << i;

    auto gv = c.gather_values(s, 1);
    for (int t = 0; t < 10; ++t)
        for (int i = 0; i < vals; ++i)
            EXPECT_FLOAT_EQ(gv[t * vals + i], -keys[t][i]);
}

TEST(KvCache, AllocatesPagesLazily) {
    auto cfg = small_cfg();
    PagedKvCache c(cfg);
    int s = c.create_sequence();
    EXPECT_EQ(c.pages_in_use(), 0) << "an empty sequence must cost no pages";

    const int vals = cfg.n_kv_heads * cfg.head_dim;
    std::vector<float> k(vals, 1.0f), v(vals, 2.0f);

    // page_tokens = 4, so the first 4 tokens fit in one page per layer.
    for (int t = 0; t < 4; ++t)
        for (int l = 0; l < cfg.n_layers; ++l) c.append(s, l, k, v);
    EXPECT_EQ(c.pages_in_use(), cfg.n_layers);

    for (int l = 0; l < cfg.n_layers; ++l) c.append(s, l, k, v);   // 5th token
    EXPECT_EQ(c.pages_in_use(), 2 * cfg.n_layers);
}

// The whole reason paging exists: freed pages come back to a shared pool and
// are immediately reusable, so a long sequence followed by a short one cannot
// strand memory.
TEST(KvCache, FreedPagesAreFullyReclaimed) {
    auto cfg = small_cfg();
    PagedKvCache c(cfg);
    const int vals = cfg.n_kv_heads * cfg.head_dim;
    std::vector<float> k(vals, 1.0f), v(vals, 2.0f);

    for (int round = 0; round < 5; ++round) {
        int s = c.create_sequence();
        for (int t = 0; t < 12; ++t)
            for (int l = 0; l < cfg.n_layers; ++l) c.append(s, l, k, v);
        EXPECT_GT(c.pages_in_use(), 0);
        c.free_sequence(s);
        EXPECT_EQ(c.pages_in_use(), 0) << "round " << round << ": pages leaked";
        EXPECT_EQ(c.free_pages(), cfg.total_pages);
    }
}

TEST(KvCache, ThrowsWhenExhausted) {
    KvCacheConfig cfg = small_cfg();
    cfg.total_pages = 2;
    cfg.n_layers = 1;
    PagedKvCache c(cfg);
    int s = c.create_sequence();
    const int vals = cfg.n_kv_heads * cfg.head_dim;
    std::vector<float> k(vals, 1.0f), v(vals, 1.0f);

    // 2 pages x 4 tokens = 8 tokens fit; the 9th must fail cleanly.
    for (int t = 0; t < 8; ++t) c.append(s, 0, k, v);
    EXPECT_THROW(c.append(s, 0, k, v), std::runtime_error);
}

TEST(KvCache, IndependentSequencesDoNotInterfere) {
    auto cfg = small_cfg();
    PagedKvCache c(cfg);
    int a = c.create_sequence();
    int b = c.create_sequence();
    const int vals = cfg.n_kv_heads * cfg.head_dim;

    std::vector<float> ka(vals, 7.0f), kb(vals, -7.0f), v(vals, 0.0f);
    for (int t = 0; t < 6; ++t) {
        for (int l = 0; l < cfg.n_layers; ++l) c.append(a, l, ka, v);
        for (int l = 0; l < cfg.n_layers; ++l) c.append(b, l, kb, v);
    }
    EXPECT_EQ(c.length(a), 6);
    EXPECT_EQ(c.length(b), 6);

    for (float x : c.gather_keys(a, 0)) EXPECT_FLOAT_EQ(x, 7.0f);
    for (float x : c.gather_keys(b, 0)) EXPECT_FLOAT_EQ(x, -7.0f);
}

TEST(KvCache, RejectsInvalidIds) {
    PagedKvCache c(small_cfg());
    int s = c.create_sequence();
    const int vals = 2 * 4;
    std::vector<float> k(vals, 1.0f), v(vals, 1.0f);

    EXPECT_THROW(c.append(99, 0, k, v), std::out_of_range);
    EXPECT_THROW(c.append(s, 99, k, v), std::out_of_range);
    EXPECT_THROW(c.append(s, 0, std::vector<float>(3), v), std::invalid_argument);
    c.free_sequence(s);
    EXPECT_THROW(c.length(s), std::out_of_range) << "a freed sequence must not be usable";
}
