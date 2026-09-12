// Attention: materialized vs fused.
//
// Every kernel is checked against an FP64 host reference that computes the
// softmax through log-sum-exp -- not against another kernel, so a mistake the
// kernels share (a transposed score matrix, a mask off by one) cannot hide.

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "flash_attention.h"

namespace {

std::vector<float> random_tensor(const fa::Shape& s, unsigned seed, float amp = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, amp);
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

struct Case {
    fa::Kernel kernel;
    int tile;
    std::string name() const {
        std::string s = fa::to_string(kernel);
        if (kernel == fa::Kernel::FusedTiled) s += " " + std::to_string(tile);
        std::string out;
        for (char c : s) out += std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
        return out;
    }
};

std::vector<Case> all_cases() {
    std::vector<Case> v = {{fa::Kernel::NaiveCublas, 0}, {fa::Kernel::FusedGlobal, 0}};
    for (int t : fa::tile_sizes()) v.push_back({fa::Kernel::FusedTiled, t});
    return v;
}

// FP32 attention against FP64: the error is rounding in the dot products and the
// running normalizer, a few parts in 1e6 relative.
constexpr double kTol = 2e-5;

}  // namespace

class EveryKernel : public ::testing::TestWithParam<Case> {};

TEST_P(EveryKernel, MatchesTheFp64Reference) {
    fa::Shape s{2, 128, 64, false};
    const auto q = random_tensor(s, 1), k = random_tensor(s, 2), v = random_tensor(s, 3);
    fa::Attention att(s);
    att.set_qkv(q, k, v);
    EXPECT_LT(rel_frobenius(att.forward(GetParam().kernel, GetParam().tile),
                            fa::reference_attention(s, q, k, v)),
              kTol);
}

TEST_P(EveryKernel, MatchesTheReferenceWhenCausal) {
    fa::Shape s{2, 128, 64, true};
    const auto q = random_tensor(s, 4), k = random_tensor(s, 5), v = random_tensor(s, 6);
    fa::Attention att(s);
    att.set_qkv(q, k, v);
    EXPECT_LT(rel_frobenius(att.forward(GetParam().kernel, GetParam().tile),
                            fa::reference_attention(s, q, k, v)),
              kTol);
}

TEST_P(EveryKernel, HandlesSequencesThatAreNotMultiplesOfTheWarpOrTile) {
    // 100 leaves a partial warp of queries and a partial tile of keys.
    for (bool causal : {false, true}) {
        fa::Shape s{3, 100, 64, causal};
        const auto q = random_tensor(s, 7), k = random_tensor(s, 8), v = random_tensor(s, 9);
        fa::Attention att(s);
        att.set_qkv(q, k, v);
        EXPECT_LT(rel_frobenius(att.forward(GetParam().kernel, GetParam().tile),
                                fa::reference_attention(s, q, k, v)),
                  kTol)
            << (causal ? "causal" : "bidirectional");
    }
}

TEST_P(EveryKernel, CausalFirstQueryReturnsTheFirstValueExactly) {
    // Query 0 may attend only to key 0, so its softmax is exactly [1] and the
    // output is exactly v_0 -- a check that needs no reference at all.
    fa::Shape s{1, 64, 64, true};
    const auto q = random_tensor(s, 10), k = random_tensor(s, 11), v = random_tensor(s, 12);
    fa::Attention att(s);
    att.set_qkv(q, k, v);
    const auto o = att.forward(GetParam().kernel, GetParam().tile);
    for (int c = 0; c < 64; ++c) EXPECT_FLOAT_EQ(o[c], v[c]) << "dim " << c;
}

TEST_P(EveryKernel, SurvivesLogitsThatOverflowANaiveSoftmax) {
    // Amplitude 40 puts logits in the hundreds; exp(700) overflows FP32 many
    // times over. Both the max-subtracted materialized softmax and the online
    // update keep every exponent <= 0, so neither may produce inf or NaN.
    fa::Shape s{1, 96, 64, false};
    const auto q = random_tensor(s, 13, 40.0f), k = random_tensor(s, 14, 40.0f);
    const auto v = random_tensor(s, 15);
    fa::Attention att(s);
    att.set_qkv(q, k, v);
    const auto o = att.forward(GetParam().kernel, GetParam().tile);
    for (float x : o) ASSERT_TRUE(std::isfinite(x));
    EXPECT_LT(rel_frobenius(o, fa::reference_attention(s, q, k, v)), 1e-4);
}

INSTANTIATE_TEST_SUITE_P(Attention, EveryKernel, ::testing::ValuesIn(all_cases()),
                         [](const auto& info) { return info.param.name(); });

// ---------------------------------------------------------------- structure

TEST(Attention, HeadsAreIndependent) {
    // Swapping two heads' inputs must swap exactly those heads' outputs.
    fa::Shape s{2, 64, 64, false};
    auto q = random_tensor(s, 20), k = random_tensor(s, 21), v = random_tensor(s, 22);
    fa::Attention att(s);
    att.set_qkv(q, k, v);
    const auto before = att.forward(fa::Kernel::FusedTiled, 32);
    const std::size_t per = std::size_t(s.seq) * s.head_dim;
    for (auto* t : {&q, &k, &v})
        std::swap_ranges(t->begin(), t->begin() + per, t->begin() + per);
    att.set_qkv(q, k, v);
    const auto after = att.forward(fa::Kernel::FusedTiled, 32);
    for (std::size_t i = 0; i < per; ++i) {
        ASSERT_FLOAT_EQ(after[i], before[i + per]);
        ASSERT_FLOAT_EQ(after[i + per], before[i]);
    }
}

TEST(Attention, NaiveMemoryIsQuadraticAndFusedIsLinear) {
    fa::Attention small(fa::Shape{8, 1024, 64, false});
    fa::Attention large(fa::Shape{8, 4096, 64, false});
    const double naive = double(large.device_bytes(fa::Kernel::NaiveCublas)) /
                         small.device_bytes(fa::Kernel::NaiveCublas);
    const double fused = double(large.device_bytes(fa::Kernel::FusedTiled)) /
                         small.device_bytes(fa::Kernel::FusedTiled);
    EXPECT_NEAR(fused, 4.0, 1e-9);
    EXPECT_GT(naive, 10.0) << "4x the sequence should cost close to 16x for naive";
}

TEST(Attention, FusedOccupancyIsReported) {
    fa::Attention att(fa::Shape{1, 64, 64, false});
    EXPECT_GT(att.blocks_per_sm(fa::Kernel::FusedGlobal), 0);
    for (int t : fa::tile_sizes()) EXPECT_GT(att.blocks_per_sm(fa::Kernel::FusedTiled, t), 0);
}

// ----------------------------------------------------------------- contract

TEST(Attention, FusedKernelsRejectOtherHeadDims) {
    fa::Shape s{1, 32, 48, false};
    fa::Attention att(s);
    att.set_qkv(random_tensor(s, 1), random_tensor(s, 2), random_tensor(s, 3));
    EXPECT_THROW(att.forward(fa::Kernel::FusedGlobal), std::invalid_argument);
    EXPECT_NO_THROW(att.forward(fa::Kernel::NaiveCublas)) << "naive supports any head_dim";
}

TEST(Attention, RejectsATileWithNoKernel) {
    fa::Shape s{1, 32, 64, false};
    fa::Attention att(s);
    att.set_qkv(random_tensor(s, 1), random_tensor(s, 2), random_tensor(s, 3));
    EXPECT_THROW(att.forward(fa::Kernel::FusedTiled, 20), std::invalid_argument);
}

TEST(Attention, RejectsBadShapesAndTensors) {
    EXPECT_THROW((fa::Attention(fa::Shape{0, 32, 64, false})), std::invalid_argument);
    fa::Attention att(fa::Shape{1, 32, 64, false});
    EXPECT_THROW(att.set_qkv({}, {}, {}), std::invalid_argument);
    EXPECT_THROW(att.forward(fa::Kernel::FusedGlobal), std::logic_error);
}
