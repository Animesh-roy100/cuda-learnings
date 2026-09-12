// Production hardening of the kernel wrappers and the paged KV cache:
// validation before launch, device capability and limit handling, native GGUF
// import, numerical guarantees, and the cache's ordering, context,
// transaction, sharing, device-read and threading rules.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "gguf.h"
#include "kernels.h"
#include "kv_cache.h"

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

std::vector<float> randvec(int n, unsigned seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}

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

// ===========================================================================
// Validation: nothing invalid reaches a launch.
// ===========================================================================

TEST(Validation, Q4MatrixInvariants) {
    const auto good = llm::quantize_q4(randvec(2 * 64, 30, 0.05f), 2, 64);
    EXPECT_NO_THROW(llm::validate(good));

    auto m = good;
    m.rows = 0;
    EXPECT_THROW(llm::validate(m), std::invalid_argument);
    m = good;
    m.cols = 48;
    EXPECT_THROW(llm::validate(m), std::invalid_argument);
    m = good;
    m.qs.pop_back();
    EXPECT_THROW(llm::validate(m), std::invalid_argument);
    m = good;
    m.scales.push_back(1.0f);
    EXPECT_THROW(llm::validate(m), std::invalid_argument);
    m = good;
    m.scales[1] = kNaN;
    EXPECT_THROW(llm::validate(m), std::invalid_argument);

    // A tampered matrix must not reach the device, or a host loop.
    m = good;
    m.qs.resize(8);
    EXPECT_THROW(llm::gemv_q4(m, randvec(64, 31)), std::invalid_argument);
    EXPECT_THROW(llm::gemv_q4_cpu(m, randvec(64, 31)), std::invalid_argument);
    EXPECT_THROW(llm::dequantize_q4(m), std::invalid_argument);
}

TEST(Validation, QuantizeRejectsNonFiniteWeightsAndEmptyShapes) {
    auto w = randvec(64, 32, 0.05f);
    w[5] = kInf;
    EXPECT_THROW(llm::quantize_q4(w, 1, 64), std::invalid_argument);
    w[5] = kNaN;
    EXPECT_THROW(llm::quantize_q4(w, 1, 64), std::invalid_argument);
    EXPECT_THROW(llm::quantize_q4({}, 0, 32), std::invalid_argument);
    EXPECT_THROW(llm::quantize_q4({}, 1, 0), std::invalid_argument);
    EXPECT_THROW(llm::quantize_q4({}, -1, -32), std::invalid_argument);
}

TEST(Validation, GemvRejectsBadArguments) {
    const auto q = llm::quantize_q4(randvec(4 * 64, 33, 0.05f), 4, 64);
    auto x = randvec(64, 34);
    x[3] = kNaN;
    EXPECT_THROW(llm::gemv_q4(q, x), std::invalid_argument) << "NaN has no INT8 value";
    x[3] = kInf;
    EXPECT_THROW(llm::gemv_q4(q, x), std::invalid_argument);

    const auto w = randvec(4 * 64, 35);
    const auto xf = randvec(64, 36);
    EXPECT_THROW(llm::gemv_f32(w, 0, 64, xf), std::invalid_argument);
    EXPECT_THROW(llm::gemv_f32(w, 4, 0, {}), std::invalid_argument);
    EXPECT_THROW(llm::gemv_f32(randvec(4 * 66, 1), 4, 66, randvec(66, 2)), std::invalid_argument);   // % 4
    EXPECT_THROW(llm::gemv_f32(w, 5, 64, xf), std::invalid_argument);                                // w size
    EXPECT_THROW(llm::gemv_f32(w, 4, 64, randvec(60, 3)), std::invalid_argument);                    // x size
    EXPECT_THROW(llm::gemv_f32(w, std::numeric_limits<int>::max(), 64, xf), std::invalid_argument);
}

TEST(Validation, NormAndRopeRejectBadArguments) {
    const auto x = randvec(64, 37);
    const auto w = randvec(64, 38);
    EXPECT_THROW(llm::rmsnorm({}, {}, 1e-5f), std::invalid_argument);
    EXPECT_THROW(llm::rmsnorm(x, randvec(63, 1), 1e-5f), std::invalid_argument);
    EXPECT_THROW(llm::rmsnorm(x, w, -1e-5f), std::invalid_argument);
    EXPECT_THROW(llm::rmsnorm(x, w, kNaN), std::invalid_argument);
    EXPECT_THROW(llm::rmsnorm(x, w, kInf), std::invalid_argument);
    EXPECT_THROW(llm::rmsnorm_cpu(x, randvec(63, 1), 1e-5f), std::invalid_argument);

    using Rope = std::function<std::vector<float>(const std::vector<float>&, const std::vector<float>&, float,
                                                  int, int, int, float)>;
    const Rope paths[] = {
        [](auto& a, auto& b, float e, int h, int d, int p, float t) { return llm::rmsnorm_rope(a, b, e, h, d, p, t); },
        [](auto& a, auto& b, float e, int h, int d, int p, float t) {
            return llm::rmsnorm_then_rope_unfused(a, b, e, h, d, p, t);
        },
        [](auto& a, auto& b, float e, int h, int d, int p, float t) {
            return llm::rmsnorm_rope_cpu(a, b, e, h, d, p, t);
        },
    };
    for (const auto& f : paths) {
        EXPECT_NO_THROW(f(x, w, 1e-5f, 4, 16, 3, 10000.0f));
        EXPECT_THROW(f(x, w, 1e-5f, 0, 16, 3, 10000.0f), std::invalid_argument);         // n_heads
        EXPECT_THROW(f(x, w, 1e-5f, 64, 1, 3, 10000.0f), std::invalid_argument);         // odd head_dim
        EXPECT_THROW(f(x, w, 1e-5f, -4, -16, 3, 10000.0f), std::invalid_argument);       // both negative
        EXPECT_THROW(f(x, w, 1e-5f, 2, 16, 3, 10000.0f), std::invalid_argument);         // 32 != 64
        EXPECT_THROW(f(x, w, 1e-5f, 4, 16, -1, 10000.0f), std::invalid_argument);        // position
        EXPECT_THROW(f(x, w, 1e-5f, 4, 16, (1 << 24) + 1, 10000.0f), std::invalid_argument);
        EXPECT_THROW(f(x, w, 1e-5f, 4, 16, 3, 0.0f), std::invalid_argument);             // theta
        EXPECT_THROW(f(x, w, 1e-5f, 4, 16, 3, kInf), std::invalid_argument);
        EXPECT_THROW(f(x, w, kNaN, 4, 16, 3, 10000.0f), std::invalid_argument);          // epsilon
        EXPECT_THROW(f(x, randvec(60, 1), 1e-5f, 4, 16, 3, 10000.0f), std::invalid_argument);
    }
}

// ===========================================================================
// Capability selection: __dp4a where it exists, a portable kernel elsewhere.
// ===========================================================================

TEST(Capability, PortableKernelIsBitwiseIdenticalToDp4a) {
    if (!llm::device_supports_dp4a()) GTEST_SKIP() << "no __dp4a on this device";
    const int M = 301, K = 1056;   // odd rows; 33 groups, so lanes wrap
    const auto q = llm::quantize_q4(randvec(M * K, 40, 0.05f), M, K);
    const auto x = randvec(K, 41);
    const auto a = llm::gemv_q4(q, x, nullptr, llm::Q4Kernel::Dp4a);
    const auto b = llm::gemv_q4(q, x, nullptr, llm::Q4Kernel::Portable);
    EXPECT_EQ(a, b) << "same integers, same accumulation order";
    EXPECT_EQ(llm::gemv_q4(q, x), a) << "Auto selects __dp4a where it exists";
}

TEST(Capability, Dp4aIsRefusedWhereTheDeviceLacksIt) {
    if (llm::device_supports_dp4a()) GTEST_SKIP() << "this device has __dp4a";
    const auto q = llm::quantize_q4(randvec(64, 42, 0.05f), 2, 32);
    EXPECT_THROW(llm::gemv_q4(q, randvec(32, 43), nullptr, llm::Q4Kernel::Dp4a), llm::DeviceUnsupported);
    EXPECT_NO_THROW(llm::gemv_q4(q, randvec(32, 43)));
}

TEST(Capability, BoundaryShapesMatchTheIntegerModel) {
    struct Shape {
        int M, K;
    };
    for (Shape s : {Shape{1, 32}, Shape{7, 32}, Shape{9, 32 * 33}, Shape{8, 32 * 64}, Shape{65, 96}}) {
        const auto q = llm::quantize_q4(randvec(s.M * s.K, 44, 0.05f), s.M, s.K);
        const auto x = randvec(s.K, 45);
        const auto ref = llm::gemv_q4_cpu(q, x);
        for (auto k : {llm::Q4Kernel::Auto, llm::Q4Kernel::Portable})
            EXPECT_LT(rel_l2(llm::gemv_q4(q, x, nullptr, k), ref), 1e-5) << "M=" << s.M << " K=" << s.K;
    }
}

// The fused kernel stages the vector in one block's shared memory; past the
// device limit it must take the global-memory path, not fail to launch.
TEST(SharedMemory, OversizedVectorsTakeTheUnfusedPathWithTheSameResult) {
    const std::size_t cap = llm::rmsnorm_rope_fused_capacity();
    ASSERT_GT(cap, 64u);
    std::printf("  fused RMSNorm+RoPE capacity on this device: %zu elements (%zu KB)\n", cap, cap * 4 / 1024);

    const int D = 64;
    const int small_heads = 8;
    bool fused = false;
    auto xs = randvec(small_heads * D, 50);
    auto ws = randvec(small_heads * D, 51, 0.5f);
    auto y = llm::rmsnorm_rope(xs, ws, 1e-5f, small_heads, D, 9, 10000.0f, nullptr, &fused);
    EXPECT_TRUE(fused);
    EXPECT_LT(rel_l2(y, llm::rmsnorm_rope_cpu(xs, ws, 1e-5f, small_heads, D, 9, 10000.0f)), 1e-5);

    const int big_heads = static_cast<int>(cap / D) + 4;   // just past the limit
    auto xb = randvec(big_heads * D, 52);
    auto wb = randvec(big_heads * D, 53, 0.5f);
    fused = true;
    auto yb = llm::rmsnorm_rope(xb, wb, 1e-5f, big_heads, D, 9, 10000.0f, nullptr, &fused);
    EXPECT_FALSE(fused);
    EXPECT_LT(rel_l2(yb, llm::rmsnorm_rope_cpu(xb, wb, 1e-5f, big_heads, D, 9, 10000.0f)), 1e-5);
}

// ===========================================================================
// Native GGUF Q4_0.
// ===========================================================================
namespace {

// One GGUF Q4_0 block: fp16 scale, then 16 bytes holding weight i < 16 in the
// low nibble of byte i and weight i >= 16 in the high nibble of byte i-16.
std::vector<std::uint8_t> gguf_block(std::uint16_t d16, const int nibbles[32]) {
    std::vector<std::uint8_t> b(18, 0);
    std::memcpy(b.data(), &d16, 2);
    for (int i = 0; i < 32; ++i)
        b[2 + (i < 16 ? i : i - 16)] |= std::uint8_t(i < 16 ? nibbles[i] : nibbles[i] << 4);
    return b;
}

}  // namespace

TEST(GgufQ4_0, ImportDecodesTheNativeBlockLayout) {
    int nib_a[32], nib_b[32];
    for (int i = 0; i < 32; ++i) {
        nib_a[i] = i % 16;
        nib_b[i] = 15 - (i * 7) % 16;
    }
    auto bytes = gguf_block(0x3800, nib_a);          // fp16 0.5
    const auto second = gguf_block(0xC000, nib_b);   // fp16 -2.0
    bytes.insert(bytes.end(), second.begin(), second.end());

    const auto m = llm::q4_from_gguf_q4_0(bytes.data(), bytes.size(), 1, 64);
    EXPECT_NO_THROW(llm::validate(m));
    const auto w = llm::dequantize_q4(m);
    for (int i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(w[i], float(nib_a[i] - 8) * 0.5f) << i;
        EXPECT_FLOAT_EQ(w[32 + i], float(nib_b[i] - 8) * -2.0f) << i;
    }
    const auto x = randvec(64, 47);
    EXPECT_LT(rel_l2(llm::gemv_q4(m, x), llm::gemv_q4_cpu(m, x)), 1e-5);
}

TEST(GgufQ4_0, ImportRejectsWhatIsNotAValidTensor) {
    int n[32] = {};
    const auto b = gguf_block(0x3800, n);
    EXPECT_THROW(llm::q4_from_gguf_q4_0(b.data(), b.size(), 1, 64), std::invalid_argument);   // 18 != 36
    EXPECT_THROW(llm::q4_from_gguf_q4_0(b.data(), b.size(), 1, 30), std::invalid_argument);   // % 32
    EXPECT_THROW(llm::q4_from_gguf_q4_0(nullptr, 18, 1, 32), std::invalid_argument);
    const auto nan = gguf_block(0x7E00, n);
    EXPECT_THROW(llm::q4_from_gguf_q4_0(nan.data(), nan.size(), 1, 32), std::invalid_argument);
    const auto inf = gguf_block(0x7C00, n);
    EXPECT_THROW(llm::q4_from_gguf_q4_0(inf.data(), inf.size(), 1, 32), std::invalid_argument);
}

TEST(GgufQ4_0, ImportFromAParsedFileChecksTypeAndRank) {
    std::vector<std::uint8_t> f;
    auto put = [&](std::uint64_t v, int n) {
        for (int i = 0; i < n; ++i) f.push_back(std::uint8_t(v >> (8 * i)));
    };
    auto tensor = [&](const std::string& name, std::vector<std::uint64_t> dims, std::uint32_t type,
                      std::uint64_t offset) {
        put(name.size(), 8);
        f.insert(f.end(), name.begin(), name.end());
        put(dims.size(), 4);
        for (auto d : dims) put(d, 8);
        put(type, 4);
        put(offset, 8);
    };
    f = {'G', 'G', 'U', 'F'};
    put(3, 4);   // version
    put(3, 8);   // tensor count
    put(0, 8);   // metadata count
    tensor("matrix", {32, 2}, 2, 0);   // Q4_0, 2 rows x 32 cols: 36 bytes at 0
    tensor("dense", {4, 1}, 0, 64);    // F32: 16 bytes at 64
    tensor("vector", {32}, 2, 96);     // Q4_0, one dimension: 18 bytes at 96
    while (f.size() % 32) f.push_back(0);

    int n[32];
    for (int i = 0; i < 32; ++i) n[i] = (i * 5) % 16;
    std::vector<std::uint8_t> blob(114, 0);
    const auto r0 = gguf_block(0x3400, n);   // 0.25
    const auto r1 = gguf_block(0x4000, n);   // 2.0
    std::memcpy(blob.data(), r0.data(), 18);
    std::memcpy(blob.data() + 18, r1.data(), 18);
    std::memcpy(blob.data() + 96, r0.data(), 18);
    f.insert(f.end(), blob.begin(), blob.end());

    const auto file = llm::GgufFile::from_memory(std::move(f));
    const auto m = llm::q4_from_gguf(file, *file.find("matrix"));
    EXPECT_EQ(m.rows, 2);
    EXPECT_EQ(m.cols, 32);
    const auto w = llm::dequantize_q4(m);
    EXPECT_FLOAT_EQ(w[1], float(n[1] - 8) * 0.25f);
    EXPECT_FLOAT_EQ(w[32 + 17], float(n[17] - 8) * 2.0f);

    EXPECT_THROW(llm::q4_from_gguf(file, *file.find("dense")), std::invalid_argument);
    EXPECT_THROW(llm::q4_from_gguf(file, *file.find("vector")), std::invalid_argument);
}

// ===========================================================================
// Numerical guarantees (NUMERICS.md).
// ===========================================================================

TEST(Numerics, KernelsAreDeterministicRunToRun) {
    const auto q = llm::quantize_q4(randvec(128 * 512, 60, 0.05f), 128, 512);
    const auto x = randvec(512, 61);
    EXPECT_EQ(llm::gemv_q4(q, x), llm::gemv_q4(q, x));
    const auto w = randvec(128 * 512, 62);
    EXPECT_EQ(llm::gemv_f32(w, 128, 512, x), llm::gemv_f32(w, 128, 512, x));
    const auto nx = randvec(2048, 63), nw = randvec(2048, 64);
    EXPECT_EQ(llm::rmsnorm(nx, nw, 1e-5f), llm::rmsnorm(nx, nw, 1e-5f));
    EXPECT_EQ(llm::rmsnorm_rope(nx, nw, 1e-5f, 32, 64, 77, 10000.0f),
              llm::rmsnorm_rope(nx, nw, 1e-5f, 32, 64, 77, 10000.0f));
}

TEST(Numerics, QuantizationRoundsToNearestAndSaturates) {
    // Weights at exact multiples of the scale survive the round trip exactly;
    // amax maps to +7 and -amax clips from -7 (it is representable) -- the
    // format's -8 code is reachable only below -amax, which cannot occur.
    std::vector<float> w(32);
    for (int i = 0; i < 32; ++i) w[i] = float((i % 15) - 7) * 0.25f;   // -1.75 .. +1.75
    const auto d = llm::dequantize_q4(llm::quantize_q4(w, 1, 32));
    for (int i = 0; i < 32; ++i) EXPECT_FLOAT_EQ(d[i], w[i]) << i;

    // Activations saturate at +-127 of the per-tensor scale: one outlier sets
    // the scale, and the kernel still matches its integer model exactly.
    auto x = randvec(64, 65, 0.01f);
    x[10] = 1000.0f;
    const auto q = llm::quantize_q4(randvec(4 * 64, 66, 0.05f), 4, 64);
    EXPECT_LT(rel_l2(llm::gemv_q4(q, x), llm::gemv_q4_cpu(q, x)), 1e-5);
}

TEST(Numerics, RmsNormEpsilonBehaviour) {
    const std::vector<float> zero(64, 0.0f), ones(64, 1.0f);
    // eps > 0 keeps a zero vector at zero.
    for (float v : llm::rmsnorm(zero, ones, 1e-5f)) EXPECT_EQ(v, 0.0f);
    // eps == 0 is allowed and divides by zero on a zero vector: NaN, on the
    // device and in the reference alike. Callers who can see zero vectors use
    // eps > 0, as every Llama checkpoint does.
    for (float v : llm::rmsnorm(zero, ones, 0.0f)) EXPECT_TRUE(std::isnan(v));
    for (float v : llm::rmsnorm_cpu(zero, ones, 0.0f)) EXPECT_TRUE(std::isnan(v));
    // A NaN input propagates rather than being hidden.
    auto x = randvec(64, 67);
    x[0] = kNaN;
    for (float v : llm::rmsnorm(x, ones, 1e-5f)) EXPECT_TRUE(std::isnan(v));
}

// The angle is pos * freq in FP32. Device powf and host std::pow may differ by
// an ULP in freq, and pos multiplies that difference: agreement degrades
// linearly with position. Measured on sm_75: 2.1e-6 at 65535, 3.3e-5 at 2^20,
// 5.3e-4 at 2^24 -- about 3.2e-11 * pos. Contexts a 4 GB card can hold stay
// below 1e-5.
TEST(Numerics, RopeAgreementDegradesLinearlyWithPosition) {
    const int H = 4, D = 64;
    const auto x = randvec(H * D, 68);
    const auto w = randvec(H * D, 69, 0.5f);
    for (int pos : {0, 1, 4095, 65535, 1 << 20, 1 << 24}) {
        const double e = rel_l2(llm::rmsnorm_rope(x, w, 1e-5f, H, D, pos, 10000.0f),
                                llm::rmsnorm_rope_cpu(x, w, 1e-5f, H, D, pos, 10000.0f));
        EXPECT_LT(e, 1e-6 + 5e-11 * pos) << "pos=" << pos;
        if (pos <= 65535) EXPECT_LT(e, 1e-5) << "pos=" << pos;
    }
}

TEST(Numerics, SoftmaxSpecialValues) {
    std::vector<float> big = {1000.0f, 1001.0f};
    llm::softmax_cpu(big);
    EXPECT_NEAR(big[1], 1.0 / (1.0 + std::exp(-1.0)), 1e-6) << "max-subtraction keeps large logits finite";

    std::vector<float> with_nan = {1.0f, kNaN, 2.0f};
    llm::softmax_cpu(with_nan);
    for (float v : with_nan) EXPECT_TRUE(std::isnan(v));

    std::vector<float> with_inf = {1.0f, kInf, 2.0f, kInf, -kInf};
    llm::softmax_cpu(with_inf);
    EXPECT_EQ(with_inf, (std::vector<float>{0.0f, 0.5f, 0.0f, 0.5f, 0.0f}));

    std::vector<float> masked = {-kInf, 0.0f, -kInf};
    llm::softmax_cpu(masked);
    EXPECT_EQ(masked, (std::vector<float>{0.0f, 1.0f, 0.0f}));

    std::vector<float> all_masked = {-kInf, -kInf};
    EXPECT_THROW(llm::softmax_cpu(all_masked), std::invalid_argument);
}

// ===========================================================================
// Paged KV cache rules.
// ===========================================================================
namespace {

llm::KvCacheConfig cache_cfg(int layers = 3, int pages = 32, int page_tokens = 4) {
    llm::KvCacheConfig c;
    c.n_layers = layers;
    c.n_kv_heads = 2;
    c.head_dim = 4;
    c.page_tokens = page_tokens;
    c.total_pages = pages;
    return c;
}

// A value that identifies (sequence tag, token, layer, element, key/value).
std::vector<float> kv(int tag, int token, int layer, bool value) {
    std::vector<float> v(8);
    for (int i = 0; i < 8; ++i) v[i] = float(tag * 100000 + token * 1000 + layer * 10 + i) * (value ? -1.0f : 1.0f);
    return v;
}

void append_token(llm::PagedKvCache& c, int seq, int tag, int token) {
    for (int l = 0; l < c.config().n_layers; ++l) c.append(seq, l, kv(tag, token, l, false), kv(tag, token, l, true));
}

// Expected contents: token t came from tags[t].
void expect_contents(const llm::PagedKvCache& c, int seq, const std::vector<int>& tags) {
    ASSERT_EQ(c.length(seq), int(tags.size()));
    for (int l = 0; l < c.config().n_layers; ++l) {
        const auto k = c.gather_keys(seq, l);
        const auto v = c.gather_values(seq, l);
        for (int t = 0; t < int(tags.size()); ++t) {
            const auto ek = kv(tags[t], t, l, false), ev = kv(tags[t], t, l, true);
            for (int i = 0; i < 8; ++i) {
                ASSERT_EQ(k[t * 8 + i], ek[i]) << "seq " << seq << " layer " << l << " token " << t;
                ASSERT_EQ(v[t * 8 + i], ev[i]) << "seq " << seq << " layer " << l << " token " << t;
            }
        }
    }
}

}  // namespace

TEST(KvCacheRules, LayersMustBeAppendedInOrder) {
    llm::PagedKvCache c(cache_cfg());
    const int s = c.create_sequence();
    EXPECT_THROW(c.append(s, 1, kv(0, 0, 1, false), kv(0, 0, 1, true)), llm::KvOrderError);
    c.append(s, 0, kv(0, 0, 0, false), kv(0, 0, 0, true));
    EXPECT_EQ(c.next_layer(s), 1);
    EXPECT_THROW(c.append(s, 0, kv(0, 0, 0, false), kv(0, 0, 0, true)), llm::KvOrderError)
        << "appending layer 0 twice would silently overwrite";
    EXPECT_EQ(c.length(s), 0) << "a token is complete only after its last layer";
    c.append(s, 1, kv(0, 0, 1, false), kv(0, 0, 1, true));
    c.append(s, 2, kv(0, 0, 2, false), kv(0, 0, 2, true));
    EXPECT_EQ(c.length(s), 1);
    EXPECT_EQ(c.next_layer(s), 0);
    expect_contents(c, s, {0});
}

TEST(KvCacheRules, MaxContextIsEnforcedWithoutSideEffects) {
    auto cfg = cache_cfg();
    cfg.max_context = 5;
    llm::PagedKvCache c(cfg);
    const int s = c.create_sequence();
    for (int t = 0; t < 5; ++t) append_token(c, s, 1, t);
    const int pages = c.pages_in_use();
    EXPECT_THROW(c.append(s, 0, kv(1, 5, 0, false), kv(1, 5, 0, true)), llm::KvContextFull);
    EXPECT_EQ(c.length(s), 5);
    EXPECT_EQ(c.next_layer(s), 0);
    EXPECT_EQ(c.pages_in_use(), pages);
    expect_contents(c, s, {1, 1, 1, 1, 1});

    auto bad = cache_cfg();
    bad.max_context = -1;
    EXPECT_THROW(llm::PagedKvCache{bad}, std::invalid_argument);
}

// The old cache allocated page by page: with 2 pages free and 3 layers needing
// one each, layers 0 and 1 got pages and layer 2 threw -- a half-written token
// holding pages nothing could use.
TEST(KvCacheRules, PageAllocationIsAllOrNothingAcrossLayers) {
    llm::PagedKvCache c(cache_cfg(3, 5, 4));
    const int a = c.create_sequence();
    for (int t = 0; t < 4; ++t) append_token(c, a, 1, t);   // 3 pages
    EXPECT_EQ(c.free_pages(), 2);

    EXPECT_THROW(c.append(a, 0, kv(1, 4, 0, false), kv(1, 4, 0, true)), llm::KvCacheExhausted);
    EXPECT_EQ(c.free_pages(), 2) << "no page may be taken by a token that cannot complete";
    EXPECT_EQ(c.next_layer(a), 0);
    expect_contents(c, a, {1, 1, 1, 1});

    const int b = c.create_sequence();   // nor can a new sequence start: 3 pages needed, 2 free
    EXPECT_THROW(c.append(b, 0, kv(2, 0, 0, false), kv(2, 0, 0, true)), llm::KvCacheExhausted);
    EXPECT_EQ(c.free_pages(), 2);
}

TEST(KvCacheRules, ExhaustionRecoversOncePagesAreFreed) {
    llm::PagedKvCache c(cache_cfg(3, 5, 4));
    const int a = c.create_sequence();
    const int b = c.create_sequence();
    append_token(c, a, 1, 0);                                                             // 3 pages
    EXPECT_THROW(c.append(b, 0, kv(2, 0, 0, false), kv(2, 0, 0, true)), llm::KvCacheExhausted);
    c.free_sequence(a);
    EXPECT_NO_THROW(append_token(c, b, 2, 0));
    expect_contents(c, b, {2});
}

TEST(KvCacheRules, ForkSharesPagesAndCopiesOnWrite) {
    llm::PagedKvCache c(cache_cfg(3, 32, 4));
    const int parent = c.create_sequence();
    for (int t = 0; t < 6; ++t) append_token(c, parent, 1, t);   // 2 pages per layer, the second half full
    const int pages = c.pages_in_use();
    ASSERT_EQ(pages, 6);

    const int child = c.fork_sequence(parent);
    EXPECT_EQ(c.pages_in_use(), pages) << "a fork copies nothing";
    for (int l = 0; l < 3; ++l) EXPECT_EQ(c.page_table(child, l), c.page_table(parent, l));
    EXPECT_EQ(c.page_refcount(c.page_table(parent, 0)[1]), 2);

    append_token(c, parent, 7, 6);   // writes into the shared half-full page: copied, per layer
    EXPECT_EQ(c.pages_in_use(), pages + 3);
    EXPECT_EQ(c.page_table(child, 0)[0], c.page_table(parent, 0)[0]) << "full pages stay shared";
    EXPECT_NE(c.page_table(child, 0)[1], c.page_table(parent, 0)[1]);

    append_token(c, child, 9, 6);   // the child now holds its page alone: written in place
    EXPECT_EQ(c.pages_in_use(), pages + 3);

    expect_contents(c, parent, {1, 1, 1, 1, 1, 1, 7});
    expect_contents(c, child, {1, 1, 1, 1, 1, 1, 9});

    c.free_sequence(parent);
    expect_contents(c, child, {1, 1, 1, 1, 1, 1, 9});
    c.free_sequence(child);
    EXPECT_EQ(c.pages_in_use(), 0);
}

TEST(KvCacheRules, ForkAtAPageBoundaryAndMidTokenRules) {
    llm::PagedKvCache c(cache_cfg(2, 32, 4));
    const int p = c.create_sequence();
    for (int t = 0; t < 4; ++t) append_token(c, p, 1, t);
    const int q = c.fork_sequence(p);
    append_token(c, p, 2, 4);
    append_token(c, q, 3, 4);
    EXPECT_EQ(c.pages_in_use(), 2 + 2 + 2) << "shared full page per layer, plus one fresh page each";
    expect_contents(c, p, {1, 1, 1, 1, 2});
    expect_contents(c, q, {1, 1, 1, 1, 3});

    c.append(p, 0, kv(2, 5, 0, false), kv(2, 5, 0, true));
    EXPECT_THROW(c.fork_sequence(p), llm::KvOrderError);
    c.free_sequence(p);   // releases the page reserved for the unfinished token too
    c.free_sequence(q);
    EXPECT_EQ(c.pages_in_use(), 0);
}

TEST(KvCacheRules, DeviceGatherReadsPagesWhereTheyAre) {
    llm::PagedKvCache c(cache_cfg(2, 32, 4));
    const int a = c.create_sequence();
    const int b = c.create_sequence();
    for (int t = 0; t < 11; ++t) {   // interleaved, so a's pages are not contiguous
        append_token(c, a, 1, t);
        append_token(c, b, 2, t);
    }
    const std::size_t bytes = 11 * 8 * sizeof(float);
    float* dk = nullptr;
    float* dv = nullptr;
    ASSERT_EQ(cudaMalloc(&dk, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dv, bytes), cudaSuccess);
    for (int l = 0; l < 2; ++l) {
        c.gather_device(a, l, dk, dv);
        std::vector<float> hk(11 * 8), hv(11 * 8);
        ASSERT_EQ(cudaMemcpy(hk.data(), dk, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(hv.data(), dv, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(hk, c.gather_keys(a, l));
        EXPECT_EQ(hv, c.gather_values(a, l));
    }
    cudaFree(dk);
    cudaFree(dv);

    // The layout addresses a page directly.
    const auto layout = c.device_layout();
    const int page = c.page_table(b, 1)[2];   // tokens 8..11 of b, layer 1
    std::vector<float> key(8);
    ASSERT_EQ(cudaMemcpy(key.data(), layout.slab + std::size_t(page) * layout.floats_per_page + 1 * layout.vals_per_token,
                         8 * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(key, kv(2, 9, 1, false));
}

TEST(KvCacheRules, ConcurrentSequencesOnSeparateThreads) {
    const int threads = 8, tokens = 21;
    llm::PagedKvCache c(cache_cfg(3, threads * 3 * 6, 4));
    std::vector<int> ids(threads);
    for (int i = 0; i < threads; ++i) ids[i] = c.create_sequence();
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i)
        pool.emplace_back([&, i] {
            for (int t = 0; t < tokens; ++t) append_token(c, ids[i], i + 1, t);
        });
    for (auto& th : pool) th.join();
    for (int i = 0; i < threads; ++i) expect_contents(c, ids[i], std::vector<int>(tokens, i + 1));
    EXPECT_EQ(c.pages_in_use(), threads * 3 * 6);
}
