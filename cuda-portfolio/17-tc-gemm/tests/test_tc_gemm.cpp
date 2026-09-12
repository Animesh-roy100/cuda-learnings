// Tiled tensor-core GEMM.
//
// Every path is checked against an FP64 host product, not against cuBLAS: a
// reference computed by one of the things under test cannot catch a bug they
// share (a transposed operand, say, which every column-major path could get
// identically wrong).

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cu/device.hpp"
#include "tc_gemm.h"

namespace {

std::vector<float> random_matrix(int n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> m(std::size_t(n) * n);
    for (auto& v : m) v = d(rng);
    return m;
}

std::vector<double> host_product(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<double> c(std::size_t(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            const double aik = a[std::size_t(i) * n + k];
            for (int j = 0; j < n; ++j)
                c[std::size_t(i) * n + j] += aik * b[std::size_t(k) * n + j];
        }
    return c;
}

// ||x - ref||_F / ||ref||_F. Per-element relative error is meaningless here:
// entries of a random product sit near zero, where any absolute error is huge
// in relative terms (the lesson from 01-gguf-inference).
double rel_frobenius(const std::vector<float>& x, const std::vector<double>& ref) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = double(x[i]) - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

// Tolerances by precision. FP32 is exact to rounding; fp16 operands carry
// ~5e-4 relative error each; int8 per-tensor quantization of uniform values
// in [-1, 1] has a step of 1/127.
double tolerance(tc::Path p) {
    if (tc::is_int8(p)) return 0.02;
    if (tc::is_fp16(p)) return 2e-3;
    return 1e-5;
}

bool has_wmma() { return cu::query_device().arch() >= 70; }

// A library that declines a shape is a documented outcome, not a failure. The
// macro must be used in a test body: GTEST_SKIP returns from the enclosing
// function.
#define MULTIPLY_OR_SKIP(g, path)                                                   [&]() -> std::vector<float> {                                                       try {                                                                               return (g).multiply(path);                                                  } catch (const tc::Unsupported&) {                                                  return {};                                                                  }                                                                           }()

}  // namespace

// ------------------------------------------------------------ correctness

class EveryPath : public ::testing::TestWithParam<tc::Path> {};

TEST_P(EveryPath, MatchesTheFp64HostProduct) {
    if (!has_wmma() && !tc::is_cublas(GetParam())) GTEST_SKIP() << "needs sm_70+";
    const int n = 128;
    const auto a = random_matrix(n, 1);
    const auto b = random_matrix(n, 2);
    const auto ref = host_product(a, b, n);

    tc::Gemm g(n);
    g.set_inputs(a, b);
    const auto c = MULTIPLY_OR_SKIP(g, GetParam());
    if (c.empty()) GTEST_SKIP() << tc::to_string(GetParam()) << " declines n=" << n;
    const double err = rel_frobenius(c, ref);
    EXPECT_LT(err, tolerance(GetParam())) << tc::to_string(GetParam());
}

TEST_P(EveryPath, IsNotAccidentallyTransposed) {
    // A symmetric-looking test can hide a transpose: A*B and B^T*A^T agree on
    // some inputs. Rectangular structure in the values breaks that symmetry.
    if (!has_wmma() && !tc::is_cublas(GetParam())) GTEST_SKIP() << "needs sm_70+";
    const int n = 32;
    std::vector<float> a(n * n), b(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = float(i + 1) / n;          // varies by row only
            b[i * n + j] = float((j % 5) + 1) / 5;    // varies by column only
        }
    const auto ref = host_product(a, b, n);
    tc::Gemm g(n);
    g.set_inputs(a, b);
    const auto c = MULTIPLY_OR_SKIP(g, GetParam());
    if (c.empty()) GTEST_SKIP() << tc::to_string(GetParam()) << " declines n=" << n;
    EXPECT_LT(rel_frobenius(c, ref), tolerance(GetParam())) << tc::to_string(GetParam());
}

TEST_P(EveryPath, HandlesBandsSmallerThanTheStagingWidth) {
    // n = 48 is below the 512-column staging band and the 16-row launch band,
    // so both remainders are exercised.
    if (!has_wmma() && !tc::is_cublas(GetParam())) GTEST_SKIP() << "needs sm_70+";
    const int n = 48;
    const auto a = random_matrix(n, 3);
    const auto b = random_matrix(n, 4);
    tc::Gemm g(n);
    g.set_inputs(a, b);
    const auto c = MULTIPLY_OR_SKIP(g, GetParam());
    // cuBLAS INT8 declines n=48 on sm_75 (see tc::Unsupported); that is
    // covered by Int8WmmaCoversShapesCublasDeclines below.
    if (c.empty()) GTEST_SKIP() << tc::to_string(GetParam()) << " declines n=" << n;
    EXPECT_LT(rel_frobenius(c, host_product(a, b, n)), tolerance(GetParam()))
        << tc::to_string(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TcGemm, EveryPath, ::testing::ValuesIn(tc::all_paths()),
                         [](const auto& info) {
                             std::string s = tc::to_string(info.param);
                             std::string out;
                             for (char ch : s)
                                 out += (std::isalnum(static_cast<unsigned char>(ch)) ? ch : '_');
                             return out;
                         });

TEST(TcGemm, SpansMultipleStagingAndLaunchBands) {
    // 1024 crosses two 512-column staging bands and 64 tile rows, i.e. four
    // separate kernel launches. A band-offset bug shows up only here.
    if (!has_wmma()) GTEST_SKIP() << "needs sm_70+";
    const int n = 1024;
    const auto a = random_matrix(n, 5);
    const auto b = random_matrix(n, 6);
    tc::Gemm g(n);
    g.set_inputs(a, b);
    const auto ref = g.multiply(tc::Path::CublasFp32);   // FP64 host at 1024^3 is slow
    // cuBLAS is used as the reference only here, after every path has been
    // checked against the independent FP64 product at smaller sizes.
    for (auto p : {tc::Path::Fp32Tiled, tc::Path::Fp16WmmaStaged, tc::Path::Fp16WmmaStagedWord,
                   tc::Path::Fp16WmmaPipelined, tc::Path::Int8WmmaStaged,
                   tc::Path::Int8WmmaStagedWord, tc::Path::Int8WmmaPipelined,
                   tc::Path::Int8Dp4a}) {
        const auto c = g.multiply(p);
        double num = 0, den = 0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            num += double(c[i] - ref[i]) * (c[i] - ref[i]);
            den += double(ref[i]) * ref[i];
        }
        EXPECT_LT(std::sqrt(num / den), tolerance(p)) << tc::to_string(p);
    }
}

TEST(TcGemm, Int8IsExactOnIntegerValuedInputs) {
    // Operands that are already multiples of their quantization step lose
    // nothing, so the INT8 paths must reproduce the product exactly.
    if (!has_wmma()) GTEST_SKIP() << "needs sm_70+";
    const int n = 64;
    std::vector<float> a(n * n), b(n * n);
    for (int i = 0; i < n * n; ++i) {
        a[i] = float((i * 37) % 255 - 127);
        b[i] = float((i * 91) % 255 - 127);
    }
    a[0] = 127.0f;   // pin the scale to exactly 1
    b[0] = 127.0f;
    const auto ref = host_product(a, b, n);
    tc::Gemm g(n);
    g.set_inputs(a, b);
    for (auto p : {tc::Path::Int8WmmaGlobal, tc::Path::Int8WmmaStaged,
                   tc::Path::Int8WmmaStagedWord, tc::Path::Int8WmmaPipelined,
                   tc::Path::Int8Dp4a, tc::Path::CublasInt8}) {
        const auto c = g.multiply(p);
        int wrong = 0;
        for (std::size_t i = 0; i < c.size(); ++i) wrong += (double(c[i]) != ref[i]);
        EXPECT_EQ(wrong, 0) << tc::to_string(p);
    }
}

TEST(TcGemm, Int8WmmaCoversShapesCublasDeclines) {
    // Every odd multiple of 16 from 48 up was refused by cuBLAS INT8 when this
    // was measured. The hand-written paths must be correct at all of them, and
    // cuBLAS must either succeed or decline cleanly -- never fail another way.
    if (!has_wmma()) GTEST_SKIP() << "needs sm_70+";
    for (int n : {48, 80, 112}) {
        const auto a = random_matrix(n, 21);
        const auto b = random_matrix(n, 22);
        const auto ref = host_product(a, b, n);
        tc::Gemm g(n);
        g.set_inputs(a, b);
        for (auto p : {tc::Path::Int8WmmaGlobal, tc::Path::Int8WmmaStaged,
                       tc::Path::Int8WmmaStagedWord, tc::Path::Int8WmmaPipelined,
                       tc::Path::Int8Dp4a})
            EXPECT_LT(rel_frobenius(g.multiply(p), ref), tolerance(p))
                << tc::to_string(p) << " n=" << n;
        try {
            g.multiply(tc::Path::CublasInt8);
        } catch (const tc::Unsupported&) {
            // declined: an acceptable, typed outcome
        }
    }
}

// ------------------------------------------------------------ quantization

TEST(Quantize, ScaleMapsTheLargestMagnitudeTo127) {
    auto q = tc::quantize_symmetric({0.5f, -2.0f, 1.0f});
    EXPECT_FLOAT_EQ(q.scale, 2.0f / 127.0f);
    EXPECT_EQ(q.q[1], -127);
    EXPECT_EQ(q.q[0], 32);   // 0.5 / (2/127) = 31.75
}

TEST(Quantize, AllZeroInputDoesNotDivideByZero) {
    auto q = tc::quantize_symmetric({0.0f, 0.0f});
    EXPECT_GT(q.scale, 0.0f);
    EXPECT_EQ(q.q[0], 0);
}

// -------------------------------------------------------------- contract

TEST(TcGemm, RejectsSizesThatAreNotMultiplesOf16) {
    EXPECT_THROW(tc::Gemm{100}, std::invalid_argument);
    EXPECT_THROW(tc::Gemm{0}, std::invalid_argument);
}

TEST(TcGemm, RejectsMismatchedOperands) {
    tc::Gemm g(16);
    EXPECT_THROW(g.set_inputs(std::vector<float>(10), std::vector<float>(256)),
                 std::invalid_argument);
}

TEST(TcGemm, RefusesToMultiplyBeforeInputsAreSet) {
    tc::Gemm g(16);
    EXPECT_THROW(g.multiply(tc::Path::Fp32Tiled), std::logic_error);
}

TEST(TcGemm, GflopsIsCubicInN) {
    EXPECT_DOUBLE_EQ(tc::Gemm::gflops(1000, 1000.0f), 1.0);
    EXPECT_DOUBLE_EQ(tc::Gemm::gflops(2000, 1000.0f), 8.0);
}

// ------------------------------------------------------------ band sweep

TEST(StagingBands, EveryCompiledBandIsCorrect) {
    // n = 512 makes the narrow bands span many staging rounds and the widest
    // exactly one, so both the band arithmetic and the remainder are exercised
    // for every one of the 36 instantiations.
    if (!has_wmma()) GTEST_SKIP() << "needs sm_70+";
    const int n = 512;
    const auto a = random_matrix(n, 31);
    const auto b = random_matrix(n, 32);
    const auto ref = host_product(a, b, n);
    tc::Gemm g(n);
    g.set_inputs(a, b);
    for (auto s : {tc::Staging::ByElement, tc::Staging::ByWord, tc::Staging::Pipelined})
        for (bool int8 : {false, true})
            for (int band : tc::staging_bands(s, int8)) {
                const double err = rel_frobenius(g.multiply_band(s, int8, band), ref);
                EXPECT_LT(err, int8 ? 0.02 : 2e-3)
                    << tc::to_string(s) << (int8 ? " int8" : " fp16") << " band " << band;
            }
}

TEST(StagingBands, DefaultBandsAreCompiledBands) {
    for (auto s : {tc::Staging::ByElement, tc::Staging::ByWord, tc::Staging::Pipelined})
        for (bool int8 : {false, true}) {
            const auto bands = tc::staging_bands(s, int8);
            EXPECT_NE(std::find(bands.begin(), bands.end(), tc::default_band(s)), bands.end())
                << tc::to_string(s) << (int8 ? " int8" : " fp16");
        }
}

TEST(StagingBands, PipelinedFp16ExcludesTheBandThatExceedsSharedMemory) {
    // 4 x 16 x 512 fp16 values = 64 KB against a 48 KB per-kernel limit.
    const auto fp16 = tc::staging_bands(tc::Staging::Pipelined, false);
    const auto int8 = tc::staging_bands(tc::Staging::Pipelined, true);
    EXPECT_EQ(std::find(fp16.begin(), fp16.end(), 512), fp16.end());
    EXPECT_NE(std::find(int8.begin(), int8.end(), 512), int8.end())
        << "int8 at 512 is 32 KB and fits";
    tc::Gemm g(64);
    g.set_inputs(random_matrix(64, 1), random_matrix(64, 2));
    EXPECT_THROW(g.multiply_band(tc::Staging::Pipelined, false, 512), std::invalid_argument);
}

TEST(StagingBands, RejectsABandWithNoKernel) {
    tc::Gemm g(64);
    g.set_inputs(random_matrix(64, 1), random_matrix(64, 2));
    EXPECT_THROW(g.multiply_band(tc::Staging::ByWord, false, 100), std::invalid_argument);
}

TEST(StagingBands, WiderBandsNeverFitMoreBlocksPerSm) {
    // Shared memory per block grows with the band, so blocks per SM must be
    // non-increasing across the sweep. If it is not, the occupancy numbers the
    // benchmark reasons from are not measuring what they claim to.
    if (!has_wmma()) GTEST_SKIP() << "needs sm_70+";
    tc::Gemm g(256);
    g.set_inputs(random_matrix(256, 1), random_matrix(256, 2));
    for (auto s : {tc::Staging::ByElement, tc::Staging::ByWord, tc::Staging::Pipelined})
        for (bool int8 : {false, true}) {
            const auto pts = g.sweep_band(s, int8, 1, 0);
            for (std::size_t i = 1; i < pts.size(); ++i)
                EXPECT_LE(pts[i].blocks_per_sm, pts[i - 1].blocks_per_sm)
                    << tc::to_string(s) << (int8 ? " int8" : " fp16") << " band "
                    << pts[i].band;
        }
}

// -------------------------------------------------------------- occupancy

TEST(Occupancy, ReportsEveryHandWrittenPath) {
    if (!has_wmma()) GTEST_SKIP() << "needs sm_70+";
    const auto occ = tc::occupancy();
    for (auto p : tc::all_paths()) {
        if (tc::is_cublas(p)) continue;
        const auto it = std::find_if(occ.begin(), occ.end(),
                                     [p](const auto& o) { return o.path == p; });
        ASSERT_NE(it, occ.end()) << tc::to_string(p);
        EXPECT_GT(it->blocks_per_sm, 0) << tc::to_string(p);
        EXPECT_GT(it->compute_warps_per_sm(), 0) << tc::to_string(p);
    }
}

// --------------------------------------------------------------- sub-byte

TEST(SubByte, ExperimentalU4FragmentIsExact) {
    if (cu::query_device().arch() < 73) GTEST_SKIP() << "sub-byte WMMA needs sm_73+";
    auto r = tc::check_u4_fragment();
    EXPECT_EQ(r.total, 64);
    EXPECT_EQ(r.wrong, 0);
}
