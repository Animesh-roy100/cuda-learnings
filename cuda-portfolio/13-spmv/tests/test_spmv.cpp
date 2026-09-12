#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "spmv.h"

using spmv::CooMatrix;
using spmv::Format;
using spmv::SparseMatrix;

namespace {

std::vector<float> random_vec(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - b[i];
        num += d * d;
        den += double(b[i]) * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

const Format kAllFormats[] = {Format::CsrScalar, Format::CsrVector,
                              Format::Ellpack, Format::Hybrid};
const char* kNames[] = {"CsrScalar", "CsrVector", "Ellpack", "Hybrid"};

}  // namespace

TEST(Spmv, RejectsMalformedInput) {
    CooMatrix bad;
    bad.rows = 0;
    EXPECT_THROW(SparseMatrix{bad}, std::invalid_argument);

    CooMatrix ragged;
    ragged.rows = 2; ragged.cols = 2;
    ragged.row_idx = {0};
    ragged.col_idx = {0, 1};
    ragged.values = {1.0f, 2.0f};
    EXPECT_THROW(SparseMatrix{ragged}, std::invalid_argument);

    CooMatrix oob;
    oob.rows = 2; oob.cols = 2;
    oob.row_idx = {0}; oob.col_idx = {5}; oob.values = {1.0f};
    EXPECT_THROW(SparseMatrix{oob}, std::invalid_argument);
}

TEST(Spmv, IdentityMatrixReturnsInput) {
    const int N = 1000;
    CooMatrix id;
    id.rows = N; id.cols = N;
    for (int i = 0; i < N; ++i) {
        id.row_idx.push_back(i);
        id.col_idx.push_back(i);
        id.values.push_back(1.0f);
    }
    SparseMatrix m(id);
    auto x = random_vec(N, 1);
    for (int f = 0; f < 4; ++f) {
        auto y = m.multiply(x, kAllFormats[f]);
        for (int i = 0; i < N; ++i)
            EXPECT_FLOAT_EQ(y[i], x[i]) << kNames[f] << " at " << i;
    }
}

// Every format must compute the SAME product. They differ only in layout, so
// a disagreement is a layout bug, not a numerical one.
TEST(Spmv, AllFormatsAgreeWithCpuOnBandedMatrix) {
    const int N = 20000;
    auto coo = CooMatrix::banded(N, 9, 1);
    SparseMatrix m(coo);
    auto x = random_vec(N, 2);
    auto cpu = SparseMatrix::multiply_cpu(coo, x);

    for (int f = 0; f < 4; ++f) {
        auto gpu = m.multiply(x, kAllFormats[f]);
        EXPECT_LT(rel_l2(gpu, cpu), 1e-5) << kNames[f];
    }
}

// The hard case: a heavy-tailed degree distribution, which is where ELLPACK
// padding explodes and the hybrid split has to be correct.
TEST(Spmv, AllFormatsAgreeOnPowerLawMatrix) {
    const int N = 30000;
    auto coo = CooMatrix::power_law(N, 8, 7);
    SparseMatrix m(coo);
    auto x = random_vec(N, 3);
    auto cpu = SparseMatrix::multiply_cpu(coo, x);

    for (int f = 0; f < 4; ++f) {
        auto gpu = m.multiply(x, kAllFormats[f]);
        EXPECT_LT(rel_l2(gpu, cpu), 1e-4) << kNames[f];
    }
}

TEST(Spmv, HybridHandlesRowsFarBeyondTheCut) {
    // One row with every column filled, the rest nearly empty: the entire
    // overflow path in one matrix.
    const int N = 500;
    CooMatrix coo;
    coo.rows = N; coo.cols = N;
    for (int c = 0; c < N; ++c) {
        coo.row_idx.push_back(0);
        coo.col_idx.push_back(c);
        coo.values.push_back(1.0f);
    }
    for (int r = 1; r < N; ++r) {
        coo.row_idx.push_back(r);
        coo.col_idx.push_back(r);
        coo.values.push_back(2.0f);
    }

    SparseMatrix m(coo);
    std::vector<float> x(N, 1.0f);
    auto cpu = SparseMatrix::multiply_cpu(coo, x);

    for (int f = 0; f < 4; ++f) {
        auto gpu = m.multiply(x, kAllFormats[f]);
        EXPECT_NEAR(gpu[0], (float)N, 1e-2f) << kNames[f] << " dense row";
        EXPECT_NEAR(gpu[1], 2.0f, 1e-5f) << kNames[f] << " sparse row";
        EXPECT_LT(rel_l2(gpu, cpu), 1e-5) << kNames[f];
    }
}

TEST(Spmv, EmptyRowsProduceZero) {
    CooMatrix coo;
    coo.rows = 10; coo.cols = 10;
    coo.row_idx = {0, 9};
    coo.col_idx = {0, 9};
    coo.values = {3.0f, 4.0f};

    SparseMatrix m(coo);
    std::vector<float> x(10, 1.0f);
    for (int f = 0; f < 4; ++f) {
        auto y = m.multiply(x, kAllFormats[f]);
        EXPECT_FLOAT_EQ(y[0], 3.0f) << kNames[f];
        EXPECT_FLOAT_EQ(y[9], 4.0f) << kNames[f];
        for (int i = 1; i < 9; ++i) EXPECT_FLOAT_EQ(y[i], 0.0f) << kNames[f] << " row " << i;
    }
}

TEST(Spmv, RejectsWrongVectorLength) {
    auto coo = CooMatrix::banded(100, 3, 1);
    SparseMatrix m(coo);
    EXPECT_THROW(m.multiply(std::vector<float>(99), Format::Hybrid), std::invalid_argument);
}

// ELLPACK pads every row to the longest one, so a heavy tail wastes enormous
// space. This is the quantitative reason the hybrid format exists.
TEST(Spmv, EllpackPaddingExplodesOnSkewHybridDoesNot) {
    const int N = 20000;
    auto coo = CooMatrix::power_law(N, 8, 11);
    SparseMatrix m(coo);

    const auto ell = m.stats(Format::Ellpack);
    const auto hyb = m.stats(Format::Hybrid);
    const auto csr = m.stats(Format::CsrScalar);

    EXPECT_GT(ell.max_row_nnz, (int)ell.mean_row_nnz * 10)
        << "generator should produce genuine skew";
    EXPECT_GT(ell.padded_entries, ell.nnz)
        << "ELL padding should exceed the real non-zeros on a skewed matrix";
    EXPECT_LT(hyb.padded_entries, ell.padded_entries)
        << "hybrid must waste less than pure ELL";
    EXPECT_LT(hyb.device_bytes, ell.device_bytes);
    EXPECT_GT(csr.nnz, 0);
}

TEST(Spmv, BandedMatrixIsTheGoodCaseForEllpack) {
    const int N = 20000;
    auto coo = CooMatrix::banded(N, 9, 1);
    SparseMatrix m(coo);
    const auto ell = m.stats(Format::Ellpack);
    // Uniform rows: padding should be a rounding error, not a multiple.
    EXPECT_LT(ell.padded_entries, ell.nnz / 100)
        << "banded matrix should barely pad at all";
}

// CG only converges on a symmetric positive definite matrix, which the 2D
// Laplacian is. Verifying A*x ~= b afterwards checks the whole pipeline:
// SpMV, the dot products, and the update kernels together.
TEST(Spmv, ConjugateGradientSolvesLaplacian) {
    const int W = 64, H = 64;
    auto coo = CooMatrix::laplacian_2d(W, H);
    SparseMatrix m(coo);

    std::vector<float> b((std::size_t)W * H, 1.0f);
    auto r = m.solve_cg(b, 500, 1e-6, Format::Hybrid);

    EXPECT_GT(r.iterations, 0);
    EXPECT_LT(r.iterations, 500) << "should converge well inside the iteration cap";

    auto ax = m.multiply(r.x, Format::Hybrid);
    EXPECT_LT(rel_l2(ax, b), 1e-3) << "A*x should reproduce b";
}

TEST(Spmv, ConjugateGradientAgreesAcrossFormats) {
    const int W = 32, H = 32;
    auto coo = CooMatrix::laplacian_2d(W, H);
    SparseMatrix m(coo);
    std::vector<float> b((std::size_t)W * H, 1.0f);

    auto ref = m.solve_cg(b, 400, 1e-6, Format::CsrScalar);
    for (int f = 1; f < 4; ++f) {
        auto r = m.solve_cg(b, 400, 1e-6, kAllFormats[f]);
        EXPECT_LT(rel_l2(r.x, ref.x), 1e-3) << kNames[f] << " solution differs";
    }
}

TEST(Spmv, ConjugateGradientRejectsBadInput) {
    auto coo = CooMatrix::laplacian_2d(8, 8);
    SparseMatrix m(coo);
    EXPECT_THROW(m.solve_cg(std::vector<float>(10), 10, 1e-6), std::invalid_argument);
}
