#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "ann_index.h"

using ann::IvfFlatIndex;
using ann::Metric;
using ann::SearchResult;

namespace {

std::vector<float> random_vectors(int n, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> v((std::size_t)n * dim);
    for (auto& x : v) x = g(rng);
    return v;
}

// Clustered data, which is what IVF is actually designed for: uniform noise
// has no cluster structure for k-means to exploit and understates recall.
std::vector<float> clustered_vectors(int n, int dim, int nclusters, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> centres((std::size_t)nclusters * dim);
    for (auto& x : centres) x = g(rng) * 8.0f;

    std::vector<float> v((std::size_t)n * dim);
    for (int i = 0; i < n; ++i) {
        const float* c = &centres[(std::size_t)(i % nclusters) * dim];
        for (int d = 0; d < dim; ++d) v[(std::size_t)i * dim + d] = c[d] + g(rng);
    }
    return v;
}

std::vector<float> slice(const std::vector<float>& v, int i, int count, int dim) {
    return std::vector<float>(v.begin() + (std::size_t)i * dim,
                              v.begin() + (std::size_t)(i + count) * dim);
}

}  // namespace

TEST(Ann, RejectsBadGeometry) {
    // float4 loads require dim % 4 == 0.
    EXPECT_THROW(IvfFlatIndex(7, 4), std::invalid_argument);
    EXPECT_THROW(IvfFlatIndex(0, 4), std::invalid_argument);
    EXPECT_THROW(IvfFlatIndex(16, 0), std::invalid_argument);
    EXPECT_NO_THROW(IvfFlatIndex(16, 4));
}

TEST(Ann, MustTrainBeforeAdd) {
    IvfFlatIndex idx(16, 4);
    EXPECT_FALSE(idx.is_trained());
    auto v = random_vectors(100, 16, 1);
    EXPECT_THROW(idx.add(v, 100), std::logic_error);
}

TEST(Ann, TrainNeedsEnoughVectors) {
    IvfFlatIndex idx(16, 64);
    auto v = random_vectors(10, 16, 2);
    EXPECT_THROW(idx.train(v, 10), std::invalid_argument);
}

TEST(Ann, EveryVectorLandsInSomeList) {
    const int N = 4000, D = 32, L = 32;
    auto v = clustered_vectors(N, D, 16, 3);
    IvfFlatIndex idx(D, L);
    idx.train(v, N);
    idx.add(v, N);

    EXPECT_EQ(idx.size(), N);
    auto sizes = idx.list_sizes();
    ASSERT_EQ((int)sizes.size(), L);
    EXPECT_EQ(std::accumulate(sizes.begin(), sizes.end(), 0), N)
        << "inverted lists must partition the dataset exactly";
    for (int s : sizes) EXPECT_GE(s, 0);
}

// A vector that is in the index must be its own nearest neighbour at distance 0.
TEST(Ann, SelfQueryReturnsSelfFirst) {
    const int N = 2000, D = 32, L = 16;
    auto v = clustered_vectors(N, D, 8, 4);
    IvfFlatIndex idx(D, L);
    idx.train(v, N);
    idx.add(v, N);

    auto q = slice(v, 0, 64, D);
    auto r = idx.search(q, 64, 5, L);   // nprobe == nlist -> exhaustive
    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(r.id_at(i, 0), i) << "query " << i << " should find itself first";
        EXPECT_NEAR(r.dist_at(i, 0), 0.0f, 1e-3f);
    }
}

// With nprobe == nlist the IVF path scans everything, so it must agree with
// brute force exactly. If these ever differ, the bug is in the search kernel,
// not in the approximation.
TEST(Ann, FullProbeMatchesBruteForceExactly) {
    const int N = 3000, D = 32, L = 16, K = 10;
    auto v = clustered_vectors(N, D, 8, 5);
    IvfFlatIndex idx(D, L);
    idx.train(v, N);
    idx.add(v, N);

    auto q = random_vectors(128, D, 99);
    auto approx = idx.search(q, 128, K, L);
    auto exact = idx.search_bruteforce(q, 128, K);

    ASSERT_EQ(approx.ids.size(), exact.ids.size());
    for (int i = 0; i < 128; ++i)
        for (int j = 0; j < K; ++j)
            EXPECT_NEAR(approx.dist_at(i, j), exact.dist_at(i, j), 1e-3f)
                << "query " << i << " rank " << j;

    EXPECT_NEAR(ann::recall_at_k(approx, exact), 1.0, 1e-9);
}

TEST(Ann, ResultsAreSortedByDistance) {
    const int N = 2000, D = 32, L = 16, K = 8;
    auto v = clustered_vectors(N, D, 8, 6);
    IvfFlatIndex idx(D, L);
    idx.train(v, N);
    idx.add(v, N);

    auto q = random_vectors(64, D, 7);
    auto r = idx.search(q, 64, K, 4);
    for (int i = 0; i < 64; ++i)
        for (int j = 1; j < K; ++j)
            EXPECT_LE(r.dist_at(i, j - 1), r.dist_at(i, j)) << "query " << i;
}

// The core trade-off: more probes, better recall. This is the curve an ANN
// index lives or dies on, so assert the direction rather than a fixed number.
TEST(Ann, RecallImprovesWithNprobe) {
    const int N = 8000, D = 32, L = 64, K = 10;
    auto v = clustered_vectors(N, D, 32, 8);
    IvfFlatIndex idx(D, L);
    idx.train(v, N);
    idx.add(v, N);

    auto q = random_vectors(200, D, 77);
    auto exact = idx.search_bruteforce(q, 200, K);

    double r1 = ann::recall_at_k(idx.search(q, 200, K, 1), exact);
    double r8 = ann::recall_at_k(idx.search(q, 200, K, 8), exact);
    double rall = ann::recall_at_k(idx.search(q, 200, K, L), exact);

    EXPECT_GT(r1, 0.0);
    EXPECT_GE(r8, r1) << "nprobe=8 recall " << r8 << " vs nprobe=1 " << r1;
    EXPECT_GE(rall, r8);
    EXPECT_NEAR(rall, 1.0, 1e-9) << "full probe must be exact";
}

TEST(Ann, CosineMetricRanksByAngleNotMagnitude) {
    const int D = 16;
    // Three directions; the second is the first scaled up 100x.
    std::vector<float> db((std::size_t)3 * D, 0.0f);
    db[0] = 1.0f;                       // vector 0: +x
    db[D + 0] = 100.0f;                 // vector 1: +x, huge magnitude
    db[2 * D + 1] = 1.0f;               // vector 2: +y

    IvfFlatIndex idx(D, 2, Metric::Cosine);
    idx.train(db, 3, 5);
    idx.add(db, 3);

    std::vector<float> q((std::size_t)D, 0.0f);
    q[0] = 0.01f;                       // tiny +x
    auto r = idx.search(q, 1, 3, 2);

    // Under cosine, magnitude is irrelevant: both +x vectors must beat +y.
    EXPECT_NE(r.id_at(0, 0), 2);
    EXPECT_NE(r.id_at(0, 1), 2);
    EXPECT_EQ(r.id_at(0, 2), 2);
    EXPECT_NEAR(r.dist_at(0, 0), 0.0f, 1e-3f);
}

TEST(Ann, RejectsBadSearchArguments) {
    const int N = 500, D = 16, L = 8;
    auto v = clustered_vectors(N, D, 4, 9);
    IvfFlatIndex idx(D, L);
    idx.train(v, N);
    idx.add(v, N);

    auto q = random_vectors(4, D, 10);
    EXPECT_THROW(idx.search(q, 4, 0, 2), std::invalid_argument);
    EXPECT_THROW(idx.search(q, 4, 33, 2), std::invalid_argument);      // k > kMaxK
    EXPECT_THROW(idx.search(q, 4, 5, 0), std::invalid_argument);
    EXPECT_THROW(idx.search(q, 4, 5, L + 1), std::invalid_argument);   // nprobe > nlist
    EXPECT_THROW(idx.search(q, 5, 5, 2), std::invalid_argument);       // size mismatch
}

TEST(Ann, HandlesKLargerThanProbedCandidates) {
    // One tiny cell: asking for more neighbours than exist must pad with -1
    // rather than read past the list.
    const int N = 40, D = 16, L = 32;
    auto v = clustered_vectors(N, D, 32, 11);
    IvfFlatIndex idx(D, L);
    idx.train(v, N, 5);
    idx.add(v, N);

    auto q = random_vectors(8, D, 12);
    auto r = idx.search(q, 8, 32, 1);
    ASSERT_EQ(r.ids.size(), 8u * 32);
    for (int i = 0; i < 8; ++i) {
        for (int j = 1; j < 32; ++j) {
            // Once a slot is empty every later slot must be empty too.
            if (r.id_at(i, j - 1) == -1) EXPECT_EQ(r.id_at(i, j), -1);
        }
    }
}
