#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

#include "spatial_index.h"

using spatial::DbscanResult;
using spatial::Point3;
using spatial::SpatialIndex;

namespace {

std::vector<Point3> uniform_cloud(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 0.999f);
    std::vector<Point3> p(n);
    for (auto& q : p) q = {u(rng), u(rng), u(rng)};
    return p;
}

// Well-separated tight blobs, so the expected cluster count is unambiguous.
std::vector<Point3> separated_blobs(int per_blob, int nblob, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 0.002f);
    std::vector<Point3> p;
    p.reserve(per_blob * nblob);
    for (int b = 0; b < nblob; ++b) {
        // Spread centres on a coarse lattice, far apart relative to eps.
        float cx = 0.15f + 0.30f * (b % 3);
        float cy = 0.15f + 0.30f * ((b / 3) % 3);
        float cz = 0.15f + 0.30f * ((b / 9) % 3);
        for (int i = 0; i < per_blob; ++i) {
            p.push_back({std::clamp(cx + g(rng), 0.0f, 0.999f),
                         std::clamp(cy + g(rng), 0.0f, 0.999f),
                         std::clamp(cz + g(rng), 0.0f, 0.999f)});
        }
    }
    return p;
}

}  // namespace

TEST(Spatial, MortonInterleavesBits) {
    // x=1 -> bit 0, y=1 -> bit 1, z=1 -> bit 2.
    EXPECT_EQ(SpatialIndex::morton3(0, 0, 0), 0u);
    EXPECT_EQ(SpatialIndex::morton3(1, 0, 0), 1u);
    EXPECT_EQ(SpatialIndex::morton3(0, 1, 0), 2u);
    EXPECT_EQ(SpatialIndex::morton3(0, 0, 1), 4u);
    EXPECT_EQ(SpatialIndex::morton3(1, 1, 1), 7u);
    EXPECT_EQ(SpatialIndex::morton3(2, 0, 0), 8u);
}

TEST(Spatial, MortonIsInjectiveOnGrid) {
    std::vector<std::uint32_t> codes;
    for (std::uint32_t z = 0; z < 16; ++z)
        for (std::uint32_t y = 0; y < 16; ++y)
            for (std::uint32_t x = 0; x < 16; ++x)
                codes.push_back(SpatialIndex::morton3(x, y, z));
    std::sort(codes.begin(), codes.end());
    EXPECT_EQ(std::adjacent_find(codes.begin(), codes.end()), codes.end())
        << "morton codes must be unique per cell";
}

TEST(Spatial, PermutationIsAValidBijection) {
    auto pts = uniform_cloud(10000, 1);
    SpatialIndex idx(pts);
    auto perm = idx.permutation();
    ASSERT_EQ(perm.size(), pts.size());

    std::vector<int> seen(pts.size(), 0);
    for (auto i : perm) {
        ASSERT_GE(i, 0);
        ASSERT_LT(i, static_cast<int>(pts.size()));
        ++seen[i];
    }
    for (auto c : seen) EXPECT_EQ(c, 1) << "every original index appears exactly once";
}

TEST(Spatial, SortedPointsMatchPermutedOriginals) {
    auto pts = uniform_cloud(5000, 2);
    SpatialIndex idx(pts);
    auto sorted = idx.sorted_points();
    auto perm = idx.permutation();
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        EXPECT_FLOAT_EQ(sorted[i].x, pts[perm[i]].x);
        EXPECT_FLOAT_EQ(sorted[i].y, pts[perm[i]].y);
        EXPECT_FLOAT_EQ(sorted[i].z, pts[perm[i]].z);
    }
}

// The whole point of the grid is that it returns EXACTLY what brute force does.
TEST(Spatial, GridKnnMatchesBruteForce) {
    auto pts = uniform_cloud(50000, 3);
    SpatialIndex idx(pts);

    std::vector<std::int32_t> q;
    for (int i = 0; i < 512; ++i) q.push_back(i * 97 % 50000);

    const int K = 8;
    auto grid = idx.knn(q, K);
    auto brute = idx.knn_bruteforce(q, K);
    ASSERT_EQ(grid.size(), brute.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
        EXPECT_NEAR(grid[i], brute[i], 1e-6f * std::max(1.0f, brute[i]))
            << "k-NN mismatch at " << i;
    }
}

TEST(Spatial, KnnDistancesAreSortedAscending) {
    auto pts = uniform_cloud(20000, 4);
    SpatialIndex idx(pts);
    std::vector<std::int32_t> q{0, 1, 999, 12345};
    const int K = 8;
    auto d = idx.knn(q, K);
    for (std::size_t s = 0; s < q.size(); ++s)
        for (int i = 1; i < K; ++i)
            EXPECT_LE(d[s * K + i - 1], d[s * K + i]);
}

TEST(Spatial, KnnRejectsBadK) {
    auto pts = uniform_cloud(100, 5);
    SpatialIndex idx(pts);
    std::vector<std::int32_t> q{0};
    EXPECT_THROW(idx.knn(q, 0), std::invalid_argument);
    EXPECT_THROW(idx.knn(q, 33), std::invalid_argument);
}

TEST(Spatial, DbscanRejectsEpsLargerThanCell) {
    auto pts = uniform_cloud(100, 6);
    SpatialIndex idx(pts);
    // The 3x3x3 stencil cannot see further than one cell, so a larger eps would
    // silently miss neighbours. Rejecting beats returning a wrong answer.
    EXPECT_THROW(idx.dbscan(SpatialIndex::kCell * 2.0f, 4), std::invalid_argument);
    EXPECT_NO_THROW(idx.dbscan(SpatialIndex::kCell * 0.5f, 4));
}

TEST(Spatial, DbscanFindsSeparatedBlobs) {
    const int PER = 2000, NB = 8;
    auto pts = separated_blobs(PER, NB, 7);
    SpatialIndex idx(pts);
    auto r = idx.dbscan(SpatialIndex::kCell * 0.5f, 8);

    EXPECT_EQ(r.num_clusters, NB) << "blobs are far apart; each must be its own cluster";

    // A Gaussian has tails, so a handful of points can genuinely fall further
    // than eps from every neighbour. Demanding exactly zero noise would be
    // asserting something false about the data, not about the algorithm.
    EXPECT_LT(r.noise_points, pts.size() / 200) << "noise should be a tiny fraction";

    // labels are indexed like the input, so blob b occupies [b*PER, (b+1)*PER).
    // Every non-noise point in a blob must carry that blob's single label, and
    // no two blobs may share one.
    std::map<int, int> blob_of_label;
    for (int b = 0; b < NB; ++b) {
        std::map<int, int> hist;
        for (int i = 0; i < PER; ++i) {
            int lab = r.labels[b * PER + i];
            if (lab != SpatialIndex::kNoise) hist[lab]++;
        }
        ASSERT_EQ(hist.size(), 1u) << "blob " << b << " split across " << hist.size()
                                   << " clusters";
        int lab = hist.begin()->first;
        EXPECT_EQ(blob_of_label.count(lab), 0u) << "blobs " << b << " and "
                                                << blob_of_label[lab] << " share a label";
        blob_of_label[lab] = b;
    }
    EXPECT_EQ(static_cast<int>(blob_of_label.size()), NB);
}

TEST(Spatial, DbscanLabelsAreDenseAndInRange) {
    auto pts = separated_blobs(1000, 4, 8);
    SpatialIndex idx(pts);
    auto r = idx.dbscan(SpatialIndex::kCell * 0.5f, 8);
    for (auto l : r.labels) {
        EXPECT_TRUE(l == SpatialIndex::kNoise || (l >= 0 && l < r.num_clusters));
    }
}

TEST(Spatial, DbscanMarksSparseCloudAsNoise) {
    // 2000 points spread over the unit cube: with a tiny eps, nothing is dense.
    auto pts = uniform_cloud(2000, 9);
    SpatialIndex idx(pts);
    auto r = idx.dbscan(SpatialIndex::kCell * 0.25f, 12);
    EXPECT_EQ(r.core_points, 0);
    EXPECT_EQ(r.noise_points, static_cast<std::int64_t>(pts.size()));
    EXPECT_EQ(r.num_clusters, 0);
}
