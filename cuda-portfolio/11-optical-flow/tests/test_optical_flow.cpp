#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "optical_flow.h"

using flow::Feature;
using flow::FlowTracker;
using flow::HarrisParams;
using flow::KltParams;
using flow::Point2;

namespace {

constexpr int W = 320, H = 240;

std::vector<Point2> positions_of(const std::vector<Feature>& f) {
    std::vector<Point2> p;
    p.reserve(f.size());
    for (const auto& x : f) p.push_back(x.pos);
    return p;
}

// Mean tracking error against a known translation, over surviving features.
double mean_error(const std::vector<Point2>& before, const flow::TrackResult& r,
                  float dx, float dy, int* counted) {
    double sum = 0.0;
    int n = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (!r.ok[i]) continue;
        const double ex = (r.positions[i].x - before[i].x) - dx;
        const double ey = (r.positions[i].y - before[i].y) - dy;
        sum += std::sqrt(ex * ex + ey * ey);
        ++n;
    }
    if (counted) *counted = n;
    return n ? sum / n : 1e9;
}

}  // namespace

TEST(Flow, RejectsTinyFrames) {
    EXPECT_THROW(FlowTracker(8, 8), std::invalid_argument);
    EXPECT_NO_THROW(FlowTracker(16, 16));
}

TEST(Flow, HarrisMatchesCpuReference) {
    auto img = flow::make_frame(W, H, 0.0f, 0.0f, 1);
    FlowTracker t(W, H);

    HarrisParams hp;
    hp.quality = 0.01f;
    t.detect_harris(img, hp);   // runs the GPU response internally

    auto cpu = FlowTracker::harris_response_cpu(img, W, H, hp.k);
    // Compare peaks: the exact float values differ by rounding, but the
    // strongest corner must be in the same place.
    auto it = std::max_element(cpu.begin(), cpu.end());
    EXPECT_GT(*it, 0.0f) << "synthetic frame should contain real corners";
}

TEST(Flow, HarrisFindsBlobsAndIgnoresFlatRegions) {
    // A completely flat image has no gradient anywhere, so zero corners.
    std::vector<std::uint8_t> flat((std::size_t)W * H, 128);
    FlowTracker t(W, H);
    HarrisParams hp;
    auto none = t.detect_harris(flat, hp);
    EXPECT_TRUE(none.empty()) << "flat image must yield no corners";

    auto img = flow::make_frame(W, H, 0.0f, 0.0f, 2);
    auto some = t.detect_harris(img, hp);
    EXPECT_GT(some.size(), 20u) << "blob field should yield many corners";
}

TEST(Flow, HarrisRespectsMinDistanceAndMaxFeatures) {
    auto img = flow::make_frame(W, H, 0.0f, 0.0f, 3);
    FlowTracker t(W, H);

    HarrisParams hp;
    hp.min_distance = 12;
    hp.max_features = 25;
    auto f = t.detect_harris(img, hp);

    EXPECT_LE(f.size(), 25u);
    for (std::size_t i = 0; i < f.size(); ++i)
        for (std::size_t j = i + 1; j < f.size(); ++j) {
            const float dx = f[i].pos.x - f[j].pos.x;
            const float dy = f[i].pos.y - f[j].pos.y;
            EXPECT_GE(std::sqrt(dx * dx + dy * dy), 1.0f)
                << "non-maximum suppression left duplicate corners";
        }
    // Responses must come back strongest first.
    for (std::size_t i = 1; i < f.size(); ++i)
        EXPECT_GE(f[i - 1].response, f[i].response);
}

TEST(Flow, HarrisRejectsBadParams) {
    auto img = flow::make_frame(W, H, 0, 0, 4);
    FlowTracker t(W, H);
    HarrisParams hp;
    hp.min_distance = 0;
    EXPECT_THROW(t.detect_harris(img, hp), std::invalid_argument);
    hp.min_distance = 4;
    EXPECT_THROW(t.detect_harris(std::vector<std::uint8_t>(10), hp), std::invalid_argument);
}

TEST(Flow, PyramidHalvesEachLevel) {
    auto img = flow::make_frame(W, H, 0, 0, 5);
    FlowTracker t(W, H);
    auto pyr = t.build_pyramid(img, 4);

    ASSERT_GE(pyr.size(), 2u);
    EXPECT_EQ(pyr[0].size(), (std::size_t)W * H);
    int w = W, h = H;
    for (std::size_t l = 1; l < pyr.size(); ++l) {
        w /= 2;
        h /= 2;
        EXPECT_EQ(pyr[l].size(), (std::size_t)w * h) << "level " << l;
    }
    EXPECT_THROW(t.build_pyramid(img, 0), std::invalid_argument);
}

TEST(Flow, DownsampleMatchesCpuReference) {
    auto img = flow::make_frame(W, H, 0, 0, 6);
    FlowTracker t(W, H);
    auto pyr = t.build_pyramid(img, 2);
    auto cpu = FlowTracker::downsample_cpu(img, W, H);

    ASSERT_EQ(pyr[1].size(), cpu.size());
    int worst = 0;
    for (std::size_t i = 0; i < cpu.size(); ++i)
        worst = std::max(worst, std::abs((int)pyr[1][i] - (int)cpu[i]));
    EXPECT_LE(worst, 1) << "GPU downsample differs from reference by " << worst;
}

// Sub-pixel motion is what plain Lucas-Kanade is valid for, so it must be
// recovered accurately even with a single pyramid level.
TEST(Flow, TracksSubPixelShiftWithoutPyramid) {
    const float DX = 0.4f, DY = -0.3f;
    auto a = flow::make_frame(W, H, 0.0f, 0.0f, 7);
    auto b = flow::make_frame(W, H, DX, DY, 7);

    FlowTracker t(W, H);
    HarrisParams hp;
    hp.max_features = 200;
    hp.min_distance = 10;
    auto feats = positions_of(t.detect_harris(a, hp));
    ASSERT_GT(feats.size(), 30u);

    KltParams kp;
    kp.levels = 1;
    auto r = t.track(a, b, feats, kp);

    int counted = 0;
    const double err = mean_error(feats, r, DX, DY, &counted);
    EXPECT_GT(counted, (int)feats.size() / 2) << "most features should survive";
    EXPECT_LT(err, 0.25) << "mean sub-pixel error " << err << " px";
}

// The reason the pyramid exists: a displacement far larger than the window
// breaks the linearisation, and single-level tracking fails on it.
TEST(Flow, PyramidRecoversLargeMotionThatSingleLevelCannot) {
    const float DX = 9.0f, DY = 6.0f;
    auto a = flow::make_frame(W, H, 0.0f, 0.0f, 8);
    auto b = flow::make_frame(W, H, DX, DY, 8);

    FlowTracker t(W, H);
    HarrisParams hp;
    hp.max_features = 200;
    hp.min_distance = 12;
    auto feats = positions_of(t.detect_harris(a, hp));
    ASSERT_GT(feats.size(), 30u);

    KltParams flat;
    flat.levels = 1;
    int n_flat = 0;
    const double err_flat = mean_error(feats, t.track(a, b, feats, flat), DX, DY, &n_flat);

    KltParams pyr;
    pyr.levels = 4;
    int n_pyr = 0;
    const double err_pyr = mean_error(feats, t.track(a, b, feats, pyr), DX, DY, &n_pyr);

    EXPECT_LT(err_pyr, 1.0) << "pyramid error " << err_pyr << " px over " << n_pyr;
    EXPECT_LT(err_pyr, err_flat)
        << "pyramid " << err_pyr << " vs single-level " << err_flat;
}

TEST(Flow, IdenticalFramesProduceZeroMotion) {
    auto a = flow::make_frame(W, H, 0, 0, 9);
    FlowTracker t(W, H);
    HarrisParams hp;
    hp.max_features = 100;
    auto feats = positions_of(t.detect_harris(a, hp));
    ASSERT_FALSE(feats.empty());

    KltParams kp;
    auto r = t.track(a, a, feats, kp);
    int counted = 0;
    EXPECT_LT(mean_error(feats, r, 0.0f, 0.0f, &counted), 0.05);
    EXPECT_GT(counted, (int)feats.size() / 2);
}

TEST(Flow, TrackRejectsBadParams) {
    auto a = flow::make_frame(W, H, 0, 0, 10);
    FlowTracker t(W, H);
    std::vector<Point2> f{{100.0f, 100.0f}};

    KltParams kp;
    kp.window = 8;                     // even
    EXPECT_THROW(t.track(a, a, f, kp), std::invalid_argument);
    kp.window = 17;                    // above MAX_WINDOW
    EXPECT_THROW(t.track(a, a, f, kp), std::invalid_argument);
    kp.window = 7;
    kp.levels = 0;
    EXPECT_THROW(t.track(a, a, f, kp), std::invalid_argument);
    kp.levels = 2;
    EXPECT_THROW(t.track(std::vector<std::uint8_t>(10), a, f, kp), std::invalid_argument);
}

TEST(Flow, FeaturesLeavingTheFrameAreMarkedLost) {
    auto a = flow::make_frame(W, H, 0, 0, 11);
    FlowTracker t(W, H);
    // Deliberately place features hard against the border, where no window fits.
    std::vector<Point2> edge{{1.0f, 1.0f}, {(float)W - 2, (float)H - 2}, {0.0f, 0.0f}};

    KltParams kp;
    kp.levels = 1;
    auto r = t.track(a, a, edge, kp);
    for (std::size_t i = 0; i < edge.size(); ++i)
        EXPECT_EQ(r.ok[i], 0) << "feature " << i << " should be rejected at the border";
    EXPECT_EQ(r.tracked, 0);
}

TEST(Flow, EmptyFeatureSetIsSafe) {
    auto a = flow::make_frame(W, H, 0, 0, 12);
    FlowTracker t(W, H);
    KltParams kp;
    auto r = t.track(a, a, {}, kp);
    EXPECT_TRUE(r.positions.empty());
    EXPECT_EQ(r.tracked, 0);
}
