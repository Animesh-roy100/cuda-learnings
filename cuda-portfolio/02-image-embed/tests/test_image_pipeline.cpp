#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "image_pipeline.h"

using vision::DynamicBatcher;
using vision::ImageRef;
using vision::PreprocessConfig;
using vision::Preprocessor;

namespace {

std::vector<std::uint8_t> make_image(int w, int h, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h * 3);
    for (auto& v : px) v = static_cast<std::uint8_t>(rng() & 0xFF);
    return px;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(double(a[i]) - b[i]));
    return d;
}

PreprocessConfig cfg224() {
    PreprocessConfig c;
    c.out_w = 224;
    c.out_h = 224;
    return c;
}

}  // namespace

TEST(Preprocess, MatchesCpuReference) {
    auto cfg = cfg224();
    auto px = make_image(640, 480, 1);
    ImageRef img{px.data(), 640, 480};

    Preprocessor p(cfg, 4);
    auto gpu = p.process({img});
    auto cpu = vision::preprocess_cpu(img, cfg);

    ASSERT_EQ(gpu.size(), cpu.size());
    // Bilinear weights differ only by float rounding between the two paths.
    EXPECT_LT(max_abs_diff(gpu, cpu), 1e-4);
}

TEST(Preprocess, OverlappedMatchesSerial) {
    auto cfg = cfg224();
    std::vector<std::vector<std::uint8_t>> store;
    std::vector<ImageRef> batch;
    for (int i = 0; i < 8; ++i) {
        store.push_back(make_image(320 + i * 16, 240 + i * 8, 100 + i));
        batch.push_back({store.back().data(), 320 + i * 16, 240 + i * 8});
    }

    Preprocessor p(cfg, 16);
    auto fast = p.process(batch);
    auto slow = p.process_serial(batch);
    ASSERT_EQ(fast.size(), slow.size());
    // Streams change WHEN work happens, never WHAT it computes.
    EXPECT_EQ(fast, slow);
}

TEST(Preprocess, OutputIsChwPlanar) {
    PreprocessConfig cfg;
    cfg.out_w = 4;
    cfg.out_h = 4;
    // Identity normalisation so we can reason about raw values.
    for (int c = 0; c < 3; ++c) { cfg.mean[c] = 0.0f; cfg.stdev[c] = 1.0f; }

    // A constant image: R=10, G=20, B=30 everywhere.
    std::vector<std::uint8_t> px(8 * 8 * 3);
    for (int i = 0; i < 8 * 8; ++i) {
        px[i * 3 + 0] = 10;
        px[i * 3 + 1] = 20;
        px[i * 3 + 2] = 30;
    }
    ImageRef img{px.data(), 8, 8};

    Preprocessor p(cfg, 1);
    auto out = p.process({img});
    ASSERT_EQ(out.size(), 4u * 4 * 3);

    // Planar layout: the first 16 floats are all R, then all G, then all B.
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(out[i], 10.0f / 255.0f, 1e-5) << "R plane at " << i;
        EXPECT_NEAR(out[16 + i], 20.0f / 255.0f, 1e-5) << "G plane at " << i;
        EXPECT_NEAR(out[32 + i], 30.0f / 255.0f, 1e-5) << "B plane at " << i;
    }
}

TEST(Preprocess, NormalisationIsApplied) {
    PreprocessConfig cfg;
    cfg.out_w = 2;
    cfg.out_h = 2;
    cfg.mean[0] = 0.5f; cfg.mean[1] = 0.25f; cfg.mean[2] = 0.0f;
    cfg.stdev[0] = 0.5f; cfg.stdev[1] = 0.25f; cfg.stdev[2] = 1.0f;

    std::vector<std::uint8_t> px(4 * 4 * 3, 255);   // all channels at 1.0
    ImageRef img{px.data(), 4, 4};

    Preprocessor p(cfg, 1);
    auto out = p.process({img});
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], (1.0f - 0.5f) / 0.5f, 1e-5);
        EXPECT_NEAR(out[4 + i], (1.0f - 0.25f) / 0.25f, 1e-5);
        EXPECT_NEAR(out[8 + i], 1.0f, 1e-5);
    }
}

TEST(Preprocess, UpscaleAndDownscaleBothWork) {
    auto cfg = cfg224();
    auto small = make_image(32, 32, 7);
    auto large = make_image(1920, 1080, 8);

    Preprocessor p(cfg, 2);
    ImageRef a{small.data(), 32, 32};
    ImageRef b{large.data(), 1920, 1080};

    auto ga = p.process({a});
    auto gb = p.process({b});
    EXPECT_LT(max_abs_diff(ga, vision::preprocess_cpu(a, cfg)), 1e-4);
    EXPECT_LT(max_abs_diff(gb, vision::preprocess_cpu(b, cfg)), 1e-4);
}

TEST(Preprocess, RejectsBadConfigAndOversizedBatch) {
    PreprocessConfig bad = cfg224();
    bad.stdev[1] = 0.0f;
    EXPECT_THROW(Preprocessor(bad, 4), std::invalid_argument);

    PreprocessConfig zero = cfg224();
    zero.out_w = 0;
    EXPECT_THROW(Preprocessor(zero, 4), std::invalid_argument);

    Preprocessor p(cfg224(), 2);
    auto px = make_image(64, 64, 9);
    std::vector<ImageRef> big(3, ImageRef{px.data(), 64, 64});
    EXPECT_THROW(p.process(big), std::invalid_argument);
}

TEST(Preprocess, EmptyBatchIsNoOp) {
    Preprocessor p(cfg224(), 4);
    auto out = p.process({});
    EXPECT_TRUE(out.empty());
}

// ---------------------------------------------------------------------------
// Dynamic batcher
// ---------------------------------------------------------------------------
TEST(Batcher, GroupsUpToMaxBatch) {
    std::atomic<int> seen{0};
    std::vector<int> sizes;
    std::mutex m;
    {
        DynamicBatcher b(8, 50'000, [&](const std::vector<int>& batch) {
            std::lock_guard<std::mutex> lk(m);
            sizes.push_back(static_cast<int>(batch.size()));
            seen.fetch_add(static_cast<int>(batch.size()));
        });
        for (int i = 0; i < 64; ++i) b.submit(i);
        b.drain();
    }
    EXPECT_EQ(seen.load(), 64);
    for (int s : sizes) EXPECT_LE(s, 8) << "no batch may exceed max_batch";
}

TEST(Batcher, EveryItemIsHandledExactlyOnce) {
    std::mutex m;
    std::vector<int> got;
    {
        DynamicBatcher b(16, 20'000, [&](const std::vector<int>& batch) {
            std::lock_guard<std::mutex> lk(m);
            got.insert(got.end(), batch.begin(), batch.end());
        });
        for (int i = 0; i < 200; ++i) b.submit(i);
        b.drain();
    }
    std::sort(got.begin(), got.end());
    ASSERT_EQ(got.size(), 200u);
    for (int i = 0; i < 200; ++i) EXPECT_EQ(got[i], i) << "item " << i << " lost or duplicated";
}

// The reason the timeout exists: a lone request must not wait for a batch that
// will never fill.
TEST(Batcher, TimeoutFlushesPartialBatch) {
    std::atomic<int> fired{0};
    std::atomic<int> count{0};
    DynamicBatcher b(64, 2'000, [&](const std::vector<int>& batch) {
        fired.fetch_add(1);
        count.fetch_add(static_cast<int>(batch.size()));
    });

    b.submit(1);   // one item, batch size 64 -- only the deadline can release it
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (count.load() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    EXPECT_EQ(count.load(), 1) << "single request should flush on timeout";
    EXPECT_GE(fired.load(), 1);
    b.drain();
}

TEST(Batcher, ReportsStatistics) {
    DynamicBatcher b(4, 10'000, [](const std::vector<int>&) {});
    for (int i = 0; i < 20; ++i) b.submit(i);
    b.drain();
    EXPECT_EQ(b.items_processed(), 20);
    EXPECT_GE(b.batches_fired(), 5);
    EXPECT_LE(b.largest_batch(), 4);
}

TEST(Batcher, SubmitAfterDrainIsIgnored) {
    DynamicBatcher b(4, 5'000, [](const std::vector<int>&) {});
    b.submit(1);
    b.drain();
    const int before = b.items_processed();
    b.submit(2);
    EXPECT_EQ(b.items_processed(), before) << "must not accept work after drain";
}
