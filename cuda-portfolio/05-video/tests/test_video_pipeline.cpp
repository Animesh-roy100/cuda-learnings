#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "video_pipeline.h"

using video::BilateralParams;
using video::Nv12Frame;
using video::VideoPipeline;

namespace {

constexpr int W = 256, H = 128;

int max_abs_diff(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    int d = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(int(a[i]) - int(b[i])));
    return d;
}

std::vector<std::uint8_t> gray_from(const std::vector<std::uint8_t>& nv12) {
    return std::vector<std::uint8_t>(nv12.begin(), nv12.begin() + (std::size_t)W * H);
}

}  // namespace

TEST(Video, RejectsOddFrameSize) {
    // NV12 chroma is half resolution in both axes, so odd dimensions have no
    // valid representation.
    EXPECT_THROW(VideoPipeline(101, 100), std::invalid_argument);
    EXPECT_THROW(VideoPipeline(100, 101), std::invalid_argument);
    EXPECT_NO_THROW(VideoPipeline(100, 100));
}

TEST(Video, Nv12ToRgbMatchesCpu) {
    auto nv12 = video::make_test_nv12(W, H, 1);
    Nv12Frame f{nv12.data(), W, H};

    VideoPipeline p(W, H);
    auto gpu = p.nv12_to_rgb(f);
    auto cpu = video::nv12_to_rgb_cpu(f);

    ASSERT_EQ(gpu.size(), cpu.size());
    // Both round the same float arithmetic; allow one LSB.
    EXPECT_LE(max_abs_diff(gpu, cpu), 1);
}

TEST(Video, Nv12GrayIsTheLumaPlane) {
    auto nv12 = video::make_test_nv12(W, H, 2);
    Nv12Frame f{nv12.data(), W, H};
    VideoPipeline p(W, H);
    auto gray = p.nv12_to_gray(f);

    ASSERT_EQ(gray.size(), (std::size_t)W * H);
    for (std::size_t i = 0; i < gray.size(); ++i) EXPECT_EQ(gray[i], nv12[i]);
}

TEST(Video, Nv12RejectsWrongSize) {
    auto nv12 = video::make_test_nv12(W, H, 3);
    VideoPipeline p(W, H);
    Nv12Frame bad{nv12.data(), W / 2, H};
    EXPECT_THROW(p.nv12_to_rgb(bad), std::invalid_argument);
}

TEST(Video, BilateralMatchesCpu) {
    auto nv12 = video::make_test_nv12(W, H, 4);
    auto gray = gray_from(nv12);

    BilateralParams bp;
    bp.radius = 3;
    bp.sigma_spatial = 3.0f;
    bp.sigma_range = 25.0f;

    VideoPipeline p(W, H);
    auto gpu = p.bilateral(gray, bp, /*use_texture=*/false);
    auto cpu = video::bilateral_cpu(gray, W, H, bp);

    ASSERT_EQ(gpu.size(), cpu.size());
    // __expf is a fast-math approximation, so allow a small rounding gap.
    EXPECT_LE(max_abs_diff(gpu, cpu), 2);
}

// The point of the texture path: hardware address clamping must give exactly
// the same answer as explicit min/max, including at the borders.
TEST(Video, TextureClampMatchesExplicitClamp) {
    auto nv12 = video::make_test_nv12(W, H, 5);
    auto gray = gray_from(nv12);

    BilateralParams bp;
    bp.radius = 4;

    VideoPipeline p(W, H);
    auto manual = p.bilateral(gray, bp, false);
    auto textured = p.bilateral(gray, bp, true);

    ASSERT_EQ(manual.size(), textured.size());
    EXPECT_LE(max_abs_diff(manual, textured), 1)
        << "hardware clamping must not change the result";

    // Check the borders specifically -- that is where clamping actually bites.
    int border_diff = 0;
    for (int x = 0; x < W; ++x) {
        border_diff = std::max(border_diff, std::abs(int(manual[x]) - int(textured[x])));
        border_diff = std::max(border_diff,
                               std::abs(int(manual[(std::size_t)(H - 1) * W + x]) -
                                        int(textured[(std::size_t)(H - 1) * W + x])));
    }
    EXPECT_LE(border_diff, 1) << "border handling differs between the two paths";
}

TEST(Video, BilateralPreservesFlatRegions) {
    // A constant image must survive any edge-preserving filter untouched.
    std::vector<std::uint8_t> flat((std::size_t)W * H, 120);
    BilateralParams bp;
    VideoPipeline p(W, H);
    auto out = p.bilateral(flat, bp, true);
    for (auto v : out) EXPECT_NEAR(int(v), 120, 1);
}

TEST(Video, BilateralRejectsBadParams) {
    std::vector<std::uint8_t> gray((std::size_t)W * H, 10);
    VideoPipeline p(W, H);
    BilateralParams bp;

    bp.radius = 99;
    EXPECT_THROW(p.bilateral(gray, bp, false), std::invalid_argument);
    bp.radius = 3;
    bp.sigma_range = 0.0f;
    EXPECT_THROW(p.bilateral(gray, bp, false), std::invalid_argument);
    bp.sigma_range = 25.0f;
    EXPECT_THROW(p.bilateral(std::vector<std::uint8_t>(10), bp, false), std::invalid_argument);
}

TEST(Video, MotionHistoryFirstFrameShowsNoMotion) {
    auto nv12 = video::make_test_nv12(W, H, 6);
    auto gray = gray_from(nv12);
    VideoPipeline p(W, H);

    auto mhi = p.motion_history(gray, 10, 32);
    // The first frame has no predecessor; seeding prev with it is what stops
    // the entire image reading as motion.
    for (auto v : mhi) EXPECT_EQ(int(v), 0);
}

TEST(Video, MotionHistoryDetectsChangeAndDecays) {
    VideoPipeline p(W, H);
    std::vector<std::uint8_t> a((std::size_t)W * H, 50);
    std::vector<std::uint8_t> b = a;
    // Move a bright block into the middle of the frame.
    for (int y = 32; y < 64; ++y)
        for (int x = 32; x < 64; ++x) b[(std::size_t)y * W + x] = 200;

    p.motion_history(a, 10, 32);            // seed
    auto m1 = p.motion_history(b, 10, 32);

    EXPECT_EQ(int(m1[(std::size_t)40 * W + 40]), 255) << "changed pixel should light up";
    EXPECT_EQ(int(m1[0]), 0) << "unchanged pixel should stay dark";

    // Hold the frame steady: the trail must decay, not persist.
    auto m2 = p.motion_history(b, 10, 32);
    EXPECT_EQ(int(m2[(std::size_t)40 * W + 40]), 255 - 32);
    auto m3 = p.motion_history(b, 10, 32);
    EXPECT_EQ(int(m3[(std::size_t)40 * W + 40]), 255 - 64);
}

TEST(Video, ResetMotionClearsState) {
    VideoPipeline p(W, H);
    std::vector<std::uint8_t> a((std::size_t)W * H, 50);
    std::vector<std::uint8_t> b((std::size_t)W * H, 200);
    p.motion_history(a, 10, 32);
    auto lit = p.motion_history(b, 10, 32);
    EXPECT_EQ(int(lit[0]), 255);

    p.reset_motion();
    auto after = p.motion_history(b, 10, 32);
    for (auto v : after) EXPECT_EQ(int(v), 0) << "reset must clear history and prev frame";
}

TEST(Video, TransferHierarchyIsPhysicallySane) {
    auto r = video::measure_transfers(4u << 20);   // 4 MB

    EXPECT_GT(r.pinned_gbps, r.pageable_gbps)
        << "pinned memory must beat pageable: pageable has to stage through a "
           "driver bounce buffer";

    // Resident VRAM must beat reading host memory over PCIe. If this ever
    // inverts, the measurement is wrong (almost always a missing warm-up),
    // not the hardware.
    EXPECT_LT(r.device_kernel_ms, r.zero_copy_kernel_ms)
        << "VRAM (~192 GB/s) cannot be slower than PCIe (~12 GB/s)";

    // Zero-copy vs upload-then-compute is NOT a guaranteed win, and asserting
    // that it is makes the suite flaky. Both paths pull the same bytes across
    // the same PCIe link, so they are inherently comparable: zero-copy saves
    // the explicit copy and the second buffer, while the upload path gets to
    // stream at full DMA width. Measured here they land within ~10% of each
    // other, and which one leads depends on size and overlap.
    //
    // What IS guaranteed is that zero-copy stays in the same league -- if it
    // were several times worse, the mapping would not be working.
    EXPECT_LT(r.zero_copy_kernel_ms, r.upload_plus_kernel_ms * 2.0)
        << "zero-copy should be comparable to upload+compute for touch-once data";
    EXPECT_GT(r.zero_copy_kernel_ms, r.device_kernel_ms)
        << "zero-copy reads over PCIe, so it cannot match resident VRAM";
}
