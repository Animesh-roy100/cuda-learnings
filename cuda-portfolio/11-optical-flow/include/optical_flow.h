#pragma once
//
// KLT feature tracking: Harris corner detection, image pyramids, and iterative
// Lucas-Kanade optical flow. No CUDA syntax in this header.
//
// The method in one paragraph: Lucas-Kanade assumes brightness constancy and
// solves a 2x2 least-squares system per feature for the displacement that best
// explains the local gradient field. That linearisation is only valid for
// sub-pixel motion, so a PYRAMID is not an optimisation here -- it is what makes
// large displacements tractable at all. Track coarse, upscale, refine.
//
#include <cstdint>
#include <vector>

namespace flow {

struct Point2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Feature {
    Point2 pos;
    float response = 0.0f;   // Harris cornerness
};

struct TrackResult {
    std::vector<Point2> positions;   // tracked location in the second frame
    std::vector<std::uint8_t> ok;    // 0 = lost (diverged, or left the frame)
    int tracked = 0;                 // count with ok != 0
};

struct HarrisParams {
    float k = 0.04f;              // Harris trace coefficient
    float quality = 0.01f;        // keep responses above quality * max
    int min_distance = 8;         // non-maximum suppression radius, pixels
    int max_features = 4096;
};

struct KltParams {
    int window = 7;               // odd; the integration window per feature
    int iterations = 20;          // Newton steps per pyramid level
    int levels = 4;               // pyramid depth, 1 = no pyramid
    float epsilon = 0.01f;        // stop when the update is smaller than this
    float max_residual = 50.0f;   // declare the track lost above this
};

class FlowTracker {
public:
    FlowTracker(int width, int height);
    ~FlowTracker();
    FlowTracker(const FlowTracker&) = delete;
    FlowTracker& operator=(const FlowTracker&) = delete;

    // 8-bit grayscale, width*height, row-major.
    std::vector<Feature> detect_harris(const std::vector<std::uint8_t>& gray,
                                       const HarrisParams& p,
                                       float* elapsed_ms = nullptr);

    // Track features from `prev` into `next`.
    TrackResult track(const std::vector<std::uint8_t>& prev,
                      const std::vector<std::uint8_t>& next,
                      const std::vector<Point2>& features,
                      const KltParams& p,
                      float* elapsed_ms = nullptr);

    // Gaussian-blurred half-resolution levels. Exposed because the pyramid is
    // the part most worth inspecting when tracking misbehaves.
    std::vector<std::vector<std::uint8_t>> build_pyramid(
        const std::vector<std::uint8_t>& gray, int levels);

    int width() const;
    int height() const;

    // Host references for the tests.
    static std::vector<float> harris_response_cpu(const std::vector<std::uint8_t>& gray,
                                                  int w, int h, float k);
    static std::vector<std::uint8_t> downsample_cpu(const std::vector<std::uint8_t>& src,
                                                    int w, int h);

private:
    struct Impl;
    Impl* impl_;
};

// Synthetic frames, so tests need no video: a field of bright blobs on a dark
// background, optionally shifted by a known amount. Ground truth motion is
// exactly the shift, which is what makes the tracking tests meaningful.
std::vector<std::uint8_t> make_frame(int w, int h, float shift_x, float shift_y,
                                     unsigned seed);

}  // namespace flow
