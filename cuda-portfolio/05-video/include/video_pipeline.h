#pragma once
//
// Real-time video analytics kernels. No CUDA syntax in this header.
//
// NVDEC hands decoded frames over as NV12 surfaces already in VRAM. Everything
// here is written to consume that layout directly, so a real pipeline never
// round-trips a frame through host memory.
//
#include <cstdint>
#include <vector>

namespace video {

// NV12: a full-resolution Y plane followed by a half-resolution interleaved
// UV plane. This is what every hardware decoder actually produces; converting
// on the CPU first is the mistake that makes video pipelines slow.
struct Nv12Frame {
    const std::uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    static std::size_t bytes(int w, int h) {
        return static_cast<std::size_t>(w) * h * 3 / 2;
    }
};

struct BilateralParams {
    int radius = 3;
    float sigma_spatial = 3.0f;
    float sigma_range = 25.0f;   // in 0-255 intensity units
};

class VideoPipeline {
public:
    VideoPipeline(int width, int height);
    ~VideoPipeline();
    VideoPipeline(const VideoPipeline&) = delete;
    VideoPipeline& operator=(const VideoPipeline&) = delete;

    // NV12 -> packed RGB8, BT.601 limited range.
    std::vector<std::uint8_t> nv12_to_rgb(const Nv12Frame& f, float* elapsed_ms = nullptr);

    // NV12 -> luma only (just the Y plane, but validated and on-device).
    std::vector<std::uint8_t> nv12_to_gray(const Nv12Frame& f, float* elapsed_ms = nullptr);

    // Edge-preserving bilateral filter on an 8-bit single-channel image.
    //
    // use_texture selects how out-of-bounds neighbours are handled:
    //   false -> explicit min/max clamping in the kernel
    //   true  -> a texture object with cudaAddressModeClamp, so the hardware
    //            address unit does it for free
    // Both must produce identical output; the benchmark prices the difference.
    std::vector<std::uint8_t> bilateral(const std::vector<std::uint8_t>& gray,
                                        const BilateralParams& p, bool use_texture,
                                        float* elapsed_ms = nullptr);

    // Motion History Image: pixels that changed this frame go to 255, and every
    // other pixel decays. Stateful across calls.
    std::vector<std::uint8_t> motion_history(const std::vector<std::uint8_t>& gray,
                                             int threshold, int decay,
                                             float* elapsed_ms = nullptr);
    void reset_motion();

    int width() const;
    int height() const;

private:
    struct Impl;
    Impl* impl_;
};

// CPU references for the tests.
std::vector<std::uint8_t> nv12_to_rgb_cpu(const Nv12Frame& f);
std::vector<std::uint8_t> bilateral_cpu(const std::vector<std::uint8_t>& gray, int w, int h,
                                        const BilateralParams& p);

// Synthetic NV12 frame, so tests and benchmarks need no video file.
std::vector<std::uint8_t> make_test_nv12(int w, int h, unsigned seed);

// ---------------------------------------------------------------------------
// Zero-copy and cross-process sharing -- the parts of the "zero-copy pipeline"
// that do not need the Video Codec SDK.
// ---------------------------------------------------------------------------
struct TransferReport {
    double pageable_gbps = 0.0;
    double pinned_gbps = 0.0;
    double zero_copy_kernel_ms = 0.0;
    double device_kernel_ms = 0.0;
    double upload_plus_kernel_ms = 0.0;
};
TransferReport measure_transfers(std::size_t bytes);

// True if a device pointer can be exported to another process on this
// platform. Documented as Linux-only in most places; measured here.
bool cuda_ipc_available();

}  // namespace video
