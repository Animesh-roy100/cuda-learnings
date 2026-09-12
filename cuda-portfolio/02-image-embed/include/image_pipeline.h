#pragma once
//
// Streaming image preprocessing for embedding inference. No CUDA syntax.
//
// The CPU path this replaces (OpenCV resize + normalize + HWC->CHW) is usually
// the bottleneck in an embedding service: the model runs in a few ms and
// preprocessing takes longer. Doing it on the GPU removes that, but only if
// the transfers overlap the compute -- otherwise you have just moved the stall.
//
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace vision {

struct PreprocessConfig {
    int out_w = 224;
    int out_h = 224;
    float mean[3] = {0.485f, 0.456f, 0.406f};   // ImageNet defaults
    float stdev[3] = {0.229f, 0.224f, 0.225f};
};

// A borrowed view of an RGB8 image, row-major, tightly packed.
struct ImageRef {
    const std::uint8_t* rgb = nullptr;
    int width = 0;
    int height = 0;
    std::size_t bytes() const { return static_cast<std::size_t>(width) * height * 3; }
};

class Preprocessor {
public:
    Preprocessor(const PreprocessConfig& cfg, int max_batch, int num_streams = 4);
    ~Preprocessor();
    Preprocessor(const Preprocessor&) = delete;
    Preprocessor& operator=(const Preprocessor&) = delete;

    // Returns NCHW float32: [batch][3][out_h][out_w].
    // Overlapped: uploads, kernels and downloads round-robin across streams.
    std::vector<float> process(const std::vector<ImageRef>& batch,
                               float* elapsed_ms = nullptr);

    // Same result, everything serialised on the default stream. Kept so the
    // benchmark can price the overlap rather than assert it.
    std::vector<float> process_serial(const std::vector<ImageRef>& batch,
                                      float* elapsed_ms = nullptr);

    std::size_t output_floats_per_image() const;
    const PreprocessConfig& config() const;

private:
    struct Impl;
    Impl* impl_;
};

// Bilinear resize + per-channel normalise + HWC->CHW, on the CPU.
// The ground truth the GPU kernel is checked against.
std::vector<float> preprocess_cpu(const ImageRef& img, const PreprocessConfig& cfg);

// ---------------------------------------------------------------------------
// Dynamic batcher
//
// Requests arrive one at a time; the GPU wants them in groups. This collects
// them until either `max_batch` are waiting or `max_delay_us` has elapsed,
// then fires one batch. The timeout is what stops a quiet period from
// stranding a single request forever.
// ---------------------------------------------------------------------------
class DynamicBatcher {
public:
    using Handler = std::function<void(const std::vector<int>&)>;

    DynamicBatcher(int max_batch, int max_delay_us, Handler on_batch);
    ~DynamicBatcher();
    DynamicBatcher(const DynamicBatcher&) = delete;
    DynamicBatcher& operator=(const DynamicBatcher&) = delete;

    void submit(int request_id);
    void drain();                 // flush and stop accepting work

    int batches_fired() const;
    int items_processed() const;
    int largest_batch() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vision
