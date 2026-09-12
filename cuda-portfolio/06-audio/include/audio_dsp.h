#pragma once
//
// Real-time audio: batched STFT spectrograms and phase-vocoder pitch shifting.
// No CUDA syntax in this header.
//
#include <cstddef>
#include <vector>

namespace audio {

struct StftConfig {
    int frame_size = 1024;   // FFT length, power of two
    int hop = 256;           // analysis hop; frame_size/hop = overlap factor
    int sample_rate = 48000;
};

class StftProcessor {
public:
    StftProcessor(const StftConfig& cfg, int max_channels, int max_samples);
    ~StftProcessor();
    StftProcessor(const StftProcessor&) = delete;
    StftProcessor& operator=(const StftProcessor&) = delete;

    // pcm is [channels][samples], interleaved by channel-major (channel c
    // occupies pcm[c*samples .. (c+1)*samples)).
    //
    // Returns magnitudes as [channels][frames][bins], bins = frame_size/2 + 1.
    std::vector<float> spectrogram(const std::vector<float>& pcm, int channels,
                                   int samples, float* elapsed_ms = nullptr);

    // Phase vocoder. ratio > 1 raises pitch, < 1 lowers it, duration preserved.
    std::vector<float> pitch_shift(const std::vector<float>& pcm, int channels,
                                   int samples, float ratio,
                                   float* elapsed_ms = nullptr);

    // Analysis then synthesis with no modification. Should reconstruct the
    // input (away from the edges) -- the cleanest test that the window
    // normalisation and overlap-add are right.
    std::vector<float> resynthesize(const std::vector<float>& pcm, int channels,
                                    int samples, float* elapsed_ms = nullptr);

    int bins() const;
    int frames_for(int samples) const;
    const StftConfig& config() const;

private:
    struct Impl;
    Impl* impl_;
};

// Hann window, used for both analysis and synthesis. With 75% overlap
// (hop = frame/4) the squared Hann windows sum to a constant, which is what
// makes overlap-add reconstruct exactly.
std::vector<float> hann_window(int n);

}  // namespace audio
