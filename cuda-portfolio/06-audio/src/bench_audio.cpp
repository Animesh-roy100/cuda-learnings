// Audio benchmark: how many concurrent streams can this card handle in real time?

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "audio_dsp.h"
#include "cu/device.hpp"

using audio::StftConfig;
using audio::StftProcessor;

int main() {
    auto dev = cu::query_device();
    cu::print_banner(dev);

    StftConfig cfg;
    cfg.frame_size = 1024;
    cfg.hop = 256;
    cfg.sample_rate = 48000;

    const int SECONDS = 1;
    const int N = cfg.sample_rate * SECONDS;

    std::printf("frame %d, hop %d (%.0f%% overlap), %d Hz\n",
                cfg.frame_size, cfg.hop, 100.0 * (1.0 - double(cfg.hop) / cfg.frame_size),
                cfg.sample_rate);
    std::printf("bins per frame: %d\n\n", cfg.frame_size / 2 + 1);

    std::printf("=== batched STFT throughput ===\n");
    for (int ch : {1, 8, 32, 128, 512}) {
        StftProcessor p(cfg, ch, N);
        std::vector<float> pcm((size_t)ch * N);
        for (int c = 0; c < ch; ++c)
            for (int i = 0; i < N; ++i)
                pcm[(size_t)c * N + i] =
                    0.5f * std::sin(2.0f * 3.14159265f * (200.0f + c) * i / cfg.sample_rate);

        float ms = 0.0f, best = 1e30f;
        for (int r = 0; r < 6; ++r) {
            p.spectrogram(pcm, ch, N, &ms);
            if (r >= 2) best = std::min(best, ms);
        }
        const double audio_ms = 1000.0 * SECONDS * ch;
        std::printf("  %4d ch x %ds : %7.2f ms  -> %6.0fx real time (%d frames)\n",
                    ch, SECONDS, best, audio_ms / best, p.frames_for(N) * ch);
    }

    std::printf("\n=== phase vocoder pitch shift ===\n");
    for (int ch : {1, 8, 32, 128}) {
        StftProcessor p(cfg, ch, N);
        std::vector<float> pcm((size_t)ch * N);
        for (size_t i = 0; i < pcm.size(); ++i)
            pcm[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * (i % N) / cfg.sample_rate);

        float ms = 0.0f, best = 1e30f;
        for (int r = 0; r < 6; ++r) {
            p.pitch_shift(pcm, ch, N, 1.5f, &ms);
            if (r >= 2) best = std::min(best, ms);
        }
        std::printf("  %4d ch x %ds : %7.2f ms  -> %6.0fx real time\n",
                    ch, SECONDS, best, 1000.0 * SECONDS * ch / best);
    }

    std::printf("\nThe FFT itself is cuFFT's problem. The CUDA work that matters is\n");
    std::printf("around it: framing thousands of overlapping windows into one\n");
    std::printf("contiguous batch so cuFFT sees a dense [batch][frame] array, and\n");
    std::printf("doing the phase arithmetic in place on cufftComplex.\n");
    std::printf("\nNote the phase-vocoder scaling: phase integration is a RECURRENCE\n");
    std::printf("along time, so it cannot be parallelised across frames. One thread\n");
    std::printf("owns one bin and walks every frame in order. Parallelism comes from\n");
    std::printf("bins x channels, which is why more channels scales well and a\n");
    std::printf("single channel does not.\n");
    return 0;
}
