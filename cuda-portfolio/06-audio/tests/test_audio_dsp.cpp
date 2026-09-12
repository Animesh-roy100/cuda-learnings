#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "audio_dsp.h"

using audio::StftConfig;
using audio::StftProcessor;

namespace {

constexpr float kPi = 3.14159265358979323846f;

std::vector<float> sine(int samples, float freq, int rate, float amp = 1.0f) {
    std::vector<float> s(samples);
    for (int i = 0; i < samples; ++i)
        s[i] = amp * std::sin(2.0f * kPi * freq * i / static_cast<float>(rate));
    return s;
}

StftConfig cfg1024() {
    StftConfig c;
    c.frame_size = 1024;
    c.hop = 256;            // 75% overlap: squared Hann sums to a constant
    c.sample_rate = 48000;
    return c;
}

// Index of the loudest bin in one frame.
int peak_bin(const std::vector<float>& mag, int frame, int bins) {
    int best = 0;
    float bv = -1.0f;
    for (int b = 0; b < bins; ++b) {
        float v = mag[(std::size_t)frame * bins + b];
        if (v > bv) { bv = v; best = b; }
    }
    return best;
}

// Dominant frequency of a signal, by brute-force correlation. Independent of
// the code under test, which is the point: a bug in the STFT cannot hide it.
float dominant_freq(const std::vector<float>& x, int rate, float lo, float hi) {
    float best_f = lo, best_p = -1.0f;
    for (float f = lo; f <= hi; f += 1.0f) {
        double re = 0, im = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            double a = 2.0 * static_cast<double>(kPi) * f * i / rate;
            re += x[i] * std::cos(a);
            im += x[i] * std::sin(a);
        }
        float p = static_cast<float>(re * re + im * im);
        if (p > best_p) { best_p = p; best_f = f; }
    }
    return best_f;
}

}  // namespace

TEST(Stft, HannWindowSumsCorrectly) {
    auto w = audio::hann_window(8);
    ASSERT_EQ(w.size(), 8u);
    EXPECT_NEAR(w[0], 0.0f, 1e-6);
    EXPECT_NEAR(w[4], 1.0f, 1e-6);          // peak at the centre
    for (float v : w) { EXPECT_GE(v, 0.0f); EXPECT_LE(v, 1.0f); }
}

TEST(Stft, BinCountAndFrameCount) {
    StftProcessor p(cfg1024(), 2, 48000);
    EXPECT_EQ(p.bins(), 513);               // frame/2 + 1
    EXPECT_EQ(p.frames_for(1024), 1);
    EXPECT_EQ(p.frames_for(1024 + 256), 2);
    EXPECT_EQ(p.frames_for(1024 + 256 * 4), 5);
}

TEST(Stft, PureToneLandsInTheExpectedBin) {
    auto cfg = cfg1024();
    const int N = 48000;
    const float f = 1000.0f;
    StftProcessor p(cfg, 1, N);

    auto x = sine(N, f, cfg.sample_rate);
    auto mag = p.spectrogram(x, 1, N);

    const int bins = p.bins();
    // bin index = f / (rate / frame_size)
    const int expect = static_cast<int>(std::lround(f / (float(cfg.sample_rate) / cfg.frame_size)));
    for (int frame : {2, 10, 50}) {
        EXPECT_NEAR(peak_bin(mag, frame, bins), expect, 1)
            << "frame " << frame << " peaked at the wrong bin";
    }
}

TEST(Stft, LouderInputGivesProportionallyLargerMagnitude) {
    auto cfg = cfg1024();
    const int N = 16384;
    StftProcessor p(cfg, 1, N);

    auto quiet = p.spectrogram(sine(N, 1000.0f, cfg.sample_rate, 0.25f), 1, N);
    auto loud = p.spectrogram(sine(N, 1000.0f, cfg.sample_rate, 0.50f), 1, N);

    const int bins = p.bins();
    const int b = peak_bin(loud, 4, bins);
    float q = quiet[(std::size_t)4 * bins + b];
    float l = loud[(std::size_t)4 * bins + b];
    EXPECT_NEAR(l / q, 2.0f, 0.05f) << "STFT magnitude must be linear in amplitude";
}

TEST(Stft, MultipleChannelsAreIndependent) {
    auto cfg = cfg1024();
    const int N = 16384;
    StftProcessor p(cfg, 2, N);

    auto a = sine(N, 1000.0f, cfg.sample_rate);
    auto b = sine(N, 4000.0f, cfg.sample_rate);
    std::vector<float> both;
    both.insert(both.end(), a.begin(), a.end());
    both.insert(both.end(), b.begin(), b.end());

    auto mag = p.spectrogram(both, 2, N);
    const int bins = p.bins();
    const int nf = p.frames_for(N);

    std::vector<float> ch0(mag.begin(), mag.begin() + (std::size_t)nf * bins);
    std::vector<float> ch1(mag.begin() + (std::size_t)nf * bins, mag.end());

    const float hz = float(cfg.sample_rate) / cfg.frame_size;
    EXPECT_NEAR(peak_bin(ch0, 4, bins), std::lround(1000.0f / hz), 1);
    EXPECT_NEAR(peak_bin(ch1, 4, bins), std::lround(4000.0f / hz), 1);
}

// Analysis then synthesis with no modification must return the input. This is
// the strongest single check on windowing, cuFFT normalisation and overlap-add
// all being right together.
TEST(Vocoder, IdentityResynthesisReconstructsSignal) {
    auto cfg = cfg1024();
    const int N = 16384;
    StftProcessor p(cfg, 1, N);

    auto x = sine(N, 440.0f, cfg.sample_rate, 0.8f);
    auto y = p.resynthesize(x, 1, N);
    ASSERT_EQ(y.size(), x.size());

    // Skip the edges, where the overlap-add has not reached full window sum.
    const int skip = cfg.frame_size;
    double err = 0.0, ref = 0.0;
    for (int i = skip; i < N - skip; ++i) {
        err += double(y[i] - x[i]) * (y[i] - x[i]);
        ref += double(x[i]) * x[i];
    }
    EXPECT_LT(std::sqrt(err / ref), 0.02) << "identity round trip should reconstruct";
}

TEST(Vocoder, RaisingPitchRaisesDominantFrequency) {
    auto cfg = cfg1024();
    const int N = 32768;
    StftProcessor p(cfg, 1, N);

    const float f0 = 440.0f;
    auto x = sine(N, f0, cfg.sample_rate, 0.8f);
    auto up = p.pitch_shift(x, 1, N, 2.0f);     // one octave up

    // Measure on a clean interior slice, away from edge artefacts.
    std::vector<float> slice(up.begin() + 4096, up.begin() + 4096 + 8192);
    float f = dominant_freq(slice, cfg.sample_rate, 300.0f, 1200.0f);
    EXPECT_NEAR(f, 2.0f * f0, 40.0f) << "measured " << f << " Hz, expected ~" << 2 * f0;
}

TEST(Vocoder, LoweringPitchLowersDominantFrequency) {
    auto cfg = cfg1024();
    const int N = 32768;
    StftProcessor p(cfg, 1, N);

    const float f0 = 880.0f;
    auto x = sine(N, f0, cfg.sample_rate, 0.8f);
    auto down = p.pitch_shift(x, 1, N, 0.5f);   // one octave down

    std::vector<float> slice(down.begin() + 4096, down.begin() + 4096 + 8192);
    float f = dominant_freq(slice, cfg.sample_rate, 200.0f, 1200.0f);
    EXPECT_NEAR(f, 0.5f * f0, 40.0f) << "measured " << f << " Hz, expected ~" << 0.5 * f0;
}

TEST(Vocoder, OutputLengthAlwaysMatchesInput) {
    auto cfg = cfg1024();
    const int N = 16384;
    StftProcessor p(cfg, 2, N);
    std::vector<float> x((std::size_t)2 * N, 0.1f);
    // Pitch shifting must preserve duration -- that is the whole point of
    // pairing the vocoder with a resample.
    for (float r : {0.5f, 0.8f, 1.0f, 1.5f, 2.0f}) {
        auto y = p.pitch_shift(x, 2, N, r);
        EXPECT_EQ(y.size(), x.size()) << "ratio " << r;
    }
}

TEST(Vocoder, SilenceStaysSilent) {
    auto cfg = cfg1024();
    const int N = 8192;
    StftProcessor p(cfg, 1, N);
    std::vector<float> x(N, 0.0f);
    auto y = p.pitch_shift(x, 1, N, 1.5f);
    for (float v : y) {
        EXPECT_FALSE(std::isnan(v));
        EXPECT_NEAR(v, 0.0f, 1e-6f);
    }
}

TEST(Stft, RejectsBadConfigAndArguments) {
    StftConfig bad = cfg1024();
    bad.frame_size = 1000;                  // not a power of two
    EXPECT_THROW(StftProcessor(bad, 1, 1024), std::invalid_argument);

    StftConfig badhop = cfg1024();
    badhop.hop = 0;
    EXPECT_THROW(StftProcessor(badhop, 1, 1024), std::invalid_argument);

    StftProcessor p(cfg1024(), 1, 8192);
    std::vector<float> x(8192, 0.0f);
    EXPECT_THROW(p.spectrogram(x, 2, 8192), std::invalid_argument);       // too many channels
    EXPECT_THROW(p.spectrogram(x, 1, 9999), std::invalid_argument);       // too many samples
    EXPECT_THROW(p.pitch_shift(x, 1, 8192, -1.0f), std::invalid_argument);
}
