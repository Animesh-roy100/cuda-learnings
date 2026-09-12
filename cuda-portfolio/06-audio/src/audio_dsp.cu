// Batched STFT and phase vocoder on cuFFT.
//
// The STFT itself is cuFFT's job; the interesting CUDA is everything around it:
// framing thousands of overlapping windows without re-reading the signal
// thousands of times, and doing the phase arithmetic on cufftComplex in place.

#include "audio_dsp.h"

#include <cufft.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace audio {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("cuFFT ") + what + " failed: " +
                                 std::to_string(static_cast<int>(r)));
}

// Frame the signal: gather overlapping windows into a contiguous batch so
// cuFFT sees [batch][frame_size]. Each output element is read once; the input
// is read `overlap` times, but coalesced, which is far cheaper than letting
// cuFFT stride into the original buffer.
__global__ void k_frame_and_window(const float* __restrict__ pcm, int samples,
                                   float* __restrict__ frames, int frame_size, int hop,
                                   int n_frames, int channels,
                                   const float* __restrict__ win) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * n_frames * frame_size;
    if (i >= total) return;

    int s = i % frame_size;
    int f = (i / frame_size) % n_frames;
    int c = i / (frame_size * n_frames);

    int src = f * hop + s;
    float v = (src < samples) ? pcm[(size_t)c * samples + src] : 0.0f;
    frames[i] = v * win[s];
}

// Magnitude of each bin.
__global__ void k_magnitude(const cufftComplex* __restrict__ spec,
                            float* __restrict__ mag, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    cufftComplex c = spec[i];
    mag[i] = sqrtf(c.x * c.x + c.y * c.y);
}

// ---------------------------------------------------------------------------
// Phase vocoder core.
//
// For each bin, the phase advance between consecutive analysis frames tells you
// the bin's TRUE frequency, not just its nominal centre. Subtract the expected
// advance, wrap the remainder into (-pi, pi], and you have the deviation. Scale
// the advance by the stretch ratio and integrate to get synthesis phase.
//
// Getting the wrapping wrong is the classic phase-vocoder bug: it produces a
// signal that sounds right in magnitude but smears transients into a metallic
// ring, because every frame's phase drifts a little further out of alignment.
// ---------------------------------------------------------------------------
__global__ void k_phase_vocoder(const cufftComplex* __restrict__ in,
                                cufftComplex* __restrict__ out,
                                float* __restrict__ last_phase,
                                float* __restrict__ sum_phase,
                                int bins, int n_frames, int channels,
                                int hop_a, float ratio) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (bin >= bins || c >= channels) return;

    const float expected = kTwoPi * static_cast<float>(hop_a) * bin / (2.0f * (bins - 1));
    const float hop_s = hop_a * ratio;

    float lp = 0.0f, sp = 0.0f;
    for (int f = 0; f < n_frames; ++f) {
        size_t idx = ((size_t)c * n_frames + f) * bins + bin;
        cufftComplex v = in[idx];
        float mag = sqrtf(v.x * v.x + v.y * v.y);
        float phase = atan2f(v.y, v.x);

        float delta = phase - lp - expected;
        // Wrap into (-pi, pi]. Without this the deviation accumulates 2*pi
        // errors and the resynthesis loses phase coherence.
        delta -= kTwoPi * floorf(delta / kTwoPi + 0.5f);
        lp = phase;

        float true_advance = expected + delta;
        sp += true_advance * ratio;

        out[idx].x = mag * cosf(sp);
        out[idx].y = mag * sinf(sp);
        (void)hop_s;
    }
    last_phase[(size_t)c * bins + bin] = lp;
    sum_phase[(size_t)c * bins + bin] = sp;
}

// Overlap-add with the synthesis window, accumulating the window-square sum so
// the result can be normalised exactly rather than by a hand-tuned constant.
__global__ void k_overlap_add(const float* __restrict__ frames, int frame_size, int hop,
                              int n_frames, int channels, int out_len,
                              const float* __restrict__ win,
                              float* __restrict__ out, float* __restrict__ wsum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * n_frames * frame_size;
    if (i >= total) return;

    int s = i % frame_size;
    int f = (i / frame_size) % n_frames;
    int c = i / (frame_size * n_frames);

    int dst = f * hop + s;
    if (dst >= out_len) return;

    float w = win[s];
    atomicAdd(&out[(size_t)c * out_len + dst], frames[i] * w);
    atomicAdd(&wsum[(size_t)c * out_len + dst], w * w);
}

__global__ void k_normalize(float* out, const float* wsum, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float w = wsum[i];
    out[i] = (w > 1e-8f) ? out[i] / w : 0.0f;
}

__global__ void k_scale(float* x, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// Linear resample. Combined with time-stretching, this is what converts a
// duration change into a PITCH change at constant duration.
__global__ void k_resample(const float* __restrict__ in, int in_len,
                           float* __restrict__ out, int out_len, int channels,
                           float ratio) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * out_len;
    if (i >= total) return;
    int c = i / out_len;
    int t = i % out_len;

    float src = t * ratio;
    int i0 = static_cast<int>(src);
    float frac = src - i0;
    float a = (i0 < in_len) ? in[(size_t)c * in_len + i0] : 0.0f;
    float b = (i0 + 1 < in_len) ? in[(size_t)c * in_len + i0 + 1] : 0.0f;
    out[i] = a + (b - a) * frac;
}

}  // namespace

std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = 0.5f * (1.0f - std::cos(kTwoPi * i / static_cast<float>(n)));
    return w;
}

// ---------------------------------------------------------------------------
struct StftProcessor::Impl {
    StftConfig cfg;
    int max_channels = 1;
    int max_samples = 0;
    int max_frames = 0;
    int bins = 0;

    cufftHandle plan_fwd = 0;
    cufftHandle plan_inv = 0;
    int plan_batch = 0;
    size_t out_cap = 0;   // floats currently allocated for d_out / d_wsum

    // The overlap-add buffer length depends on the STRETCH ratio, which is not
    // known at construction: at ratio 2 the intermediate signal is roughly
    // twice the input. Sizing it from max_samples alone silently overflows, so
    // it grows on demand instead.
    void ensure_out(size_t floats) {
        if (out_cap >= floats) return;
        if (d_out) cudaFree(d_out);
        if (d_wsum) cudaFree(d_wsum);
        CU_CHECK(cudaMalloc(&d_out, floats * sizeof(float)));
        CU_CHECK(cudaMalloc(&d_wsum, floats * sizeof(float)));
        out_cap = floats;
    }

    float* d_pcm = nullptr;
    float* d_frames = nullptr;
    float* d_out = nullptr;
    float* d_wsum = nullptr;
    float* d_resampled = nullptr;
    float* d_win = nullptr;
    cufftComplex* d_spec = nullptr;
    cufftComplex* d_spec2 = nullptr;
    float* d_lastphase = nullptr;
    float* d_sumphase = nullptr;

    void make_plans(int batch) {
        if (plan_batch == batch) return;
        if (plan_fwd) cufftDestroy(plan_fwd);
        if (plan_inv) cufftDestroy(plan_inv);
        int n[1] = {cfg.frame_size};
        cufft_check(cufftPlanMany(&plan_fwd, 1, n, nullptr, 1, cfg.frame_size,
                                  nullptr, 1, bins, CUFFT_R2C, batch), "plan R2C");
        cufft_check(cufftPlanMany(&plan_inv, 1, n, nullptr, 1, bins,
                                  nullptr, 1, cfg.frame_size, CUFFT_C2R, batch), "plan C2R");
        plan_batch = batch;
    }
};

StftProcessor::StftProcessor(const StftConfig& cfg, int max_channels, int max_samples)
    : impl_(new Impl) {
    if (cfg.frame_size <= 0 || (cfg.frame_size & (cfg.frame_size - 1)) != 0)
        throw std::invalid_argument("frame_size must be a power of two");
    if (cfg.hop <= 0 || cfg.hop > cfg.frame_size)
        throw std::invalid_argument("hop must be in (0, frame_size]");
    if (max_channels <= 0 || max_samples <= 0)
        throw std::invalid_argument("max_channels and max_samples must be positive");

    impl_->cfg = cfg;
    impl_->max_channels = max_channels;
    impl_->max_samples = max_samples;
    impl_->bins = cfg.frame_size / 2 + 1;
    impl_->max_frames = (max_samples + cfg.hop - 1) / cfg.hop + 1;

    const size_t nf = (size_t)max_channels * impl_->max_frames;
    const size_t out_len = (size_t)max_channels * (max_samples + cfg.frame_size);

    CU_CHECK(cudaMalloc(&impl_->d_pcm, (size_t)max_channels * max_samples * sizeof(float)));
    CU_CHECK(cudaMalloc(&impl_->d_frames, nf * cfg.frame_size * sizeof(float)));
    CU_CHECK(cudaMalloc(&impl_->d_spec, nf * impl_->bins * sizeof(cufftComplex)));
    CU_CHECK(cudaMalloc(&impl_->d_spec2, nf * impl_->bins * sizeof(cufftComplex)));
    impl_->ensure_out(out_len);
    CU_CHECK(cudaMalloc(&impl_->d_resampled, out_len * sizeof(float)));
    CU_CHECK(cudaMalloc(&impl_->d_win, cfg.frame_size * sizeof(float)));
    CU_CHECK(cudaMalloc(&impl_->d_lastphase, (size_t)max_channels * impl_->bins * sizeof(float)));
    CU_CHECK(cudaMalloc(&impl_->d_sumphase, (size_t)max_channels * impl_->bins * sizeof(float)));

    auto win = hann_window(cfg.frame_size);
    CU_CHECK(cudaMemcpy(impl_->d_win, win.data(), win.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
}

StftProcessor::~StftProcessor() {
    if (!impl_) return;
    if (impl_->plan_fwd) cufftDestroy(impl_->plan_fwd);
    if (impl_->plan_inv) cufftDestroy(impl_->plan_inv);
    cudaFree(impl_->d_pcm);
    cudaFree(impl_->d_frames);
    cudaFree(impl_->d_spec);
    cudaFree(impl_->d_spec2);
    cudaFree(impl_->d_out);
    cudaFree(impl_->d_wsum);
    cudaFree(impl_->d_resampled);
    cudaFree(impl_->d_win);
    cudaFree(impl_->d_lastphase);
    cudaFree(impl_->d_sumphase);
    delete impl_;
}

int StftProcessor::bins() const { return impl_->bins; }
const StftConfig& StftProcessor::config() const { return impl_->cfg; }

int StftProcessor::frames_for(int samples) const {
    if (samples < impl_->cfg.frame_size) return 1;
    return (samples - impl_->cfg.frame_size) / impl_->cfg.hop + 1;
}

std::vector<float> StftProcessor::spectrogram(const std::vector<float>& pcm, int channels,
                                              int samples, float* elapsed_ms) {
    if (channels <= 0 || channels > impl_->max_channels)
        throw std::invalid_argument("channels out of range");
    if (samples <= 0 || samples > impl_->max_samples)
        throw std::invalid_argument("samples out of range");
    if (pcm.size() != (size_t)channels * samples)
        throw std::invalid_argument("pcm size must be channels*samples");

    const auto& c = impl_->cfg;
    const int nf = frames_for(samples);
    const int batch = channels * nf;
    impl_->make_plans(batch);

    CU_CHECK(cudaMemcpy(impl_->d_pcm, pcm.data(), pcm.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

    const int T = 256;
    const int total_frame = batch * c.frame_size;

    cu::EventTimer t;
    t.start();
    k_frame_and_window<<<(total_frame + T - 1) / T, T>>>(
        impl_->d_pcm, samples, impl_->d_frames, c.frame_size, c.hop, nf, channels,
        impl_->d_win);
    cufft_check(cufftExecR2C(impl_->plan_fwd, impl_->d_frames, impl_->d_spec), "exec R2C");

    const int total_bins = batch * impl_->bins;
    float* d_mag = reinterpret_cast<float*>(impl_->d_spec2);   // reuse
    k_magnitude<<<(total_bins + T - 1) / T, T>>>(impl_->d_spec, d_mag, total_bins);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> out((size_t)total_bins);
    CU_CHECK(cudaMemcpy(out.data(), d_mag, out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
    return out;
}

std::vector<float> StftProcessor::resynthesize(const std::vector<float>& pcm, int channels,
                                               int samples, float* elapsed_ms) {
    return pitch_shift(pcm, channels, samples, 1.0f, elapsed_ms);
}

std::vector<float> StftProcessor::pitch_shift(const std::vector<float>& pcm, int channels,
                                              int samples, float ratio, float* elapsed_ms) {
    if (channels <= 0 || channels > impl_->max_channels)
        throw std::invalid_argument("channels out of range");
    if (samples <= 0 || samples > impl_->max_samples)
        throw std::invalid_argument("samples out of range");
    if (pcm.size() != (size_t)channels * samples)
        throw std::invalid_argument("pcm size must be channels*samples");
    if (ratio <= 0.0f) throw std::invalid_argument("ratio must be positive");

    const auto& c = impl_->cfg;
    const int nf = frames_for(samples);
    const int batch = channels * nf;
    impl_->make_plans(batch);

    CU_CHECK(cudaMemcpy(impl_->d_pcm, pcm.data(), pcm.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

    const int T = 256;
    const int total_frame = batch * c.frame_size;
    const int hop_s = static_cast<int>(std::lround(c.hop * ratio));
    const int stretched_len = (nf - 1) * hop_s + c.frame_size;

    cu::EventTimer t;
    t.start();

    k_frame_and_window<<<(total_frame + T - 1) / T, T>>>(
        impl_->d_pcm, samples, impl_->d_frames, c.frame_size, c.hop, nf, channels,
        impl_->d_win);
    cufft_check(cufftExecR2C(impl_->plan_fwd, impl_->d_frames, impl_->d_spec), "exec R2C");

    CU_CHECK(cudaMemset(impl_->d_lastphase, 0,
                        (size_t)channels * impl_->bins * sizeof(float)));
    CU_CHECK(cudaMemset(impl_->d_sumphase, 0,
                        (size_t)channels * impl_->bins * sizeof(float)));

    // One thread per bin, walking frames sequentially: phase integration is a
    // recurrence along the time axis and cannot be parallelised across frames.
    dim3 blk(256), grd((impl_->bins + 255) / 256, channels);
    k_phase_vocoder<<<grd, blk>>>(impl_->d_spec, impl_->d_spec2, impl_->d_lastphase,
                                  impl_->d_sumphase, impl_->bins, nf, channels,
                                  c.hop, ratio);

    cufft_check(cufftExecC2R(impl_->plan_inv, impl_->d_spec2, impl_->d_frames), "exec C2R");
    // cuFFT is unnormalised: a forward+inverse round trip scales by N.
    k_scale<<<(total_frame + T - 1) / T, T>>>(impl_->d_frames, total_frame,
                                              1.0f / c.frame_size);

    const size_t out_total = (size_t)channels * stretched_len;
    impl_->ensure_out(out_total);
    CU_CHECK(cudaMemset(impl_->d_out, 0, out_total * sizeof(float)));
    CU_CHECK(cudaMemset(impl_->d_wsum, 0, out_total * sizeof(float)));
    k_overlap_add<<<(total_frame + T - 1) / T, T>>>(
        impl_->d_frames, c.frame_size, hop_s, nf, channels, stretched_len,
        impl_->d_win, impl_->d_out, impl_->d_wsum);
    k_normalize<<<(int)((out_total + T - 1) / T), T>>>(impl_->d_out, impl_->d_wsum,
                                                       (int)out_total);

    // Time-stretched by `ratio`; resampling by the same factor restores the
    // original duration and leaves the pitch shifted.
    const int total_rs = channels * samples;
    k_resample<<<(total_rs + T - 1) / T, T>>>(impl_->d_out, stretched_len,
                                              impl_->d_resampled, samples, channels, ratio);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> out((size_t)channels * samples);
    CU_CHECK(cudaMemcpy(out.data(), impl_->d_resampled, out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
    return out;
}

}  // namespace audio
