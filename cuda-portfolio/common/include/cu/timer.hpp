#pragma once
//
// GPU timing. Host clocks measure the wrong thing because kernel launches are
// asynchronous -- std::chrono around a launch times the enqueue, not the work.
// CUDA events are recorded in the stream and timed by the device itself.
//
#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <vector>

#include "cu/check.hpp"

namespace cu {

class EventTimer {
public:
    EventTimer() {
        CU_CHECK(cudaEventCreate(&start_));
        CU_CHECK(cudaEventCreate(&stop_));
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void start(cudaStream_t s = nullptr) { CU_CHECK(cudaEventRecord(start_, s)); }
    float stop(cudaStream_t s = nullptr) {
        CU_CHECK(cudaEventRecord(stop_, s));
        CU_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CU_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{}, stop_{};
};

struct BenchResult {
    float mean_ms;
    float min_ms;
    float median_ms;
};

// Warm-up then repeat. The warm-up is not optional: the first launch of any
// kernel pays one-time setup, and timing that against an already-resident
// kernel makes whichever ran second look artificially fast.
inline BenchResult benchmark(const std::function<void()>& body,
                             int iterations = 20,
                             int warmup = 3) {
    for (int i = 0; i < warmup; ++i) body();
    CU_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(iterations);
    EventTimer t;
    for (int i = 0; i < iterations; ++i) {
        t.start();
        body();
        samples.push_back(t.stop());
    }
    std::sort(samples.begin(), samples.end());

    float sum = 0.0f;
    for (float s : samples) sum += s;
    BenchResult r;
    r.mean_ms = sum / static_cast<float>(samples.size());
    r.min_ms = samples.front();
    r.median_ms = samples[samples.size() / 2];
    return r;
}

}  // namespace cu
