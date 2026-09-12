// Image preprocessing kernels + overlapped stream pipeline.

#include "image_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace vision {
namespace {

// Bilinear resize + normalise + HWC->CHW, all in one pass.
//
// Writing CHW is what makes this coalesced. One thread per OUTPUT pixel, with
// threadIdx.x running along the output row, means lane i writes element i of a
// contiguous plane -- a single 128-byte transaction per warp per channel.
// Emitting HWC instead would have consecutive lanes write 12 bytes apart and
// cost three times the transactions.
//
// The reads are inherently scattered (that is what resampling is), but they hit
// the texture-ish access pattern the L2 handles well, and the output side is
// where the bandwidth actually goes.
__global__ void k_resize_normalize_chw(const unsigned char* __restrict__ src,
                                       int src_w, int src_h,
                                       float* __restrict__ dst, int dst_w, int dst_h,
                                       float m0, float m1, float m2,
                                       float s0, float s1, float s2) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    // Half-pixel centres: aligning corners instead shifts the image by half a
    // pixel and quietly disagrees with every CPU reference implementation.
    const float sx = (x + 0.5f) * src_w / dst_w - 0.5f;
    const float sy = (y + 0.5f) * src_h / dst_h - 0.5f;

    const int x0 = max(0, min(src_w - 1, (int)floorf(sx)));
    const int y0 = max(0, min(src_h - 1, (int)floorf(sy)));
    const int x1 = min(src_w - 1, x0 + 1);
    const int y1 = min(src_h - 1, y0 + 1);
    const float fx = fminf(fmaxf(sx - x0, 0.0f), 1.0f);
    const float fy = fminf(fmaxf(sy - y0, 0.0f), 1.0f);

    const float mean[3] = {m0, m1, m2};
    const float istd[3] = {1.0f / s0, 1.0f / s1, 1.0f / s2};
    const size_t plane = (size_t)dst_w * dst_h;

#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float p00 = src[((size_t)y0 * src_w + x0) * 3 + c];
        const float p01 = src[((size_t)y0 * src_w + x1) * 3 + c];
        const float p10 = src[((size_t)y1 * src_w + x0) * 3 + c];
        const float p11 = src[((size_t)y1 * src_w + x1) * 3 + c];

        const float top = p00 + (p01 - p00) * fx;
        const float bot = p10 + (p11 - p10) * fx;
        const float v = (top + (bot - top) * fy) * (1.0f / 255.0f);

        dst[c * plane + (size_t)y * dst_w + x] = (v - mean[c]) * istd[c];
    }
}

}  // namespace

// ---------------------------------------------------------------------------
struct Preprocessor::Impl {
    PreprocessConfig cfg;
    int max_batch = 1;
    int num_streams = 4;

    std::vector<cudaStream_t> streams;
    // One staging slot per stream, so stream i can upload while stream j runs.
    std::vector<unsigned char*> d_src;
    std::vector<std::size_t> d_src_cap;

    float* d_dst = nullptr;
    float* h_pinned = nullptr;          // pinned output staging
    unsigned char* h_src_pinned = nullptr;
    std::size_t h_src_cap = 0;

    std::size_t out_floats = 0;

    void ensure_src(int slot, std::size_t bytes) {
        if (d_src_cap[slot] >= bytes) return;
        // Null the pointer and capacity BEFORE re-allocating. If cudaMalloc
        // fails, a stale pointer here would be freed a second time by ~Impl.
        cudaFree(d_src[slot]);
        d_src[slot] = nullptr;
        d_src_cap[slot] = 0;
        CU_CHECK(cudaMalloc(&d_src[slot], bytes));
        d_src_cap[slot] = bytes;
    }
    void ensure_host_src(std::size_t bytes) {
        if (h_src_cap >= bytes) return;
        if (h_src_pinned) cudaFreeHost(h_src_pinned);
        h_src_pinned = nullptr;   // see ensure_src
        h_src_cap = 0;
        // Pinned, because pageable memory forces the driver to stage through
        // an internal bounce buffer and roughly halves PCIe throughput.
        CU_CHECK(cudaHostAlloc(&h_src_pinned, bytes, cudaHostAllocDefault));
        h_src_cap = bytes;
    }

    // Owns every resource, so a constructor that throws partway through
    // releases what it had already acquired. A class destructor never runs
    // for an object whose constructor threw. Every release is null-safe.
    ~Impl() {
        // Streams are created one at a time; a constructor that fails partway
        // leaves the rest null. cudaStreamDestroy(nullptr) is an error, and
        // an error left unread is misreported by the next CU_CHECK_KERNEL.
        for (auto s : streams)
            if (s) cudaStreamDestroy(s);
        for (auto p : d_src) cudaFree(p);
        cudaFree(d_dst);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (h_src_pinned) cudaFreeHost(h_src_pinned);
    }
};

Preprocessor::Preprocessor(const PreprocessConfig& cfg, int max_batch, int num_streams)
    : impl_(new Impl) {
    try {
        if (max_batch <= 0) throw std::invalid_argument("max_batch must be positive");
        if (num_streams <= 0) throw std::invalid_argument("num_streams must be positive");
        if (cfg.out_w <= 0 || cfg.out_h <= 0)
            throw std::invalid_argument("output size must be positive");
        for (int c = 0; c < 3; ++c)
            if (cfg.stdev[c] == 0.0f) throw std::invalid_argument("stdev must be non-zero");

        impl_->cfg = cfg;
        impl_->max_batch = max_batch;
        impl_->num_streams = num_streams;
        impl_->out_floats = static_cast<std::size_t>(cfg.out_w) * cfg.out_h * 3;

        impl_->streams.resize(num_streams);
        impl_->d_src.assign(num_streams, nullptr);
        impl_->d_src_cap.assign(num_streams, 0);
        for (int i = 0; i < num_streams; ++i) CU_CHECK(cudaStreamCreate(&impl_->streams[i]));

        CU_CHECK(cudaMalloc(&impl_->d_dst, impl_->out_floats * max_batch * sizeof(float)));
        CU_CHECK(cudaHostAlloc(&impl_->h_pinned, impl_->out_floats * max_batch * sizeof(float),
                               cudaHostAllocDefault));
    } catch (...) {
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

Preprocessor::~Preprocessor() { delete impl_; }

std::size_t Preprocessor::output_floats_per_image() const { return impl_->out_floats; }
const PreprocessConfig& Preprocessor::config() const { return impl_->cfg; }

std::vector<float> Preprocessor::process(const std::vector<ImageRef>& batch,
                                         float* elapsed_ms) {
    const int n = static_cast<int>(batch.size());
    std::vector<float> out(static_cast<std::size_t>(n) * impl_->out_floats);
    if (n == 0) return out;
    if (n > impl_->max_batch) throw std::invalid_argument("batch exceeds max_batch");

    std::size_t biggest = 0;
    for (const auto& im : batch) biggest = std::max(biggest, im.bytes());
    impl_->ensure_host_src(biggest * impl_->num_streams);

    const auto& c = impl_->cfg;
    dim3 blk(32, 8);
    dim3 grd((c.out_w + blk.x - 1) / blk.x, (c.out_h + blk.y - 1) / blk.y);

    cu::EventTimer t;
    t.start();
    for (int i = 0; i < n; ++i) {
        const int s = i % impl_->num_streams;
        const auto& im = batch[i];
        impl_->ensure_src(s, im.bytes());

        // Stage into pinned memory so the copy is a true async DMA. Copying
        // from the caller's pageable buffer would serialise on a driver-side
        // bounce copy and destroy the overlap this function exists for.
        unsigned char* stage = impl_->h_src_pinned + static_cast<std::size_t>(s) * biggest;
        std::memcpy(stage, im.rgb, im.bytes());

        CU_CHECK(cudaMemcpyAsync(impl_->d_src[s], stage, im.bytes(),
                                 cudaMemcpyHostToDevice, impl_->streams[s]));
        k_resize_normalize_chw<<<grd, blk, 0, impl_->streams[s]>>>(
            impl_->d_src[s], im.width, im.height,
            impl_->d_dst + static_cast<std::size_t>(i) * impl_->out_floats,
            c.out_w, c.out_h, c.mean[0], c.mean[1], c.mean[2],
            c.stdev[0], c.stdev[1], c.stdev[2]);
        CU_CHECK(cudaMemcpyAsync(impl_->h_pinned + static_cast<std::size_t>(i) * impl_->out_floats,
                                 impl_->d_dst + static_cast<std::size_t>(i) * impl_->out_floats,
                                 impl_->out_floats * sizeof(float),
                                 cudaMemcpyDeviceToHost, impl_->streams[s]));
    }
    for (auto s : impl_->streams) CU_CHECK(cudaStreamSynchronize(s));
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    CU_CHECK(cudaGetLastError());
    std::memcpy(out.data(), impl_->h_pinned, out.size() * sizeof(float));
    return out;
}

std::vector<float> Preprocessor::process_serial(const std::vector<ImageRef>& batch,
                                                float* elapsed_ms) {
    const int n = static_cast<int>(batch.size());
    std::vector<float> out(static_cast<std::size_t>(n) * impl_->out_floats);
    if (n == 0) return out;
    if (n > impl_->max_batch) throw std::invalid_argument("batch exceeds max_batch");

    const auto& c = impl_->cfg;
    dim3 blk(32, 8);
    dim3 grd((c.out_w + blk.x - 1) / blk.x, (c.out_h + blk.y - 1) / blk.y);

    cu::EventTimer t;
    t.start();
    for (int i = 0; i < n; ++i) {
        const auto& im = batch[i];
        impl_->ensure_src(0, im.bytes());
        CU_CHECK(cudaMemcpy(impl_->d_src[0], im.rgb, im.bytes(), cudaMemcpyHostToDevice));
        k_resize_normalize_chw<<<grd, blk>>>(
            impl_->d_src[0], im.width, im.height,
            impl_->d_dst + static_cast<std::size_t>(i) * impl_->out_floats,
            c.out_w, c.out_h, c.mean[0], c.mean[1], c.mean[2],
            c.stdev[0], c.stdev[1], c.stdev[2]);
        CU_CHECK(cudaMemcpy(out.data() + static_cast<std::size_t>(i) * impl_->out_floats,
                            impl_->d_dst + static_cast<std::size_t>(i) * impl_->out_floats,
                            impl_->out_floats * sizeof(float), cudaMemcpyDeviceToHost));
    }
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;
    CU_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
std::vector<float> preprocess_cpu(const ImageRef& img, const PreprocessConfig& cfg) {
    std::vector<float> out(static_cast<std::size_t>(cfg.out_w) * cfg.out_h * 3);
    const std::size_t plane = static_cast<std::size_t>(cfg.out_w) * cfg.out_h;

    for (int y = 0; y < cfg.out_h; ++y) {
        for (int x = 0; x < cfg.out_w; ++x) {
            const float sx = (x + 0.5f) * img.width / cfg.out_w - 0.5f;
            const float sy = (y + 0.5f) * img.height / cfg.out_h - 0.5f;

            const int x0 = std::max(0, std::min(img.width - 1, (int)std::floor(sx)));
            const int y0 = std::max(0, std::min(img.height - 1, (int)std::floor(sy)));
            const int x1 = std::min(img.width - 1, x0 + 1);
            const int y1 = std::min(img.height - 1, y0 + 1);
            const float fx = std::min(std::max(sx - x0, 0.0f), 1.0f);
            const float fy = std::min(std::max(sy - y0, 0.0f), 1.0f);

            for (int c = 0; c < 3; ++c) {
                const float p00 = img.rgb[((std::size_t)y0 * img.width + x0) * 3 + c];
                const float p01 = img.rgb[((std::size_t)y0 * img.width + x1) * 3 + c];
                const float p10 = img.rgb[((std::size_t)y1 * img.width + x0) * 3 + c];
                const float p11 = img.rgb[((std::size_t)y1 * img.width + x1) * 3 + c];
                const float top = p00 + (p01 - p00) * fx;
                const float bot = p10 + (p11 - p10) * fx;
                const float v = (top + (bot - top) * fy) / 255.0f;
                out[c * plane + (std::size_t)y * cfg.out_w + x] =
                    (v - cfg.mean[c]) / cfg.stdev[c];
            }
        }
    }
    return out;
}

}  // namespace vision
