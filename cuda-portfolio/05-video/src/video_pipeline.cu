// Video analytics kernels.

#include "video_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace video {
namespace {

__device__ __forceinline__ unsigned char clamp8(float v) {
    return static_cast<unsigned char>(fminf(fmaxf(v, 0.0f), 255.0f));
}

// NV12 -> RGB, BT.601 limited range. The chroma plane is half resolution in
// both axes, so four luma samples share one UV pair.
__global__ void k_nv12_to_rgb(const unsigned char* __restrict__ nv12, int w, int h,
                              unsigned char* __restrict__ rgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const unsigned char* Y = nv12;
    const unsigned char* UV = nv12 + (size_t)w * h;

    float yy = Y[(size_t)y * w + x] - 16.0f;
    size_t uv = (size_t)(y / 2) * w + (x / 2) * 2;
    float u = UV[uv] - 128.0f;
    float v = UV[uv + 1] - 128.0f;

    float r = 1.164f * yy + 1.596f * v;
    float g = 1.164f * yy - 0.392f * u - 0.813f * v;
    float b = 1.164f * yy + 2.017f * u;

    size_t o = ((size_t)y * w + x) * 3;
    rgb[o + 0] = clamp8(r);
    rgb[o + 1] = clamp8(g);
    rgb[o + 2] = clamp8(b);
}

__global__ void k_extract_y(const unsigned char* __restrict__ nv12, int w, int h,
                            unsigned char* __restrict__ gray) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) gray[i] = nv12[i];
}

// Bilateral with EXPLICIT bounds checking. Every tap costs two clamps and the
// branch sits inside the innermost loop.
__global__ void k_bilateral_clamped(const unsigned char* __restrict__ src, int w, int h,
                                    unsigned char* __restrict__ dst, int radius,
                                    float inv2ss, float inv2sr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float centre = src[(size_t)y * w + x];
    float wsum = 0.0f, acc = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = min(max(x + dx, 0), w - 1);
            int sy = min(max(y + dy, 0), h - 1);
            float s = src[(size_t)sy * w + sx];
            float ds = float(dx * dx + dy * dy);
            float dr = (s - centre) * (s - centre);
            float weight = __expf(-ds * inv2ss - dr * inv2sr);
            acc += weight * s;
            wsum += weight;
        }
    dst[(size_t)y * w + x] = clamp8(acc / wsum);
}

// Same filter, reading through a TEXTURE OBJECT with cudaAddressModeClamp.
// The address unit clamps out-of-range coordinates in hardware, so the inner
// loop has no min/max and no branch at all -- the boundary handling is free.
__global__ void k_bilateral_texture(cudaTextureObject_t tex, int w, int h,
                                    unsigned char* __restrict__ dst, int radius,
                                    float inv2ss, float inv2sr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float centre = tex2D<unsigned char>(tex, x, y);
    float wsum = 0.0f, acc = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            float s = tex2D<unsigned char>(tex, x + dx, y + dy);   // clamped in HW
            float ds = float(dx * dx + dy * dy);
            float dr = (s - centre) * (s - centre);
            float weight = __expf(-ds * inv2ss - dr * inv2sr);
            acc += weight * s;
            wsum += weight;
        }
    dst[(size_t)y * w + x] = clamp8(acc / wsum);
}

__global__ void k_motion_history(const unsigned char* __restrict__ cur,
                                 unsigned char* __restrict__ prev,
                                 unsigned char* __restrict__ mhi,
                                 int n, int threshold, int decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = cur[i], p = prev[i];
    int moved = abs(c - p) > threshold;
    int v = mhi[i];
    mhi[i] = static_cast<unsigned char>(moved ? 255 : max(0, v - decay));
    prev[i] = static_cast<unsigned char>(c);
}

__global__ void k_sum_bytes(const unsigned char* __restrict__ src, int n,
                            unsigned long long* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    unsigned long long s = 0;
    for (; i < n; i += stride) s += src[i];

    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    __shared__ unsigned long long w[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) w[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        unsigned long long v = (lane < nw) ? w[lane] : 0ULL;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) atomicAdd(out, v);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
struct VideoPipeline::Impl {
    int w = 0, h = 0;
    unsigned char* d_nv12 = nullptr;
    unsigned char* d_rgb = nullptr;
    unsigned char* d_gray = nullptr;
    unsigned char* d_out = nullptr;
    unsigned char* d_prev = nullptr;
    unsigned char* d_mhi = nullptr;
    bool motion_started = false;

    // Pitched allocation + texture object, for the hardware-clamped path.
    unsigned char* d_pitched = nullptr;
    size_t pitch = 0;
    cudaTextureObject_t tex = 0;

    // Owns every resource, so a constructor that throws partway through
    // releases what it had already acquired. A class destructor never runs
    // for an object whose constructor threw. Every release is null-safe.
    ~Impl() {
        if (tex) cudaDestroyTextureObject(tex);
        cudaFree(d_nv12);
        cudaFree(d_rgb);
        cudaFree(d_gray);
        cudaFree(d_out);
        cudaFree(d_prev);
        cudaFree(d_mhi);
        cudaFree(d_pitched);
    }
};

VideoPipeline::VideoPipeline(int width, int height) : impl_(new Impl) {
    try {
        if (width <= 0 || height <= 0 || (width % 2) || (height % 2))
            throw std::invalid_argument("frame size must be positive and even (NV12 chroma)");
        impl_->w = width;
        impl_->h = height;
        const size_t n = (size_t)width * height;

        CU_CHECK(cudaMalloc(&impl_->d_nv12, Nv12Frame::bytes(width, height)));
        CU_CHECK(cudaMalloc(&impl_->d_rgb, n * 3));
        CU_CHECK(cudaMalloc(&impl_->d_gray, n));
        CU_CHECK(cudaMalloc(&impl_->d_out, n));
        CU_CHECK(cudaMalloc(&impl_->d_prev, n));
        CU_CHECK(cudaMalloc(&impl_->d_mhi, n));
        CU_CHECK(cudaMemset(impl_->d_prev, 0, n));
        CU_CHECK(cudaMemset(impl_->d_mhi, 0, n));

        CU_CHECK(cudaMallocPitch(&impl_->d_pitched, &impl_->pitch, width, height));

        cudaResourceDesc res{};
        res.resType = cudaResourceTypePitch2D;
        res.res.pitch2D.devPtr = impl_->d_pitched;
        res.res.pitch2D.width = width;
        res.res.pitch2D.height = height;
        res.res.pitch2D.pitchInBytes = impl_->pitch;
        res.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();

        cudaTextureDesc td{};
        td.addressMode[0] = cudaAddressModeClamp;   // free boundary handling
        td.addressMode[1] = cudaAddressModeClamp;
        td.filterMode = cudaFilterModePoint;        // exact texels, no interpolation
        td.readMode = cudaReadModeElementType;
        td.normalizedCoords = 0;

        CU_CHECK(cudaCreateTextureObject(&impl_->tex, &res, &td, nullptr));
    } catch (...) {
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

VideoPipeline::~VideoPipeline() { delete impl_; }

int VideoPipeline::width() const { return impl_->w; }
int VideoPipeline::height() const { return impl_->h; }
void VideoPipeline::reset_motion() {
    const size_t n = (size_t)impl_->w * impl_->h;
    CU_CHECK(cudaMemset(impl_->d_prev, 0, n));
    CU_CHECK(cudaMemset(impl_->d_mhi, 0, n));
    impl_->motion_started = false;
}

std::vector<std::uint8_t> VideoPipeline::nv12_to_rgb(const Nv12Frame& f, float* elapsed_ms) {
    if (f.width != impl_->w || f.height != impl_->h)
        throw std::invalid_argument("frame size does not match pipeline");
    const size_t n = (size_t)impl_->w * impl_->h;

    CU_CHECK(cudaMemcpy(impl_->d_nv12, f.data, Nv12Frame::bytes(f.width, f.height),
                        cudaMemcpyHostToDevice));
    dim3 blk(32, 8), grd((impl_->w + 31) / 32, (impl_->h + 7) / 8);

    cu::EventTimer t;
    t.start();
    k_nv12_to_rgb<<<grd, blk>>>(impl_->d_nv12, impl_->w, impl_->h, impl_->d_rgb);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<std::uint8_t> out(n * 3);
    CU_CHECK(cudaMemcpy(out.data(), impl_->d_rgb, n * 3, cudaMemcpyDeviceToHost));
    return out;
}

std::vector<std::uint8_t> VideoPipeline::nv12_to_gray(const Nv12Frame& f, float* elapsed_ms) {
    if (f.width != impl_->w || f.height != impl_->h)
        throw std::invalid_argument("frame size does not match pipeline");
    const size_t n = (size_t)impl_->w * impl_->h;

    CU_CHECK(cudaMemcpy(impl_->d_nv12, f.data, Nv12Frame::bytes(f.width, f.height),
                        cudaMemcpyHostToDevice));
    cu::EventTimer t;
    t.start();
    k_extract_y<<<(int)((n + 255) / 256), 256>>>(impl_->d_nv12, impl_->w, impl_->h,
                                                 impl_->d_gray);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<std::uint8_t> out(n);
    CU_CHECK(cudaMemcpy(out.data(), impl_->d_gray, n, cudaMemcpyDeviceToHost));
    return out;
}

std::vector<std::uint8_t> VideoPipeline::bilateral(const std::vector<std::uint8_t>& gray,
                                                   const BilateralParams& p,
                                                   bool use_texture, float* elapsed_ms) {
    const size_t n = (size_t)impl_->w * impl_->h;
    if (gray.size() != n) throw std::invalid_argument("gray size does not match frame");
    if (p.radius < 0 || p.radius > 15) throw std::invalid_argument("radius must be 0..15");
    if (p.sigma_spatial <= 0.0f || p.sigma_range <= 0.0f)
        throw std::invalid_argument("sigmas must be positive");

    const float inv2ss = 1.0f / (2.0f * p.sigma_spatial * p.sigma_spatial);
    const float inv2sr = 1.0f / (2.0f * p.sigma_range * p.sigma_range);
    dim3 blk(32, 8), grd((impl_->w + 31) / 32, (impl_->h + 7) / 8);

    cu::EventTimer t;
    if (use_texture) {
        CU_CHECK(cudaMemcpy2D(impl_->d_pitched, impl_->pitch, gray.data(), impl_->w,
                              impl_->w, impl_->h, cudaMemcpyHostToDevice));
        t.start();
        k_bilateral_texture<<<grd, blk>>>(impl_->tex, impl_->w, impl_->h, impl_->d_out,
                                          p.radius, inv2ss, inv2sr);
    } else {
        CU_CHECK(cudaMemcpy(impl_->d_gray, gray.data(), n, cudaMemcpyHostToDevice));
        t.start();
        k_bilateral_clamped<<<grd, blk>>>(impl_->d_gray, impl_->w, impl_->h, impl_->d_out,
                                          p.radius, inv2ss, inv2sr);
    }
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<std::uint8_t> out(n);
    CU_CHECK(cudaMemcpy(out.data(), impl_->d_out, n, cudaMemcpyDeviceToHost));
    return out;
}

std::vector<std::uint8_t> VideoPipeline::motion_history(const std::vector<std::uint8_t>& gray,
                                                        int threshold, int decay,
                                                        float* elapsed_ms) {
    const size_t n = (size_t)impl_->w * impl_->h;
    if (gray.size() != n) throw std::invalid_argument("gray size does not match frame");

    CU_CHECK(cudaMemcpy(impl_->d_gray, gray.data(), n, cudaMemcpyHostToDevice));
    if (!impl_->motion_started) {
        // The first frame has no predecessor; seed prev with it so the whole
        // image does not read as motion.
        CU_CHECK(cudaMemcpy(impl_->d_prev, impl_->d_gray, n, cudaMemcpyDeviceToDevice));
        impl_->motion_started = true;
    }

    cu::EventTimer t;
    t.start();
    k_motion_history<<<(int)((n + 255) / 256), 256>>>(impl_->d_gray, impl_->d_prev,
                                                      impl_->d_mhi, (int)n, threshold, decay);
    CU_CHECK_KERNEL();
    float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<std::uint8_t> out(n);
    CU_CHECK(cudaMemcpy(out.data(), impl_->d_mhi, n, cudaMemcpyDeviceToHost));
    return out;
}

// ---------------------------------------------------------------------------
std::vector<std::uint8_t> nv12_to_rgb_cpu(const Nv12Frame& f) {
    const int w = f.width, h = f.height;
    std::vector<std::uint8_t> out((size_t)w * h * 3);
    const std::uint8_t* Y = f.data;
    const std::uint8_t* UV = f.data + (size_t)w * h;

    auto clamp = [](float v) {
        return static_cast<std::uint8_t>(std::min(std::max(v, 0.0f), 255.0f));
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float yy = Y[(size_t)y * w + x] - 16.0f;
            size_t uv = (size_t)(y / 2) * w + (x / 2) * 2;
            float u = UV[uv] - 128.0f;
            float v = UV[uv + 1] - 128.0f;
            size_t o = ((size_t)y * w + x) * 3;
            out[o + 0] = clamp(1.164f * yy + 1.596f * v);
            out[o + 1] = clamp(1.164f * yy - 0.392f * u - 0.813f * v);
            out[o + 2] = clamp(1.164f * yy + 2.017f * u);
        }
    return out;
}

std::vector<std::uint8_t> bilateral_cpu(const std::vector<std::uint8_t>& gray, int w, int h,
                                        const BilateralParams& p) {
    std::vector<std::uint8_t> out((size_t)w * h);
    const float inv2ss = 1.0f / (2.0f * p.sigma_spatial * p.sigma_spatial);
    const float inv2sr = 1.0f / (2.0f * p.sigma_range * p.sigma_range);

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float centre = gray[(size_t)y * w + x];
            float wsum = 0.0f, acc = 0.0f;
            for (int dy = -p.radius; dy <= p.radius; ++dy)
                for (int dx = -p.radius; dx <= p.radius; ++dx) {
                    int sx = std::min(std::max(x + dx, 0), w - 1);
                    int sy = std::min(std::max(y + dy, 0), h - 1);
                    float s = gray[(size_t)sy * w + sx];
                    float ds = float(dx * dx + dy * dy);
                    float dr = (s - centre) * (s - centre);
                    float weight = std::exp(-ds * inv2ss - dr * inv2sr);
                    acc += weight * s;
                    wsum += weight;
                }
            out[(size_t)y * w + x] =
                static_cast<std::uint8_t>(std::min(std::max(acc / wsum, 0.0f), 255.0f));
        }
    return out;
}

std::vector<std::uint8_t> make_test_nv12(int w, int h, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> f(Nv12Frame::bytes(w, h));
    // Smooth gradient plus a few bright blocks, so the bilateral filter has
    // both flat regions and real edges to preserve.
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int v = 16 + (x * 180) / std::max(1, w);
            if (((x / 64) + (y / 64)) % 2 == 0) v = std::min(235, v + 40);
            f[(size_t)y * w + x] = static_cast<std::uint8_t>(v);
        }
    std::uint8_t* UV = f.data() + (size_t)w * h;
    for (int y = 0; y < h / 2; ++y)
        for (int x = 0; x < w / 2; ++x) {
            UV[(size_t)y * w + x * 2 + 0] = static_cast<std::uint8_t>(100 + (rng() % 40));
            UV[(size_t)y * w + x * 2 + 1] = static_cast<std::uint8_t>(120 + (rng() % 40));
        }
    return f;
}

// ---------------------------------------------------------------------------
TransferReport measure_transfers(std::size_t bytes) {
    TransferReport r;
    std::vector<std::uint8_t> pageable(bytes, 7);
    std::uint8_t* pinned = nullptr;
    std::uint8_t* mapped = nullptr;
    std::uint8_t* mapped_dev = nullptr;
    void* dbuf = nullptr;
    unsigned long long* dsum = nullptr;

    CU_CHECK(cudaHostAlloc(&pinned, bytes, cudaHostAllocDefault));
    CU_CHECK(cudaHostAlloc(&mapped, bytes, cudaHostAllocMapped));
    CU_CHECK(cudaHostGetDevicePointer((void**)&mapped_dev, mapped, 0));
    CU_CHECK(cudaMalloc(&dbuf, bytes));
    CU_CHECK(cudaMalloc(&dsum, sizeof(unsigned long long)));
    std::fill(pinned, pinned + bytes, 7);
    std::fill(mapped, mapped + bytes, 7);

    const int REP = 20;
    cu::EventTimer t;

    CU_CHECK(cudaMemcpy(dbuf, pageable.data(), bytes, cudaMemcpyHostToDevice));  // warm
    t.start();
    for (int i = 0; i < REP; ++i)
        CU_CHECK(cudaMemcpy(dbuf, pageable.data(), bytes, cudaMemcpyHostToDevice));
    float ms = t.stop() / REP;
    r.pageable_gbps = bytes / (ms / 1e3) / 1e9;

    t.start();
    for (int i = 0; i < REP; ++i)
        CU_CHECK(cudaMemcpy(dbuf, pinned, bytes, cudaMemcpyHostToDevice));
    ms = t.stop() / REP;
    r.pinned_gbps = bytes / (ms / 1e3) / 1e9;

    const int T = 256, B = 1024;
    const int n = static_cast<int>(bytes);
    // Warm BOTH kernels before timing either: a cold first launch carries
    // one-time setup and makes whichever ran second look artificially fast.
    k_sum_bytes<<<B, T>>>(static_cast<unsigned char*>(dbuf), n, dsum);
    k_sum_bytes<<<B, T>>>(mapped_dev, n, dsum);
    CU_CHECK(cudaDeviceSynchronize());

    t.start();
    for (int i = 0; i < REP; ++i) k_sum_bytes<<<B, T>>>(static_cast<unsigned char*>(dbuf), n, dsum);
    r.device_kernel_ms = t.stop() / REP;

    t.start();
    for (int i = 0; i < REP; ++i) {
        CU_CHECK(cudaMemcpy(dbuf, pinned, bytes, cudaMemcpyHostToDevice));
        k_sum_bytes<<<B, T>>>(static_cast<unsigned char*>(dbuf), n, dsum);
    }
    r.upload_plus_kernel_ms = t.stop() / REP;

    t.start();
    for (int i = 0; i < REP; ++i) k_sum_bytes<<<B, T>>>(mapped_dev, n, dsum);
    r.zero_copy_kernel_ms = t.stop() / REP;

    CU_CHECK(cudaGetLastError());
    cudaFreeHost(pinned);
    cudaFreeHost(mapped);
    cudaFree(dbuf);
    cudaFree(dsum);
    return r;
}

bool cuda_ipc_available() {
    void* p = nullptr;
    if (cudaMalloc(&p, 1024) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaIpcMemHandle_t h;
    cudaError_t e = cudaIpcGetMemHandle(&h, p);
    cudaFree(p);
    // Not sticky -- the context is fine -- but a failed call leaves its error
    // recorded, and the next CU_CHECK_KERNEL would report it as a kernel
    // failure. This probe is expected to fail on platforms without IPC, so it
    // must consume what it leaves behind.
    cudaGetLastError();
    return e == cudaSuccess;
}

}  // namespace video
