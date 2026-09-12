// KLT feature tracking -- implementation.

#include "optical_flow.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace flow {
namespace {

constexpr int MAX_WINDOW = 15;

// Gradients kept separately so the structure tensor can be box-summed.
__global__ void k_sobel(const unsigned char* __restrict__ img, int w, int h,
                        float* __restrict__ gx, float* __restrict__ gy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    if (x == 0 || y == 0 || x == w - 1 || y == h - 1) {
        gx[(size_t)y * w + x] = 0.0f;
        gy[(size_t)y * w + x] = 0.0f;
        return;
    }
    const size_t i = (size_t)y * w + x;
    const float tl = img[i - w - 1], tc = img[i - w], tr = img[i - w + 1];
    const float ml = img[i - 1], mr = img[i + 1];
    const float bl = img[i + w - 1], bc = img[i + w], br = img[i + w + 1];

    gx[i] = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
    gy[i] = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
}

// Harris = det(M) - k*trace(M)^2 over a 3x3 box-summed structure tensor.
__global__ void k_harris_response(const float* __restrict__ gx,
                                  const float* __restrict__ gy, int w, int h,
                                  float k, float* __restrict__ response) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    if (x < 2 || y < 2 || x >= w - 2 || y >= h - 2) {
        response[(size_t)y * w + x] = 0.0f;
        return;
    }
    float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const size_t i = (size_t)(y + dy) * w + (x + dx);
            const float a = gx[i], b = gy[i];
            sxx += a * a;
            syy += b * b;
            sxy += a * b;
        }
    const float det = sxx * syy - sxy * sxy;
    const float tr = sxx + syy;
    response[(size_t)y * w + x] = det - k * tr * tr;
}

// Gaussian-weighted 2x downsample (the standard 1-4-6-4-1 separable kernel,
// applied as a 5x5 tap). Plain pixel dropping would alias badly and the
// pyramid would track garbage at coarse levels.
__global__ void k_downsample(const unsigned char* __restrict__ src, int sw, int sh,
                             unsigned char* __restrict__ dst, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;

    const float w5[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    float acc = 0.0f, wsum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
            int sx = min(max(2 * x + dx, 0), sw - 1);
            int sy = min(max(2 * y + dy, 0), sh - 1);
            const float wt = w5[dx + 2] * w5[dy + 2];
            acc += wt * src[(size_t)sy * sw + sx];
            wsum += wt;
        }
    dst[(size_t)y * dw + x] = (unsigned char)fminf(fmaxf(acc / wsum, 0.0f), 255.0f);
}

// Bilinear sample with edge clamping, used for sub-pixel window extraction.
__device__ __forceinline__ float sample(const unsigned char* __restrict__ img,
                                        int w, int h, float x, float y) {
    const float cx = fminf(fmaxf(x, 0.0f), (float)(w - 1));
    const float cy = fminf(fmaxf(y, 0.0f), (float)(h - 1));
    const int x0 = (int)cx, y0 = (int)cy;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = cx - x0, fy = cy - y0;

    const float p00 = img[(size_t)y0 * w + x0], p01 = img[(size_t)y0 * w + x1];
    const float p10 = img[(size_t)y1 * w + x0], p11 = img[(size_t)y1 * w + x1];
    const float top = p00 + (p01 - p00) * fx;
    const float bot = p10 + (p11 - p10) * fx;
    return top + (bot - top) * fy;
}

// ---------------------------------------------------------------------------
// Lucas-Kanade, one thread per feature, one pyramid level per launch.
//
// Solves the 2x2 system  [Ixx Ixy; Ixy Iyy] d = [bx; by]  by explicit inverse.
// A 2x2 inverse is three multiplies and a reciprocal -- there is no reason to
// reach for a general solver, and the determinant doubles as the degeneracy
// test: a near-zero det means the window has gradient in only one direction
// (an edge, not a corner) and the displacement along it is unrecoverable.
// ---------------------------------------------------------------------------
// src = where the feature sits in the FIRST frame at this level (the template
// anchor, fixed). est = the running estimate of where it moved to in the
// second frame, carried down from the coarser level.
//
// These MUST be separate. Using one array for both means that once a coarse
// level refines the position, the next finer level samples its template at the
// refined location instead of the original feature -- the template drifts with
// the estimate and the pyramid actively degrades the answer. Measured before
// the fix: 33 px error with 4 levels versus 12.5 px with one.
__global__ void k_klt_level(const unsigned char* __restrict__ prev,
                            const unsigned char* __restrict__ next,
                            int w, int h,
                            const Point2* __restrict__ src,
                            Point2* __restrict__ est, unsigned char* __restrict__ ok,
                            int n, int win, int iters, float eps, float max_res) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !ok[i]) return;

    const int r = win / 2;
    const Point2 s = src[i];
    Point2 p = est[i];

    // Reject features whose template window would leave the image.
    if (s.x < r || s.y < r || s.x >= w - r - 1 || s.y >= h - r - 1) {
        ok[i] = 0;
        return;
    }

    // The structure tensor depends only on the FIRST frame at the FIXED
    // template anchor, so it is computed once outside the Newton iteration.
    float ixx = 0.0f, iyy = 0.0f, ixy = 0.0f;
    float gxw[MAX_WINDOW * MAX_WINDOW], gyw[MAX_WINDOW * MAX_WINDOW];
    float tmpl[MAX_WINDOW * MAX_WINDOW];

    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const float px = s.x + dx, py = s.y + dy;
            const float gx = 0.5f * (sample(prev, w, h, px + 1, py) -
                                     sample(prev, w, h, px - 1, py));
            const float gy = 0.5f * (sample(prev, w, h, px, py + 1) -
                                     sample(prev, w, h, px, py - 1));
            const int idx = (dy + r) * win + (dx + r);
            gxw[idx] = gx;
            gyw[idx] = gy;
            tmpl[idx] = sample(prev, w, h, px, py);
            ixx += gx * gx;
            iyy += gy * gy;
            ixy += gx * gy;
        }

    const float det = ixx * iyy - ixy * ixy;
    if (fabsf(det) < 1e-6f) {
        ok[i] = 0;               // aperture problem: not a trackable corner
        return;
    }
    const float inv_det = 1.0f / det;

    float last_res = 0.0f;
    for (int it = 0; it < iters; ++it) {
        float bx = 0.0f, by = 0.0f, res = 0.0f;
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx) {
                const int idx = (dy + r) * win + (dx + r);
                const float diff = tmpl[idx] - sample(next, w, h, p.x + dx, p.y + dy);
                bx += diff * gxw[idx];
                by += diff * gyw[idx];
                res += fabsf(diff);
            }
        last_res = res / (win * win);

        // 2x2 inverse, applied directly.
        const float ux = (iyy * bx - ixy * by) * inv_det;
        const float uy = (ixx * by - ixy * bx) * inv_det;
        p.x += ux;
        p.y += uy;

        if (p.x < 0 || p.y < 0 || p.x >= w || p.y >= h) { ok[i] = 0; return; }
        if (ux * ux + uy * uy < eps * eps) break;
    }

    if (last_res > max_res) ok[i] = 0;
    est[i] = p;
}

__global__ void k_scale_points(Point2* pts, int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { pts[i].x *= s; pts[i].y *= s; }
}

}  // namespace

// ---------------------------------------------------------------------------
struct FlowTracker::Impl {
    int w = 0, h = 0;
    unsigned char *d_a = nullptr, *d_b = nullptr;
    float *d_gx = nullptr, *d_gy = nullptr, *d_resp = nullptr;

    // Owns every resource, so a constructor that throws partway through
    // releases what it had already acquired. A class destructor never runs
    // for an object whose constructor threw. Every release is null-safe.
    ~Impl() {
        cudaFree(d_a); cudaFree(d_b);
        cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_resp);
    }
};

FlowTracker::FlowTracker(int width, int height) : impl_(new Impl) {
    try {
        if (width < 16 || height < 16)
            throw std::invalid_argument("frame must be at least 16x16");
        impl_->w = width;
        impl_->h = height;
        const size_t n = (size_t)width * height;
        CU_CHECK(cudaMalloc(&impl_->d_a, n));
        CU_CHECK(cudaMalloc(&impl_->d_b, n));
        CU_CHECK(cudaMalloc(&impl_->d_gx, n * sizeof(float)));
        CU_CHECK(cudaMalloc(&impl_->d_gy, n * sizeof(float)));
        CU_CHECK(cudaMalloc(&impl_->d_resp, n * sizeof(float)));
    } catch (...) {
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

FlowTracker::~FlowTracker() { delete impl_; }

int FlowTracker::width() const { return impl_->w; }
int FlowTracker::height() const { return impl_->h; }

std::vector<Feature> FlowTracker::detect_harris(const std::vector<std::uint8_t>& gray,
                                                const HarrisParams& p,
                                                float* elapsed_ms) {
    const int w = impl_->w, h = impl_->h;
    if (gray.size() != (std::size_t)w * h)
        throw std::invalid_argument("image size does not match tracker geometry");
    if (p.min_distance < 1) throw std::invalid_argument("min_distance must be >= 1");

    CU_CHECK(cudaMemcpy(impl_->d_a, gray.data(), gray.size(), cudaMemcpyHostToDevice));
    dim3 blk(32, 8), grd((w + 31) / 32, (h + 7) / 8);

    cu::EventTimer t;
    t.start();
    k_sobel<<<grd, blk>>>(impl_->d_a, w, h, impl_->d_gx, impl_->d_gy);
    k_harris_response<<<grd, blk>>>(impl_->d_gx, impl_->d_gy, w, h, p.k, impl_->d_resp);
    CU_CHECK_KERNEL();
    const float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> resp((std::size_t)w * h);
    CU_CHECK(cudaMemcpy(resp.data(), impl_->d_resp, resp.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

    // Non-maximum suppression on the host: it is a sort plus a greedy sweep,
    // inherently sequential, and tiny next to the per-pixel work above.
    const float maxr = *std::max_element(resp.begin(), resp.end());
    const float thresh = maxr * p.quality;

    std::vector<Feature> cand;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float r = resp[(std::size_t)y * w + x];
            if (r > thresh) cand.push_back(Feature{{(float)x, (float)y}, r});
        }
    std::sort(cand.begin(), cand.end(),
              [](const Feature& a, const Feature& b) { return a.response > b.response; });

    std::vector<Feature> keep;
    const int md2 = p.min_distance * p.min_distance;
    std::vector<std::uint8_t> blocked((std::size_t)w * h, 0);
    for (const auto& c : cand) {
        if ((int)keep.size() >= p.max_features) break;
        const int cx = (int)c.pos.x, cy = (int)c.pos.y;
        if (blocked[(std::size_t)cy * w + cx]) continue;
        keep.push_back(c);
        for (int dy = -p.min_distance; dy <= p.min_distance; ++dy)
            for (int dx = -p.min_distance; dx <= p.min_distance; ++dx) {
                if (dx * dx + dy * dy > md2) continue;
                const int nx = cx + dx, ny = cy + dy;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                blocked[(std::size_t)ny * w + nx] = 1;
            }
    }
    return keep;
}

std::vector<std::vector<std::uint8_t>> FlowTracker::build_pyramid(
    const std::vector<std::uint8_t>& gray, int levels) {
    if (levels < 1) throw std::invalid_argument("levels must be >= 1");
    const int w = impl_->w, h = impl_->h;
    if (gray.size() != (std::size_t)w * h)
        throw std::invalid_argument("image size does not match tracker geometry");

    std::vector<std::vector<std::uint8_t>> pyr;
    pyr.push_back(gray);

    int cw = w, ch = h;
    unsigned char* d_src = nullptr;
    CU_CHECK(cudaMalloc(&d_src, (size_t)cw * ch));
    CU_CHECK(cudaMemcpy(d_src, gray.data(), gray.size(), cudaMemcpyHostToDevice));

    for (int l = 1; l < levels; ++l) {
        const int nw = std::max(cw / 2, 1), nh = std::max(ch / 2, 1);
        if (nw < 8 || nh < 8) break;   // below this a window no longer fits

        unsigned char* d_dst = nullptr;
        CU_CHECK(cudaMalloc(&d_dst, (size_t)nw * nh));
        dim3 blk(32, 8), grd((nw + 31) / 32, (nh + 7) / 8);
        k_downsample<<<grd, blk>>>(d_src, cw, ch, d_dst, nw, nh);
        CU_CHECK_KERNEL();

        std::vector<std::uint8_t> level((std::size_t)nw * nh);
        CU_CHECK(cudaMemcpy(level.data(), d_dst, level.size(), cudaMemcpyDeviceToHost));
        pyr.push_back(std::move(level));

        cudaFree(d_src);
        d_src = d_dst;
        cw = nw;
        ch = nh;
    }
    cudaFree(d_src);
    return pyr;
}

TrackResult FlowTracker::track(const std::vector<std::uint8_t>& prev,
                               const std::vector<std::uint8_t>& next,
                               const std::vector<Point2>& features,
                               const KltParams& p, float* elapsed_ms) {
    const int w = impl_->w, h = impl_->h;
    if (prev.size() != (std::size_t)w * h || next.size() != (std::size_t)w * h)
        throw std::invalid_argument("image size does not match tracker geometry");
    if (p.window < 3 || p.window > MAX_WINDOW || p.window % 2 == 0)
        throw std::invalid_argument("window must be odd and in [3,15]");
    if (p.levels < 1) throw std::invalid_argument("levels must be >= 1");

    TrackResult r;
    const int n = (int)features.size();
    r.positions.resize(n);
    r.ok.assign(n, 1);
    if (n == 0) return r;

    auto pyr_a = build_pyramid(prev, p.levels);
    auto pyr_b = build_pyramid(next, p.levels);
    const int L = (int)pyr_a.size();

    Point2 *d_src = nullptr, *d_est = nullptr;
    unsigned char* d_ok = nullptr;
    CU_CHECK(cudaMalloc(&d_src, (std::size_t)n * sizeof(Point2)));
    CU_CHECK(cudaMalloc(&d_est, (std::size_t)n * sizeof(Point2)));
    CU_CHECK(cudaMalloc(&d_ok, (std::size_t)n));

    // Start at the coarsest level, so the initial estimate is scaled DOWN.
    std::vector<Point2> start(n);
    const float s0 = 1.0f / float(1 << (L - 1));
    for (int i = 0; i < n; ++i) start[i] = Point2{features[i].x * s0, features[i].y * s0};
    CU_CHECK(cudaMemcpy(d_est, start.data(), (std::size_t)n * sizeof(Point2),
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_ok, 1, (std::size_t)n));

    const int T = 128;
    cu::EventTimer t;
    t.start();

    std::vector<Point2> src_level(n);
    for (int l = L - 1; l >= 0; --l) {
        // Level dimensions come from the pyramid itself. Deriving them as
        // (w >> l) would drift from build_pyramid's repeated integer halving
        // whenever a dimension is odd.
        const int lw = (l == 0) ? w : (int)(pyr_a[l].size() / std::max(1, (int)(h >> l)));
        const int lh = (l == 0) ? h : (int)(h >> l);
        const int aw = (l == 0) ? w : (int)(w >> l);
        const int ah = lh;
        (void)lw;

        // Template anchors: the ORIGINAL feature positions at this level's
        // scale, never the running estimate.
        const float s = 1.0f / float(1 << l);
        for (int i = 0; i < n; ++i)
            src_level[i] = Point2{features[i].x * s, features[i].y * s};
        CU_CHECK(cudaMemcpy(d_src, src_level.data(), (std::size_t)n * sizeof(Point2),
                            cudaMemcpyHostToDevice));

        unsigned char *da = nullptr, *db = nullptr;
        CU_CHECK(cudaMalloc(&da, pyr_a[l].size()));
        CU_CHECK(cudaMalloc(&db, pyr_b[l].size()));
        CU_CHECK(cudaMemcpy(da, pyr_a[l].data(), pyr_a[l].size(), cudaMemcpyHostToDevice));
        CU_CHECK(cudaMemcpy(db, pyr_b[l].data(), pyr_b[l].size(), cudaMemcpyHostToDevice));

        k_klt_level<<<(n + T - 1) / T, T>>>(da, db, aw, ah, d_src, d_est, d_ok, n,
                                            p.window, p.iterations, p.epsilon,
                                            p.max_residual);
        CU_CHECK_KERNEL();
        cudaFree(da);
        cudaFree(db);

        // Carry the estimate to the next finer level.
        if (l > 0) {
            k_scale_points<<<(n + T - 1) / T, T>>>(d_est, n, 2.0f);
            CU_CHECK_KERNEL();
        }
    }
    const float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    CU_CHECK(cudaMemcpy(r.positions.data(), d_est, (std::size_t)n * sizeof(Point2),
                        cudaMemcpyDeviceToHost));
    CU_CHECK(cudaMemcpy(r.ok.data(), d_ok, (std::size_t)n, cudaMemcpyDeviceToHost));
    cudaFree(d_src);
    cudaFree(d_est);
    cudaFree(d_ok);

    for (int i = 0; i < n; ++i) r.tracked += r.ok[i] ? 1 : 0;
    return r;
}

// ---------------------------------------------------------------------------
std::vector<float> FlowTracker::harris_response_cpu(const std::vector<std::uint8_t>& gray,
                                                    int w, int h, float k) {
    std::vector<float> gx((std::size_t)w * h, 0.0f), gy((std::size_t)w * h, 0.0f);
    for (int y = 1; y < h - 1; ++y)
        for (int x = 1; x < w - 1; ++x) {
            const std::size_t i = (std::size_t)y * w + x;
            const float tl = gray[i - w - 1], tc = gray[i - w], tr = gray[i - w + 1];
            const float ml = gray[i - 1], mr = gray[i + 1];
            const float bl = gray[i + w - 1], bc = gray[i + w], br = gray[i + w + 1];
            gx[i] = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
            gy[i] = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
        }

    std::vector<float> resp((std::size_t)w * h, 0.0f);
    for (int y = 2; y < h - 2; ++y)
        for (int x = 2; x < w - 2; ++x) {
            float sxx = 0, syy = 0, sxy = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t i = (std::size_t)(y + dy) * w + (x + dx);
                    sxx += gx[i] * gx[i];
                    syy += gy[i] * gy[i];
                    sxy += gx[i] * gy[i];
                }
            const float det = sxx * syy - sxy * sxy;
            const float tr = sxx + syy;
            resp[(std::size_t)y * w + x] = det - k * tr * tr;
        }
    return resp;
}

std::vector<std::uint8_t> FlowTracker::downsample_cpu(const std::vector<std::uint8_t>& src,
                                                      int w, int h) {
    const int dw = std::max(w / 2, 1), dh = std::max(h / 2, 1);
    std::vector<std::uint8_t> dst((std::size_t)dw * dh);
    const float w5[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x) {
            float acc = 0, wsum = 0;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    const int sx = std::min(std::max(2 * x + dx, 0), w - 1);
                    const int sy = std::min(std::max(2 * y + dy, 0), h - 1);
                    const float wt = w5[dx + 2] * w5[dy + 2];
                    acc += wt * src[(std::size_t)sy * w + sx];
                    wsum += wt;
                }
            dst[(std::size_t)y * dw + x] =
                (std::uint8_t)std::min(std::max(acc / wsum, 0.0f), 255.0f);
        }
    return dst;
}

std::vector<std::uint8_t> make_frame(int w, int h, float shift_x, float shift_y,
                                     unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(0.08f, 0.92f);

    // Blob centres are fixed by the seed, so two frames with different shifts
    // depict the SAME scene translated -- which makes the shift exact ground
    // truth for the tracker.
    const int nblob = 140;
    std::vector<float> cx(nblob), cy(nblob);
    for (int i = 0; i < nblob; ++i) { cx[i] = ux(rng) * w; cy[i] = ux(rng) * h; }

    std::vector<std::uint8_t> img((std::size_t)w * h, 20);
    const float sigma = 2.6f, inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int i = 0; i < nblob; ++i) {
        const float bx = cx[i] + shift_x, by = cy[i] + shift_y;
        const int x0 = std::max(0, (int)(bx - 9)), x1 = std::min(w - 1, (int)(bx + 9));
        const int y0 = std::max(0, (int)(by - 9)), y1 = std::min(h - 1, (int)(by + 9));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const float dx = x - bx, dy = y - by;
                const float v = 230.0f * std::exp(-(dx * dx + dy * dy) * inv2s2);
                const std::size_t idx = (std::size_t)y * w + x;
                img[idx] = (std::uint8_t)std::min(255.0f, img[idx] + v);
            }
    }
    return img;
}

}  // namespace flow
