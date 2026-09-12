// Sparse matrix-vector multiply -- implementation.

#include "spmv.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#include "cu/check.hpp"
#include "cu/timer.hpp"

namespace spmv {
namespace {

using i32 = std::int32_t;

__device__ __forceinline__ float warp_reduce(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// CSR, one thread per row.
//
// Simple and compact, and it falls apart on degree skew: a warp finishes no
// sooner than its longest row, so one 10000-entry row stalls 31 idle lanes.
// Reads of A are also uncoalesced -- lane i walks row i, and neighbouring rows
// live far apart in the value array.
// ---------------------------------------------------------------------------
__global__ void k_csr_scalar(const i32* __restrict__ row_ptr,
                             const i32* __restrict__ col_idx,
                             const float* __restrict__ vals,
                             const float* __restrict__ x,
                             float* __restrict__ y, i32 n) {
    const i32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float sum = 0.0f;
    const i32 begin = row_ptr[r], end = row_ptr[r + 1];
    for (i32 e = begin; e < end; ++e) sum += vals[e] * __ldg(&x[col_idx[e]]);
    y[r] = sum;
}

// CSR, one warp per row. Lanes stride along a single row, so reads of A and
// col_idx are contiguous. Costs a shuffle reduction and wastes lanes on short
// rows, which is exactly the trade the benchmark measures.
__global__ void k_csr_vector(const i32* __restrict__ row_ptr,
                             const i32* __restrict__ col_idx,
                             const float* __restrict__ vals,
                             const float* __restrict__ x,
                             float* __restrict__ y, i32 n) {
    const i32 warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= n) return;

    const i32 begin = row_ptr[warp], end = row_ptr[warp + 1];
    float sum = 0.0f;
    for (i32 e = begin + lane; e < end; e += 32) sum += vals[e] * __ldg(&x[col_idx[e]]);
    sum = warp_reduce(sum);
    if (lane == 0) y[warp] = sum;
}

// ---------------------------------------------------------------------------
// ELLPACK: every row padded to `width`, stored COLUMN-MAJOR.
//
// The column-major layout is the entire point. At step j, thread r reads
// element [j*n + r] -- so consecutive threads read consecutive addresses and
// the warp issues one clean transaction. A row-major ELL would stride by
// `width` per thread and lose that immediately.
//
// The cost is padding: a single long row inflates `width` for every row.
// ---------------------------------------------------------------------------
__global__ void k_ellpack(const i32* __restrict__ col_idx,
                          const float* __restrict__ vals,
                          const float* __restrict__ x,
                          float* __restrict__ y, i32 n, int width) {
    const i32 r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    float sum = 0.0f;
    for (int j = 0; j < width; ++j) {
        const size_t k = (size_t)j * n + r;
        const i32 c = col_idx[k];
        if (c >= 0) sum += vals[k] * __ldg(&x[c]);
    }
    y[r] = sum;
}

// Hybrid: ELL part first, then the CSR overflow added on top.
__global__ void k_hybrid_overflow(const i32* __restrict__ row_ptr,
                                  const i32* __restrict__ col_idx,
                                  const float* __restrict__ vals,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, i32 n_over,
                                  const i32* __restrict__ over_rows) {
    const i32 idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (idx >= n_over) return;

    const i32 r = over_rows[idx];
    const i32 begin = row_ptr[idx], end = row_ptr[idx + 1];
    float sum = 0.0f;
    for (i32 e = begin + lane; e < end; e += 32) sum += vals[e] * __ldg(&x[col_idx[e]]);
    sum = warp_reduce(sum);
    // atomicAdd because the ELL kernel already wrote y[r]; only the few
    // overflow rows touch it, so contention is negligible.
    if (lane == 0 && sum != 0.0f) atomicAdd(&y[r], sum);
}

__global__ void k_axpy(float* __restrict__ y, const float* __restrict__ x,
                       float a, i32 n) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_xpay(float* __restrict__ y, const float* __restrict__ x,
                       float a, i32 n) {
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + a * y[i];
}

__global__ void k_dot(const float* __restrict__ a, const float* __restrict__ b,
                      i32 n, double* __restrict__ out) {
    i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 stride = gridDim.x * blockDim.x;
    float s = 0.0f;
    for (; i < n; i += stride) s += a[i] * b[i];

    s = warp_reduce(s);
    __shared__ float w[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) w[wid] = s;
    __syncthreads();
    if (wid == 0) {
        const int nw = (blockDim.x + 31) / 32;
        float v = (lane < nw) ? w[lane] : 0.0f;
        v = warp_reduce(v);
        // Accumulate in double: CG's convergence test compares residuals that
        // shrink by orders of magnitude, and FP32 accumulation over millions
        // of terms loses exactly the digits the test depends on.
        if (lane == 0) atomicAdd(out, (double)v);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
CooMatrix CooMatrix::banded(i32 n, int bandwidth, unsigned seed) {
    (void)seed;
    CooMatrix m;
    m.rows = n;
    m.cols = n;
    const int half = bandwidth / 2;
    for (i32 r = 0; r < n; ++r)
        for (int d = -half; d <= half; ++d) {
            const i32 c = r + d;
            if (c < 0 || c >= n) continue;
            m.row_idx.push_back(r);
            m.col_idx.push_back(c);
            m.values.push_back(d == 0 ? 2.0f : -0.5f);
        }
    return m;
}

CooMatrix CooMatrix::power_law(i32 n, int avg_nnz_per_row, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    CooMatrix m;
    m.rows = n;
    m.cols = n;

    for (i32 r = 0; r < n; ++r) {
        // A heavy tail: most rows short, a handful enormous.
        int deg = 1 + (int)(u01(rng) * avg_nnz_per_row);
        if (u01(rng) < 0.001f) deg = avg_nnz_per_row * 200;
        deg = std::min<int>(deg, n);

        for (int k = 0; k < deg; ++k) {
            m.row_idx.push_back(r);
            m.col_idx.push_back((i32)(rng() % (unsigned)n));
            m.values.push_back(u01(rng) * 2.0f - 1.0f);
        }
    }
    return m;
}

CooMatrix CooMatrix::laplacian_2d(i32 w, i32 h) {
    CooMatrix m;
    const i32 n = w * h;
    m.rows = n;
    m.cols = n;
    for (i32 y = 0; y < h; ++y)
        for (i32 x = 0; x < w; ++x) {
            const i32 r = y * w + x;
            m.row_idx.push_back(r); m.col_idx.push_back(r); m.values.push_back(4.0f);
            if (x > 0)     { m.row_idx.push_back(r); m.col_idx.push_back(r - 1); m.values.push_back(-1.0f); }
            if (x < w - 1) { m.row_idx.push_back(r); m.col_idx.push_back(r + 1); m.values.push_back(-1.0f); }
            if (y > 0)     { m.row_idx.push_back(r); m.col_idx.push_back(r - w); m.values.push_back(-1.0f); }
            if (y < h - 1) { m.row_idx.push_back(r); m.col_idx.push_back(r + w); m.values.push_back(-1.0f); }
        }
    return m;
}

// ---------------------------------------------------------------------------
struct SparseMatrix::Impl {
    i32 n = 0, cols = 0;
    std::int64_t nnz = 0;
    int max_row = 0;
    double mean_row = 0.0;

    // CSR
    i32 *d_row_ptr = nullptr, *d_col = nullptr;
    float* d_val = nullptr;

    // ELLPACK (column major, width = max_row)
    int ell_width = 0;
    i32* d_ell_col = nullptr;
    float* d_ell_val = nullptr;

    // Hybrid
    int hyb_width = 0;
    i32* d_hyb_col = nullptr;
    float* d_hyb_val = nullptr;
    i32 n_over = 0;
    i32 *d_over_rows = nullptr, *d_over_ptr = nullptr, *d_over_col = nullptr;
    float* d_over_val = nullptr;

    std::size_t bytes_csr = 0, bytes_ell = 0, bytes_hyb = 0;
    std::int64_t ell_padding = 0, hyb_padding = 0;

    // Owns every resource, so a constructor that throws partway through
    // releases what it had already acquired. A class destructor never runs
    // for an object whose constructor threw. Every release is null-safe.
    ~Impl() {
        cudaFree(d_row_ptr); cudaFree(d_col); cudaFree(d_val);
        cudaFree(d_ell_col); cudaFree(d_ell_val);
        cudaFree(d_hyb_col); cudaFree(d_hyb_val);
        cudaFree(d_over_rows); cudaFree(d_over_ptr);
        cudaFree(d_over_col); cudaFree(d_over_val);
    }
};

SparseMatrix::SparseMatrix(const CooMatrix& coo) : impl_(new Impl) {
    try {
        if (coo.rows <= 0 || coo.cols <= 0) throw std::invalid_argument("empty matrix");
        if (coo.row_idx.size() != coo.values.size() || coo.col_idx.size() != coo.values.size())
            throw std::invalid_argument("COO arrays must have equal length");

        impl_->n = coo.rows;
        impl_->cols = coo.cols;
        impl_->nnz = coo.nnz();
        const i32 n = coo.rows;

        // --- COO -> CSR by counting sort ---
        std::vector<i32> count(n, 0);
        for (auto r : coo.row_idx) {
            if (r < 0 || r >= n) throw std::invalid_argument("row index out of range");
            count[r]++;
        }
        std::vector<i32> row_ptr(n + 1, 0);
        for (i32 r = 0; r < n; ++r) row_ptr[r + 1] = row_ptr[r] + count[r];

        std::vector<i32> col((std::size_t)impl_->nnz);
        std::vector<float> val((std::size_t)impl_->nnz);
        std::vector<i32> cursor(row_ptr.begin(), row_ptr.end() - 1);
        for (std::size_t e = 0; e < coo.values.size(); ++e) {
            if (coo.col_idx[e] < 0 || coo.col_idx[e] >= coo.cols)
                throw std::invalid_argument("column index out of range");
            const i32 pos = cursor[coo.row_idx[e]]++;
            col[pos] = coo.col_idx[e];
            val[pos] = coo.values[e];
        }

        impl_->max_row = 0;
        for (i32 r = 0; r < n; ++r) impl_->max_row = std::max(impl_->max_row, count[r]);
        impl_->mean_row = double(impl_->nnz) / n;

        auto up_i = [](i32** d, const std::vector<i32>& h) {
            CU_CHECK(cudaMalloc(d, std::max<std::size_t>(h.size(), 1) * sizeof(i32)));
            if (!h.empty())
                CU_CHECK(cudaMemcpy(*d, h.data(), h.size() * sizeof(i32), cudaMemcpyHostToDevice));
        };
        auto up_f = [](float** d, const std::vector<float>& h) {
            CU_CHECK(cudaMalloc(d, std::max<std::size_t>(h.size(), 1) * sizeof(float)));
            if (!h.empty())
                CU_CHECK(cudaMemcpy(*d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        };
        up_i(&impl_->d_row_ptr, row_ptr);
        up_i(&impl_->d_col, col);
        up_f(&impl_->d_val, val);
        impl_->bytes_csr = row_ptr.size() * sizeof(i32) + col.size() * sizeof(i32) +
                           val.size() * sizeof(float);

        // --- ELLPACK, column major, padded to the longest row ---
        impl_->ell_width = impl_->max_row;
        const std::size_t ell_n = (std::size_t)impl_->ell_width * n;
        {
            std::vector<i32> ecol(ell_n, -1);
            std::vector<float> eval(ell_n, 0.0f);
            for (i32 r = 0; r < n; ++r) {
                const i32 begin = row_ptr[r], end = row_ptr[r + 1];
                for (i32 e = begin; e < end; ++e) {
                    const std::size_t j = e - begin;
                    ecol[j * n + r] = col[e];
                    eval[j * n + r] = val[e];
                }
            }
            up_i(&impl_->d_ell_col, ecol);
            up_f(&impl_->d_ell_val, eval);
            impl_->bytes_ell = ell_n * (sizeof(i32) + sizeof(float));
            impl_->ell_padding = (std::int64_t)ell_n - impl_->nnz;
        }

        // --- Hybrid: cut at a width that covers most rows ---
        // The cut is the mean plus a margin. Chasing the exact optimum is not the
        // point; covering the bulk of rows while leaving the heavy tail to CSR is.
        {
            std::vector<i32> sorted(count);
            std::sort(sorted.begin(), sorted.end());
            const int p90 = sorted[(std::size_t)(n * 0.90)];
            impl_->hyb_width = std::max(1, std::min(p90, impl_->max_row));

            const std::size_t hn = (std::size_t)impl_->hyb_width * n;
            std::vector<i32> hcol(hn, -1);
            std::vector<float> hval(hn, 0.0f);

            std::vector<i32> over_rows, over_ptr{0}, over_col;
            std::vector<float> over_val;
            for (i32 r = 0; r < n; ++r) {
                const i32 begin = row_ptr[r], end = row_ptr[r + 1];
                const i32 len = end - begin;
                const i32 in_ell = std::min<i32>(len, impl_->hyb_width);
                for (i32 j = 0; j < in_ell; ++j) {
                    hcol[(std::size_t)j * n + r] = col[begin + j];
                    hval[(std::size_t)j * n + r] = val[begin + j];
                }
                if (len > impl_->hyb_width) {
                    over_rows.push_back(r);
                    for (i32 e = begin + impl_->hyb_width; e < end; ++e) {
                        over_col.push_back(col[e]);
                        over_val.push_back(val[e]);
                    }
                    over_ptr.push_back((i32)over_col.size());
                }
            }
            impl_->n_over = (i32)over_rows.size();
            up_i(&impl_->d_hyb_col, hcol);
            up_f(&impl_->d_hyb_val, hval);
            up_i(&impl_->d_over_rows, over_rows);
            up_i(&impl_->d_over_ptr, over_ptr);
            up_i(&impl_->d_over_col, over_col);
            up_f(&impl_->d_over_val, over_val);

            impl_->bytes_hyb = hn * (sizeof(i32) + sizeof(float)) +
                               over_col.size() * (sizeof(i32) + sizeof(float)) +
                               over_rows.size() * sizeof(i32) + over_ptr.size() * sizeof(i32);
            impl_->hyb_padding = (std::int64_t)hn - (impl_->nnz - (std::int64_t)over_col.size());
        }
    } catch (...) {
        delete impl_;   // releases anything acquired before the throw
        impl_ = nullptr;
        throw;
    }
}

SparseMatrix::~SparseMatrix() { delete impl_; }

i32 SparseMatrix::rows() const { return impl_->n; }
i32 SparseMatrix::cols() const { return impl_->cols; }
std::int64_t SparseMatrix::nnz() const { return impl_->nnz; }

FormatStats SparseMatrix::stats(Format fmt) const {
    FormatStats s;
    s.nnz = impl_->nnz;
    s.max_row_nnz = impl_->max_row;
    s.mean_row_nnz = impl_->mean_row;
    switch (fmt) {
        case Format::Ellpack:
            s.device_bytes = impl_->bytes_ell;
            s.padded_entries = impl_->ell_padding;
            s.ell_width = impl_->ell_width;
            break;
        case Format::Hybrid:
            s.device_bytes = impl_->bytes_hyb;
            s.padded_entries = impl_->hyb_padding;
            s.ell_width = impl_->hyb_width;
            break;
        default:
            s.device_bytes = impl_->bytes_csr;
            s.padded_entries = 0;
            break;
    }
    return s;
}

std::vector<float> SparseMatrix::multiply(const std::vector<float>& x, Format fmt,
                                          float* elapsed_ms) const {
    if ((i32)x.size() != impl_->cols)
        throw std::invalid_argument("x length must equal the column count");

    float *d_x = nullptr, *d_y = nullptr;
    CU_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_y, (std::size_t)impl_->n * sizeof(float)));
    CU_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_y, 0, (std::size_t)impl_->n * sizeof(float)));

    const int T = 256;
    const int B = (impl_->n + T - 1) / T;
    const int Bw = (impl_->n + (T / 32) - 1) / (T / 32);

    cu::EventTimer t;
    t.start();
    switch (fmt) {
        case Format::CsrScalar:
            k_csr_scalar<<<B, T>>>(impl_->d_row_ptr, impl_->d_col, impl_->d_val,
                                   d_x, d_y, impl_->n);
            break;
        case Format::CsrVector:
            k_csr_vector<<<Bw, T>>>(impl_->d_row_ptr, impl_->d_col, impl_->d_val,
                                    d_x, d_y, impl_->n);
            break;
        case Format::Ellpack:
            k_ellpack<<<B, T>>>(impl_->d_ell_col, impl_->d_ell_val, d_x, d_y,
                                impl_->n, impl_->ell_width);
            break;
        case Format::Hybrid:
            k_ellpack<<<B, T>>>(impl_->d_hyb_col, impl_->d_hyb_val, d_x, d_y,
                                impl_->n, impl_->hyb_width);
            if (impl_->n_over > 0) {
                const int Bo = (impl_->n_over + (T / 32) - 1) / (T / 32);
                k_hybrid_overflow<<<Bo, T>>>(impl_->d_over_ptr, impl_->d_over_col,
                                             impl_->d_over_val, d_x, d_y,
                                             impl_->n_over, impl_->d_over_rows);
            }
            break;
    }
    CU_CHECK_KERNEL();
    const float ms = t.stop();
    if (elapsed_ms) *elapsed_ms = ms;

    std::vector<float> y((std::size_t)impl_->n);
    CU_CHECK(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_x);
    cudaFree(d_y);
    return y;
}

SparseMatrix::CgResult SparseMatrix::solve_cg(const std::vector<float>& b, int max_iter,
                                              double tol, Format fmt) const {
    if ((i32)b.size() != impl_->n)
        throw std::invalid_argument("b length must equal the row count");
    if (impl_->n != impl_->cols)
        throw std::invalid_argument("conjugate gradient requires a square matrix");

    const i32 n = impl_->n;
    CgResult res;
    res.x.assign(n, 0.0f);

    float *d_x = nullptr, *d_r = nullptr, *d_p = nullptr, *d_ap = nullptr;
    double* d_scalar = nullptr;
    CU_CHECK(cudaMalloc(&d_x, (std::size_t)n * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_r, (std::size_t)n * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_p, (std::size_t)n * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_ap, (std::size_t)n * sizeof(float)));
    CU_CHECK(cudaMalloc(&d_scalar, sizeof(double)));

    CU_CHECK(cudaMemset(d_x, 0, (std::size_t)n * sizeof(float)));
    CU_CHECK(cudaMemcpy(d_r, b.data(), (std::size_t)n * sizeof(float), cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_p, b.data(), (std::size_t)n * sizeof(float), cudaMemcpyHostToDevice));

    const int T = 256;
    const int B = (n + T - 1) / T;
    const int Bw = (n + (T / 32) - 1) / (T / 32);
    const int Bd = std::min(256, B);

    auto dot = [&](const float* a, const float* c) {
        CU_CHECK(cudaMemset(d_scalar, 0, sizeof(double)));
        k_dot<<<Bd, T>>>(a, c, n, d_scalar);
        CU_CHECK_KERNEL();
        double h = 0.0;
        CU_CHECK(cudaMemcpy(&h, d_scalar, sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    };
    auto spmv_into = [&](const float* xin, float* yout) {
        CU_CHECK(cudaMemset(yout, 0, (std::size_t)n * sizeof(float)));
        switch (fmt) {
            case Format::CsrScalar:
                k_csr_scalar<<<B, T>>>(impl_->d_row_ptr, impl_->d_col, impl_->d_val,
                                       xin, yout, n);
                break;
            case Format::CsrVector:
                k_csr_vector<<<Bw, T>>>(impl_->d_row_ptr, impl_->d_col, impl_->d_val,
                                        xin, yout, n);
                break;
            case Format::Ellpack:
                k_ellpack<<<B, T>>>(impl_->d_ell_col, impl_->d_ell_val, xin, yout,
                                    n, impl_->ell_width);
                break;
            case Format::Hybrid:
                k_ellpack<<<B, T>>>(impl_->d_hyb_col, impl_->d_hyb_val, xin, yout,
                                    n, impl_->hyb_width);
                if (impl_->n_over > 0) {
                    const int Bo = (impl_->n_over + (T / 32) - 1) / (T / 32);
                    k_hybrid_overflow<<<Bo, T>>>(impl_->d_over_ptr, impl_->d_over_col,
                                                 impl_->d_over_val, xin, yout,
                                                 impl_->n_over, impl_->d_over_rows);
                }
                break;
        }
        CU_CHECK_KERNEL();
    };

    cu::EventTimer t;
    t.start();

    double rs_old = dot(d_r, d_r);
    const double rs0 = rs_old;
    int it = 0;
    for (; it < max_iter; ++it) {
        if (std::sqrt(rs_old) <= tol * std::sqrt(rs0) || rs_old == 0.0) break;
        spmv_into(d_p, d_ap);

        const double pAp = dot(d_p, d_ap);
        if (pAp == 0.0) break;
        const double alpha = rs_old / pAp;

        k_axpy<<<B, T>>>(d_x, d_p, (float)alpha, n);
        k_axpy<<<B, T>>>(d_r, d_ap, (float)(-alpha), n);
        CU_CHECK_KERNEL();

        const double rs_new = dot(d_r, d_r);
        k_xpay<<<B, T>>>(d_p, d_r, (float)(rs_new / rs_old), n);
        CU_CHECK_KERNEL();
        rs_old = rs_new;
    }
    res.elapsed_ms = t.stop();
    res.iterations = it;
    res.residual = std::sqrt(rs_old);

    CU_CHECK(cudaMemcpy(res.x.data(), d_x, (std::size_t)n * sizeof(float),
                        cudaMemcpyDeviceToHost));
    cudaFree(d_x); cudaFree(d_r); cudaFree(d_p); cudaFree(d_ap); cudaFree(d_scalar);
    return res;
}

std::vector<float> SparseMatrix::multiply_cpu(const CooMatrix& coo,
                                              const std::vector<float>& x) {
    std::vector<double> acc((std::size_t)coo.rows, 0.0);
    for (std::size_t e = 0; e < coo.values.size(); ++e)
        acc[coo.row_idx[e]] += (double)coo.values[e] * x[coo.col_idx[e]];
    std::vector<float> y((std::size_t)coo.rows);
    for (std::size_t i = 0; i < acc.size(); ++i) y[i] = (float)acc[i];
    return y;
}

}  // namespace spmv
