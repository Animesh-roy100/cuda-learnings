// Monte Carlo option pricing -- implementation.
//
// This is the one compute-bound workload in the portfolio: a handful of
// parameters become billions of paths, so it is pure ALU throughput.
//
// Two Turing realities shape every decision here:
//   FP32 ~2.8 TFLOP/s, FP64 ~0.09 TFLOP/s (1/32 rate on consumer silicon).
// Using double by reflex costs 30x. So everything runs in FP32 -- and the
// accumulation therefore needs Kahan compensation, because summing 10^8 FP32
// payoffs naively loses low-order bits once the running total dwarfs the
// increment.

#include "mc_pricing.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <stdexcept>

#include "cu/check.hpp"

namespace mc {
namespace {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Kahan compensated summation, in registers.
// Naive FP32 accumulation over 10^8 terms drifts badly: once `sum` is large,
// `sum + small` rounds the small term away entirely. Kahan keeps the lost
// low-order part in `c` and folds it back next iteration.
// ---------------------------------------------------------------------------
struct Kahan {
    float sum = 0.0f;
    float c = 0.0f;

    __device__ __forceinline__ void add(float v) {
        float y = v - c;
        float t = sum + y;
        c = (t - sum) - y;      // recovers exactly what rounding discarded
        sum = t;
    }
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// One atomicAdd per block, not per thread: millions of atomics on one address
// would serialise and dominate the runtime.
//
// The LEADING __syncthreads() is load-bearing. This function is called five
// times in a row (payoff, payoff^2, delta, gamma, vega) and reuses one shared
// array. Without a barrier at entry, warps 1..7 race ahead into call N+1 and
// overwrite warp_tot[] while warp 0 is still reading it for call N. That race
// is timing-dependent: it stayed hidden at 4M paths and produced an answer
// 817 standard errors wrong at 100M.
//
// Accumulating into double costs one atomic per block and removes any question
// about FP32 drift in the final total -- cheap insurance for a pricing engine.
__device__ void block_add(float val, double* out) {
    __shared__ float warp_tot[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;

    __syncthreads();                    // previous call has finished reading
    val = warp_sum(val);
    if (lane == 0) warp_tot[wid] = val;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) / 32;
        float v = (lane < nw) ? warp_tot[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, static_cast<double>(v));
    }
}

struct Accum {
    float payoff, payoff2, delta, gamma, vega;
};

// ---------------------------------------------------------------------------
// Terminal-price payoffs (European): one normal draw, exact GBM terminal law.
// Greeks come from pathwise derivatives (delta, vega) and the likelihood-ratio
// method (gamma). Pathwise is unbiased and nearly free; gamma cannot use it
// because the payoff kink has no second derivative, hence LRM.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void euro_sample(const Option& o, float z,
                                            float drift, float vol, float disc,
                                            float sqT, Accum& a) {
    float ST = o.S0 * __expf(drift + vol * z);
    int itm = (ST > o.K);
    float pay = itm ? (ST - o.K) * disc : 0.0f;

    a.payoff = pay;
    a.payoff2 = pay * pay;
    a.delta = itm ? (ST / o.S0) * disc : 0.0f;                    // pathwise
    a.vega = itm ? ST * (sqT * z - o.sigma * o.T) * disc : 0.0f;  // pathwise
    // Likelihood-ratio gamma for GBM.
    float lrm = (z * z - z * vol - 1.0f) / (o.S0 * o.S0 * o.sigma * o.sigma * o.T);
    a.gamma = pay * lrm;
}

__global__ void k_european(Option o, long long npaths, bool antithetic,
                           double* out, unsigned long long seed) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    float sqT = sqrtf(o.T);
    float drift = (o.r - 0.5f * o.sigma * o.sigma) * o.T;
    float vol = o.sigma * sqT;
    float disc = __expf(-o.r * o.T);

    Kahan kp, kp2, kd, kg, kv;
    for (long long i = tid; i < npaths; i += stride) {
        float z = curand_normal(&st);
        Accum a;
        euro_sample(o, z, drift, vol, disc, sqT, a);

        if (antithetic) {
            // Reusing -z halves the variance of a monotone payoff for the same
            // number of random draws. The cheapest variance reduction there is.
            Accum b;
            euro_sample(o, -z, drift, vol, disc, sqT, b);
            a.payoff = 0.5f * (a.payoff + b.payoff);
            a.delta = 0.5f * (a.delta + b.delta);
            a.gamma = 0.5f * (a.gamma + b.gamma);
            a.vega = 0.5f * (a.vega + b.vega);
            a.payoff2 = a.payoff * a.payoff;
        }
        kp.add(a.payoff);
        kp2.add(a.payoff2);
        kd.add(a.delta);
        kg.add(a.gamma);
        kv.add(a.vega);
    }
    block_add(kp.sum, out + 0);
    block_add(kp2.sum, out + 1);
    block_add(kd.sum, out + 2);
    block_add(kg.sum, out + 3);
    block_add(kv.sum, out + 4);
}

// ---------------------------------------------------------------------------
// Path-dependent payoffs: Asian (arithmetic average) and up-and-out barrier.
// Both need real time stepping, so arithmetic intensity rises sharply.
// ---------------------------------------------------------------------------
__global__ void k_path_dependent(Option o, long long npaths, int nsteps,
                                 double* out, unsigned long long seed) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    float dt = o.T / nsteps;
    float drift = (o.r - 0.5f * o.sigma * o.sigma) * dt;
    float vol = o.sigma * sqrtf(dt);
    float disc = __expf(-o.r * o.T);
    bool asian = (o.style == Style::Asian);

    Kahan kp, kp2;
    for (long long i = tid; i < npaths; i += stride) {
        float S = o.S0, acc = 0.0f;
        bool knocked = false;
        for (int t = 0; t < nsteps; ++t) {
            S *= __expf(drift + vol * curand_normal(&st));
            acc += S;
            if (!asian && S >= o.barrier) { knocked = true; break; }
        }
        float pay = 0.0f;
        if (asian) {
            pay = fmaxf(acc / nsteps - o.K, 0.0f) * disc;
        } else if (!knocked) {
            pay = fmaxf(S - o.K, 0.0f) * disc;
        }
        kp.add(pay);
        kp2.add(pay * pay);
    }
    block_add(kp.sum, out + 0);
    block_add(kp2.sum, out + 1);
}

// One block per contract: a 10k-contract book becomes ONE launch instead of
// 10k tiny ones, each of which would leave 13 of 14 SMs idle.
__global__ void k_portfolio(const Option* __restrict__ book, int n_opts,
                            long long paths_each, double* __restrict__ out,
                            unsigned long long seed) {
    int oid = blockIdx.x;
    if (oid >= n_opts) return;
    Option o = book[oid];

    curandStatePhilox4_32_10_t st;
    curand_init(seed + oid, threadIdx.x, 0, &st);

    float sqT = sqrtf(o.T);
    float drift = (o.r - 0.5f * o.sigma * o.sigma) * o.T;
    float vol = o.sigma * sqT;
    float disc = __expf(-o.r * o.T);

    Kahan kp, kp2, kd, kg, kv;
    for (long long i = threadIdx.x; i < paths_each; i += blockDim.x) {
        float z = curand_normal(&st);
        Accum a;
        euro_sample(o, z, drift, vol, disc, sqT, a);
        kp.add(a.payoff);
        kp2.add(a.payoff2);
        kd.add(a.delta);
        kg.add(a.gamma);
        kv.add(a.vega);
    }
    block_add(kp.sum, out + oid * 5 + 0);
    block_add(kp2.sum, out + oid * 5 + 1);
    block_add(kd.sum, out + oid * 5 + 2);
    block_add(kg.sum, out + oid * 5 + 3);
    block_add(kv.sum, out + oid * 5 + 4);
}

double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); }
double d1_of(double S, double K, double r, double sig, double T) {
    return (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * std::sqrt(T));
}

}  // namespace

// ---------------------------------------------------------------------------
struct Engine::Impl {
    std::uint64_t seed;
    bool antithetic = false;
    int threads = 256;
    int blocks = 14 * 16;      // ~16 blocks per SM keeps all 14 SMs busy
};

Engine::Engine(std::uint64_t seed) : impl_(new Impl) { impl_->seed = seed; }
Engine::~Engine() { delete impl_; }
void Engine::set_antithetic(bool on) { impl_->antithetic = on; }
bool Engine::antithetic() const { return impl_->antithetic; }

Price Engine::price(const Option& opt, std::int64_t paths, int steps) const {
    if (paths <= 0) throw std::invalid_argument("paths must be positive");
    // Validated BEFORE the allocation below. It used to be checked after, so
    // every rejected call leaked the output buffer.
    if (opt.style != Style::European && steps <= 0)
        throw std::invalid_argument("steps must be positive");

    double* d_out = nullptr;
    double h[5] = {0, 0, 0, 0, 0};
    try {
        CU_CHECK(cudaMalloc(&d_out, 5 * sizeof(double)));
        CU_CHECK(cudaMemset(d_out, 0, 5 * sizeof(double)));

        if (opt.style == Style::European) {
            k_european<<<impl_->blocks, impl_->threads>>>(opt, paths, impl_->antithetic,
                                                          d_out, impl_->seed);
        } else {
            k_path_dependent<<<impl_->blocks, impl_->threads>>>(opt, paths, steps,
                                                                d_out, impl_->seed);
        }
        CU_CHECK_KERNEL();
        CU_CHECK(cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_out);
        throw;
    }
    cudaFree(d_out);

    double n = static_cast<double>(paths);
    Price p;
    p.value = h[0] / n;
    double var = h[1] / n - p.value * p.value;
    p.std_error = std::sqrt((var > 0 ? var : 0) / n);
    p.delta = h[2] / n;
    p.gamma = h[3] / n;
    p.vega = h[4] / n;
    return p;
}

std::vector<Price> Engine::price_portfolio(const std::vector<Option>& book,
                                           std::int64_t paths_each) const {
    const int n = static_cast<int>(book.size());
    std::vector<Price> out(n);
    if (n == 0) return out;

    Option* d_book = nullptr;
    double* d_out = nullptr;
    CU_CHECK(cudaMalloc(&d_book, sizeof(Option) * n));
    CU_CHECK(cudaMalloc(&d_out, sizeof(double) * 5 * n));
    CU_CHECK(cudaMemcpy(d_book, book.data(), sizeof(Option) * n, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(d_out, 0, sizeof(double) * 5 * n));

    k_portfolio<<<n, impl_->threads>>>(d_book, n, paths_each, d_out, impl_->seed);
    CU_CHECK_KERNEL();

    std::vector<double> h(5 * n);
    CU_CHECK(cudaMemcpy(h.data(), d_out, sizeof(double) * 5 * n, cudaMemcpyDeviceToHost));
    cudaFree(d_book);
    cudaFree(d_out);

    double pe = static_cast<double>(paths_each);
    for (int i = 0; i < n; ++i) {
        out[i].value = h[i * 5 + 0] / pe;
        double var = h[i * 5 + 1] / pe - out[i].value * out[i].value;
        out[i].std_error = std::sqrt((var > 0 ? var : 0) / pe);
        out[i].delta = h[i * 5 + 2] / pe;
        out[i].gamma = h[i * 5 + 3] / pe;
        out[i].vega = h[i * 5 + 4] / pe;
    }
    return out;
}

double Engine::bs_call(double S, double K, double r, double sig, double T) {
    double d1 = d1_of(S, K, r, sig, T);
    double d2 = d1 - sig * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}
double Engine::bs_delta(double S, double K, double r, double sig, double T) {
    return norm_cdf(d1_of(S, K, r, sig, T));
}
double Engine::bs_gamma(double S, double K, double r, double sig, double T) {
    return norm_pdf(d1_of(S, K, r, sig, T)) / (S * sig * std::sqrt(T));
}
double Engine::bs_vega(double S, double K, double r, double sig, double T) {
    return S * norm_pdf(d1_of(S, K, r, sig, T)) * std::sqrt(T);
}

}  // namespace mc
