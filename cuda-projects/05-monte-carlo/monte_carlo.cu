// ============================================================================
// Project 5 - Monte Carlo option pricing and risk
//
// Monte Carlo is the one workload in this set that is genuinely COMPUTE bound.
// Everything else here streams memory; this simulates millions of independent
// price paths from a handful of parameters, so it is pure ALU throughput.
// That makes it the right place to show what the GTX 1650 is and is not good at:
//
//   FP32  ~2.8 TFLOP/s   (1 unit per CUDA core per clock)
//   FP64  ~0.09 TFLOP/s  (1/32 rate -- consumer Turing cripples double precision)
//
// A 32x penalty is not a micro-optimisation. Using double here by reflex is the
// single most expensive mistake you can make on this card, and the benchmark
// below measures exactly that.
//
// Verification: European options have a closed form (Black-Scholes), so we can
// check the simulation against ground truth and report the standard error.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e_ = (call);                                              \
        if (e_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,      \
                         cudaGetErrorString(e_));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// Contract parameters
struct Opt {
    float S0, K, r, sigma, T;
};

// ---------------------------------------------------------------------------
// Black-Scholes closed form, on the host, as ground truth.
// ---------------------------------------------------------------------------
static double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

static double bs_call(double S, double K, double r, double sig, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * std::sqrt(T));
    double d2 = d1 - sig * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}
static double bs_delta(double S, double K, double r, double sig, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * std::sqrt(T));
    return norm_cdf(d1);
}
static double bs_vega(double S, double K, double r, double sig, double T) {
    double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * std::sqrt(T));
    double pdf = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * 3.14159265358979323846);
    return S * pdf * std::sqrt(T);
}

// ---------------------------------------------------------------------------
// Block reduction: warp shuffle then one atomicAdd per block.
// Doing an atomicAdd per THREAD instead would serialise millions of updates
// onto one address and dominate the runtime.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

template <typename T>
__device__ void block_accumulate(T val, T* out) {
    __shared__ T warp_tot[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    val = warp_sum(val);
    if (lane == 0) warp_tot[wid] = val;
    __syncthreads();
    if (wid == 0) {
        int nwarps = (blockDim.x + 31) / 32;
        T v = (lane < nwarps) ? warp_tot[lane] : T(0);
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, v);
    }
}

// ---------------------------------------------------------------------------
// European call, FP32. One normal draw per path (terminal distribution is
// known exactly for GBM, so no time stepping is needed).
// ---------------------------------------------------------------------------
__global__ void mc_euro_f32(Opt o, long long npaths, float* sum, float* sumsq,
                            unsigned long long seed) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    float drift = (o.r - 0.5f * o.sigma * o.sigma) * o.T;
    float vol   = o.sigma * sqrtf(o.T);
    float disc  = expf(-o.r * o.T);

    // Plain grid-stride loop: exactly one path per iteration, so the path count
    // matches npaths exactly and the mean is unbiased. (curand_normal4 would
    // amortise the Philox call over 4 draws, but hand-advancing the loop index
    // to consume them makes the accounting easy to get subtly wrong.)
    float s = 0.0f, s2 = 0.0f;
    for (long long i = tid; i < npaths; i += stride) {
        float z   = curand_normal(&st);
        float ST  = o.S0 * __expf(drift + vol * z);
        float pay = fmaxf(ST - o.K, 0.0f) * disc;
        s += pay; s2 += pay * pay;
    }
    block_accumulate(s,  sum);
    block_accumulate(s2, sumsq);
}

// ---------------------------------------------------------------------------
// Same option, FP64. Identical maths, 32x slower ALU path.
// ---------------------------------------------------------------------------
__global__ void mc_euro_f64(Opt o, long long npaths, double* sum, double* sumsq,
                            unsigned long long seed) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    double drift = ((double)o.r - 0.5 * (double)o.sigma * o.sigma) * o.T;
    double vol   = (double)o.sigma * std::sqrt((double)o.T);
    double disc  = std::exp(-(double)o.r * o.T);

    double s = 0.0, s2 = 0.0;
    for (long long i = tid; i < npaths; i += stride) {
        double z = curand_normal_double(&st);
        double ST = o.S0 * exp(drift + vol * z);
        double pay = fmax(ST - o.K, 0.0) * disc;
        s += pay; s2 += pay * pay;
    }
    block_accumulate(s,  sum);
    block_accumulate(s2, sumsq);
}

// ---------------------------------------------------------------------------
// Antithetic variates: for every draw Z also price -Z and average the pair.
// Same cost per normal, roughly half the variance for a monotone payoff --
// free accuracy, and the cheapest variance reduction there is.
// ---------------------------------------------------------------------------
__global__ void mc_euro_antithetic(Opt o, long long npairs, float* sum, float* sumsq,
                                   unsigned long long seed) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    float drift = (o.r - 0.5f * o.sigma * o.sigma) * o.T;
    float vol   = o.sigma * sqrtf(o.T);
    float disc  = expf(-o.r * o.T);

    float s = 0.0f, s2 = 0.0f;
    for (long long i = tid; i < npairs; i += stride) {
        float z = curand_normal(&st);
        float a = fmaxf(o.S0 * __expf(drift + vol *  z) - o.K, 0.0f) * disc;
        float b = fmaxf(o.S0 * __expf(drift + vol * -z) - o.K, 0.0f) * disc;
        float pay = 0.5f * (a + b);
        s += pay; s2 += pay * pay;
    }
    block_accumulate(s,  sum);
    block_accumulate(s2, sumsq);
}

// ---------------------------------------------------------------------------
// Greeks by pathwise differentiation. For a call, dPayoff/dS0 = (ST/S0) when
// in the money, 0 otherwise -- an unbiased estimator that costs almost nothing,
// unlike bump-and-revalue which needs a second full simulation and then
// subtracts two nearly equal numbers.
// ---------------------------------------------------------------------------
__global__ void mc_greeks(Opt o, long long npaths, float* sum_p, float* sum_delta,
                          float* sum_vega, unsigned long long seed) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    float sqT   = sqrtf(o.T);
    float drift = (o.r - 0.5f * o.sigma * o.sigma) * o.T;
    float vol   = o.sigma * sqT;
    float disc  = expf(-o.r * o.T);

    float sp = 0.0f, sd = 0.0f, sv = 0.0f;
    for (long long i = tid; i < npaths; i += stride) {
        float z  = curand_normal(&st);
        float ST = o.S0 * __expf(drift + vol * z);
        int itm  = (ST > o.K);
        sp += itm ? (ST - o.K) * disc : 0.0f;
        sd += itm ? (ST / o.S0) * disc : 0.0f;                  // pathwise delta
        sv += itm ? ST * (sqT * z - o.sigma * o.T) * disc : 0.0f; // pathwise vega
    }
    block_accumulate(sp, sum_p);
    block_accumulate(sd, sum_delta);
    block_accumulate(sv, sum_vega);
}

// ---------------------------------------------------------------------------
// Asian option: payoff depends on the AVERAGE price, so it needs real time
// stepping and has no simple closed form. This is where Monte Carlo earns
// its keep -- and where the arithmetic intensity goes up sharply.
// ---------------------------------------------------------------------------
__global__ void mc_asian(Opt o, long long npaths, int nsteps, float* sum, float* sumsq,
                         unsigned long long seed) {
    long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    float dt    = o.T / nsteps;
    float drift = (o.r - 0.5f * o.sigma * o.sigma) * dt;
    float vol   = o.sigma * sqrtf(dt);
    float disc  = expf(-o.r * o.T);

    float s = 0.0f, s2 = 0.0f;
    for (long long i = tid; i < npaths; i += stride) {
        float S = o.S0, acc = 0.0f;
        for (int t = 0; t < nsteps; ++t) {
            S *= __expf(drift + vol * curand_normal(&st));
            acc += S;
        }
        float pay = fmaxf(acc / nsteps - o.K, 0.0f) * disc;
        s += pay; s2 += pay * pay;
    }
    block_accumulate(s,  sum);
    block_accumulate(s2, sumsq);
}

// ---------------------------------------------------------------------------
struct Stat { double price, stderr_, ms; };

template <typename T>
static Stat finish(T* d_sum, T* d_sumsq, long long n, float ms) {
    T hs = 0, hq = 0;
    CUDA_CHECK(cudaMemcpy(&hs, d_sum,   sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hq, d_sumsq, sizeof(T), cudaMemcpyDeviceToHost));
    double mean = (double)hs / n;
    double var  = (double)hq / n - mean * mean;
    if (var < 0) var = 0;
    Stat st;
    st.price = mean;
    st.stderr_ = std::sqrt(var / n);
    st.ms = ms;
    return st;
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    std::printf("%s  sm_%d%d  %d SMs\n\n", p.name, p.major, p.minor, p.multiProcessorCount);

    Opt o;
    o.S0 = 100.0f; o.K = 105.0f; o.r = 0.05f; o.sigma = 0.20f; o.T = 1.0f;
    double truth = bs_call(o.S0, o.K, o.r, o.sigma, o.T);
    std::printf("European call  S0=%.0f K=%.0f r=%.2f sigma=%.2f T=%.1f\n",
                o.S0, o.K, o.r, o.sigma, o.T);
    std::printf("Black-Scholes exact price: %.6f\n\n", truth);

    const int T_ = 256;
    const int B_ = 14 * 16;                 // ~16 blocks per SM
    const long long NP = 100000000LL;       // 100M paths

    float *d_s, *d_q, *d_d, *d_v;
    double *d_sd, *d_qd;
    CUDA_CHECK(cudaMalloc(&d_s, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sd, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_qd, sizeof(double)));

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    float ms = 0.0f;

    // ---- FP32 ----
    CUDA_CHECK(cudaMemset(d_s, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_q, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(a));
    mc_euro_f32<<<B_, T_>>>(o, NP, d_s, d_q, 1234ULL);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaGetLastError());
    Stat f32 = finish(d_s, d_q, NP, ms);

    // ---- FP64, 100x fewer paths so it finishes this decade ----
    const long long NP64 = NP / 100;
    CUDA_CHECK(cudaMemset(d_sd, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_qd, 0, sizeof(double)));
    CUDA_CHECK(cudaEventRecord(a));
    mc_euro_f64<<<B_, T_>>>(o, NP64, d_sd, d_qd, 1234ULL);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaGetLastError());
    Stat f64 = finish(d_sd, d_qd, NP64, ms);

    // ---- antithetic (NP/2 pairs = NP draws of work, NP paths of information) ----
    CUDA_CHECK(cudaMemset(d_s, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_q, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(a));
    mc_euro_antithetic<<<B_, T_>>>(o, NP / 2, d_s, d_q, 1234ULL);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaGetLastError());
    Stat anti = finish(d_s, d_q, NP / 2, ms);

    std::printf("=== European call, %lld paths ===\n", NP);
    std::printf("  %-14s %10.6f  +/- %.6f  %8.1f ms  %7.0f M paths/s  err %+.6f\n",
                "fp32", f32.price, f32.stderr_, f32.ms, NP / (f32.ms / 1000.0) / 1e6,
                f32.price - truth);
    std::printf("  %-14s %10.6f  +/- %.6f  %8.1f ms  %7.0f M paths/s  err %+.6f  (%lld paths)\n",
                "fp64", f64.price, f64.stderr_, f64.ms, NP64 / (f64.ms / 1000.0) / 1e6,
                f64.price - truth, NP64);
    std::printf("  %-14s %10.6f  +/- %.6f  %8.1f ms  %7.0f M paths/s  err %+.6f\n",
                "fp32 antithet", anti.price, anti.stderr_, anti.ms,
                NP / (anti.ms / 1000.0) / 1e6, anti.price - truth);

    double thr32 = NP / (f32.ms / 1000.0);
    double thr64 = NP64 / (f64.ms / 1000.0);
    std::printf("\n  FP32 is %.1fx the FP64 path rate. Turing consumer silicon runs\n"
                "  double precision at 1/32 rate -- this is that, measured.\n", thr32 / thr64);
    std::printf("  Antithetic cuts the standard error %.2fx for the same path count.\n",
                f32.stderr_ / anti.stderr_);

    // Statistical check: is the FP32 estimate consistent with the exact price?
    double z = std::fabs(f32.price - truth) / f32.stderr_;
    std::printf("  fp32 estimate is %.2f standard errors from exact  -> %s\n",
                z, z < 3.0 ? "CONSISTENT" : "SUSPICIOUS");

    // ---- Greeks ----
    CUDA_CHECK(cudaMemset(d_s, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_d, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(a));
    mc_greeks<<<B_, T_>>>(o, NP, d_s, d_d, d_v, 4321ULL);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaGetLastError());
    float hp = 0, hd = 0, hv = 0;
    CUDA_CHECK(cudaMemcpy(&hp, d_s, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hd, d_d, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hv, d_v, sizeof(float), cudaMemcpyDeviceToHost));

    double d_exact = bs_delta(o.S0, o.K, o.r, o.sigma, o.T);
    double v_exact = bs_vega(o.S0, o.K, o.r, o.sigma, o.T);
    std::printf("\n=== Greeks by pathwise differentiation (%lld paths, %.1f ms) ===\n", NP, ms);
    std::printf("  delta  MC %8.6f   exact %8.6f   err %+.6f\n",
                hd / NP, d_exact, hd / NP - d_exact);
    std::printf("  vega   MC %8.4f   exact %8.4f   err %+.4f\n",
                hv / NP, v_exact, hv / NP - v_exact);

    // ---- Asian ----
    const long long NPA = 2000000LL;
    const int STEPS = 252;                  // daily fixings for a year
    CUDA_CHECK(cudaMemset(d_s, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_q, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(a));
    mc_asian<<<B_, T_>>>(o, NPA, STEPS, d_s, d_q, 777ULL);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaGetLastError());
    Stat as = finish(d_s, d_q, NPA, ms);

    std::printf("\n=== Asian (arithmetic average) call, %lld paths x %d steps ===\n", NPA, STEPS);
    std::printf("  price %.6f +/- %.6f   %.1f ms   %.0f M steps/s\n",
                as.price, as.stderr_, as.ms, (double)NPA * STEPS / (as.ms / 1000.0) / 1e6);
    std::printf("  (cheaper than the European call, as it must be: averaging\n"
                "   suppresses the terminal variance the payoff depends on)\n");

    std::printf("\nThis is the one compute-bound project in the set. Everything else\n"
                "streams memory; here the GPU does real arithmetic per byte, which is\n"
                "why FP32 vs FP64 shows up so brutally.\n");

    cudaFree(d_s); cudaFree(d_q); cudaFree(d_d); cudaFree(d_v);
    cudaFree(d_sd); cudaFree(d_qd);
    return 0;
}
