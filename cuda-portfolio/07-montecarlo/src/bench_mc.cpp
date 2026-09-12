// Risk-engine benchmark: single-contract throughput and a full book revaluation.

#include <cstdio>
#include <vector>

#include "cu/device.hpp"
#include "cu/timer.hpp"
#include "mc_pricing.h"

using mc::Engine;
using mc::Option;
using mc::Price;
using mc::Style;

int main() {
    auto dev = cu::query_device();
    cu::print_banner(dev);
    std::printf("Compute-bound workload. FP32 only: Turing runs FP64 at 1/32 rate.\n\n");

    Option o;
    o.S0 = 100.0f; o.K = 105.0f; o.r = 0.05f; o.sigma = 0.20f; o.T = 1.0f;
    double truth = Engine::bs_call(o.S0, o.K, o.r, o.sigma, o.T);
    std::printf("European call S0=100 K=105 r=5%% sigma=20%% T=1\n");
    std::printf("Black-Scholes exact: %.6f\n\n", truth);

    const std::int64_t N = 100'000'000;
    cu::EventTimer t;

    Engine plain(1234ULL);
    t.start();
    Price p = plain.price(o, N);
    float ms = t.stop();

    Engine anti(1234ULL);
    anti.set_antithetic(true);
    t.start();
    Price pa = anti.price(o, N);
    float ms_a = t.stop();

    std::printf("=== %lld paths ===\n", static_cast<long long>(N));
    std::printf("  plain       %10.6f +/- %.6f  %7.1f ms  %6.0f M paths/s  err %+.6f\n",
                p.value, p.std_error, ms, N / (ms / 1e3) / 1e6, p.value - truth);
    std::printf("  antithetic  %10.6f +/- %.6f  %7.1f ms  %6.0f M paths/s  err %+.6f\n",
                pa.value, pa.std_error, ms_a, N / (ms_a / 1e3) / 1e6, pa.value - truth);
    std::printf("  antithetic cuts standard error %.2fx\n",
                p.std_error / pa.std_error);
    std::printf("  price sits %.2f standard errors from exact\n\n",
                std::abs(p.value - truth) / p.std_error);

    std::printf("=== Greeks (pathwise delta/vega, likelihood-ratio gamma) ===\n");
    std::printf("  %-6s %12s %12s %12s\n", "", "monte carlo", "exact", "error");
    std::printf("  %-6s %12.6f %12.6f %+12.6f\n", "delta", p.delta,
                Engine::bs_delta(o.S0, o.K, o.r, o.sigma, o.T),
                p.delta - Engine::bs_delta(o.S0, o.K, o.r, o.sigma, o.T));
    std::printf("  %-6s %12.6f %12.6f %+12.6f\n", "gamma", p.gamma,
                Engine::bs_gamma(o.S0, o.K, o.r, o.sigma, o.T),
                p.gamma - Engine::bs_gamma(o.S0, o.K, o.r, o.sigma, o.T));
    std::printf("  %-6s %12.4f %12.4f %+12.4f\n\n", "vega", p.vega,
                Engine::bs_vega(o.S0, o.K, o.r, o.sigma, o.T),
                p.vega - Engine::bs_vega(o.S0, o.K, o.r, o.sigma, o.T));

    // --- exotics ---
    Option asian = o; asian.style = Style::Asian;
    Option barrier = o; barrier.style = Style::BarrierUpOut; barrier.barrier = 130.0f;

    t.start();
    Price pas = plain.price(asian, 2'000'000, 252);
    float ms_as = t.stop();
    t.start();
    Price pb = plain.price(barrier, 2'000'000, 252);
    float ms_b = t.stop();

    std::printf("=== exotics (2M paths x 252 steps) ===\n");
    std::printf("  asian         %10.6f +/- %.6f  %7.1f ms  %5.0f M steps/s\n",
                pas.value, pas.std_error, ms_as, 2e6 * 252 / (ms_as / 1e3) / 1e6);
    std::printf("  barrier 130   %10.6f +/- %.6f  %7.1f ms  %5.0f M steps/s\n",
                pb.value, pb.std_error, ms_b, 2e6 * 252 / (ms_b / 1e3) / 1e6);
    std::printf("  (both below the %.4f vanilla, as they must be)\n\n", truth);

    // --- portfolio revaluation ---
    const int BOOK = 10000;
    std::vector<Option> book;
    book.reserve(BOOK);
    for (int i = 0; i < BOOK; ++i) {
        Option x = o;
        x.K = 70.0f + 0.006f * static_cast<float>(i);
        x.sigma = 0.15f + 0.0001f * static_cast<float>(i % 500);
        book.push_back(x);
    }

    const std::int64_t PE = 100'000;
    t.start();
    auto prices = plain.price_portfolio(book, PE);
    float ms_pf = t.stop();

    double worst = 0.0;
    for (int i = 0; i < BOOK; ++i) {
        double tr = Engine::bs_call(book[i].S0, book[i].K, book[i].r,
                                    book[i].sigma, book[i].T);
        double z = prices[i].std_error > 0
                       ? std::abs(prices[i].value - tr) / prices[i].std_error
                       : 0.0;
        if (z > worst) worst = z;
    }

    std::printf("=== portfolio revaluation: %d contracts x %lld paths ===\n",
                BOOK, static_cast<long long>(PE));
    std::printf("  %.1f ms for the whole book (%.0f M paths/s)\n",
                ms_pf, (double)BOOK * PE / (ms_pf / 1e3) / 1e6);
    std::printf("  %.2f ms per 1000 contracts -- a live risk screen can repaint\n",
                ms_pf / (BOOK / 1000.0));
    std::printf("  worst contract deviates %.2f sigma from closed form\n", worst);
    std::printf("\nOne launch, one block per contract. Pricing each contract in its\n"
                "own kernel would leave 13 of 14 SMs idle per launch.\n");
    return 0;
}
