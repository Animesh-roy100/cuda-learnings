// Validation against closed-form Black-Scholes.
//
// Monte Carlo is statistical, so assertions are in units of STANDARD ERRORS,
// not absolute tolerance. A fixed epsilon either passes trivially or flakes;
// "within 4 sigma" is the statistically meaningful statement, and fails about
// 1 run in 16000 by chance.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "mc_pricing.h"

using mc::Engine;
using mc::Option;
using mc::Price;
using mc::Style;

namespace {
constexpr std::int64_t kPaths = 4'000'000;

Option base_call() {
    Option o;
    o.S0 = 100.0f; o.K = 105.0f; o.r = 0.05f; o.sigma = 0.20f; o.T = 1.0f;
    o.style = Style::European;
    return o;
}
}  // namespace

TEST(MonteCarlo, EuropeanMatchesBlackScholes) {
    Engine e(20260912ULL);
    Option o = base_call();
    Price p = e.price(o, kPaths);
    double truth = Engine::bs_call(o.S0, o.K, o.r, o.sigma, o.T);

    ASSERT_GT(p.std_error, 0.0);
    double z = std::fabs(p.value - truth) / p.std_error;
    EXPECT_LT(z, 4.0) << "price " << p.value << " vs exact " << truth
                      << " (" << z << " sigma)";
}

// Regression test for a shared-memory race in the block reduction.
//
// block_add() reuses one __shared__ array across five consecutive calls. With
// no barrier at entry, warps racing ahead into call N+1 overwrote the array
// while warp 0 was still reading it for call N. The corruption was TIMING
// dependent: invisible at 4M paths, and 817 standard errors wrong at 100M,
// where longer per-thread loops let warps drift further apart.
//
// The lesson generalises: concurrency bugs scale in with occupancy and loop
// length, so a correctness suite that only runs small inputs proves very little.
TEST(MonteCarlo, EuropeanCorrectAtHighPathCount) {
    Engine e(1234ULL);
    Option o = base_call();
    Price p = e.price(o, 100'000'000);
    double truth = Engine::bs_call(o.S0, o.K, o.r, o.sigma, o.T);

    ASSERT_GT(p.std_error, 0.0);
    double z = std::fabs(p.value - truth) / p.std_error;
    EXPECT_LT(z, 4.0) << "price " << p.value << " vs exact " << truth
                      << " (" << z << " sigma)";

    // Greeks must survive the same reduction path.
    EXPECT_NEAR(p.delta, Engine::bs_delta(o.S0, o.K, o.r, o.sigma, o.T), 0.005);
    EXPECT_NEAR(p.gamma, Engine::bs_gamma(o.S0, o.K, o.r, o.sigma, o.T), 0.005);
    EXPECT_NEAR(p.vega, Engine::bs_vega(o.S0, o.K, o.r, o.sigma, o.T), 0.25);
}

TEST(MonteCarlo, DeepInTheMoneyApproachesIntrinsic) {
    Engine e(7ULL);
    Option o = base_call();
    o.K = 10.0f;                       // essentially certain to be exercised
    Price p = e.price(o, 1'000'000);
    double intrinsic = o.S0 - o.K * std::exp(-o.r * o.T);
    EXPECT_NEAR(p.value, intrinsic, 0.05);
    EXPECT_NEAR(p.delta, 1.0, 0.02);   // delta pins at 1
}

TEST(MonteCarlo, DeepOutOfTheMoneyIsNearlyWorthless) {
    Engine e(7ULL);
    Option o = base_call();
    o.K = 1000.0f;
    Price p = e.price(o, 1'000'000);
    EXPECT_LT(p.value, 0.01);
    EXPECT_LT(p.delta, 0.01);
}

TEST(MonteCarlo, PathwiseGreeksMatchClosedForm) {
    Engine e(20260912ULL);
    Option o = base_call();
    Price p = e.price(o, kPaths);

    double d = Engine::bs_delta(o.S0, o.K, o.r, o.sigma, o.T);
    double v = Engine::bs_vega(o.S0, o.K, o.r, o.sigma, o.T);
    double g = Engine::bs_gamma(o.S0, o.K, o.r, o.sigma, o.T);

    EXPECT_NEAR(p.delta, d, 0.01) << "pathwise delta";
    EXPECT_NEAR(p.vega, v, 0.5) << "pathwise vega";
    // Gamma uses the likelihood-ratio method: unbiased but much noisier than
    // the pathwise estimators, so it gets a looser band.
    EXPECT_NEAR(p.gamma, g, 0.01) << "LRM gamma";
}

TEST(MonteCarlo, AntitheticReducesStandardError) {
    Option o = base_call();

    Engine plain(99ULL);
    Price a = plain.price(o, kPaths);

    Engine anti(99ULL);
    anti.set_antithetic(true);
    Price b = anti.price(o, kPaths);

    EXPECT_LT(b.std_error, a.std_error);
    // Both must still agree with the truth.
    double truth = Engine::bs_call(o.S0, o.K, o.r, o.sigma, o.T);
    EXPECT_LT(std::fabs(b.value - truth) / b.std_error, 4.0);
}

TEST(MonteCarlo, AsianIsCheaperThanEuropean) {
    Engine e(5ULL);
    Option euro = base_call();
    Option asian = base_call();
    asian.style = Style::Asian;

    Price pe = e.price(euro, 500'000);
    Price pa = e.price(asian, 500'000, 252);

    // Averaging suppresses the terminal variance the payoff depends on, so an
    // Asian call must be worth strictly less than its European twin.
    EXPECT_LT(pa.value, pe.value);
    EXPECT_GT(pa.value, 0.0);
}

TEST(MonteCarlo, BarrierIsCheaperThanVanillaAndZeroWhenBarrierIsTight) {
    Engine e(11ULL);
    Option vanilla = base_call();
    Price pv = e.price(vanilla, 500'000, 252);

    Option knock = base_call();
    knock.style = Style::BarrierUpOut;
    knock.barrier = 130.0f;
    Price pk = e.price(knock, 500'000, 252);
    EXPECT_LT(pk.value, pv.value);
    EXPECT_GE(pk.value, 0.0);

    // A barrier at the strike can never pay: to finish above K the path must
    // cross K, which knocks it out first.
    Option dead = base_call();
    dead.style = Style::BarrierUpOut;
    dead.barrier = dead.K;
    Price pd = e.price(dead, 200'000, 252);
    EXPECT_NEAR(pd.value, 0.0, 1e-9);
}

TEST(MonteCarlo, PortfolioMatchesIndividualPricing) {
    Engine e(31337ULL);
    std::vector<Option> book;
    for (int i = 0; i < 64; ++i) {
        Option o = base_call();
        o.K = 80.0f + static_cast<float>(i);
        book.push_back(o);
    }

    auto prices = e.price_portfolio(book, 200'000);
    ASSERT_EQ(prices.size(), book.size());

    for (std::size_t i = 0; i < book.size(); ++i) {
        double truth = Engine::bs_call(book[i].S0, book[i].K, book[i].r,
                                       book[i].sigma, book[i].T);
        ASSERT_GT(prices[i].std_error, 0.0) << "at " << i;
        double z = std::fabs(prices[i].value - truth) / prices[i].std_error;
        EXPECT_LT(z, 4.5) << "contract " << i << " K=" << book[i].K
                          << " mc=" << prices[i].value << " exact=" << truth;
    }
}

TEST(MonteCarlo, PriceIsMonotoneDecreasingInStrike) {
    Engine e(1234ULL);
    std::vector<Option> book;
    for (int i = 0; i < 16; ++i) {
        Option o = base_call();
        o.K = 60.0f + 10.0f * static_cast<float>(i);
        book.push_back(o);
    }
    auto p = e.price_portfolio(book, 400'000);
    for (std::size_t i = 1; i < p.size(); ++i) {
        EXPECT_LE(p[i].value, p[i - 1].value + 1e-3)
            << "call price must fall as strike rises, at i=" << i;
    }
}
