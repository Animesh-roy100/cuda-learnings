#pragma once
//
// Monte Carlo option pricing and risk -- public interface.
// No CUDA syntax: host code and tests compile as plain C++20.
//
#include <cstdint>
#include <vector>

namespace mc {

enum class Style {
    European,      // payoff on terminal price; closed form exists (validation)
    Asian,         // payoff on the arithmetic average; needs time stepping
    BarrierUpOut,  // knocked out if the path ever exceeds `barrier`
};

struct Option {
    float S0 = 100.0f;
    float K = 100.0f;
    float r = 0.05f;
    float sigma = 0.20f;
    float T = 1.0f;
    float barrier = 0.0f;   // used only by BarrierUpOut
    Style style = Style::European;
};

struct Price {
    double value = 0.0;
    double std_error = 0.0;
    double delta = 0.0;
    double gamma = 0.0;
    double vega = 0.0;
};

class Engine {
public:
    explicit Engine(std::uint64_t seed = 1234ULL);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // steps is ignored for European (terminal distribution is exact).
    Price price(const Option& opt, std::int64_t paths, int steps = 252) const;

    // Prices many contracts in one launch -- the realistic risk-engine shape,
    // where one portfolio revaluation must not become thousands of tiny kernels.
    std::vector<Price> price_portfolio(const std::vector<Option>& book,
                                       std::int64_t paths_each) const;

    void set_antithetic(bool on);
    bool antithetic() const;

    // Closed-form Black-Scholes, for validating the simulation.
    static double bs_call(double S, double K, double r, double sig, double T);
    static double bs_delta(double S, double K, double r, double sig, double T);
    static double bs_gamma(double S, double K, double r, double sig, double T);
    static double bs_vega(double S, double K, double r, double sig, double T);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace mc
