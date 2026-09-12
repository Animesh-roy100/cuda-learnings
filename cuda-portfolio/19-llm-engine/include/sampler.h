#pragma once
//
// Token sampling policy. Host-only.
//
// Numerical rules, all tested:
//   * any NaN logit is an error -- a NaN means the forward pass is broken, and
//     sampling around it would hide that
//   * +inf logits: the +inf entries share all probability (uniformly under
//     sampling, lowest id under greedy)
//   * -inf logits are legal and mean "never"
//   * temperature 0 is greedy; ties resolve to the lowest token id, so greedy
//     decoding is deterministic
//
#include <cstdint>
#include <random>
#include <vector>

namespace llm {

struct SamplingParams {
    float temperature = 0.0f;   // 0 = greedy
    int top_k = 40;             // <= 0 disables
    float top_p = 1.0f;         // 1 disables
    std::uint64_t seed = 1;
};

class Sampler {
public:
    explicit Sampler(const SamplingParams& p);
    const SamplingParams& params() const { return p_; }
    int sample(const std::vector<float>& logits);

private:
    SamplingParams p_;
    std::mt19937_64 rng_;
};

}  // namespace llm
