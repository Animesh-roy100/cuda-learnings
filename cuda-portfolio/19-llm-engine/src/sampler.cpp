#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

#include "errors.h"

namespace llm {

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.seed) {
    if (!std::isfinite(p.temperature) || p.temperature < 0.0f)
        throw Error(ErrorKind::InvalidArgument, "sampler: temperature must be finite and >= 0");
    if (!std::isfinite(p.top_p) || p.top_p <= 0.0f || p.top_p > 1.0f)
        throw Error(ErrorKind::InvalidArgument, "sampler: top_p must be in (0, 1]");
}

int Sampler::sample(const std::vector<float>& logits) {
    ErrorContext ctx;
    ctx.operation = "sample";
    if (logits.empty()) throw Error(ErrorKind::InvalidArgument, "no logits", ctx);

    bool any_pos_inf = false, any_finite = false;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float v = logits[i];
        if (std::isnan(v))
            throw Error(ErrorKind::Device, "NaN logit at token " + std::to_string(i) +
                                               " -- the forward pass produced an invalid value",
                        ctx);
        any_pos_inf |= (v == INFINITY);
        any_finite |= std::isfinite(v);
    }
    if (!any_pos_inf && !any_finite)
        throw Error(ErrorKind::InvalidArgument, "every logit is -inf; nothing can be sampled", ctx);

    // +inf dominates everything: restrict to those entries.
    if (any_pos_inf) {
        std::vector<int> ids;
        for (std::size_t i = 0; i < logits.size(); ++i)
            if (logits[i] == INFINITY) ids.push_back(int(i));
        if (p_.temperature == 0.0f) return ids.front();
        std::uniform_int_distribution<std::size_t> pick(0, ids.size() - 1);
        return ids[pick(rng_)];
    }

    if (p_.temperature == 0.0f) {
        // max_element returns the first maximum: lowest id wins ties.
        return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<int> idx;
    idx.reserve(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i)
        if (std::isfinite(logits[i])) idx.push_back(int(i));
    const int k = p_.top_k > 0 ? std::min<int>(p_.top_k, int(idx.size())) : int(idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    });
    idx.resize(k);

    // Softmax over the kept candidates with the max subtracted, in double.
    const double mx = double(logits[idx[0]]) / p_.temperature;
    std::vector<double> w(k);
    double z = 0.0;
    for (int i = 0; i < k; ++i) z += w[i] = std::exp(double(logits[idx[i]]) / p_.temperature - mx);

    int keep = k;
    if (p_.top_p < 1.0f) {
        double cum = 0.0;
        for (int i = 0; i < k; ++i) {
            cum += w[i] / z;
            if (cum >= p_.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    std::discrete_distribution<int> d(w.begin(), w.begin() + keep);
    return idx[d(rng_)];
}

}  // namespace llm
