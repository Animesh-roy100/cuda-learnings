// Host FP32 Llama forward pass, for validating the device engine.
//
// Written to be obviously correct rather than fast: textbook matrix products
// in double-free FP32, rotate-in-place RoPE, a softmax per query. It shares
// nothing with the device path but the file parser and the block
// dequantizers, which have their own tests.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "gguf.h"

namespace llm {
namespace {

struct Dense {
    std::vector<float> w;   // row-major [rows x cols]
    int rows = 0, cols = 0;
};

Dense load(const GgufFile& f, const std::string& name) {
    const auto* t = f.find(name);
    if (!t) throw std::runtime_error("reference: missing " + name);
    Dense d;
    d.cols = int(t->dims[0]);
    d.rows = t->dims.size() > 1 ? int(t->dims[1]) : 1;
    d.w.resize(t->num_elements());
    const auto* src = static_cast<const std::uint8_t*>(f.tensor_data(*t));
    switch (t->type) {
        case GgmlType::F32: std::copy_n(reinterpret_cast<const float*>(src), d.w.size(), d.w.data()); break;
        case GgmlType::Q4_0: dequantize_q4_0(src, d.w.size(), d.w.data()); break;
        case GgmlType::Q6_K: dequantize_q6_k(src, d.w.size(), d.w.data()); break;
        default: throw std::runtime_error("reference: unsupported type for " + name);
    }
    return d;
}

std::vector<float> mul(const Dense& m, const std::vector<float>& x) {
    std::vector<float> y(m.rows, 0.0f);
    for (int r = 0; r < m.rows; ++r) {
        const float* row = m.w.data() + std::size_t(r) * m.cols;
        float s = 0.0f;
        for (int c = 0; c < m.cols; ++c) s += row[c] * x[c];
        y[r] = s;
    }
    return y;
}

std::vector<float> rmsnorm(const std::vector<float>& x, const Dense& w, float eps) {
    double ss = 0.0;
    for (float v : x) ss += double(v) * v;
    const float inv = float(1.0 / std::sqrt(ss / x.size() + eps));
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * w.w[i];
    return y;
}

void rope(std::vector<float>& v, int heads, int hd, int pos, float base) {
    for (int h = 0; h < heads; ++h)
        for (int i = 0; i < hd / 2; ++i) {
            const double theta = pos * std::pow(double(base), -2.0 * i / hd);
            const float c = float(std::cos(theta)), s = float(std::sin(theta));
            float& a = v[std::size_t(h) * hd + 2 * i];
            float& b = v[std::size_t(h) * hd + 2 * i + 1];
            const float a0 = a, b0 = b;
            a = a0 * c - b0 * s;
            b = a0 * s + b0 * c;
        }
}

}  // namespace

std::vector<float> reference_logits(const GgufFile& f, const std::vector<int>& tokens) {
    if (tokens.empty()) throw std::invalid_argument("reference: no tokens");
    const int L = int(*f.meta_int("llama.block_count"));
    const int D = int(*f.meta_int("llama.embedding_length"));
    const int H = int(*f.meta_int("llama.attention.head_count"));
    const int KV = int(*f.meta_int("llama.attention.head_count_kv"));
    const int HD = D / H;
    const float base = float(f.meta_float("llama.rope.freq_base").value_or(10000.0));
    const float eps = float(f.meta_float("llama.attention.layer_norm_rms_epsilon").value_or(1e-5));
    const int T = int(tokens.size());

    // Embeddings for every position.
    std::vector<std::vector<float>> xs(T, std::vector<float>(D));
    {
        const auto* e = f.find("token_embd.weight");
        const auto* src = static_cast<const std::uint8_t*>(f.tensor_data(*e));
        for (int t = 0; t < T; ++t)
            dequantize_q4_0(src + std::size_t(tokens[t]) * (D / 32) * 18, D, xs[t].data());
    }

    // Layer by layer across all positions, so each layer's weights are
    // dequantized once and freed before the next.
    for (int l = 0; l < L; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const Dense an = load(f, p + "attn_norm.weight");
        const Dense wq = load(f, p + "attn_q.weight");
        const Dense wk = load(f, p + "attn_k.weight");
        const Dense wv = load(f, p + "attn_v.weight");
        const Dense wo = load(f, p + "attn_output.weight");

        std::vector<std::vector<float>> K(T), V(T), Q(T);
        for (int t = 0; t < T; ++t) {
            const auto h = rmsnorm(xs[t], an, eps);
            Q[t] = mul(wq, h);
            K[t] = mul(wk, h);
            V[t] = mul(wv, h);
            rope(Q[t], H, HD, t, base);
            rope(K[t], KV, HD, t, base);
        }
        const float scale = 1.0f / std::sqrt(float(HD));
        for (int t = 0; t < T; ++t) {
            std::vector<float> att(D, 0.0f);
            for (int h = 0; h < H; ++h) {
                const int kvh = h / (H / KV);
                std::vector<double> logit(t + 1);
                double mx = -1e300;
                for (int j = 0; j <= t; ++j) {
                    double s = 0.0;
                    for (int c = 0; c < HD; ++c)
                        s += double(Q[t][std::size_t(h) * HD + c]) * K[j][std::size_t(kvh) * HD + c];
                    logit[j] = s * scale;
                    mx = std::max(mx, logit[j]);
                }
                double z = 0.0;
                for (double v : logit) z += std::exp(v - mx);
                for (int j = 0; j <= t; ++j) {
                    const float wgt = float(std::exp(logit[j] - mx) / z);
                    for (int c = 0; c < HD; ++c)
                        att[std::size_t(h) * HD + c] += wgt * V[j][std::size_t(kvh) * HD + c];
                }
            }
            const auto o = mul(wo, att);
            for (int i = 0; i < D; ++i) xs[t][i] += o[i];
        }

        const Dense fn = load(f, p + "ffn_norm.weight");
        const Dense wg = load(f, p + "ffn_gate.weight");
        const Dense wu = load(f, p + "ffn_up.weight");
        const Dense wd = load(f, p + "ffn_down.weight");
        for (int t = 0; t < T; ++t) {
            const auto h = rmsnorm(xs[t], fn, eps);
            auto g = mul(wg, h);
            const auto u = mul(wu, h);
            for (std::size_t i = 0; i < g.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
            const auto d = mul(wd, g);
            for (int i = 0; i < D; ++i) xs[t][i] += d[i];
        }
    }

    const Dense on = load(f, "output_norm.weight");
    const auto hidden = rmsnorm(xs[T - 1], on, eps);
    const Dense ow = load(f, "output.weight");
    return mul(ow, hidden);
}

}  // namespace llm
