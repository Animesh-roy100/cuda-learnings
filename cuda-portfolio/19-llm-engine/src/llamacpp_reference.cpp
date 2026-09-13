#include "llamacpp_reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>

#include "errors.h"

namespace llm {
namespace {

[[noreturn]] void invalid(const std::string& why) {
    ErrorContext c;
    c.operation = "read llama.cpp reference";
    throw Error(ErrorKind::InvalidArgument, why, c);
}

std::int32_t read_i32(const std::vector<std::uint8_t>& b, std::size_t at) {
    std::int32_t v;
    std::memcpy(&v, b.data() + at, 4);
    return v;
}

}  // namespace

LlamaCppReference LlamaCppReference::read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) invalid("cannot open " + path);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return parse(std::move(bytes));
}

LlamaCppReference LlamaCppReference::parse(std::vector<std::uint8_t> bytes) {
    if (bytes.size() < 20 || std::memcmp(bytes.data(), "_logits_", 8) != 0)
        invalid("not a llama.cpp logits file (magic \"_logits_\" missing)");
    LlamaCppReference r;
    r.n_ctx_ = read_i32(bytes, 8);
    r.n_vocab_ = read_i32(bytes, 12);
    r.n_chunk_ = read_i32(bytes, 16);
    if (r.n_ctx_ < 4 || r.n_ctx_ > (1 << 20)) invalid("implausible n_ctx " + std::to_string(r.n_ctx_));
    if (r.n_vocab_ < 2 || r.n_vocab_ > (1 << 22)) invalid("implausible n_vocab " + std::to_string(r.n_vocab_));
    if (r.n_chunk_ < 1 || r.n_chunk_ > (1 << 20)) invalid("implausible n_chunk " + std::to_string(r.n_chunk_));

    const std::uint64_t n_tokens = std::uint64_t(r.n_chunk_) * std::uint64_t(r.n_ctx_);
    const std::uint64_t nv = 2 * ((std::uint64_t(r.n_vocab_) + 1) / 2) + 4;
    const std::uint64_t expected =
        20 + n_tokens * 4 + std::uint64_t(r.n_chunk_) * std::uint64_t(r.records_per_chunk()) * nv * 2;
    if (bytes.size() != expected)
        invalid("file is " + std::to_string(bytes.size()) + " bytes; its header (n_ctx " + std::to_string(r.n_ctx_) +
                ", n_vocab " + std::to_string(r.n_vocab_) + ", n_chunk " + std::to_string(r.n_chunk_) +
                ") implies " + std::to_string(expected));

    r.tokens_.resize(std::size_t(n_tokens));
    for (std::size_t i = 0; i < r.tokens_.size(); ++i) {
        r.tokens_[i] = read_i32(bytes, 20 + 4 * i);
        if (r.tokens_[i] < 0 || r.tokens_[i] >= r.n_vocab_)
            invalid("token " + std::to_string(i) + " is " + std::to_string(r.tokens_[i]) + ", outside the vocabulary");
    }
    r.records_offset_ = std::size_t(20 + n_tokens * 4);
    r.bytes_ = std::move(bytes);
    return r;
}

std::vector<int> LlamaCppReference::chunk_input(int chunk, int bos) const {
    if (chunk < 0 || chunk >= n_chunk_) invalid("chunk " + std::to_string(chunk) + " out of range");
    std::vector<int> t(tokens_.begin() + std::ptrdiff_t(chunk) * n_ctx_,
                       tokens_.begin() + std::ptrdiff_t(chunk + 1) * n_ctx_);
    t[0] = bos;
    return t;
}

void LlamaCppReference::log_probs(int chunk, int record, std::vector<float>& out) const {
    if (chunk < 0 || chunk >= n_chunk_ || record < 0 || record >= records_per_chunk())
        invalid("record " + std::to_string(chunk) + "/" + std::to_string(record) + " out of range");
    const std::size_t nv = 2 * ((std::size_t(n_vocab_) + 1) / 2) + 4;
    const std::size_t at = records_offset_ + (std::size_t(chunk) * records_per_chunk() + record) * nv * 2;
    float scale, min_log_prob;
    std::memcpy(&scale, bytes_.data() + at, 4);
    std::memcpy(&min_log_prob, bytes_.data() + at + 4, 4);
    out.resize(n_vocab_);
    for (int i = 0; i < n_vocab_; ++i) {
        std::uint16_t q;
        std::memcpy(&q, bytes_.data() + at + 8 + 2 * std::size_t(i), 2);
        out[i] = scale * q + min_log_prob;
    }
}

void KlDivergence::add(const float* logits, const std::vector<float>& base, int tok) {
    const int n = static_cast<int>(base.size());
    if (tok < 0 || tok >= n) invalid("next token out of range");
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n; ++i)
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) sum_exp += std::exp(double(logits[i]) - max_logit);
    const double log_z = max_logit + std::log(sum_exp);   // log-softmax(x)_i = x_i - log_z

    double kld = 0.0;
    int imax_base = 0;
    for (int i = 0; i < n; ++i) {
        if (base[i] > base[imax_base]) imax_base = i;
        if (base[i] > -16.0f) kld += std::exp(double(base[i])) * (double(base[i]) - (double(logits[i]) - log_z));
    }
    const double nll = log_z - logits[tok];
    const double nll_base = -double(base[tok]);
    const double p_diff = std::exp(-nll) - std::exp(-nll_base);

    ++count_;
    same_top_ += imax == imax_base;
    sum_kld_ += kld;
    max_kld_ = std::max(max_kld_, kld);
    klds_.push_back(kld);
    sum_nll_ += nll;
    sum_nll_base_ += nll_base;
    sum_p_diff2_ += p_diff * p_diff;
    max_p_diff_ = std::max(max_p_diff_, std::fabs(p_diff));
}

double KlDivergence::kld_quantile(double q) const {
    if (klds_.empty()) return 0.0;
    std::vector<double> s = klds_;
    std::sort(s.begin(), s.end());
    const std::size_t i = std::min(s.size() - 1, std::size_t(std::clamp(q, 0.0, 1.0) * double(s.size() - 1) + 0.5));
    return s[i];
}

double KlDivergence::ppl_ours() const { return count_ ? std::exp(sum_nll_ / count_) : 0.0; }
double KlDivergence::ppl_base() const { return count_ ? std::exp(sum_nll_base_ / count_) : 0.0; }
double KlDivergence::rms_p_diff() const { return count_ ? std::sqrt(sum_p_diff2_ / count_) : 0.0; }

}  // namespace llm
