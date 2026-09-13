#pragma once
//
// The trusted end-to-end reference: log-probabilities written by llama.cpp.
//
//   llama-perplexity -m model.gguf -f text -c 512 -b 512 --chunks 8 \
//                    --kl-divergence-base reference.kld
//
// The file is llama.cpp's own format for validating a runtime against a base
// (tools/perplexity/perplexity.cpp, build b10932):
//
//   "_logits_" | int32 n_ctx | int32 n_vocab | int32 n_chunk
//   int32 tokens[n_chunk * n_ctx]              -- llama.cpp's tokenization, BOS first
//   n_chunk x (n_ctx - 1 - n_ctx/2) records, each nv = 2*((n_vocab+1)/2) + 4 uint16:
//       float scale | float min_log_prob | uint16 q[n_vocab]
//       log p[i] = scale * q[i] + min_log_prob   (q == 0 means "at or below the floor")
//
// Record r of a chunk holds the distribution after position n_ctx/2 + r, which
// predicts token n_ctx/2 + r + 1. Each chunk is evaluated from an empty cache
// with its first token replaced by BOS. Host-only.
//
#include <cstdint>
#include <string>
#include <vector>

namespace llm {

class LlamaCppReference {
public:
    // Validates the header and that the file is exactly as long as it says.
    static LlamaCppReference read(const std::string& path);
    static LlamaCppReference parse(std::vector<std::uint8_t> bytes);   // tests

    int n_ctx() const { return n_ctx_; }
    int n_vocab() const { return n_vocab_; }
    int n_chunk() const { return n_chunk_; }
    int first() const { return n_ctx_ / 2; }
    int records_per_chunk() const { return n_ctx_ - 1 - n_ctx_ / 2; }

    const std::vector<int>& tokens() const { return tokens_; }
    // The tokens the model saw for one chunk: BOS at position 0.
    std::vector<int> chunk_input(int chunk, int bos) const;

    // Decoded log-probabilities for one record; entries at the floor are the
    // floor value (<= -16 nats below the max, which llama.cpp ignores).
    void log_probs(int chunk, int record, std::vector<float>& out) const;

private:
    int n_ctx_ = 0, n_vocab_ = 0, n_chunk_ = 0;
    std::vector<int> tokens_;
    std::vector<std::uint8_t> bytes_;
    std::size_t records_offset_ = 0;
};

// Accumulates the statistics llama.cpp reports with --kl-divergence, computed
// the same way: KL(base || ours) over base entries above -16 nats, top-token
// agreement, both perplexities, and the probability each assigns the true
// next token.
class KlDivergence {
public:
    void add(const float* our_logits, const std::vector<float>& base_log_probs, int next_token);

    int count() const { return count_; }
    double mean_kld() const { return count_ ? sum_kld_ / count_ : 0.0; }
    double max_kld() const { return max_kld_; }
    double kld_quantile(double q) const;       // q in [0, 1]
    double same_top_fraction() const { return count_ ? double(same_top_) / count_ : 0.0; }
    double ppl_ours() const;
    double ppl_base() const;
    double rms_p_diff() const;                 // sqrt(mean((p_ours - p_base)^2)) of the true token
    double max_p_diff() const { return max_p_diff_; }

private:
    int count_ = 0, same_top_ = 0;
    double sum_kld_ = 0, max_kld_ = 0, sum_nll_ = 0, sum_nll_base_ = 0, sum_p_diff2_ = 0, max_p_diff_ = 0;
    std::vector<double> klds_;
};

}  // namespace llm
