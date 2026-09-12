#pragma once
//
// Llama tokenizer, read from GGUF metadata. Host-only.
//
// The TinyLlama GGUF used here stores a score of 0 for all 32000 vocabulary
// entries and ships 61249 BPE merges instead -- so a SentencePiece-style
// encoder that picks merges by score would pick arbitrarily. This one merges by
// RANK: the earlier a merge appears in tokenizer.ggml.merges, the sooner it is
// applied. Merge 0 is "▁ t", which produces token 260 "▁t".
//
// Conventions, all taken from the file rather than assumed:
//   * spaces become U+2581 "▁", and one is prepended (SentencePiece's
//     add-dummy-prefix)
//   * a character with no vocabulary entry falls back to <0xXX> byte tokens,
//     of which the vocabulary has exactly 256 (token_type 6)
//   * control tokens (<s>, </s>) are recognized in the input text by their
//     literal spelling, because chat templates write "</s>" as text
//
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

class GgufFile;

class Tokenizer {
public:
    explicit Tokenizer(const GgufFile& f);

    int vocab_size() const { return static_cast<int>(pieces_.size()); }
    int bos() const { return bos_; }
    int eos() const { return eos_; }

    std::vector<int> encode(const std::string& text, bool add_bos) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string piece(int id) const;   // one token's text, spaces restored

private:
    std::vector<int> encode_plain(const std::string& text) const;

    std::vector<std::string> pieces_;
    std::vector<std::int64_t> types_;
    std::unordered_map<std::string, int> ids_;
    std::unordered_map<std::string, int> merge_rank_;   // "left right" -> rank
    int byte_token_[256] = {};
    int bos_ = 1, eos_ = 2;
};

}  // namespace llm
