// Merge-rank BPE over UTF-8 characters, with byte fallback.

#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "gguf.h"

namespace llm {
namespace {

const std::string kSpace = "\xE2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK

constexpr std::int64_t kTypeControl = 3;
constexpr std::int64_t kTypeByte = 6;

// Length of the UTF-8 sequence starting with byte c. Malformed lead bytes are
// treated as single bytes so they reach byte fallback instead of being dropped.
std::size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

}  // namespace

Tokenizer::Tokenizer(const GgufFile& f) {
    const auto* tokens = f.meta_string_array("tokenizer.ggml.tokens");
    const auto* types = f.meta_int_array("tokenizer.ggml.token_type");
    const auto* merges = f.meta_string_array("tokenizer.ggml.merges");
    if (!tokens || tokens->empty()) throw std::runtime_error("tokenizer: no vocabulary in GGUF");
    if (!merges) throw std::runtime_error("tokenizer: no merges in GGUF (score-based vocab not supported)");

    pieces_ = *tokens;
    types_ = types ? *types : std::vector<std::int64_t>(pieces_.size(), 1);
    if (types_.size() != pieces_.size()) throw std::runtime_error("tokenizer: token_type length mismatch");

    ids_.reserve(pieces_.size());
    for (int i = 0; i < static_cast<int>(pieces_.size()); ++i) ids_.emplace(pieces_[i], i);

    merge_rank_.reserve(merges->size());
    for (int r = 0; r < static_cast<int>(merges->size()); ++r) merge_rank_.emplace((*merges)[r], r);

    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        auto it = ids_.find(name);
        if (it == ids_.end()) throw std::runtime_error(std::string("tokenizer: missing byte token ") + name);
        byte_token_[b] = it->second;
    }
    if (auto v = f.meta_int("tokenizer.ggml.bos_token_id")) bos_ = static_cast<int>(*v);
    if (auto v = f.meta_int("tokenizer.ggml.eos_token_id")) eos_ = static_cast<int>(*v);
}

std::vector<int> Tokenizer::encode_plain(const std::string& text) const {
    if (text.empty()) return {};
    // SentencePiece normalization: dummy prefix, spaces to U+2581.
    std::string norm = kSpace;
    for (char c : text) norm += (c == ' ') ? kSpace : std::string(1, c);

    std::vector<std::string> sym;
    for (std::size_t i = 0; i < norm.size();) {
        std::size_t n = std::min(utf8_len(static_cast<unsigned char>(norm[i])), norm.size() - i);
        sym.emplace_back(norm.substr(i, n));
        i += n;
    }

    // Repeatedly apply the lowest-ranked merge present anywhere in the sequence.
    // Quadratic in the number of symbols, which is fine for prompt-length text
    // and makes the rank semantics impossible to get subtly wrong.
    while (sym.size() > 1) {
        int best = INT_MAX;
        std::size_t at = 0;
        for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_rank_.end() && it->second < best) {
                best = it->second;
                at = i;
            }
        }
        if (best == INT_MAX) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + static_cast<std::ptrdiff_t>(at) + 1);
    }

    std::vector<int> out;
    out.reserve(sym.size());
    for (const auto& s : sym) {
        auto it = ids_.find(s);
        if (it != ids_.end()) {
            out.push_back(it->second);
        } else {
            for (unsigned char b : s) out.push_back(byte_token_[b]);
        }
    }
    return out;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> out;
    if (add_bos) out.push_back(bos_);
    // Split on literal control-token spellings, which chat templates emit as
    // text. Text between them is encoded normally.
    std::size_t start = 0;
    while (start <= text.size()) {
        std::size_t best_pos = std::string::npos;
        int best_id = -1;
        std::size_t best_len = 0;
        for (int id : {bos_, eos_}) {
            const auto& p = pieces_[id];
            std::size_t pos = text.find(p, start);
            if (pos != std::string::npos && pos < best_pos) {
                best_pos = pos;
                best_id = id;
                best_len = p.size();
            }
        }
        if (best_pos == std::string::npos) {
            auto tail = encode_plain(text.substr(start));
            out.insert(out.end(), tail.begin(), tail.end());
            break;
        }
        auto part = encode_plain(text.substr(start, best_pos - start));
        out.insert(out.end(), part.begin(), part.end());
        out.push_back(best_id);
        // Each segment after a control token gets its own dummy prefix, as
        // llama.cpp's SentencePiece session does.
        start = best_pos + best_len;
    }
    return out;
}

std::string Tokenizer::piece(int id) const {
    if (id < 0 || id >= vocab_size()) throw std::out_of_range("tokenizer: token id out of range");
    if (types_[id] == kTypeControl) return "";
    const std::string& p = pieces_[id];
    if (types_[id] == kTypeByte && p.size() == 6) {
        return std::string(1, static_cast<char>(std::stoi(p.substr(3, 2), nullptr, 16)));
    }
    std::string s;
    for (std::size_t i = 0; i < p.size();) {
        if (p.compare(i, kSpace.size(), kSpace) == 0) {
            s += ' ';
            i += kSpace.size();
        } else {
            s += p[i++];
        }
    }
    return s;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) s += piece(id);
    // Undo the dummy prefix: the first real token of a segment starts with the
    // space SentencePiece added.
    if (!s.empty() && s[0] == ' ') s.erase(0, 1);
    return s;
}

}  // namespace llm
