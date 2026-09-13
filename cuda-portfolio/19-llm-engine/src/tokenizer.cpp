// Merge-rank BPE over UTF-8 characters, with byte fallback.

#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <functional>
#include <queue>
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

Tokenizer::Tokenizer(const GgufFile& f, Algorithm algorithm) {
    const auto* tokens = f.meta_string_array("tokenizer.ggml.tokens");
    const auto* types = f.meta_int_array("tokenizer.ggml.token_type");
    const auto* merges = f.meta_string_array("tokenizer.ggml.merges");
    const auto* scores = f.meta_float_array("tokenizer.ggml.scores");
    if (!tokens || tokens->empty()) throw std::runtime_error("tokenizer: no vocabulary in GGUF");
    if (algorithm == Algorithm::Auto) algorithm = merges ? Algorithm::Merges : Algorithm::Scores;
    if (algorithm == Algorithm::Merges && !merges)
        throw std::runtime_error("tokenizer: merge order requested but the GGUF has no tokenizer.ggml.merges");
    if (algorithm == Algorithm::Scores && !scores)
        throw std::runtime_error("tokenizer: score order requested but the GGUF has no tokenizer.ggml.scores");
    algorithm_ = algorithm;

    pieces_ = *tokens;
    types_ = types ? *types : std::vector<std::int64_t>(pieces_.size(), 1);
    if (types_.size() != pieces_.size()) throw std::runtime_error("tokenizer: token_type length mismatch");
    if (scores) {
        scores_ = *scores;
        if (scores_.size() != pieces_.size()) throw std::runtime_error("tokenizer: scores length mismatch");
    }

    ids_.reserve(pieces_.size());
    for (int i = 0; i < static_cast<int>(pieces_.size()); ++i) ids_.emplace(pieces_[i], i);

    if (merges) {
        merge_rank_.reserve(merges->size());
        for (int r = 0; r < static_cast<int>(merges->size()); ++r) merge_rank_.emplace((*merges)[r], r);
    }

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

    // Symbols are spans of `norm` in a doubly linked list; a merged-away
    // symbol has length 0.
    struct Sym {
        std::size_t start, len;
        int prev, next;
    };
    std::vector<Sym> sym;
    for (std::size_t i = 0; i < norm.size();) {
        std::size_t n = std::min(utf8_len(static_cast<unsigned char>(norm[i])), norm.size() - i);
        const int k = static_cast<int>(sym.size());
        sym.push_back({i, n, k - 1, k + 1});
        i += n;
    }
    sym.back().next = -1;

    // Always apply the best merge present anywhere in the sequence, the
    // leftmost on a tie. Merges: best is the lowest merge rank. Scores: a pair
    // is mergeable when its concatenation is a vocabulary entry, and best is
    // that entry's highest score. A priority queue of candidate pairs keyed
    // (key, left symbol) gives exactly that order in O(n log n); candidates
    // made stale by an earlier merge are recognized by their recorded length
    // and skipped. (A rescan per merge is simpler and quadratic: minutes for a
    // few thousand tokens, and a stall on any long prompt.)
    struct Cand {
        double key;   // lower merges first
        int left, right;
        std::size_t len;
        bool operator>(const Cand& o) const { return key != o.key ? key > o.key : left > o.left; }
    };
    std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> queue;
    auto consider = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        const std::size_t len = sym[l].len + sym[r].len;
        if (algorithm_ == Algorithm::Merges) {
            auto it = merge_rank_.find(norm.substr(sym[l].start, sym[l].len) + " " +
                                       norm.substr(sym[r].start, sym[r].len));
            if (it != merge_rank_.end()) queue.push({double(it->second), l, r, len});
        } else {
            auto it = ids_.find(norm.substr(sym[l].start, len));
            if (it != ids_.end()) queue.push({-scores_[it->second], l, r, len});
        }
    };
    for (int i = 0; i + 1 < static_cast<int>(sym.size()); ++i) consider(i, i + 1);

    while (!queue.empty()) {
        const Cand c = queue.top();
        queue.pop();
        Sym& l = sym[c.left];
        Sym& r = sym[c.right];
        if (l.len == 0 || r.len == 0 || l.next != c.right || l.len + r.len != c.len) continue;
        l.len += r.len;
        r.len = 0;
        l.next = r.next;
        if (r.next >= 0) sym[r.next].prev = c.left;
        consider(l.prev, c.left);
        consider(c.left, l.next);
    }

    std::vector<int> out;
    for (int i = 0; i >= 0; i = sym[i].next) {
        const std::string s = norm.substr(sym[i].start, sym[i].len);
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
