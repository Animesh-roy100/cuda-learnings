// End-to-end comparison against llama.cpp: the same model, the same text, the
// same chunks -- does this runtime produce the distributions llama.cpp does?
//
//   llama-perplexity -m model.gguf -f wiki.test.raw -c 512 -b 512 --chunks 8 \
//                    --kl-divergence-base reference.kld
//   compare_llamacpp --reference reference.kld --text wiki.test.raw \
//                    [--model model.gguf] [--activations float|int8|both] [--chunks N]
//                    [--max-mean-kld 0.01] [--min-same-top 0.95]
//
// Reports tokenizer agreement on the text, then per activation mode: mean,
// median, p99 and max KL divergence, top-token agreement, both perplexities,
// and the RMS difference in the probability of the true next token. Exits 1
// if a threshold is missed, so it can gate a release.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "llamacpp_reference.h"
#include "runtime.h"

namespace {

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

int main(int argc, char** argv) try {
    std::string reference, text_path, model_path = llm::default_model_path(), modes = "both";
    int chunks = 0;
    double max_mean_kld = 0.01, min_same_top = 0.95;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--reference") reference = next();
        else if (a == "--text") text_path = next();
        else if (a == "--model") model_path = next();
        else if (a == "--activations") modes = next();
        else if (a == "--chunks") chunks = std::stoi(next());
        else if (a == "--max-mean-kld") max_mean_kld = std::stod(next());
        else if (a == "--min-same-top") min_same_top = std::stod(next());
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (reference.empty()) throw std::invalid_argument("--reference is required");

    const auto ref = llm::LlamaCppReference::read(reference);
    auto model = llm::GgufModel::open(model_path);
    const auto& tok = model->tokenizer();
    std::printf("reference: n_ctx %d, n_vocab %d, %d chunks (%d scored tokens each)\n", ref.n_ctx(), ref.n_vocab(),
                ref.n_chunk(), ref.records_per_chunk());
    if (ref.n_vocab() < model->config().vocab_size)
        throw std::runtime_error("reference vocabulary is smaller than the model's");
    if (chunks <= 0 || chunks > ref.n_chunk()) chunks = ref.n_chunk();
    bool ok = true;

    // ------------------------------------------------------------ tokenizer
    // llama.cpp tokenizes a "llama" GGUF by vocabulary score and ignores the
    // merges; this runtime defaults to merge order, which is the order of the
    // tokenizer the model was trained with. Score order must reproduce
    // llama.cpp exactly -- that is the parity gate. Merge order is reported.
    const llm::Tokenizer by_score(model->file(), llm::Tokenizer::Algorithm::Scores);
    if (!text_path.empty()) {
        // llama.cpp tokenized the whole file; the first n_chunk*n_ctx tokens
        // depend only on a prefix of it. Take a generous prefix, cut at a line.
        std::string text = read_text(text_path);
        std::size_t cut = std::min(text.size(), std::size_t(ref.n_chunk()) * ref.n_ctx() * 16);
        while (cut < text.size() && text[cut] != '\n') ++cut;
        const std::string prefix = text.substr(0, cut);
        for (const llm::Tokenizer* t : {&by_score, &tok}) {
            const bool gate = t == &by_score;
            const auto t0 = std::chrono::steady_clock::now();
            const auto ours = t->encode(prefix, true);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const std::size_t n = ref.tokens().size();
            std::size_t first_bad = n;
            for (std::size_t i = 0; i < n && first_bad == n; ++i)
                if (i >= ours.size() || ours[i] != ref.tokens()[i]) first_bad = i;
            std::printf("tokenizer (%s order): %zu bytes -> %zu tokens in %.0f ms; ", gate ? "score" : "merge", cut,
                        ours.size(), ms);
            if (first_bad == n) {
                std::printf("all %zu tokens identical to llama.cpp\n", n);
            } else {
                std::printf("identical to llama.cpp for %zu of %zu tokens, then ours %d \"%s\" vs %d \"%s\"%s\n",
                            first_bad, n, first_bad < ours.size() ? ours[first_bad] : -1,
                            first_bad < ours.size() ? tok.piece(ours[first_bad]).c_str() : "",
                            ref.tokens()[first_bad], tok.piece(ref.tokens()[first_bad]).c_str(),
                            gate ? "  FAIL" : "  (expected: different merge order)");
                if (gate) ok = false;
            }
        }
    }

    // ------------------------------------------------ which tokenization fits
    // Per-token perplexities are not comparable across tokenizations; the
    // likelihood of the same BYTES is. Encode each chunk's text both ways and
    // score the whole sequence with this runtime.
    {
        llm::RuntimeOptions o;
        o.context = 2 * ref.n_ctx();
        o.max_batch = 1;
        llm::Runtime rt(model, o);
        double nll[2] = {0, 0};
        std::size_t bytes = 0, count[2] = {0, 0};
        const int quality_chunks = std::min(chunks, 4);
        for (int c = 0; c < quality_chunks; ++c) {
            const auto begin = ref.tokens().begin() + std::ptrdiff_t(c) * ref.n_ctx() + 1;
            const std::string text = by_score.decode(std::vector<int>(begin, begin + ref.n_ctx() - 1));
            bytes += text.size();
            int k = 0;
            for (const llm::Tokenizer* t : {&tok, &by_score}) {
                const auto ids = t->encode(text, true);
                count[k] += ids.size() - 1;
                auto seq = rt.new_sequence();
                for (std::size_t p = 0; p + 1 < ids.size(); ++p) {
                    const auto logits = rt.step({seq.get()}, {ids[p]}).front();
                    const float mx = *std::max_element(logits.begin(), logits.end());
                    double z = 0;
                    for (float l : logits) z += std::exp(double(l) - mx);
                    nll[k] += mx + std::log(z) - logits[ids[p + 1]];
                }
                ++k;
            }
        }
        std::printf("\nsame text (%zu bytes, %d chunks) scored under each tokenization (int8):\n", bytes,
                    quality_chunks);
        const char* names[] = {"merge order (default)", "score order (llama.cpp)"};
        for (int k = 0; k < 2; ++k)
            std::printf("  %-24s %6zu tokens  %.4f bits/byte\n", names[k], count[k],
                        nll[k] / std::log(2.0) / double(bytes));
    }

    // ------------------------------------------------------------ logits
    std::vector<llm::Activations> acts;
    if (modes == "float" || modes == "both") acts.push_back(llm::Activations::Float);
    if (modes == "int8" || modes == "both") acts.push_back(llm::Activations::Int8);
    if (acts.empty()) throw std::invalid_argument("--activations must be float, int8 or both");

    std::printf("\n%-8s %6s %10s %10s %10s %10s %9s %9s %9s %9s\n", "mode", "tokens", "mean KLD", "median", "p99",
                "max", "same top", "PPL ours", "PPL l.cpp", "RMS dp");
    std::vector<float> base;
    for (auto act : acts) {
        llm::RuntimeOptions o;
        o.context = ref.n_ctx();
        o.max_batch = 1;
        o.activations = act;
        llm::Runtime rt(model, o);
        llm::KlDivergence kl;
        for (int c = 0; c < chunks; ++c) {
            const auto input = ref.chunk_input(c, tok.bos());
            auto seq = rt.new_sequence();
            for (int p = 0; p + 1 < ref.n_ctx(); ++p) {
                const auto logits = rt.step({seq.get()}, {input[p]});
                if (p < ref.first()) continue;
                ref.log_probs(c, p - ref.first(), base);
                kl.add(logits[0].data(), base, input[p + 1]);
            }
        }
        const bool pass = kl.mean_kld() <= max_mean_kld && kl.same_top_fraction() >= min_same_top;
        ok = ok && pass;
        std::printf("%-8s %6d %10.6f %10.6f %10.6f %10.6f %8.2f%% %9.4f %9.4f %8.4f%%  %s\n",
                    act == llm::Activations::Float ? "float" : "int8", kl.count(), kl.mean_kld(),
                    kl.kld_quantile(0.5), kl.kld_quantile(0.99), kl.max_kld(), 100.0 * kl.same_top_fraction(),
                    kl.ppl_ours(), kl.ppl_base(), 100.0 * kl.rms_p_diff(), pass ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
    std::printf("\nthresholds: mean KLD <= %g, same top >= %.1f%%\n", max_mean_kld, 100.0 * min_same_top);
    return ok ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FATAL: %s\n", e.what());
    return 2;
}
