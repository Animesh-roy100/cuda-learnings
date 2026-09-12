// Runtime benchmark: quality, single-sequence speed, batched throughput, and
// where a step's time goes.

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "generator.h"
#include "runtime.h"

namespace {

// Public domain: the opening of "Alice's Adventures in Wonderland" (1865).
const char* kPassage =
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of "
    "having nothing to do: once or twice she had peeped into the book her sister was "
    "reading, but it had no pictures or conversations in it, \"and what is the use of a "
    "book,\" thought Alice \"without pictures or conversations?\" So she was considering "
    "in her own mind (as well as she could, for the hot day made her feel very sleepy and "
    "stupid), whether the pleasure of making a daisy-chain would be worth the trouble of "
    "getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran "
    "close by her.";

const char* kPrompt = "Explain in three sentences why GPUs are good at matrix multiplication.";

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) try {
    const std::string path = argc > 1 ? argv[1] : llm::default_model_path();
    auto model = llm::GgufModel::open(path);

    // ---------------------------------------------------- quality and speed
    struct Row {
        const char* name;
        llm::Activations act;
        bool graphs;
    };
    const Row rows[] = {
        {"float (W4A16), no graphs", llm::Activations::Float, false},
        {"int8  (W4A8),  no graphs", llm::Activations::Int8, false},
        {"int8  (W4A8),  CUDA graphs", llm::Activations::Int8, true},
    };
    std::printf("%-28s %11s %9s %13s %13s\n", "configuration", "perplexity", "ttft ms", "prefill tok/s",
                "decode tok/s");
    std::string sample;
    for (const auto& r : rows) {
        llm::RuntimeOptions o;
        o.activations = r.act;
        o.cuda_graphs = r.graphs;
        o.max_batch = 4;
        llm::Runtime rt(model, o);
        if (&r == &rows[0]) {
            const auto& d = rt.device();
            std::printf("[%s sm_%d, driver %s, CUDA runtime %d.%d]\n", d.name.c_str(), d.compute_capability,
                        d.driver_release.c_str(), d.runtime_version / 1000, d.runtime_version % 1000 / 10);
        }
        const double ppl = llm::perplexity(rt, kPassage);
        llm::GenerationRequest req{llm::chat_prompt(kPrompt), 8, {}};
        llm::generate(rt, req);   // warm-up, including graph capture
        req.max_new_tokens = 96;
        const auto res = llm::generate(rt, req);
        if (r.graphs) sample = res.text;
        std::printf("%-28s %11.3f %9.1f %13.1f %13.1f\n", r.name, ppl, res.metrics.time_to_first_token_ms,
                    res.metrics.prefill_tokens_per_s(), res.metrics.decode_tokens_per_s());
    }

    // ------------------------------------------------------ batched throughput
    {
        llm::RuntimeOptions o;
        o.max_batch = 4;
        llm::Runtime rt(model, o);
        std::printf("\nBatched decode throughput (int8, CUDA graphs, position 128):\n");
        std::printf("  %5s %12s %16s %14s\n", "batch", "ms / step", "tokens/s total", "per sequence");
        for (int b : {1, 2, 4}) {
            std::vector<std::unique_ptr<llm::Sequence>> seqs;
            std::vector<llm::Sequence*> ptrs;
            for (int i = 0; i < b; ++i) {
                seqs.push_back(rt.new_sequence());
                ptrs.push_back(seqs.back().get());
            }
            const std::vector<int> toks(b, 1);
            for (int i = 0; i < 128; ++i) rt.step(ptrs, toks);
            const auto t0 = std::chrono::steady_clock::now();
            const int steps = 32;
            for (int i = 0; i < steps; ++i) rt.step(ptrs, toks);
            const double ms = ms_since(t0) / steps;
            std::printf("  %5d %9.2f ms %16.1f %14.1f\n", b, ms, 1000.0 * b / ms, 1000.0 / ms);
        }
    }

    // -------------------------------------------------------- stage breakdown
    {
        llm::RuntimeOptions o;
        o.max_batch = 1;
        o.profile_stages = true;
        llm::Runtime rt(model, o);
        auto seq = rt.new_sequence();
        for (int i = 0; i < 128; ++i) rt.step({seq.get()}, {1});
        rt.reset_stage_times();
        for (int i = 0; i < 32; ++i) rt.step({seq.get()}, {1});
        const auto t = rt.stage_times();
        const double total = t.embed_upload + t.norms + t.projections + t.rope_and_cache + t.attention +
                             t.activation + t.logits + t.download;
        std::printf("\nWhere a step goes (batch 1, position ~128, profiled -- graphs off):\n");
        auto line = [&](const char* name, double v) {
            std::printf("  %-22s %8.3f ms  %5.1f%%\n", name, v / t.steps, 100.0 * v / total);
        };
        line("Q4 projections", t.projections);
        line("attention (paged)", t.attention);
        line("activations / adds", t.activation);
        line("norms", t.norms);
        line("RoPE + KV write", t.rope_and_cache);
        line("logits (cuBLAS)", t.logits);
        line("embed + upload", t.embed_upload);
        line("download", t.download);
    }

    std::printf("\nGreedy sample (int8, CUDA graphs):\n> %s\n\n%s\n", kPrompt, sample.c_str());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
