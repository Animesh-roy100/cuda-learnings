// Generate replies with TinyLlama-Chat on the production runtime.
//
//   chat "Explain what a GPU kernel is in two sentences."
//   chat --temp 0.7 --max 200 "Write a haiku about memory bandwidth."
//   chat --json "..."                       # also print the metrics record
//   chat "first question" "second question" # both prompts batched together
//   chat --model path/to.gguf --float --no-graphs --context 512 "..."

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "generator.h"
#include "runtime.h"

int main(int argc, char** argv) try {
    llm::RuntimeOptions opt;
    opt.max_batch = 4;
    llm::SamplingParams sampling;
    std::string model = llm::default_model_path();
    std::vector<std::string> prompts;
    int max_new = 128;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--max") max_new = std::stoi(next());
        else if (a == "--temp") sampling.temperature = std::stof(next());
        else if (a == "--top-p") sampling.top_p = std::stof(next());
        else if (a == "--seed") sampling.seed = std::stoull(next());
        else if (a == "--context") opt.context = std::stoi(next());
        else if (a == "--float") opt.activations = llm::Activations::Float;
        else if (a == "--no-graphs") opt.cuda_graphs = false;
        else if (a == "--json") json = true;
        else prompts.push_back(a);
    }
    if (prompts.empty()) prompts.push_back("What is a GPU, in two sentences?");
    if (int(prompts.size()) > opt.max_batch) opt.max_batch = int(prompts.size());

    auto m = llm::GgufModel::open(model);
    llm::Runtime rt(m, opt);
    const auto& d = rt.device();
    const auto& c = rt.config();
    std::printf("%s  sm_%d  driver %s (CUDA %d.%d)  runtime CUDA %d.%d\n", d.name.c_str(), d.compute_capability,
                d.driver_release.c_str(), d.driver_version / 1000, d.driver_version % 1000 / 10,
                d.runtime_version / 1000, d.runtime_version % 1000 / 10);
    std::printf("%s: %d layers, dim %d, %d heads / %d kv -- weights %.0f MB, KV cache %.0f MB\n",
                model.c_str(), c.layers, c.hidden_size, c.attention_heads, c.kv_heads,
                rt.memory().weights / 1048576.0, rt.memory().kv_cache / 1048576.0);
    std::printf("activations: %s, CUDA graphs: %s\n\n",
                rt.activations() == llm::Activations::Int8 ? "int8 (W4A8)" : "float (W4A16)",
                opt.cuda_graphs ? "on" : "off");

    if (prompts.size() == 1) {
        llm::GenerationRequest req{llm::chat_prompt(prompts[0]), max_new, sampling};
        std::printf("> %s\n\n", prompts[0].c_str());
        bool first = true;
        const auto r = llm::generate(rt, req, [&](int, const std::string& piece) {
            std::fputs(first && !piece.empty() && piece[0] == ' ' ? piece.c_str() + 1 : piece.c_str(), stdout);
            first = false;
            std::fflush(stdout);
            return true;
        });
        const auto& mt = r.metrics;
        std::printf("\n\n[ttft %.0f ms | prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s | %s]\n",
                    mt.time_to_first_token_ms, mt.prompt_tokens, mt.prefill_tokens_per_s(),
                    mt.generated_tokens, mt.decode_tokens_per_s(), llm::to_string(mt.finish));
        if (json) std::printf("%s\n", llm::metrics_json(rt, mt).c_str());
    } else {
        std::vector<llm::GenerationRequest> reqs;
        for (const auto& p : prompts) reqs.push_back({llm::chat_prompt(p), max_new, sampling});
        const auto results = llm::generate_batch(rt, reqs);
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& mt = results[i].metrics;
            std::printf("> %s\n\n%s\n\n[ttft %.0f ms | decode %d tok %.1f tok/s | %s]\n\n", prompts[i].c_str(),
                        results[i].text.c_str(), mt.time_to_first_token_ms, mt.generated_tokens,
                        mt.decode_tokens_per_s(), llm::to_string(mt.finish));
            if (json) std::printf("%s\n", llm::metrics_json(rt, mt).c_str());
        }
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "\nFATAL: %s\n", e.what());
    return 1;
}
