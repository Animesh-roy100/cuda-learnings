#include "generator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

namespace llm {

const char* to_string(FinishReason r) {
    switch (r) {
        case FinishReason::Length: return "length";
        case FinishReason::EndOfSequence: return "end_of_sequence";
        case FinishReason::Cancelled: return "cancelled";
        case FinishReason::ContextFull: return "context_full";
    }
    return "?";
}

namespace {

using clock_type = std::chrono::steady_clock;
double ms_since(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// Per-request state inside a batch.
struct Active {
    const GenerationRequest* req = nullptr;
    GenerationResult* out = nullptr;
    std::unique_ptr<Sequence> seq;
    std::vector<int> prompt;
    std::size_t fed = 0;              // prompt tokens stepped so far
    int next_token = -1;              // sampled token waiting to be stepped
    std::unique_ptr<Sampler> sampler;
    bool done = false;
    clock_type::time_point start;
    clock_type::time_point decode_start;
};

}  // namespace

std::vector<GenerationResult> generate_batch(Runtime& rt, const std::vector<GenerationRequest>& reqs,
                                             const CancellationToken* cancel) {
    const auto& tok = rt.model().tokenizer();
    ErrorContext ctx;
    ctx.operation = "generate";
    if (int(reqs.size()) > rt.options().max_batch)
        throw Error(ErrorKind::InvalidArgument,
                    std::to_string(reqs.size()) + " requests exceed max_batch " +
                        std::to_string(rt.options().max_batch),
                    ctx);

    std::vector<GenerationResult> results(reqs.size());
    std::vector<Active> active(reqs.size());
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        auto& a = active[i];
        a.req = &reqs[i];
        a.out = &results[i];
        a.prompt = tok.encode(reqs[i].prompt, reqs[i].add_bos);
        if (a.prompt.empty()) throw Error(ErrorKind::InvalidArgument, "empty prompt", ctx);
        if (reqs[i].max_new_tokens < 0) throw Error(ErrorKind::InvalidArgument, "negative max_new_tokens", ctx);
        if (int(a.prompt.size()) > rt.options().context)
            throw Error(ErrorKind::ContextFull, "prompt of " + std::to_string(a.prompt.size()) +
                                                    " tokens exceeds the context length", ctx);
        a.sampler = std::make_unique<Sampler>(reqs[i].sampling);
        a.seq = rt.new_sequence();
        a.out->metrics.prompt_tokens = int(a.prompt.size());
        a.start = clock_type::now();
    }

    auto finish = [&](Active& a, FinishReason r) {
        if (a.done) return;
        a.done = true;
        a.out->metrics.finish = r;
        if (a.fed >= a.prompt.size()) a.out->metrics.decode_ms = ms_since(a.decode_start);
        a.seq.reset();   // pages back to the pool as soon as a request ends
        std::string text = tok.decode(a.out->tokens);
        a.out->text = std::move(text);
    };

    while (true) {
        std::vector<Active*> batch;
        for (auto& a : active)
            if (!a.done) batch.push_back(&a);
        if (batch.empty()) break;
        if (cancel && cancel->cancelled()) {
            for (auto* a : batch) finish(*a, FinishReason::Cancelled);
            break;
        }

        std::vector<Sequence*> seqs;
        std::vector<int> tokens;
        for (auto* a : batch) {
            if (a->seq->position() >= a->seq->capacity()) {
                finish(*a, FinishReason::ContextFull);
                continue;
            }
            seqs.push_back(a->seq.get());
            tokens.push_back(a->fed < a->prompt.size() ? a->prompt[a->fed] : a->next_token);
        }
        batch.erase(std::remove_if(batch.begin(), batch.end(), [](Active* a) { return a->done; }), batch.end());
        if (batch.empty()) continue;

        const auto logits = rt.step(seqs, tokens);

        for (std::size_t i = 0; i < batch.size(); ++i) {
            auto& a = *batch[i];
            const bool prefilling = a.fed < a.prompt.size();
            if (prefilling) {
                ++a.fed;
                if (a.fed < a.prompt.size()) continue;
                a.out->metrics.prefill_ms = ms_since(a.start);
                a.decode_start = clock_type::now();
            }
            if (a.out->metrics.generated_tokens >= a.req->max_new_tokens) {
                finish(a, FinishReason::Length);
                continue;
            }
            const int next = a.sampler->sample(logits[i]);
            if (a.out->metrics.generated_tokens == 0) a.out->metrics.time_to_first_token_ms = ms_since(a.start);
            if (next == tok.eos()) {
                finish(a, FinishReason::EndOfSequence);
                continue;
            }
            a.out->tokens.push_back(next);
            ++a.out->metrics.generated_tokens;
            a.next_token = next;
            if (a.out->metrics.generated_tokens >= a.req->max_new_tokens) finish(a, FinishReason::Length);
        }
    }
    return results;
}

GenerationResult generate(Runtime& rt, const GenerationRequest& req, const TokenCallback& on_token,
                          const CancellationToken* cancel) {
    if (!on_token) return generate_batch(rt, {req}, cancel).front();

    // With a callback the loop is written out so each piece streams as it is
    // sampled and the callback can stop generation.
    const auto& tok = rt.model().tokenizer();
    GenerationResult out;
    auto seq = rt.new_sequence();
    Sampler sampler(req.sampling);
    const auto prompt = tok.encode(req.prompt, req.add_bos);
    ErrorContext ctx;
    ctx.operation = "generate";
    if (prompt.empty()) throw Error(ErrorKind::InvalidArgument, "empty prompt", ctx);
    if (int(prompt.size()) > rt.options().context)
        throw Error(ErrorKind::ContextFull, "prompt exceeds the context length", ctx);
    out.metrics.prompt_tokens = int(prompt.size());

    const auto t0 = clock_type::now();
    std::vector<float> logits;
    for (int id : prompt) {
        if (cancel && cancel->cancelled()) {
            out.metrics.finish = FinishReason::Cancelled;
            return out;
        }
        logits = rt.step({seq.get()}, {id}).front();
    }
    out.metrics.prefill_ms = ms_since(t0);
    const auto t1 = clock_type::now();

    out.metrics.finish = FinishReason::Length;
    while (out.metrics.generated_tokens < req.max_new_tokens) {
        if (cancel && cancel->cancelled()) {
            out.metrics.finish = FinishReason::Cancelled;
            break;
        }
        const int next = sampler.sample(logits);
        if (out.metrics.generated_tokens == 0) out.metrics.time_to_first_token_ms = ms_since(t0);
        if (next == tok.eos()) {
            out.metrics.finish = FinishReason::EndOfSequence;
            break;
        }
        out.tokens.push_back(next);
        ++out.metrics.generated_tokens;
        if (!on_token(next, tok.piece(next))) {
            out.metrics.finish = FinishReason::Cancelled;
            break;
        }
        if (out.metrics.generated_tokens >= req.max_new_tokens) break;
        if (seq->position() >= seq->capacity()) {
            out.metrics.finish = FinishReason::ContextFull;
            break;
        }
        logits = rt.step({seq.get()}, {next}).front();
    }
    out.metrics.decode_ms = ms_since(t1);
    out.text = tok.decode(out.tokens);
    return out;
}

double perplexity(Runtime& rt, const std::string& text, int max_tokens) {
    const auto ids = rt.model().tokenizer().encode(text, true);
    const int n = std::min({int(ids.size()), max_tokens, rt.options().context});
    if (n < 2) throw Error(ErrorKind::InvalidArgument, "perplexity needs at least two tokens");
    auto seq = rt.new_sequence();
    double nll = 0.0;
    for (int i = 0; i + 1 < n; ++i) {
        const auto logits = rt.step({seq.get()}, {ids[i]}).front();
        const double mx = *std::max_element(logits.begin(), logits.end());
        double z = 0.0;
        for (float l : logits) z += std::exp(double(l) - mx);
        nll += (mx + std::log(z)) - logits[ids[i + 1]];
    }
    return std::exp(nll / (n - 1));
}

namespace {
std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}
}  // namespace

std::string metrics_json(const Runtime& rt, const GenerationMetrics& m) {
    const auto& d = rt.device();
    const auto& c = rt.config();
    const auto& mem = rt.memory();
    const auto kv = rt.kv_stats();
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(3);
    s << "{"
      << "\"device\":{\"name\":\"" << esc(d.name) << "\",\"compute_capability\":" << d.compute_capability
      << ",\"driver_release\":\"" << esc(d.driver_release) << "\",\"driver_version\":" << d.driver_version
      << ",\"cuda_runtime_version\":" << d.runtime_version << "},"
      << "\"model\":{\"path\":\"" << esc(rt.model().path()) << "\",\"architecture\":\"" << c.architecture
      << "\",\"layers\":" << c.layers << ",\"weights\":\"Q4_0\",\"activations\":\""
      << (rt.activations() == Activations::Int8 ? "int8" : "float") << "\"},"
      << "\"limits\":{\"context\":" << rt.options().context << ",\"max_batch\":" << rt.options().max_batch
      << ",\"cuda_graphs\":" << (rt.options().cuda_graphs ? "true" : "false") << "},"
      << "\"memory_mb\":{\"weights\":" << mem.weights / double(1 << 20) << ",\"kv_cache\":"
      << mem.kv_cache / double(1 << 20) << ",\"workspace\":" << mem.workspace / double(1 << 20) << "},"
      << "\"kv\":{\"pages_in_use\":" << kv.pages_in_use << ",\"total_pages\":" << kv.total_pages
      << ",\"utilization\":" << kv.utilization() << ",\"live_sequences\":" << kv.live_sequences << "},"
      << "\"generation\":{\"prompt_tokens\":" << m.prompt_tokens << ",\"generated_tokens\":" << m.generated_tokens
      << ",\"time_to_first_token_ms\":" << m.time_to_first_token_ms << ",\"prefill_tokens_per_s\":"
      << m.prefill_tokens_per_s() << ",\"decode_tokens_per_s\":" << m.decode_tokens_per_s()
      << ",\"finish\":\"" << to_string(m.finish) << "\"}"
      << "}";
    return s.str();
}

}  // namespace llm
