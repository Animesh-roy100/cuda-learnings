// Model configuration and tensor manifest validation.

#include "model_config.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace llm {
namespace {

ErrorContext ctx(const std::string& path, const std::string& tensor = {},
                 std::vector<std::uint64_t> shape = {}) {
    ErrorContext c;
    c.operation = "load";
    c.model = path;
    c.tensor = tensor;
    c.shape = std::move(shape);
    return c;
}

int require_int(const GgufFile& f, const std::string& key, const std::string& path) {
    const auto v = f.meta_int(key);
    if (!v) throw Error(ErrorKind::InvalidModel, "missing metadata key " + key, ctx(path));
    if (*v <= 0 || *v > (std::int64_t(1) << 30))
        throw Error(ErrorKind::InvalidModel,
                    key + " is " + std::to_string(*v) + ", outside (0, 2^30]", ctx(path));
    return static_cast<int>(*v);
}

std::string type_list(const std::vector<GgmlType>& types) {
    std::string s;
    for (std::size_t i = 0; i < types.size(); ++i) s += (i ? "/" : "") + std::string(type_traits(types[i]).name);
    return s;
}

std::string dims_string(const std::vector<std::uint64_t>& d) {
    std::string s = "[";
    for (std::size_t i = 0; i < d.size(); ++i) s += (i ? ", " : "") + std::to_string(d[i]);
    return s + "]";
}

}  // namespace

ModelConfig load_config(const GgufFile& f, const std::string& path) {
    ModelConfig c;
    c.architecture = f.meta_string("general.architecture").value_or("");
    if (c.architecture.empty())
        throw Error(ErrorKind::InvalidModel, "missing general.architecture", ctx(path));
    if (c.architecture != "llama")
        throw Error(ErrorKind::Unsupported,
                    "architecture '" + c.architecture + "' (only llama is implemented)", ctx(path));

    c.layers = require_int(f, "llama.block_count", path);
    c.hidden_size = require_int(f, "llama.embedding_length", path);
    c.intermediate_size = require_int(f, "llama.feed_forward_length", path);
    c.attention_heads = require_int(f, "llama.attention.head_count", path);
    c.kv_heads = f.meta_int("llama.attention.head_count_kv")
                     ? require_int(f, "llama.attention.head_count_kv", path)
                     : c.attention_heads;   // no GQA key means multi-head attention
    c.context_length = require_int(f, "llama.context_length", path);
    c.head_dim = c.hidden_size / c.attention_heads;
    if (f.meta_int("llama.rope.dimension_count")) {
        const int rope_dims = require_int(f, "llama.rope.dimension_count", path);
        if (rope_dims != c.head_dim)
            throw Error(ErrorKind::Unsupported,
                        "partial rotary embedding (rope.dimension_count " + std::to_string(rope_dims) +
                            " != head_dim " + std::to_string(c.head_dim) + ")",
                        ctx(path));
    }

    c.rms_epsilon = float(f.meta_float("llama.attention.layer_norm_rms_epsilon").value_or(1e-5));
    c.rope_theta = float(f.meta_float("llama.rope.freq_base").value_or(10000.0));
    if (f.meta_string("llama.rope.scaling.type").value_or("none") != "none")
        throw Error(ErrorKind::Unsupported, "RoPE scaling is not implemented", ctx(path));

    const auto* tokens = f.meta_string_array("tokenizer.ggml.tokens");
    if (!tokens || tokens->empty())
        throw Error(ErrorKind::InvalidModel, "missing tokenizer.ggml.tokens", ctx(path));
    c.vocab_size = static_cast<int>(tokens->size());

    validate_config(c, path);
    return c;
}

void validate_config(const ModelConfig& c, const std::string& path) {
    std::vector<std::string> bad;
    auto need = [&](bool ok, const std::string& what) {
        if (!ok) bad.push_back(what);
    };
    need(c.layers > 0, "layers > 0");
    need(c.hidden_size > 0, "hidden_size > 0");
    need(c.intermediate_size > 0, "intermediate_size > 0");
    need(c.attention_heads > 0, "attention_heads > 0");
    need(c.kv_heads > 0, "kv_heads > 0");
    need(c.head_dim > 0, "head_dim > 0");
    need(c.vocab_size > 0, "vocab_size > 0");
    need(c.context_length > 0, "context_length > 0");
    need(c.attention_heads > 0 && c.hidden_size == c.attention_heads * c.head_dim,
         "hidden_size == attention_heads * head_dim");
    need(c.kv_heads <= c.attention_heads, "kv_heads <= attention_heads");
    need(c.kv_heads > 0 && c.attention_heads % c.kv_heads == 0, "attention_heads % kv_heads == 0");
    need(std::isfinite(c.rms_epsilon) && c.rms_epsilon > 0.0f, "rms_epsilon finite and > 0");
    need(std::isfinite(c.rope_theta) && c.rope_theta > 0.0f, "rope_theta finite and > 0");
    if (!bad.empty()) {
        std::string msg = "configuration violates:";
        for (const auto& b : bad) msg += " " + b + ";";
        throw Error(ErrorKind::InvalidModel, msg, ctx(path));
    }

    // Structurally valid, but outside what the kernels are compiled for.
    std::vector<std::string> unsupported;
    if (c.head_dim != 64) unsupported.push_back("head_dim " + std::to_string(c.head_dim) + " (kernels: 64)");
    if (c.attention_heads > 32)
        unsupported.push_back(std::to_string(c.attention_heads) + " attention heads (max 32)");
    if (c.hidden_size % 32) unsupported.push_back("hidden_size not a multiple of 32");
    if (c.intermediate_size % 32) unsupported.push_back("intermediate_size not a multiple of 32");
    if (!unsupported.empty()) {
        std::string msg = "model shape outside this runtime's kernels:";
        for (const auto& u : unsupported) msg += " " + u + ";";
        throw Error(ErrorKind::Unsupported, msg, ctx(path));
    }
}

std::vector<TensorRequirement> required_tensors(const ModelConfig& c) {
    const std::uint64_t D = c.hidden_size, F = c.intermediate_size, V = c.vocab_size,
                        KV = c.kv_dim();
    const std::vector<GgmlType> q4 = {GgmlType::Q4_0};
    const std::vector<GgmlType> f32 = {GgmlType::F32};
    std::vector<TensorRequirement> r;
    r.push_back({"token_embd.weight", {D, V}, q4});
    r.push_back({"output_norm.weight", {D}, f32});
    r.push_back({"output.weight", {D, V}, {GgmlType::Q6_K, GgmlType::Q4_0, GgmlType::F32}});
    for (int l = 0; l < c.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        r.push_back({p + "attn_norm.weight", {D}, f32});
        r.push_back({p + "attn_q.weight", {D, D}, q4});
        r.push_back({p + "attn_k.weight", {D, KV}, q4});
        r.push_back({p + "attn_v.weight", {D, KV}, q4});
        r.push_back({p + "attn_output.weight", {D, D}, q4});
        r.push_back({p + "ffn_norm.weight", {D}, f32});
        r.push_back({p + "ffn_gate.weight", {D, F}, q4});
        r.push_back({p + "ffn_up.weight", {D, F}, q4});
        r.push_back({p + "ffn_down.weight", {F, D}, q4});
    }
    return r;
}

void validate_tensors(const GgufFile& f, const ModelConfig& c, const std::string& path) {
    std::vector<std::string> problems;
    for (const auto& req : required_tensors(c)) {
        const auto* t = f.find(req.name);
        if (!t) {
            problems.push_back(req.name + ": missing");
            continue;
        }
        if (std::find(req.types.begin(), req.types.end(), t->type) == req.types.end())
            problems.push_back(req.name + ": type " + type_traits(t->type).name + ", expected " +
                               type_list(req.types));
        if (t->dims != req.dims)
            problems.push_back(req.name + ": shape " + dims_string(t->dims) + ", expected " +
                               dims_string(req.dims));
    }
    if (!problems.empty()) {
        std::string msg = std::to_string(problems.size()) + " tensor problem(s):";
        const std::size_t shown = std::min<std::size_t>(problems.size(), 12);
        for (std::size_t i = 0; i < shown; ++i) msg += " " + problems[i] + ";";
        if (shown < problems.size()) msg += " ... and " + std::to_string(problems.size() - shown) + " more";
        throw Error(ErrorKind::InvalidModel, msg, ctx(path));
    }
}

std::uint64_t weight_device_bytes(const GgufFile& f, const ModelConfig& c) {
    std::uint64_t total = 0;
    for (const auto& req : required_tensors(c)) {
        if (req.name == "token_embd.weight") continue;   // stays in the mapped file
        const auto* t = f.find(req.name);
        if (!t) continue;
        const std::uint64_t n = t->num_elements();
        if (t->type == GgmlType::Q4_0 && req.name != "output.weight")
            total += n / 2 + (n / 32) * 4;
        else
            total += n * 4;   // F32, or dequantized to F32 on load
    }
    return total;
}

}  // namespace llm
