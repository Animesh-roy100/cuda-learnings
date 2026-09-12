#include <stdexcept>

#include "runtime.h"

namespace llm {

std::shared_ptr<const GgufModel> GgufModel::open(const std::string& path) {
    ErrorContext ctx;
    ctx.operation = "load";
    ctx.model = path;

    std::shared_ptr<GgufModel> m(new GgufModel());
    m->path_ = path;
    try {
        m->file_ = std::make_unique<GgufFile>(GgufFile::open(path));
    } catch (const Error&) {
        throw;
    } catch (const std::exception& e) {
        // The parser throws plain runtime_errors with specific reasons; give
        // them the runtime's error type and the file's context.
        throw Error(ErrorKind::InvalidModel, e.what(), ctx);
    }
    m->config_ = load_config(*m->file_, path);
    validate_tensors(*m->file_, m->config_, path);
    try {
        m->tokenizer_ = std::make_unique<Tokenizer>(*m->file_);
    } catch (const std::exception& e) {
        throw Error(ErrorKind::InvalidModel, std::string("tokenizer: ") + e.what(), ctx);
    }
    if (m->tokenizer_->vocab_size() != m->config_.vocab_size)
        throw Error(ErrorKind::InvalidModel, "tokenizer and configuration disagree on vocabulary size", ctx);
    return m;
}

}  // namespace llm
