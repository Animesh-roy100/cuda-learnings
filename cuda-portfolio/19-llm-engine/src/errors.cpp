#include "errors.h"

#include <sstream>

namespace llm {

const char* to_string(ErrorKind k) {
    switch (k) {
        case ErrorKind::InvalidArgument: return "invalid argument";
        case ErrorKind::InvalidModel: return "invalid model";
        case ErrorKind::Unsupported: return "unsupported";
        case ErrorKind::OutOfMemory: return "out of memory";
        case ErrorKind::ContextFull: return "context full";
        case ErrorKind::Cancelled: return "cancelled";
        case ErrorKind::Device: return "device error";
    }
    return "error";
}

namespace {

std::string render(ErrorKind kind, const std::string& message, const ErrorContext& c) {
    std::ostringstream s;
    s << to_string(kind);
    if (!c.operation.empty()) s << " during " << c.operation;
    s << ": " << message;
    std::string sep = " [";
    auto field = [&](const char* name, const std::string& value) {
        s << sep << name << "=" << value;
        sep = ", ";
    };
    if (!c.model.empty()) field("model", c.model);
    if (!c.tensor.empty()) field("tensor", c.tensor);
    if (!c.shape.empty()) {
        std::string dims;
        for (std::size_t i = 0; i < c.shape.size(); ++i)
            dims += (i ? "x" : "") + std::to_string(c.shape[i]);
        field("shape", dims);
    }
    if (c.device) field("device", std::to_string(*c.device));
    if (!c.cuda_error.empty()) field("cuda", c.cuda_error);
    if (c.sequence) field("sequence", std::to_string(*c.sequence));
    if (c.position) field("position", std::to_string(*c.position));
    if (sep == ", ") s << "]";
    return s.str();
}

}  // namespace

Error::Error(ErrorKind kind, std::string message, ErrorContext ctx)
    : std::runtime_error(render(kind, message, ctx)),
      kind_(kind),
      message_(std::move(message)),
      ctx_(std::move(ctx)) {}

}  // namespace llm
