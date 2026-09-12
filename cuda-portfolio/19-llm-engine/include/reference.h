#pragma once
//
// Host FP32 reference forward pass. No CUDA syntax, no dependency on the
// runtime: it shares only the parser and the block dequantizers with the
// device path, which is what makes it a reference.
//
#include <vector>

namespace llm {

class GgufFile;

// Next-token logits after the given tokens, computed on the host in FP32 from
// fully dequantized weights. Seconds per token.
std::vector<float> reference_logits(const GgufFile& f, const std::vector<int>& tokens);

}  // namespace llm
