#pragma once
//
// Host-side dequantization of GGUF block formats. No CUDA syntax.
//
#include <cstdint>
#include <vector>

namespace llm {

float fp16_to_float(std::uint16_t h);

// Q4_0: blocks of 32 weights, each an fp16 scale then 16 bytes of nibbles. The
// low nibbles hold weights 0..15 and the high nibbles 16..31 -- split halves,
// not interleaved. Value = (nibble - 8) * scale.
void dequantize_q4_0(const std::uint8_t* blocks, std::size_t n_elems, float* out);

// Q4_0 repacked for the device: nibbles as one contiguous 16-byte-per-block
// stream and scales as floats, so a warp's reads coalesce. Same split-half
// nibble order as the file.
struct Q4Packed {
    std::vector<std::uint8_t> nibbles;   // n_blocks * 16
    std::vector<float> scales;           // n_blocks
};
Q4Packed repack_q4_0(const std::uint8_t* blocks, std::size_t n_elems);

// Q6_K: super-blocks of 256 weights = 128 bytes of low 4 bits (ql), 64 bytes of
// high 2 bits (qh), 16 int8 sub-block scales, one fp16 super-scale (which is
// negative about half the time, as are the sub-scales). Value =
// super * sub_scale[j/16] * (q - 32), q a 6-bit code.
//
// The bits are laid out in two halves of 128 weights. Within half c and offset
// k (0..127):
//   low 4 bits   ql[64c + (k mod 64)], low nibble if k < 64, else high nibble
//   high 2 bits  qh[32c + (k mod 32)] >> 2*(k / 32)
// This was determined from the file, not assumed: of 36 candidate arrangements
// scored by the perplexity of the next-token predictions they produce, this one
// gave 4.99 on a short factual passage and every arrangement with a different
// high-bit layout gave 377 to 352,000. See the 19-llm-engine README.
void dequantize_q6_k(const std::uint8_t* blocks, std::size_t n_elems, float* out);

}  // namespace llm
