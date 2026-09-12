// Host-side dequantization of GGUF block formats.

#include "quant.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llm {

float fp16_to_float(std::uint16_t h) {
    const int sign = (h >> 15) & 1;
    const int exp = (h >> 10) & 0x1F;
    const int mant = h & 0x3FF;
    float v;
    if (exp == 0) {
        v = std::ldexp(float(mant), -24);   // subnormal: mant * 2^-10 * 2^-14
    } else if (exp == 31) {
        v = mant ? NAN : INFINITY;
    } else {
        v = std::ldexp(float(mant | 0x400), exp - 25);   // (1 + mant/1024) * 2^(exp-15)
    }
    return sign ? -v : v;
}

void dequantize_q4_0(const std::uint8_t* blocks, std::size_t n_elems, float* out) {
    if (n_elems % 32) throw std::invalid_argument("Q4_0: element count not a multiple of 32");
    const std::size_t nb = n_elems / 32;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = blocks + b * 18;
        std::uint16_t d16;
        std::memcpy(&d16, blk, 2);
        const float d = fp16_to_float(d16);
        const std::uint8_t* q = blk + 2;
        for (int j = 0; j < 16; ++j) {
            out[b * 32 + j] = float(int(q[j] & 0x0F) - 8) * d;
            out[b * 32 + j + 16] = float(int(q[j] >> 4) - 8) * d;
        }
    }
}

Q4Packed repack_q4_0(const std::uint8_t* blocks, std::size_t n_elems) {
    if (n_elems % 32) throw std::invalid_argument("Q4_0: element count not a multiple of 32");
    const std::size_t nb = n_elems / 32;
    Q4Packed p;
    p.nibbles.resize(nb * 16);
    p.scales.resize(nb);
    for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = blocks + b * 18;
        std::uint16_t d16;
        std::memcpy(&d16, blk, 2);
        p.scales[b] = fp16_to_float(d16);
        std::memcpy(p.nibbles.data() + b * 16, blk + 2, 16);
    }
    return p;
}

void dequantize_q6_k(const std::uint8_t* blocks, std::size_t n_elems, float* out) {
    if (n_elems % 256) throw std::invalid_argument("Q6_K: element count not a multiple of 256");
    const std::size_t nb = n_elems / 256;
    for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = blocks + b * 210;
        const std::uint8_t* ql = blk;
        const std::uint8_t* qh = blk + 128;
        const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
        std::uint16_t d16;
        std::memcpy(&d16, blk + 208, 2);
        const float d = fp16_to_float(d16);

        for (int j = 0; j < 256; ++j) {
            const int half = j / 128, k = j % 128;
            const std::uint8_t lob = ql[64 * half + (k % 64)];
            const int lo = k < 64 ? (lob & 0x0F) : (lob >> 4);
            const int hi = (qh[32 * half + (k % 32)] >> (2 * (k / 32))) & 0x03;
            out[b * 256 + j] = d * float(sc[j / 16]) * float((lo | (hi << 4)) - 32);
        }
    }
}

}  // namespace llm
