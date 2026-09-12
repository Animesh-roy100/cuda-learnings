#pragma once
//
// Tiled tensor-core GEMM, FP16 and INT8, measured against cuBLAS.
//
// C = A * B for square N x N matrices, row-major. Every path computes the same
// product; they differ in precision, in which instruction does the multiply,
// and in how the operands reach it. No CUDA syntax in this header.
//
// The paths form controlled comparisons, one variable at a time:
//
//   Fp32Tiled           shared-memory tiled SGEMM in CUDA cores -- the baseline
//   Fp16WmmaGlobal      16x16x16 WMMA fragments loaded straight from global
//   Fp16WmmaStaged      the same, with each band of operands first staged into
//                       shared memory element by element
//   Fp16WmmaStagedWord  staged as whole 4-byte words instead
//   Int8WmmaGlobal      INT8 tensor-core multiply, from global
//   Int8WmmaStaged      INT8, staged element by element
//   Int8WmmaStagedWord  INT8, staged as whole words
//   Fp16WmmaPipelined   word staging overlapped with compute: two warps per
//                       block, one staging the next band into a second buffer
//                       while the other multiplies the current one
//   Int8WmmaPipelined   the same, INT8
//   Int8Dp4a            INT8 in CUDA cores via __dp4a, from global -- the same
//                       instruction 01-gguf-inference's GEMV is built on
//   CublasFp32          cublasSgemm
//   CublasFp16          cublasGemmEx: fp16 operands, fp32 compute (SgemmEx)
//   CublasFp16Hgemm     cublasGemmEx: fp16 operands, result and compute
//   CublasInt8          cublasGemmEx: int8 operands, int32 result
//
// Global vs Staged isolates the memory path, Staged vs StagedWord isolates
// shared-memory bank conflicts within it, and StagedWord vs Pipelined isolates
// whether overlapping the staging with the compute recovers the rest. Int8Dp4a vs Int8WmmaGlobal isolates
// the instruction: identical data, CUDA cores versus tensor cores. Hand-written
// versus cuBLAS is the gap this project exists to explain.
//
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc {

enum class Path {
    Fp32Tiled,
    Fp16WmmaGlobal,
    Fp16WmmaStaged,
    Fp16WmmaStagedWord,
    Fp16WmmaPipelined,
    Int8WmmaGlobal,
    Int8WmmaStaged,
    Int8WmmaStagedWord,
    Int8WmmaPipelined,
    Int8Dp4a,
    CublasFp32,
    CublasFp16,
    CublasFp16Hgemm,
    CublasInt8,
};

const char* to_string(Path p);
bool is_int8(Path p);
bool is_fp16(Path p);
bool is_cublas(Path p);
std::vector<Path> all_paths();

// Thrown when a library path refuses a shape it cannot handle. Distinct from a
// failure: the call was valid, the library simply has no kernel for it.
//
// Measured with cuBLAS 13.4 on sm_75, INT8 GemmEx accepts N = 16 and every
// multiple of 32, and returns CUBLAS_STATUS_NOT_SUPPORTED for every odd
// multiple of 16 from 48 upwards -- although the documented requirement is
// only that dimensions be multiples of 4. The hand-written INT8 paths accept
// any multiple of 16.
struct Unsupported : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Symmetric per-tensor quantization: x ~= scale * q, q in [-127, 127].
struct Quantized {
    std::vector<std::int8_t> q;
    float scale = 1.0f;
};
Quantized quantize_symmetric(const std::vector<float>& x);

// How the staged paths get operands into shared memory.
enum class Staging { ByElement, ByWord, Pipelined };
const char* to_string(Staging s);

// Staging band widths compiled for this staging mode and precision. The band
// sizes the __shared__ arrays, so it is a compile-time parameter, and it trades
// warps for work: a wider band means more fragments between barriers but more
// shared memory per block, hence fewer blocks resident on each SM.
//
// Not every band exists for every mode. The device linker limits a kernel to
// 48 KB of shared data on sm_75 (it reports "0xc000 max"), and the pipelined
// fp16 kernel at band 512 would need 64 KB -- two staged buffers of two arrays
// of 16 x 512 fp16 values.
std::vector<int> staging_bands(Staging s, bool int8);

// The band each staged Path uses -- chosen from the sweep in the benchmark.
int default_band(Staging s);

struct BandPoint {
    int band = 0;
    int blocks_per_sm = 0;
    int compute_warps_per_sm = 0;
    float ms = 0.0f;
};

class Gemm {
public:
    // n must be a positive multiple of 16: every fragment is 16x16x16.
    explicit Gemm(int n);
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    int n() const;

    // Row-major N*N operands. Uploads fp32, fp16 and int8 copies of both.
    void set_inputs(const std::vector<float>& a, const std::vector<float>& b);

    // The product, row-major, as float. INT8 paths are dequantized with the
    // operand scales, so their error against FP32 is directly comparable.
    std::vector<float> multiply(Path p);

    // Median milliseconds for the multiply alone -- no upload, no download, no
    // dequantization. Kernels are launched in bands short enough to stay under
    // the Windows display-driver timeout.
    float time(Path p, int iterations = 5, int warmup = 2);

    // A staged path at an explicit band width, for the band sweep.
    std::vector<float> multiply_band(Staging s, bool int8, int band);
    std::vector<BandPoint> sweep_band(Staging s, bool int8, int iterations = 3, int warmup = 1);

    // Billions of multiply-adds per second: N^3 of them per product.
    static double gflops(int n, float ms);

private:
    std::vector<float> multiply(Path p, bool already_run);
    struct Impl;
    Impl* impl_;
};

// How many compute warps each hand-written path keeps resident per SM.
//
// Blocks-per-SM comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor for
// each kernel at the block size it launches with. Compute warps are the warps
// in a block that actually multiply: all of them, except in the pipelined
// kernel, where one warp of every two only stages. Needs no profiler and no
// elevation, which is why it is used here to test an explanation that would
// otherwise need a Nsight Compute roofline.
struct Occupancy {
    Path path;
    int threads_per_block = 0;
    int blocks_per_sm = 0;
    int compute_warps_per_block = 0;
    int compute_warps_per_sm() const { return blocks_per_sm * compute_warps_per_block; }
};
std::vector<Occupancy> occupancy();

// The experimental 4-bit fragment (8x8x32, u4 operands, int accumulator) on a
// small known matrix. Correctness only: the sub-byte WMMA types exist on sm_75
// and were deprecated afterwards, so this is a historical artifact, not a path
// anyone should build on.
struct SubByteCheck {
    int wrong = 0;
    int total = 0;
};
SubByteCheck check_u4_fragment();

}  // namespace tc
