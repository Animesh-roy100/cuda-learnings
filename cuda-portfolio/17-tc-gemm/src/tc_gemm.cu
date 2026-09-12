// Tiled tensor-core GEMM.

#include "tc_gemm.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "cu/check.hpp"
#include "cu/timer.hpp"

using namespace nvcuda;

namespace tc {
namespace {

// A launch that runs past ~2 s trips the Windows display driver's timeout
// detection and resets the GPU. At N=4096 the tiled FP32 kernel needs several
// seconds in total, so every hand-written path is launched in bands of tile
// rows, each well under that limit.
constexpr int kTileRowsPerLaunch = 16;

// Band widths the named staged paths use, taken from the benchmark's band sweep
// at N=2048 (best mean of fp16 and int8 relative to loading from global). The
// first version used 512/512/256, the worst end of that sweep: wide bands cost
// so much shared memory that only 2-4 blocks fit per SM.
constexpr int kDefaultBandByElement = 128;
constexpr int kDefaultBandByWord = 128;
constexpr int kDefaultBandPipelined = 64;


// ---------------------------------------------------------------------------
// FP32 baseline: classic shared-memory tiled SGEMM. B is stored transposed so
// every kernel reads both operands row-wise.
// ---------------------------------------------------------------------------
__global__ void k_fp32_tiled(const float* __restrict__ A, const float* __restrict__ Bt,
                             float* __restrict__ C, int N, int row_off) {
    // Width 17, not 16: Bs is indexed lane-varying-first, and at width 16 every
    // lane's column would land in the same two banks (see 16-layout-advanced).
    __shared__ float As[16][17];
    __shared__ float Bs[16][17];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = (row_off + blockIdx.y) * 16 + ty;
    const int col = blockIdx.x * 16 + tx;
    if (row >= N) return;

    float acc = 0.0f;
    for (int k0 = 0; k0 < N; k0 += 16) {
        As[ty][tx] = A[row * N + k0 + tx];
        Bs[tx][ty] = Bt[col * N + k0 + ty];
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// WMMA from global. One warp per 16x16 output tile, four warps per block.
// ---------------------------------------------------------------------------
template <typename T, typename Acc>
__global__ void k_wmma_global(const T* __restrict__ A, const T* __restrict__ Bt,
                              Acc* __restrict__ C, int N, int m_off) {
    const int m = m_off + blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x;
    if (m * 16 >= N || n * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> c;
    wmma::fill_fragment(c, Acc(0));
    for (int k = 0; k < N; k += 16) {
        // Each fragment spans 16 rows N elements apart: 16 strided lines per
        // load, read straight from DRAM.
        wmma::load_matrix_sync(a, A + m * 16 * N + k, N);
        wmma::load_matrix_sync(b, Bt + n * 16 * N + k, N);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + m * 16 * N + n * 16, c, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// WMMA with shared-memory staging. One warp per block.
//
// Per band of BAND columns, the whole warp copies the 16 rows of A and the 16
// columns of B that its tile needs into shared memory -- every lane loading the
// positions congruent to its lane number, so each row's 32 lanes read one
// contiguous run -- and then feeds BAND/16 fragments from L1 instead of DRAM.
//
// BAND is a template parameter because it sizes the __shared__ arrays, and the
// shared memory a kernel declares decides how many blocks an SM can hold at once.
// This was the dominant effect in this project, and it was found only by asking
// the occupancy API: at BAND=512 the fp16 staging arrays are 32 KB, just two
// blocks fit per SM, and the kernel ran with 2 compute warps per SM against 32
// for the kernel that loads from global. Gemm::sweep_band measures the trade.
//
// A second, smaller cost: shared memory banks are 4-byte words, so an fp16 array
// puts two elements in each word and int8 puts four. Lanes staging neighbouring
// elements share banks -- a 2-way conflict for fp16, 4-way for int8.
// k_wmma_staged_word is the control for that.
// ---------------------------------------------------------------------------
template <typename T, typename Acc, int BAND>
__global__ void k_wmma_staged(const T* __restrict__ A, const T* __restrict__ Bt,
                              Acc* __restrict__ C, int N, int m_off) {
    __shared__ T As[16 * BAND];
    __shared__ T Bs[16 * BAND];

    const int m = m_off + blockIdx.y;
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    if (m * 16 >= N || n * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> c;
    wmma::fill_fragment(c, Acc(0));

    for (int k0 = 0; k0 < N; k0 += BAND) {
        const int kc = min(BAND, N - k0);
        for (int r = 0; r < 16; ++r) {
            const T* arow = A + (m * 16 + r) * N + k0;
            const T* brow = Bt + (n * 16 + r) * N + k0;
            for (int x = lane; x < kc; x += 32) {
                As[r * kc + x] = arow[x];
                Bs[r * kc + x] = brow[x];
            }
        }
        __syncthreads();
        for (int kk = 0; kk < kc; kk += 16) {
            // Row-major A: element (r, j) at base + r*kc + j.
            // Col-major B: element (j, s) at base + j + s*kc, which is exactly
            // how the transposed band was staged.
            wmma::load_matrix_sync(a, As + kk, kc);
            wmma::load_matrix_sync(b, Bs + kk, kc);
            wmma::mma_sync(c, a, b, c);
        }
    }
    wmma::store_matrix_sync(C + m * 16 * N + n * 16, c, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// WMMA with WORD-ALIGNED staging -- the bank-conflict control.
//
// Identical bands and fragments; only the unit each lane moves changes. The
// operand rows start on 4-byte boundaries (N is a multiple of 16), so a row can
// be read as 32-bit words, and each lane stages one whole word -- two fp16 or
// four int8 values -- into a word-typed shared array. Neighbouring lanes always
// address different words, so no two lanes share a bank.
//
// The arrays hold exactly the words the band needs. The first version sized
// them for fp16 regardless of T, which gave the int8 instantiation twice the
// shared memory it used and half the blocks per SM it could have had.
// ---------------------------------------------------------------------------
template <typename T, typename Acc, int BAND>
__global__ void k_wmma_staged_word(const T* __restrict__ A, const T* __restrict__ Bt,
                                   Acc* __restrict__ C, int N, int m_off) {
    __shared__ unsigned Aw[16 * BAND * sizeof(T) / 4];
    __shared__ unsigned Bw[16 * BAND * sizeof(T) / 4];

    const int m = m_off + blockIdx.y;
    const int n = blockIdx.x;
    const int lane = threadIdx.x;
    if (m * 16 >= N || n * 16 >= N) return;
    const int per = 4 / int(sizeof(T));   // elements per word

    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> c;
    wmma::fill_fragment(c, Acc(0));

    for (int k0 = 0; k0 < N; k0 += BAND) {
        const int kc = min(BAND, N - k0);
        const int wpr = kc / per;   // words per staged row
        for (int r = 0; r < 16; ++r) {
            const unsigned* aw = reinterpret_cast<const unsigned*>(A + (m * 16 + r) * N + k0);
            const unsigned* bw = reinterpret_cast<const unsigned*>(Bt + (n * 16 + r) * N + k0);
            for (int j = lane; j < wpr; j += 32) {
                Aw[r * wpr + j] = aw[j];
                Bw[r * wpr + j] = bw[j];
            }
        }
        __syncthreads();
        // Word r*wpr + j holds elements r*kc + j*per onwards, so read as T the
        // staged band has exactly the layout the element-wise kernel builds.
        const T* As = reinterpret_cast<const T*>(Aw);
        const T* Bs = reinterpret_cast<const T*>(Bw);
        for (int kk = 0; kk < kc; kk += 16) {
            wmma::load_matrix_sync(a, As + kk, kc);
            wmma::load_matrix_sync(b, Bs + kk, kc);
            wmma::mma_sync(c, a, b, c);
        }
    }
    wmma::store_matrix_sync(C + m * 16 * N + n * 16, c, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Software pipelining: staging overlapped with compute.
//
// With two warps per block the roles can run at the same time: warp 1 stages
// band t+1 into the buffer warp 0 is NOT reading, while warp 0 multiplies band t
// out of the other. One barrier per band hands the staged buffer across.
//
// This is how GEMM on sm_75 gets overlap without cp.async: sm_80 lets DRAM write
// straight into shared memory while the warp keeps computing, so one buffer
// suffices. Without that instruction, overlap is built from a second warp and a
// second buffer -- and the second buffer doubles the shared memory per block.
//
// Both warps take whole branches, so no warp ever diverges internally.
// ---------------------------------------------------------------------------
template <typename T>
__device__ void stage_words(const T* A, const T* Bt, unsigned* aw_dst, unsigned* bw_dst,
                            int N, int m, int n, int k0, int kc, int lane) {
    const int wpr = kc / (4 / int(sizeof(T)));
    for (int r = 0; r < 16; ++r) {
        const unsigned* aw = reinterpret_cast<const unsigned*>(A + (m * 16 + r) * N + k0);
        const unsigned* bw = reinterpret_cast<const unsigned*>(Bt + (n * 16 + r) * N + k0);
        for (int j = lane; j < wpr; j += 32) {
            aw_dst[r * wpr + j] = aw[j];
            bw_dst[r * wpr + j] = bw[j];
        }
    }
}

template <typename T, typename Acc, int BAND>
__global__ void k_wmma_pipelined(const T* __restrict__ A, const T* __restrict__ Bt,
                                 Acc* __restrict__ C, int N, int m_off) {
    __shared__ unsigned Aw[2][16 * BAND * sizeof(T) / 4];
    __shared__ unsigned Bw[2][16 * BAND * sizeof(T) / 4];

    const int m = m_off + blockIdx.y;
    const int n = blockIdx.x;
    if (m * 16 >= N || n * 16 >= N) return;
    const int warp = threadIdx.x >> 5;   // 0 multiplies, 1 stages
    const int lane = threadIdx.x & 31;
    const int bands = (N + BAND - 1) / BAND;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> c;
    if (warp == 0) wmma::fill_fragment(c, Acc(0));

    if (warp == 1) stage_words(A, Bt, Aw[0], Bw[0], N, m, n, 0, min(BAND, N), lane);
    __syncthreads();

    for (int t = 0; t < bands; ++t) {
        const int k0 = t * BAND;
        const int kc = min(BAND, N - k0);
        if (warp == 0) {
            const T* As = reinterpret_cast<const T*>(Aw[t & 1]);
            const T* Bs = reinterpret_cast<const T*>(Bw[t & 1]);
            for (int kk = 0; kk < kc; kk += 16) {
                wmma::load_matrix_sync(a, As + kk, kc);
                wmma::load_matrix_sync(b, Bs + kk, kc);
                wmma::mma_sync(c, a, b, c);
            }
        } else if (t + 1 < bands) {
            const int k1 = k0 + BAND;
            stage_words(A, Bt, Aw[(t + 1) & 1], Bw[(t + 1) & 1], N, m, n, k1,
                        min(BAND, N - k1), lane);
        }
        __syncthreads();
    }
    if (warp == 0)
        wmma::store_matrix_sync(C + m * 16 * N + n * 16, c, N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// Dispatch over the compiled band widths. Every (staging, precision, band)
// combination is its own instantiation, because each sizes its shared arrays.
// ---------------------------------------------------------------------------
// Measured, not documented: the device linker refuses a kernel whose __shared__
// declarations exceed this ("uses too much shared data (0x10000 bytes, 0xc000
// max)") -- 48 KB, not the 64 KB of L1 a Turing SM has.
constexpr std::size_t kMaxSharedBytes = 0xc000;

template <typename T, int BAND>
constexpr std::size_t shared_bytes(Staging s) {
    const std::size_t words = 16 * std::size_t(BAND) * sizeof(T) / 4;
    switch (s) {
        case Staging::ByElement: return 2 * 16 * std::size_t(BAND) * sizeof(T);
        case Staging::ByWord: return 2 * words * 4;
        case Staging::Pipelined: return 4 * words * 4;
    }
    return 0;
}

template <typename T, int BAND>
constexpr bool fits(Staging s) { return shared_bytes<T, BAND>(s) <= kMaxSharedBytes; }

int staging_threads(Staging s) { return s == Staging::Pipelined ? 64 : 32; }

template <typename T, typename Acc, int BAND>
void launch_staging(Staging s, dim3 grid, const T* A, const T* Bt, Acc* C, int N, int off) {
    switch (s) {
        case Staging::ByElement:
            k_wmma_staged<T, Acc, BAND><<<grid, dim3(32, 1)>>>(A, Bt, C, N, off);
            return;
        case Staging::ByWord:
            k_wmma_staged_word<T, Acc, BAND><<<grid, dim3(32, 1)>>>(A, Bt, C, N, off);
            return;
        case Staging::Pipelined:
            // Compile-time exclusion: an over-budget instantiation would not
            // just fail at runtime, it would fail to LINK.
            if constexpr (fits<T, BAND>(Staging::Pipelined)) {
                k_wmma_pipelined<T, Acc, BAND><<<grid, dim3(64, 1)>>>(A, Bt, C, N, off);
                return;
            }
            throw std::invalid_argument("tc: pipelined band exceeds the shared-memory limit");
    }
}

template <typename T, typename Acc, int BAND>
const void* staging_kernel(Staging s) {
    switch (s) {
        case Staging::ByElement:
            return reinterpret_cast<const void*>(k_wmma_staged<T, Acc, BAND>);
        case Staging::ByWord:
            return reinterpret_cast<const void*>(k_wmma_staged_word<T, Acc, BAND>);
        case Staging::Pipelined:
            if constexpr (fits<T, BAND>(Staging::Pipelined))
                return reinterpret_cast<const void*>(k_wmma_pipelined<T, Acc, BAND>);
            throw std::invalid_argument("tc: pipelined band exceeds the shared-memory limit");
    }
    return nullptr;
}

#define TC_FOR_EACH_BAND(X) X(16) X(32) X(64) X(128) X(256) X(512)



template <typename T, typename Acc>
void launch_staging_band(Staging s, int band, dim3 grid, const T* A, const T* Bt, Acc* C,
                         int N, int off) {
    switch (band) {
#define TC_CASE(B) case B: launch_staging<T, Acc, B>(s, grid, A, Bt, C, N, off); return;
        TC_FOR_EACH_BAND(TC_CASE)
#undef TC_CASE
    }
    throw std::invalid_argument("tc: no kernel compiled for staging band " + std::to_string(band));
}

template <typename T, typename Acc>
const void* staging_kernel_band(Staging s, int band) {
    switch (band) {
#define TC_CASE(B) case B: return staging_kernel<T, Acc, B>(s);
        TC_FOR_EACH_BAND(TC_CASE)
#undef TC_CASE
    }
    throw std::invalid_argument("tc: no kernel compiled for staging band " + std::to_string(band));
}

// ---------------------------------------------------------------------------
// INT8 in CUDA cores: one output element per thread, four products per
// __dp4a. Reads the same int8 operands as the INT8 WMMA paths, from global, so
// against Int8WmmaGlobal the only difference is which instruction multiplies.
//
// __dp4a here is signed x signed -- measured in 15-warp-primitives -- unlike
// x86 VNNI's unsigned x signed, so the operands need no bias correction.
// ---------------------------------------------------------------------------
__global__ void k_int8_dp4a(const signed char* __restrict__ A, const signed char* __restrict__ Bt,
                            int* __restrict__ C, int N, int row_off) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = (row_off + blockIdx.y) * 16 + ty;
    const int col = blockIdx.x * 16 + tx;
    if (row >= N) return;
    const int* aw = reinterpret_cast<const int*>(A + row * N);
    const int* bw = reinterpret_cast<const int*>(Bt + col * N);
    int s = 0;
    for (int j = 0; j < N / 4; ++j) s = __dp4a(aw[j], bw[j], s);
    C[row * N + col] = s;
}

__global__ void k_dequant(const int* __restrict__ in, float* __restrict__ out,
                          int count, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = float(in[i]) * scale;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        CU_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr;
        o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        std::swap(ptr_, o.ptr_);
        std::swap(count_, o.count_);
        return *this;
    }
    T* get() const { return ptr_; }
    void upload(const T* h) {
        CU_CHECK(cudaMemcpy(ptr_, h, count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* h) const {
        CU_CHECK(cudaMemcpy(h, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

void cublas_check(cublasStatus_t s, const char* what) {
    if (s == CUBLAS_STATUS_NOT_SUPPORTED)
        throw Unsupported(std::string("cuBLAS ") + what + ": no kernel for this shape");
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS ") + what + " failed: status " +
                                 std::to_string(static_cast<int>(s)));
}

std::vector<float> transpose(const std::vector<float>& x, int n) {
    std::vector<float> t(x.size());
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) t[std::size_t(j) * n + i] = x[std::size_t(i) * n + j];
    return t;
}

}  // namespace

// ---------------------------------------------------------------------------

const char* to_string(Staging s) {
    switch (s) {
        case Staging::ByElement: return "staged by element";
        case Staging::ByWord: return "staged by word";
        case Staging::Pipelined: return "pipelined";
    }
    return "?";
}

std::vector<int> staging_bands(Staging s, bool int8) {
    std::vector<int> v;
#define TC_PUSH(B)                                                              \
    if (int8 ? fits<signed char, B>(s) : fits<__half, B>(s)) v.push_back(B);
    TC_FOR_EACH_BAND(TC_PUSH)
#undef TC_PUSH
    return v;
}

int default_band(Staging s) {
    switch (s) {
        case Staging::ByElement: return kDefaultBandByElement;
        case Staging::ByWord: return kDefaultBandByWord;
        case Staging::Pipelined: return kDefaultBandPipelined;
    }
    return 0;
}

const char* to_string(Path p) {
    switch (p) {
        case Path::Fp32Tiled: return "fp32 tiled (CUDA cores)";
        case Path::Fp16WmmaGlobal: return "fp16 wmma, from global";
        case Path::Fp16WmmaStaged: return "fp16 wmma, staged by element";
        case Path::Fp16WmmaStagedWord: return "fp16 wmma, staged by word";
        case Path::Fp16WmmaPipelined: return "fp16 wmma, pipelined";
        case Path::Int8WmmaGlobal: return "int8 wmma, from global";
        case Path::Int8WmmaStaged: return "int8 wmma, staged by element";
        case Path::Int8WmmaStagedWord: return "int8 wmma, staged by word";
        case Path::Int8WmmaPipelined: return "int8 wmma, pipelined";
        case Path::Int8Dp4a: return "int8 __dp4a (CUDA cores)";
        case Path::CublasFp32: return "cuBLAS sgemm";
        case Path::CublasFp16: return "cuBLAS fp16, 32F compute";
        case Path::CublasFp16Hgemm: return "cuBLAS fp16, 16F compute";
        case Path::CublasInt8: return "cuBLAS int8 (GemmEx)";
    }
    return "?";
}

bool is_int8(Path p) {
    return p == Path::Int8WmmaGlobal || p == Path::Int8WmmaStaged ||
           p == Path::Int8WmmaStagedWord || p == Path::Int8WmmaPipelined ||
           p == Path::Int8Dp4a || p == Path::CublasInt8;
}

bool is_fp16(Path p) {
    return p == Path::Fp16WmmaGlobal || p == Path::Fp16WmmaStaged ||
           p == Path::Fp16WmmaStagedWord || p == Path::Fp16WmmaPipelined ||
           p == Path::CublasFp16 || p == Path::CublasFp16Hgemm;
}

bool is_cublas(Path p) {
    return p == Path::CublasFp32 || p == Path::CublasFp16 || p == Path::CublasFp16Hgemm ||
           p == Path::CublasInt8;
}

std::vector<Path> all_paths() {
    return {Path::Fp32Tiled,
            Path::Fp16WmmaGlobal, Path::Fp16WmmaStaged, Path::Fp16WmmaStagedWord,
            Path::Fp16WmmaPipelined,
            Path::Int8WmmaGlobal, Path::Int8WmmaStaged, Path::Int8WmmaStagedWord,
            Path::Int8WmmaPipelined,
            Path::Int8Dp4a,
            Path::CublasFp32, Path::CublasFp16, Path::CublasFp16Hgemm, Path::CublasInt8};
}

Quantized quantize_symmetric(const std::vector<float>& x) {
    Quantized r;
    float amax = 0.0f;
    for (float v : x) amax = std::max(amax, std::fabs(v));
    r.scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    r.q.resize(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float q = std::nearbyint(x[i] / r.scale);
        r.q[i] = static_cast<std::int8_t>(std::clamp(q, -127.0f, 127.0f));
    }
    return r;
}

double Gemm::gflops(int n, float ms) {
    if (ms <= 0.0f) return 0.0;
    const double ops = double(n) * n * n;
    return ops / (ms / 1000.0) / 1e9;
}

struct Gemm::Impl {
    int n = 0;
    std::size_t cells = 0;
    float scale_a = 1.0f, scale_b = 1.0f;
    bool have_inputs = false;

    // Row-major operands for cuBLAS; the transposed B for the hand kernels.
    DeviceBuffer<float> a32, b32, bt32;
    DeviceBuffer<__half> a16, b16, bt16;
    DeviceBuffer<signed char> a8, b8, bt8;
    DeviceBuffer<float> c32;
    DeviceBuffer<int> c_i32;
    DeviceBuffer<__half> c16;   // Hgemm's result

    cublasHandle_t blas = nullptr;

    ~Impl() {
        if (blas) cublasDestroy(blas);
    }

    void launch_bands(const std::function<void(int, int)>& launch) {
        const int tiles = n / 16;
        for (int off = 0; off < tiles; off += kTileRowsPerLaunch) {
            launch(off, std::min(kTileRowsPerLaunch, tiles - off));
            CU_CHECK_KERNEL();
        }
    }

    template <typename T, typename Acc>
    void run_staging(Staging s, int band, const T* A, const T* Bt, Acc* C) {
        const int tiles = n / 16;
        launch_bands([&](int off, int rows) {
            launch_staging_band<T, Acc>(s, band, dim3(tiles, rows), A, Bt, C, n, off);
        });
    }

    void run_band(Staging s, bool int8, int band) {
        if (!have_inputs) throw std::logic_error("Gemm: set_inputs() first");
        if (int8)
            run_staging<signed char, int>(s, band, a8.get(), bt8.get(), c_i32.get());
        else
            run_staging<__half, float>(s, band, a16.get(), bt16.get(), c32.get());
    }

    void run(Path p) {
        if (!have_inputs) throw std::logic_error("Gemm: set_inputs() first");
        const int N = n;
        const int tiles = N / 16;
        switch (p) {
            case Path::Fp32Tiled:
                launch_bands([&](int off, int rows) {
                    k_fp32_tiled<<<dim3(tiles, rows), dim3(16, 16)>>>(
                        a32.get(), bt32.get(), c32.get(), N, off);
                });
                break;
            case Path::Fp16WmmaGlobal:
                launch_bands([&](int off, int rows) {
                    k_wmma_global<__half, float><<<dim3(tiles, (rows + 3) / 4), dim3(32, 4)>>>(
                        a16.get(), bt16.get(), c32.get(), N, off);
                });
                break;
            case Path::Fp16WmmaStaged:
                run_band(Staging::ByElement, false, default_band(Staging::ByElement));
                break;
            case Path::Fp16WmmaStagedWord:
                run_band(Staging::ByWord, false, default_band(Staging::ByWord));
                break;
            case Path::Fp16WmmaPipelined:
                run_band(Staging::Pipelined, false, default_band(Staging::Pipelined));
                break;
            case Path::Int8WmmaGlobal:
                launch_bands([&](int off, int rows) {
                    k_wmma_global<signed char, int><<<dim3(tiles, (rows + 3) / 4), dim3(32, 4)>>>(
                        a8.get(), bt8.get(), c_i32.get(), N, off);
                });
                break;
            case Path::Int8WmmaStaged:
                run_band(Staging::ByElement, true, default_band(Staging::ByElement));
                break;
            case Path::Int8WmmaStagedWord:
                run_band(Staging::ByWord, true, default_band(Staging::ByWord));
                break;
            case Path::Int8WmmaPipelined:
                run_band(Staging::Pipelined, true, default_band(Staging::Pipelined));
                break;
            case Path::Int8Dp4a:
                launch_bands([&](int off, int rows) {
                    k_int8_dp4a<<<dim3(tiles, rows), dim3(16, 16)>>>(
                        a8.get(), bt8.get(), c_i32.get(), N, off);
                });
                break;
            // cuBLAS is column-major. A row-major array read column-major is the
            // transpose, so row-major C = A*B is column-major C^T = B^T * A^T:
            // pass B then A, and no copies are needed.
            case Path::CublasFp32: {
                const float alpha = 1.0f, beta = 0.0f;
                cublas_check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                         b32.get(), N, a32.get(), N, &beta, c32.get(), N),
                             "Sgemm");
                CU_CHECK(cudaDeviceSynchronize());
                break;
            }
            case Path::CublasFp16: {
                const float alpha = 1.0f, beta = 0.0f;
                cublas_check(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                          b16.get(), CUDA_R_16F, N, a16.get(), CUDA_R_16F, N,
                                          &beta, c32.get(), CUDA_R_32F, N,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                             "GemmEx fp16");
                CU_CHECK(cudaDeviceSynchronize());
                break;
            }
            case Path::CublasFp16Hgemm: {
                // All-fp16: the configuration cuBLAS routes to tensor cores.
                // Alpha and beta must match the 16F compute type.
                const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
                cublas_check(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                          b16.get(), CUDA_R_16F, N, a16.get(), CUDA_R_16F, N,
                                          &beta, c16.get(), CUDA_R_16F, N,
                                          CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT),
                             "GemmEx fp16/16F");
                CU_CHECK(cudaDeviceSynchronize());
                break;
            }
            case Path::CublasInt8: {
                const int alpha = 1, beta = 0;
                cublas_check(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                          b8.get(), CUDA_R_8I, N, a8.get(), CUDA_R_8I, N,
                                          &beta, c_i32.get(), CUDA_R_32I, N,
                                          CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
                             "GemmEx int8");
                CU_CHECK(cudaDeviceSynchronize());
                break;
            }
        }
    }
};

Gemm::Gemm(int n) : impl_(new Impl) {
    try {
        if (n <= 0 || n % 16 != 0)
            throw std::invalid_argument("Gemm: n must be a positive multiple of 16");
        if (n > 16384)
            throw std::invalid_argument("Gemm: n above 16384 overflows int indexing");
        impl_->n = n;
        impl_->cells = std::size_t(n) * n;
        const std::size_t c = impl_->cells;
        impl_->a32 = DeviceBuffer<float>(c);
        impl_->b32 = DeviceBuffer<float>(c);
        impl_->bt32 = DeviceBuffer<float>(c);
        impl_->a16 = DeviceBuffer<__half>(c);
        impl_->b16 = DeviceBuffer<__half>(c);
        impl_->bt16 = DeviceBuffer<__half>(c);
        impl_->a8 = DeviceBuffer<signed char>(c);
        impl_->b8 = DeviceBuffer<signed char>(c);
        impl_->bt8 = DeviceBuffer<signed char>(c);
        impl_->c32 = DeviceBuffer<float>(c);
        impl_->c_i32 = DeviceBuffer<int>(c);
        impl_->c16 = DeviceBuffer<__half>(c);
        cublas_check(cublasCreate(&impl_->blas), "create");
        // The default math mode already permits tensor cores in cuBLAS 11+;
        // set explicitly so the comparison does not depend on a library default.
        cublas_check(cublasSetMathMode(impl_->blas, CUBLAS_DEFAULT_MATH), "math mode");
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

Gemm::~Gemm() { delete impl_; }

int Gemm::n() const { return impl_->n; }

void Gemm::set_inputs(const std::vector<float>& a, const std::vector<float>& b) {
    const std::size_t c = impl_->cells;
    if (a.size() != c || b.size() != c)
        throw std::invalid_argument("Gemm::set_inputs: operands must be n*n");

    const auto bt = transpose(b, impl_->n);
    impl_->a32.upload(a.data());
    impl_->b32.upload(b.data());
    impl_->bt32.upload(bt.data());

    std::vector<__half> h(c);
    for (std::size_t i = 0; i < c; ++i) h[i] = __float2half(a[i]);
    impl_->a16.upload(h.data());
    for (std::size_t i = 0; i < c; ++i) h[i] = __float2half(b[i]);
    impl_->b16.upload(h.data());
    for (std::size_t i = 0; i < c; ++i) h[i] = __float2half(bt[i]);
    impl_->bt16.upload(h.data());

    const auto qa = quantize_symmetric(a);
    const auto qb = quantize_symmetric(b);
    const auto qbt = quantize_symmetric(bt);   // same scale as qb: same values
    impl_->scale_a = qa.scale;
    impl_->scale_b = qb.scale;
    impl_->a8.upload(reinterpret_cast<const signed char*>(qa.q.data()));
    impl_->b8.upload(reinterpret_cast<const signed char*>(qb.q.data()));
    impl_->bt8.upload(reinterpret_cast<const signed char*>(qbt.q.data()));
    impl_->have_inputs = true;
}

std::vector<float> Gemm::multiply_band(Staging s, bool int8, int band) {
    impl_->run_band(s, int8, band);
    return multiply(int8 ? Path::Int8WmmaStaged : Path::Fp16WmmaStaged, /*already_run=*/true);
}

std::vector<BandPoint> Gemm::sweep_band(Staging s, bool int8, int iterations, int warmup) {
    std::vector<BandPoint> out;
    for (int band : staging_bands(s, int8)) {
        BandPoint b;
        b.band = band;
        const void* k = int8 ? staging_kernel_band<signed char, int>(s, band)
                             : staging_kernel_band<__half, float>(s, band);
        CU_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b.blocks_per_sm, k,
                                                               staging_threads(s), 0));
        b.compute_warps_per_sm = b.blocks_per_sm;   // one compute warp per block
        b.ms = cu::benchmark([&] { impl_->run_band(s, int8, band); }, iterations, warmup)
                   .median_ms;
        out.push_back(b);
    }
    return out;
}

std::vector<float> Gemm::multiply(Path p) { return multiply(p, false); }

std::vector<float> Gemm::multiply(Path p, bool already_run) {
    if (!already_run) impl_->run(p);
    std::vector<float> out(impl_->cells);
    if (p == Path::CublasFp16Hgemm) {
        std::vector<__half> h(impl_->cells);
        impl_->c16.download(h.data());
        for (std::size_t i = 0; i < h.size(); ++i) out[i] = __half2float(h[i]);
        return out;
    }
    if (is_int8(p)) {
        const int count = static_cast<int>(impl_->cells);
        const int t = 1024;
        k_dequant<<<(count + t - 1) / t, t>>>(impl_->c_i32.get(), impl_->c32.get(), count,
                                              impl_->scale_a * impl_->scale_b);
        CU_CHECK_KERNEL();
    }
    impl_->c32.download(out.data());
    return out;
}

float Gemm::time(Path p, int iterations, int warmup) {
    return cu::benchmark([&] { impl_->run(p); }, iterations, warmup).median_ms;
}

// ---------------------------------------------------------------------------

std::vector<Occupancy> occupancy() {
    struct Entry {
        Path path;
        const void* kernel;
        int threads;
        int compute_warps;
    };
    const Entry entries[] = {
        {Path::Fp32Tiled, reinterpret_cast<const void*>(k_fp32_tiled), 256, 8},
        {Path::Fp16WmmaGlobal,
         reinterpret_cast<const void*>(k_wmma_global<__half, float>), 128, 4},
        {Path::Fp16WmmaStaged,
         staging_kernel_band<__half, float>(Staging::ByElement, kDefaultBandByElement), 32, 1},
        {Path::Fp16WmmaStagedWord,
         staging_kernel_band<__half, float>(Staging::ByWord, kDefaultBandByWord), 32, 1},
        {Path::Fp16WmmaPipelined,
         staging_kernel_band<__half, float>(Staging::Pipelined, kDefaultBandPipelined), 64, 1},
        {Path::Int8WmmaGlobal,
         reinterpret_cast<const void*>(k_wmma_global<signed char, int>), 128, 4},
        {Path::Int8WmmaStaged,
         staging_kernel_band<signed char, int>(Staging::ByElement, kDefaultBandByElement), 32, 1},
        {Path::Int8WmmaStagedWord,
         staging_kernel_band<signed char, int>(Staging::ByWord, kDefaultBandByWord), 32, 1},
        {Path::Int8WmmaPipelined,
         staging_kernel_band<signed char, int>(Staging::Pipelined, kDefaultBandPipelined), 64, 1},
        {Path::Int8Dp4a, reinterpret_cast<const void*>(k_int8_dp4a), 256, 8},
    };
    std::vector<Occupancy> out;
    for (const auto& e : entries) {
        Occupancy o;
        o.path = e.path;
        o.threads_per_block = e.threads;
        o.compute_warps_per_block = e.compute_warps;
        CU_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o.blocks_per_sm, e.kernel,
                                                               e.threads, 0));
        out.push_back(o);
    }
    return out;
}

namespace {
namespace ex = wmma::experimental::precision;

__global__ void k_u4(const unsigned* A, const unsigned* B, int* C) {
    wmma::fragment<wmma::matrix_a, 8, 8, 32, ex::u4, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 8, 8, 32, ex::u4, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 8, 8, 32, int> c;
    wmma::fill_fragment(c, 0);
    wmma::load_matrix_sync(a, A, 32);   // 8 x 32, leading dimension = columns
    wmma::load_matrix_sync(b, B, 32);   // 32 x 8 col-major, leading dim = rows
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 8, wmma::mem_row_major);
}

unsigned u4_a(int r, int x) { return unsigned((r * 3 + x * 5) & 15); }
unsigned u4_b(int x, int c) { return unsigned((x * 7 + c * 11) & 15); }

}  // namespace

SubByteCheck check_u4_fragment() {
    const int R = 8, K = 32, N = 8;
    // u4 values are packed eight to an unsigned, low nibble first.
    std::vector<unsigned> ha(64, 0), hb(64, 0);
    for (int r = 0; r < R; ++r)
        for (int x = 0; x < K; ++x) {
            const int e = r * K + x;
            ha[e / 8] |= u4_a(r, x) << (4 * (e % 8));
        }
    for (int c = 0; c < N; ++c)
        for (int x = 0; x < K; ++x) {
            const int e = x + c * K;
            hb[e / 8] |= u4_b(x, c) << (4 * (e % 8));
        }

    DeviceBuffer<unsigned> da(64), db(64);
    DeviceBuffer<int> dc(64);
    da.upload(ha.data());
    db.upload(hb.data());
    k_u4<<<1, 32>>>(da.get(), db.get(), dc.get());
    CU_CHECK_KERNEL();
    std::vector<int> hc(64);
    dc.download(hc.data());

    SubByteCheck r;
    r.total = R * N;
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < N; ++j) {
            long ref = 0;
            for (int x = 0; x < K; ++x) ref += long(u4_a(i, x)) * u4_b(x, j);
            r.wrong += (hc[i * N + j] != ref);
        }
    return r;
}

}  // namespace tc
