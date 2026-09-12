#pragma once
//
// Sparse matrix-vector multiply and iterative solvers. No CUDA syntax here.
//
// SpMV is the kernel that decides the runtime of almost every iterative method
// in scientific computing, and it is the textbook case of an IRREGULAR memory
// problem: A is stored compactly, but x is indexed by the column array, so
// every access into x is a scatter the hardware cannot coalesce.
//
// Three storage formats are implemented because no single one wins:
//   CSR       compact, but a thread-per-row kernel stalls on degree skew
//   ELLPACK   perfectly coalesced, but pads every row to the longest one
//   Hybrid    ELLPACK for the common case, CSR for the few long rows
//
#include <cstddef>
#include <cstdint>
#include <vector>

namespace spmv {

// Coordinate triplets, the format matrices are usually born in.
struct CooMatrix {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::vector<std::int32_t> row_idx;
    std::vector<std::int32_t> col_idx;
    std::vector<float> values;

    std::int64_t nnz() const { return (std::int64_t)values.size(); }

    // Banded: every row has the same width. The best case for ELLPACK.
    static CooMatrix banded(std::int32_t n, int bandwidth, unsigned seed);
    // Power-law row degrees: a few very long rows. The worst case, and the
    // reason hybrid formats exist at all.
    static CooMatrix power_law(std::int32_t n, int avg_nnz_per_row, unsigned seed);
    // 5-point Laplacian on a w x h grid: symmetric positive definite, so
    // conjugate gradient is guaranteed to converge on it.
    static CooMatrix laplacian_2d(std::int32_t w, std::int32_t h);
};

enum class Format {
    CsrScalar,   // one thread per row
    CsrVector,   // one warp per row
    Ellpack,     // padded, column-major, fully coalesced
    Hybrid,      // ELL for rows up to a cut, CSR for the overflow
};

struct FormatStats {
    std::int64_t nnz = 0;
    std::int64_t padded_entries = 0;   // ELL padding waste
    int max_row_nnz = 0;
    double mean_row_nnz = 0.0;
    std::size_t device_bytes = 0;
    int ell_width = 0;                 // hybrid cut, 0 if not applicable
};

class SparseMatrix {
public:
    explicit SparseMatrix(const CooMatrix& coo);
    ~SparseMatrix();
    SparseMatrix(const SparseMatrix&) = delete;
    SparseMatrix& operator=(const SparseMatrix&) = delete;

    // y = A*x
    std::vector<float> multiply(const std::vector<float>& x, Format fmt,
                                float* elapsed_ms = nullptr) const;

    // Conjugate gradient. Requires a symmetric positive definite matrix;
    // laplacian_2d() produces one.
    struct CgResult {
        std::vector<float> x;
        int iterations = 0;
        double residual = 0.0;
        float elapsed_ms = 0.0f;
    };
    CgResult solve_cg(const std::vector<float>& b, int max_iter, double tol,
                      Format fmt = Format::Hybrid) const;

    FormatStats stats(Format fmt) const;
    std::int32_t rows() const;
    std::int32_t cols() const;
    std::int64_t nnz() const;

    // Host reference.
    static std::vector<float> multiply_cpu(const CooMatrix& coo,
                                           const std::vector<float>& x);

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace spmv
