// GGUF parser. Host-only C++20 -- no CUDA in this translation unit.

#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace llm {
namespace {

// Metadata value type tags, fixed by the format.
enum class MetaType : std::uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12,
};

// A bounds-checked cursor. Every read validates against the end of the buffer:
// a truncated or hostile file must produce a clean error, never a read past
// the mapping (which on Windows is an access violation, not a bad value).
class Cursor {
public:
    Cursor(const std::uint8_t* p, std::size_t n) : p_(p), n_(n) {}

    void need(std::size_t k) const {
        if (off_ + k > n_)
            throw std::runtime_error("GGUF: truncated file (wanted " + std::to_string(k) +
                                     " bytes at offset " + std::to_string(off_) + ")");
    }
    template <typename T>
    T pod() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_ + off_, sizeof(T));
        off_ += sizeof(T);
        return v;
    }
    std::string str() {
        std::uint64_t len = pod<std::uint64_t>();
        if (len > (1u << 30)) throw std::runtime_error("GGUF: implausible string length");
        need(static_cast<std::size_t>(len));
        std::string s(reinterpret_cast<const char*>(p_ + off_), static_cast<std::size_t>(len));
        off_ += static_cast<std::size_t>(len);
        return s;
    }
    std::size_t offset() const { return off_; }
    void seek(std::size_t o) {
        if (o > n_) throw std::runtime_error("GGUF: seek past end");
        off_ = o;
    }

private:
    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t off_ = 0;
};

MetaValue read_value(Cursor& c, MetaType t);

void skip_value(Cursor& c, MetaType t) { (void)read_value(c, t); }

MetaValue read_value(Cursor& c, MetaType t) {
    switch (t) {
        case MetaType::UINT8:   return static_cast<std::int64_t>(c.pod<std::uint8_t>());
        case MetaType::INT8:    return static_cast<std::int64_t>(c.pod<std::int8_t>());
        case MetaType::UINT16:  return static_cast<std::int64_t>(c.pod<std::uint16_t>());
        case MetaType::INT16:   return static_cast<std::int64_t>(c.pod<std::int16_t>());
        case MetaType::UINT32:  return static_cast<std::int64_t>(c.pod<std::uint32_t>());
        case MetaType::INT32:   return static_cast<std::int64_t>(c.pod<std::int32_t>());
        case MetaType::UINT64:  return static_cast<std::int64_t>(c.pod<std::uint64_t>());
        case MetaType::INT64:   return static_cast<std::int64_t>(c.pod<std::int64_t>());
        case MetaType::BOOL:    return static_cast<bool>(c.pod<std::uint8_t>() != 0);
        case MetaType::FLOAT32: return static_cast<double>(c.pod<float>());
        case MetaType::FLOAT64: return c.pod<double>();
        case MetaType::STRING:  return c.str();
        case MetaType::ARRAY: {
            // Arrays are consumed and discarded: this engine needs only scalars
            // and strings from the header. They must still be parsed correctly
            // to find where the next key begins.
            auto elem = static_cast<MetaType>(c.pod<std::uint32_t>());
            std::uint64_t n = c.pod<std::uint64_t>();
            if (n > (1ull << 32)) throw std::runtime_error("GGUF: implausible array length");
            for (std::uint64_t i = 0; i < n; ++i) skip_value(c, elem);
            return std::monostate{};
        }
        default:
            throw std::runtime_error("GGUF: unknown metadata type " +
                                     std::to_string(static_cast<std::uint32_t>(t)));
    }
}

// Tensor sizes are products of four untrusted 64-bit values. Every product is
// checked, and capped well below 2^64 so that later byte arithmetic (elements
// times bytes per block, offset plus size) cannot wrap either.
constexpr std::uint64_t kMaxElements = std::uint64_t(1) << 50;

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const std::string& what) {
    if (a != 0 && b > kMaxElements / a)
        throw std::runtime_error("GGUF: " + what + " overflows (" + std::to_string(a) + " x " +
                                 std::to_string(b) + ")");
    return a * b;
}

}  // namespace

TypeTraits type_traits(GgmlType t) {
    switch (t) {
        case GgmlType::F32:  return {1, 4, "F32"};
        case GgmlType::F16:  return {1, 2, "F16"};
        // Q4_0: 32 weights share one fp16 scale -> 2 + 16 = 18 bytes.
        case GgmlType::Q4_0: return {32, 18, "Q4_0"};
        case GgmlType::Q4_1: return {32, 20, "Q4_1"};
        case GgmlType::Q5_0: return {32, 22, "Q5_0"};
        case GgmlType::Q5_1: return {32, 24, "Q5_1"};
        case GgmlType::Q8_0: return {32, 34, "Q8_0"};
        case GgmlType::Q8_1: return {32, 36, "Q8_1"};
        // K-quant super-blocks of 256, sizes from the ggml block structs.
        case GgmlType::Q2_K: return {256, 84, "Q2_K"};
        case GgmlType::Q3_K: return {256, 110, "Q3_K"};
        case GgmlType::Q4_K: return {256, 144, "Q4_K"};
        case GgmlType::Q5_K: return {256, 176, "Q5_K"};
        case GgmlType::Q6_K: return {256, 210, "Q6_K"};
        case GgmlType::Q8_K: return {256, 292, "Q8_K"};
        default: return {1, 0, "unknown"};
    }
}

std::uint64_t GgufTensor::num_elements() const {
    std::uint64_t n = 1;
    for (auto d : dims) n *= d;
    return dims.empty() ? 0 : n;
}

std::uint64_t GgufTensor::num_bytes() const {
    auto tr = type_traits(type);
    if (tr.block_bytes == 0) return 0;
    std::uint64_t n = num_elements();
    return (n / tr.block_elems) * tr.block_bytes;
}

// ---------------------------------------------------------------------------
struct GgufFile::Impl {
    std::vector<std::uint8_t> owned;      // used by from_memory
    const std::uint8_t* base = nullptr;
    std::size_t size = 0;

    std::uint32_t version = 0;
    std::vector<GgufTensor> tensors;
    std::map<std::string, MetaValue> meta;
    std::map<std::string, std::vector<std::string>> string_arrays;
    std::map<std::string, std::vector<double>> float_arrays;
    std::map<std::string, std::vector<std::int64_t>> int_arrays;
    std::size_t data_offset = 0;

#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    void* view = nullptr;
#else
    int fd = -1;
    void* view = nullptr;
#endif

    ~Impl() {
#ifdef _WIN32
        if (view) UnmapViewOfFile(view);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
        if (view && view != MAP_FAILED) munmap(view, size);
        if (fd >= 0) close(fd);
#endif
    }

    void parse();
};

void GgufFile::Impl::parse() {
    Cursor c(base, size);

    char magic[4];
    c.need(4);
    std::memcpy(magic, base, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) throw std::runtime_error("GGUF: bad magic");
    c.seek(4);

    version = c.pod<std::uint32_t>();
    if (version < 2 || version > 3)
        throw std::runtime_error("GGUF: unsupported version " + std::to_string(version));

    std::uint64_t n_tensors = c.pod<std::uint64_t>();
    std::uint64_t n_meta = c.pod<std::uint64_t>();
    if (n_tensors > (1u << 20) || n_meta > (1u << 20))
        throw std::runtime_error("GGUF: implausible header counts");

    for (std::uint64_t i = 0; i < n_meta; ++i) {
        std::string key = c.str();
        auto t = static_cast<MetaType>(c.pod<std::uint32_t>());
        if (t != MetaType::ARRAY) {
            meta[key] = read_value(c, t);
            continue;
        }
        // Arrays are read here rather than in read_value, which has no key to
        // file them under. Every element is still bounds-checked by Cursor.
        auto elem = static_cast<MetaType>(c.pod<std::uint32_t>());
        std::uint64_t n = c.pod<std::uint64_t>();
        if (n > (1ull << 32)) throw std::runtime_error("GGUF: implausible array length");
        if (elem == MetaType::STRING) {
            auto& v = string_arrays[key];
            v.reserve(static_cast<std::size_t>(std::min<std::uint64_t>(n, 1u << 20)));
            for (std::uint64_t j = 0; j < n; ++j) v.push_back(c.str());
        } else if (elem == MetaType::FLOAT32 || elem == MetaType::FLOAT64) {
            auto& v = float_arrays[key];
            for (std::uint64_t j = 0; j < n; ++j)
                v.push_back(std::get<double>(read_value(c, elem)));
        } else if (elem == MetaType::ARRAY) {
            for (std::uint64_t j = 0; j < n; ++j) skip_value(c, elem);
        } else if (elem == MetaType::BOOL) {
            for (std::uint64_t j = 0; j < n; ++j) skip_value(c, elem);
        } else {
            auto& v = int_arrays[key];
            for (std::uint64_t j = 0; j < n; ++j)
                v.push_back(std::get<std::int64_t>(read_value(c, elem)));
        }
    }

    // Alignment comes from metadata, which is untrusted: it must be a power of
    // two, or both the data offset and every tensor's alignment are meaningless.
    std::uint64_t align = 32;
    if (auto it = meta.find("general.alignment"); it != meta.end()) {
        auto* v = std::get_if<std::int64_t>(&it->second);
        if (!v || *v <= 0 || (*v & (*v - 1)) != 0 || *v > (1 << 20))
            throw std::runtime_error("GGUF: general.alignment must be a power of two");
        align = static_cast<std::uint64_t>(*v);
    }

    tensors.reserve(static_cast<std::size_t>(n_tensors));
    std::map<std::string, int> seen;
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = c.str();
        if (!seen.emplace(t.name, 0).second)
            throw std::runtime_error("GGUF: duplicate tensor name '" + t.name + "'");
        std::uint32_t nd = c.pod<std::uint32_t>();
        if (nd == 0 || nd > 4) throw std::runtime_error("GGUF: bad tensor rank for '" + t.name + "'");
        std::uint64_t elements = 1;
        for (std::uint32_t d = 0; d < nd; ++d) {
            const std::uint64_t dim = c.pod<std::uint64_t>();
            if (dim == 0) throw std::runtime_error("GGUF: tensor '" + t.name + "' has a zero dimension");
            elements = checked_mul(elements, dim, "element count of '" + t.name + "'");
            t.dims.push_back(dim);
        }
        t.type = static_cast<GgmlType>(c.pod<std::uint32_t>());
        t.offset = c.pod<std::uint64_t>();

        auto tr = type_traits(t.type);
        if (tr.block_bytes == 0)
            throw std::runtime_error("GGUF: unsupported tensor type in '" + t.name + "'");
        if (elements % tr.block_elems != 0)
            throw std::runtime_error("GGUF: '" + t.name + "' element count is not a multiple "
                                     "of the block size for " + tr.name);
        (void)checked_mul(elements / tr.block_elems, tr.block_bytes, "byte size of '" + t.name + "'");
        if (t.offset % align != 0)
            throw std::runtime_error("GGUF: tensor '" + t.name + "' offset is not aligned to " +
                                     std::to_string(align));
        tensors.push_back(std::move(t));
    }

    std::size_t off = c.offset();
    data_offset = static_cast<std::size_t>((off + align - 1) / align * align);
    if (data_offset > size) throw std::runtime_error("GGUF: data section starts past end");

    // Every tensor must lie wholly inside the blob, and no two may overlap.
    // Checking once here means tensor_data() can hand out pointers without
    // re-validating. The comparisons are arranged so none of them can wrap.
    const std::uint64_t blob = size - data_offset;
    std::vector<std::pair<std::uint64_t, const GgufTensor*>> by_offset;
    for (const auto& t : tensors) {
        const std::uint64_t bytes = t.num_bytes();
        if (t.offset > blob || bytes > blob - t.offset)
            throw std::runtime_error("GGUF: tensor '" + t.name + "' runs past end of file");
        by_offset.emplace_back(t.offset, &t);
    }
    std::sort(by_offset.begin(), by_offset.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (std::size_t i = 1; i < by_offset.size(); ++i) {
        const auto* prev = by_offset[i - 1].second;
        if (prev->offset + prev->num_bytes() > by_offset[i].first)
            throw std::runtime_error("GGUF: tensors '" + prev->name + "' and '" +
                                     by_offset[i].second->name + "' overlap");
    }
}

GgufFile::GgufFile() : impl_(std::make_unique<Impl>()) {}
GgufFile::~GgufFile() = default;
GgufFile::GgufFile(GgufFile&&) noexcept = default;
GgufFile& GgufFile::operator=(GgufFile&&) noexcept = default;

GgufFile GgufFile::from_memory(std::vector<std::uint8_t> bytes) {
    GgufFile f;
    f.impl_->owned = std::move(bytes);
    f.impl_->base = f.impl_->owned.data();
    f.impl_->size = f.impl_->owned.size();
    f.impl_->parse();
    return f;
}

GgufFile GgufFile::open(const std::string& path) {
    GgufFile f;
    auto& I = *f.impl_;

#ifdef _WIN32
    I.file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (I.file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("GGUF: cannot open " + path);
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(I.file, &sz) || sz.QuadPart == 0)
        throw std::runtime_error("GGUF: empty or unreadable " + path);
    I.size = static_cast<std::size_t>(sz.QuadPart);
    I.mapping = CreateFileMappingA(I.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!I.mapping) throw std::runtime_error("GGUF: CreateFileMapping failed");
    I.view = MapViewOfFile(I.mapping, FILE_MAP_READ, 0, 0, 0);
    if (!I.view) throw std::runtime_error("GGUF: MapViewOfFile failed");
#else
    I.fd = ::open(path.c_str(), O_RDONLY);
    if (I.fd < 0) throw std::runtime_error("GGUF: cannot open " + path);
    struct stat st {};
    if (fstat(I.fd, &st) != 0 || st.st_size == 0)
        throw std::runtime_error("GGUF: empty or unreadable " + path);
    I.size = static_cast<std::size_t>(st.st_size);
    I.view = mmap(nullptr, I.size, PROT_READ, MAP_PRIVATE, I.fd, 0);
    if (I.view == MAP_FAILED) throw std::runtime_error("GGUF: mmap failed");
#endif

    I.base = static_cast<const std::uint8_t*>(I.view);
    I.parse();
    return f;
}

std::uint32_t GgufFile::version() const { return impl_->version; }
const std::vector<GgufTensor>& GgufFile::tensors() const { return impl_->tensors; }
std::uint64_t GgufFile::data_size() const { return impl_->size - impl_->data_offset; }

const GgufTensor* GgufFile::find(const std::string& name) const {
    for (const auto& t : impl_->tensors)
        if (t.name == name) return &t;
    return nullptr;
}

std::optional<std::int64_t> GgufFile::meta_int(const std::string& key) const {
    auto it = impl_->meta.find(key);
    if (it == impl_->meta.end()) return std::nullopt;
    if (auto* v = std::get_if<std::int64_t>(&it->second)) return *v;
    return std::nullopt;
}
std::optional<double> GgufFile::meta_float(const std::string& key) const {
    auto it = impl_->meta.find(key);
    if (it == impl_->meta.end()) return std::nullopt;
    if (auto* v = std::get_if<double>(&it->second)) return *v;
    if (auto* v = std::get_if<std::int64_t>(&it->second)) return static_cast<double>(*v);
    return std::nullopt;
}
std::optional<std::string> GgufFile::meta_string(const std::string& key) const {
    auto it = impl_->meta.find(key);
    if (it == impl_->meta.end()) return std::nullopt;
    if (auto* v = std::get_if<std::string>(&it->second)) return *v;
    return std::nullopt;
}

const std::vector<std::string>* GgufFile::meta_string_array(const std::string& key) const {
    auto it = impl_->string_arrays.find(key);
    return it == impl_->string_arrays.end() ? nullptr : &it->second;
}
const std::vector<double>* GgufFile::meta_float_array(const std::string& key) const {
    auto it = impl_->float_arrays.find(key);
    return it == impl_->float_arrays.end() ? nullptr : &it->second;
}
const std::vector<std::int64_t>* GgufFile::meta_int_array(const std::string& key) const {
    auto it = impl_->int_arrays.find(key);
    return it == impl_->int_arrays.end() ? nullptr : &it->second;
}

const void* GgufFile::tensor_data(const GgufTensor& t) const {
    return impl_->base + impl_->data_offset + t.offset;
}

}  // namespace llm
