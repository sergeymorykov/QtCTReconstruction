#pragma once
// AlignedBuf.h — общие утилиты для выравнивания памяти и restrict-указателей,
// используемые в горячих CPU/OpenMP-путях (RadonTransform, FilteredBackprojection,
// OptimizedBackprojectionCPU).
//
// CT_RESTRICT — портабельный аналог __restrict__:
//   MSVC: __restrict
//   GCC/Clang: __restrict__
//
// AlignedAllocator<T, N> — std::allocator-совместимый аллокатор с N-байтовым
// выравниванием. Применяется к thread-local std::vector<float> FFT-буферам
// (proj_padded, q) для устранения штрафа за невыровненные SIMD-загрузки
// (особенно для AVX-512, где штраф самый ощутимый).

#include <cstddef>
#include <cstdlib>
#include <new>
#include <type_traits>

#if defined(_MSC_VER)
  #define CT_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
  #define CT_RESTRICT __restrict__
#else
  #define CT_RESTRICT
#endif

namespace ct {

// Кросс-платформенная аллокация выровненной памяти.
inline void* aligned_malloc(std::size_t bytes, std::size_t alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, alignment);
#else
    // std::aligned_alloc требует bytes % alignment == 0.
    const std::size_t rounded = ((bytes + alignment - 1) / alignment) * alignment;
    return std::aligned_alloc(alignment, rounded);
#endif
}

inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// std::allocator-совместимый аллокатор с N-байтным выравниванием.
template <typename T, std::size_t N = 64>
struct AlignedAllocator {
    using value_type = T;
    static constexpr std::size_t alignment = N;

    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, N>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* p = ::ct::aligned_malloc(n * sizeof(T), N);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t /*n*/) noexcept {
        ::ct::aligned_free(p);
    }

    template <typename U> struct rebind { using other = AlignedAllocator<U, N>; };
};

template <typename T, typename U, std::size_t N>
bool operator==(const AlignedAllocator<T, N>&, const AlignedAllocator<U, N>&) noexcept { return true; }
template <typename T, typename U, std::size_t N>
bool operator!=(const AlignedAllocator<T, N>&, const AlignedAllocator<U, N>&) noexcept { return false; }

} // namespace ct
