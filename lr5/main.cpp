#include <emmintrin.h>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static inline void compare_scalar(const int8_t* a, const int8_t* b, uint8_t* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = (a[i] >= b[i]) ? 1u : 0u;
}

static inline void compare_sse2(const int8_t* a, const int8_t* b, uint8_t* c, size_t n) {
    const __m128i ones = _mm_set1_epi8(1);
    const __m128i zeros = _mm_setzero_si128();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m128i gt = _mm_cmpgt_epi8(va, vb);
        __m128i eq = _mm_cmpeq_epi8(va, vb);
        __m128i ge = _mm_or_si128(gt, eq);
        __m128i res = _mm_and_si128(ge, ones);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i), res);
    }
    for (; i < n; ++i) c[i] = (a[i] >= b[i]) ? 1u : 0u;
}

static inline void compare_sse2_unroll2(const int8_t* a, const int8_t* b, uint8_t* c, size_t n) {
    const __m128i ones = _mm_set1_epi8(1);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m128i va0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m128i vb0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m128i va1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i + 16));
        __m128i vb1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i + 16));

        __m128i ge0 = _mm_or_si128(_mm_cmpgt_epi8(va0, vb0), _mm_cmpeq_epi8(va0, vb0));
        __m128i ge1 = _mm_or_si128(_mm_cmpgt_epi8(va1, vb1), _mm_cmpeq_epi8(va1, vb1));

        _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i), _mm_and_si128(ge0, ones));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i + 16), _mm_and_si128(ge1, ones));
    }
    for (; i < n; ++i) c[i] = (a[i] >= b[i]) ? 1u : 0u;
}

static inline void compare_sse2_unroll4(const int8_t* a, const int8_t* b, uint8_t* c, size_t n) {
    const __m128i ones = _mm_set1_epi8(1);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        for (int k = 0; k < 4; ++k) {
            const size_t off = i + 16 * k;
            __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + off));
            __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + off));
            __m128i ge = _mm_or_si128(_mm_cmpgt_epi8(va, vb), _mm_cmpeq_epi8(va, vb));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + off), _mm_and_si128(ge, ones));
        }
    }
    for (; i < n; ++i) c[i] = (a[i] >= b[i]) ? 1u : 0u;
}

static inline void compare_sse2_unroll8(const int8_t* a, const int8_t* b, uint8_t* c, size_t n) {
    const __m128i ones = _mm_set1_epi8(1);
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        for (int k = 0; k < 8; ++k) {
            const size_t off = i + 16 * k;
            __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + off));
            __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + off));
            __m128i ge = _mm_or_si128(_mm_cmpgt_epi8(va, vb), _mm_cmpeq_epi8(va, vb));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(c + off), _mm_and_si128(ge, ones));
        }
    }
    for (; i < n; ++i) c[i] = (a[i] >= b[i]) ? 1u : 0u;
}

using Fn = void(*)(const int8_t*, const int8_t*, uint8_t*, size_t);

double benchmark(Fn fn, const std::vector<int8_t>& a, const std::vector<int8_t>& b, std::vector<uint8_t>& c, int reps=30) {
    double best_ms = 1e100;
    for (int r = 0; r < reps; ++r) {
        auto t1 = Clock::now();
        fn(a.data(), b.data(), c.data(), a.size());
        auto t2 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (ms < best_ms) best_ms = ms;
    }
    return best_ms;
}

bool equals(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) {
    return x == y;
}

int main() {
    const size_t n = 1'000'000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> a(n), b(n);
    for (size_t i = 0; i < n; ++i) { a[i] = static_cast<int8_t>(dist(rng)); b[i] = static_cast<int8_t>(dist(rng)); }

    std::vector<uint8_t> ref(n), out(n);
    compare_scalar(a.data(), b.data(), ref.data(), n);

    compare_sse2(a.data(), b.data(), out.data(), n);
    bool ok1 = equals(ref, out);
    compare_sse2_unroll2(a.data(), b.data(), out.data(), n);
    bool ok2 = equals(ref, out);
    compare_sse2_unroll4(a.data(), b.data(), out.data(), n);
    bool ok4 = equals(ref, out);
    compare_sse2_unroll8(a.data(), b.data(), out.data(), n);
    bool ok8 = equals(ref, out);

    std::cout << std::boolalpha;
    std::cout << "correct_sse2=" << ok1 << "\n";
    std::cout << "correct_unroll2=" << ok2 << "\n";
    std::cout << "correct_unroll4=" << ok4 << "\n";
    std::cout << "correct_unroll8=" << ok8 << "\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "scalar=" << benchmark(compare_scalar, a, b, out) << " ms\n";
    std::cout << "sse2=" << benchmark(compare_sse2, a, b, out) << " ms\n";
    std::cout << "sse2_unroll2=" << benchmark(compare_sse2_unroll2, a, b, out) << " ms\n";
    std::cout << "sse2_unroll4=" << benchmark(compare_sse2_unroll4, a, b, out) << " ms\n";
    std::cout << "sse2_unroll8=" << benchmark(compare_sse2_unroll8, a, b, out) << " ms\n";
    return 0;
}
