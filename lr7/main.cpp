#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <immintrin.h>

void blur_scalar(const unsigned char* src, unsigned char* dst, int w, int h) {
    for (int y = 0; y < h - 1; ++y) {
        for (int x = 0; x < w - 1; ++x) {
            int a = src[y * w + x];
            int b = src[y * w + x + 1];
            int c = src[(y + 1) * w + x];
            int d = src[(y + 1) * w + x + 1];

            int result = (a + b + c + d) / 4;
            dst[y * w + x] = static_cast<unsigned char>(result);
        }
    }
}

void blur_simd(const unsigned char* src, unsigned char* dst, int w, int h) {
    for (int y = 0; y < h - 1; ++y) {
        int x = 0;

        for (; x <= w - 1 - 16; x += 16) {
            __m128i a = _mm_loadu_si128((const __m128i*)&src[y * w + x]);
            __m128i b = _mm_loadu_si128((const __m128i*)&src[y * w + x + 1]);
            __m128i c = _mm_loadu_si128((const __m128i*)&src[(y + 1) * w + x]);
            __m128i d = _mm_loadu_si128((const __m128i*)&src[(y + 1) * w + x + 1]);

            __m128i zero = _mm_setzero_si128();

            __m128i a_lo = _mm_unpacklo_epi8(a, zero);
            __m128i a_hi = _mm_unpackhi_epi8(a, zero);
            __m128i b_lo = _mm_unpacklo_epi8(b, zero);
            __m128i b_hi = _mm_unpackhi_epi8(b, zero);
            __m128i c_lo = _mm_unpacklo_epi8(c, zero);
            __m128i c_hi = _mm_unpackhi_epi8(c, zero);
            __m128i d_lo = _mm_unpacklo_epi8(d, zero);
            __m128i d_hi = _mm_unpackhi_epi8(d, zero);

            __m128i sum_lo = _mm_add_epi16(_mm_add_epi16(a_lo, b_lo), _mm_add_epi16(c_lo, d_lo));
            __m128i sum_hi = _mm_add_epi16(_mm_add_epi16(a_hi, b_hi), _mm_add_epi16(c_hi, d_hi));

            __m128i res_lo = _mm_srli_epi16(sum_lo, 2);
            __m128i res_hi = _mm_srli_epi16(sum_hi, 2);

            __m128i result = _mm_packus_epi16(res_lo, res_hi);

            _mm_storeu_si128((__m128i*)&dst[y * w + x], result);
        }

        for (; x < w - 1; ++x) {
            int a = src[y * w + x];
            int b = src[y * w + x + 1];
            int c = src[(y + 1) * w + x];
            int d = src[(y + 1) * w + x + 1];

            dst[y * w + x] = static_cast<unsigned char>((a + b + c + d) / 4);
        }
    }
}

int main() {
    const int w = 2048;
    const int h = 2048;

    std::vector<unsigned char> src(w * h);
    std::vector<unsigned char> dst_scalar(w * h, 0);
    std::vector<unsigned char> dst_simd(w * h, 0);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            src[y * w + x] = static_cast<unsigned char>((x + y) % 256);
        }
    }

    auto start_scalar = std::chrono::high_resolution_clock::now();
    blur_scalar(src.data(), dst_scalar.data(), w, h);
    auto end_scalar = std::chrono::high_resolution_clock::now();

    auto start_simd = std::chrono::high_resolution_clock::now();
    blur_simd(src.data(), dst_simd.data(), w, h);
    auto end_simd = std::chrono::high_resolution_clock::now();

    double time_scalar = std::chrono::duration<double>(end_scalar - start_scalar).count();
    double time_simd = std::chrono::duration<double>(end_simd - start_simd).count();

    bool correct = true;
    for (int i = 0; i < w * h; ++i) {
        if (dst_scalar[i] != dst_simd[i]) {
            correct = false;
            break;
        }
    }

    std::cout << "Scalar time: " << time_scalar << " seconds\n";
    std::cout << "SIMD time: " << time_simd << " seconds\n";
    std::cout << "Acceleration: " << time_scalar / time_simd << "x\n";
    std::cout << "Result check: " << (correct ? "PASSED" : "FAILED") << "\n";

    return 0;
}
