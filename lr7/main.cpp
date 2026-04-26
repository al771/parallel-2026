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

        for (; x <= w - 1 - 4; x += 4) {
            int a_val;
            int b_val;
            int c_val;
            int d_val;

            std::memcpy(&a_val, &src[y * w + x], 4);
            std::memcpy(&b_val, &src[y * w + x + 1], 4);
            std::memcpy(&c_val, &src[(y + 1) * w + x], 4);
            std::memcpy(&d_val, &src[(y + 1) * w + x + 1], 4);

            __m128i a_bytes = _mm_cvtsi32_si128(a_val);
            __m128i b_bytes = _mm_cvtsi32_si128(b_val);
            __m128i c_bytes = _mm_cvtsi32_si128(c_val);
            __m128i d_bytes = _mm_cvtsi32_si128(d_val);

            __m128i a = _mm_cvtepu8_epi32(a_bytes);
            __m128i b = _mm_cvtepu8_epi32(b_bytes);
            __m128i c = _mm_cvtepu8_epi32(c_bytes);
            __m128i d = _mm_cvtepu8_epi32(d_bytes);

            __m128i sum1 = _mm_add_epi32(a, b);
            __m128i sum2 = _mm_add_epi32(c, d);
            __m128i sum = _mm_add_epi32(sum1, sum2);

            __m128i result = _mm_srli_epi32(sum, 2);

            __m128i result16 = _mm_packus_epi32(result, result);
            __m128i result8 = _mm_packus_epi16(result16, result16);

            int out;
            out = _mm_cvtsi128_si32(result8);
            std::memcpy(&dst[y * w + x], &out, 4);
        }

        for (; x < w - 1; ++x) {
            int a = src[y * w + x];
            int b = src[y * w + x + 1];
            int c = src[(y + 1) * w + x];
            int d = src[(y + 1) * w + x + 1];

            int result = (a + b + c + d) / 4;
            dst[y * w + x] = static_cast<unsigned char>(result);
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
