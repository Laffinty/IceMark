// AVX2 + FMA kernels.
//
// This is the ONLY translation unit compiled with /arch:AVX2. Nothing in it
// may be reached before a runtime CPUID check (cpu_supports_avx2_fma), since
// executing AVX2 on a host without it is an illegal instruction, not a slow
// path.
#include "kernel.hpp"

#include <immintrin.h>

namespace icemark {

float dot_f32_avx2(const float* a, const float* b, size_t n) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = a0, a2 = a0, a3 = a0;

    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), a3);
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);

    float acc = _mm_cvtss_f32(h);
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

int32_t dot_i8_avx2(const int8_t* a, const int8_t* b, size_t n) {
    __m256i acc = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        // signed x signed -> int32 via widening then madd
        __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a + i)));
        __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }

    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);

    int32_t total = _mm_cvtsi128_si32(s);
    for (; i < n; ++i) total += (int32_t)a[i] * (int32_t)b[i];
    return total;
}

} // namespace icemark