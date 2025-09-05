#pragma once

#include <immintrin.h>

/**
 * Compute horizontal sum of all elements in AVX register
 */
inline float horizontal_sum_avx(__m256 vec) {
    __m256 sum1 = _mm256_hadd_ps(vec, vec);
    __m256 sum2 = _mm256_hadd_ps(sum1, sum1);
    __m128 sum3 = _mm256_extractf128_ps(sum2, 1);
    __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum2), sum3);
    return _mm_cvtss_f32(sum4);
}
