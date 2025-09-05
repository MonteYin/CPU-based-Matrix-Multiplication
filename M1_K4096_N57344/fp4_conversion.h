#pragma once

#include "fp4_types.h"
#include <cmath>
#include <immintrin.h>

/**
 * Convert FP4 E2M1 format to float using mathematical computation
 */
float fp4_to_float(uint8_t fp4_val);

/**
 * Convert FP4 E2M1 format to float using lookup table
 */
inline float fp4_to_float_fast(uint8_t fp4_val) {
    return fp4_e2m1_lookup[fp4_val & 0x0F];
}

/**
 * Convert FP4 E2M1 format to float using AVX
 */
inline void fp4_to_float_avx(
    const fp4_t* B_packed_ptr,
    int offset,
    const __m256& lookup_low,
    const __m256& lookup_high,
    __m256& result1,
    __m256& result2
) {
    // mask for extracting lower 4 bits
    const __m128i low_nibble_mask = _mm_set1_epi8(0x0F);
    // mask for checking bit 3
    const __m256i high_bit_mask = _mm256_set1_epi32(0x8);
    
    // load 16 bytes
    __m128i packed_data = _mm_loadu_si128((const __m128i*)(B_packed_ptr + offset / 2));
    
    // extract lower and upper nibbles from each byte
    __m128i low_nibbles_8bit = _mm_and_si128(packed_data, low_nibble_mask);
    __m128i high_nibbles_8bit = _mm_and_si128(_mm_srli_epi16(packed_data, 4), low_nibble_mask);
    
    // convert 8-bit indices to 32-bit
    __m256i low_indices = _mm256_cvtepu8_epi32(low_nibbles_8bit);
    __m256i high_indices = _mm256_cvtepu8_epi32(high_nibbles_8bit);
    
    // split indices
    __m256i low_perm_idx = _mm256_and_si256(low_indices, _mm256_set1_epi32(0x7));
    __m256i high_perm_idx = _mm256_and_si256(high_indices, _mm256_set1_epi32(0x7));
    __m256i low_select_high = _mm256_and_si256(low_indices, high_bit_mask);
    __m256i high_select_high = _mm256_and_si256(high_indices, high_bit_mask);
    
    // lookup table permutations
    __m256 low_result_low = _mm256_permutevar8x32_ps(lookup_low, low_perm_idx);
    __m256 low_result_high = _mm256_permutevar8x32_ps(lookup_high, low_perm_idx);
    __m256 high_result_low = _mm256_permutevar8x32_ps(lookup_low, high_perm_idx);
    __m256 high_result_high = _mm256_permutevar8x32_ps(lookup_high, high_perm_idx);
    
    // selection masks
    __m256 low_mask_f = _mm256_castsi256_ps(_mm256_cmpeq_epi32(low_select_high, _mm256_setzero_si256()));
    __m256 high_mask_f = _mm256_castsi256_ps(_mm256_cmpeq_epi32(high_select_high, _mm256_setzero_si256()));
    
    // select values based on bit 3
    __m256 dequant_vals_low = _mm256_blendv_ps(low_result_high, low_result_low, low_mask_f);
    __m256 dequant_vals_high = _mm256_blendv_ps(high_result_high, high_result_low, high_mask_f);
    
    // reorder values to match format
    __m256 temp1 = _mm256_unpacklo_ps(dequant_vals_high, dequant_vals_low);
    __m256 temp2 = _mm256_unpackhi_ps(dequant_vals_high, dequant_vals_low);
    result1 = _mm256_permute2f128_ps(temp1, temp2, 0x20);
    result2 = _mm256_permute2f128_ps(temp1, temp2, 0x31);
}
