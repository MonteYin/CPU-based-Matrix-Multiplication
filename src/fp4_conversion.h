#pragma once

#include "fp4_types.h"
#include <cmath>

/**
 * Convert FP4 E2M1 format to float using mathematical computation
 */
float fp4_to_float(uint8_t fp4_val);

/**
 * Convert FP4 E2M1 format to float using lookup table (fast)
 */
inline float fp4_to_float_fast(uint8_t fp4_val) {
    return fp4_e2m1_lookup[fp4_val & 0x0F];
}

/**
 * Convert batch of FP4 values to float array (unrolled for performance)
 */
inline void fp4_batch_to_float_unrolled(const fp4_t* packed_fp4, float* output) {
    // First 8 values
    output[0] = fp4_e2m1_lookup[(packed_fp4[0] >> 4) & 0x0F];
    output[1] = fp4_e2m1_lookup[packed_fp4[0] & 0x0F];
    output[2] = fp4_e2m1_lookup[(packed_fp4[1] >> 4) & 0x0F];
    output[3] = fp4_e2m1_lookup[packed_fp4[1] & 0x0F];
    output[4] = fp4_e2m1_lookup[(packed_fp4[2] >> 4) & 0x0F];
    output[5] = fp4_e2m1_lookup[packed_fp4[2] & 0x0F];
    output[6] = fp4_e2m1_lookup[(packed_fp4[3] >> 4) & 0x0F];
    output[7] = fp4_e2m1_lookup[packed_fp4[3] & 0x0F];
    
    // Second 8 values
    output[8] = fp4_e2m1_lookup[(packed_fp4[4] >> 4) & 0x0F];
    output[9] = fp4_e2m1_lookup[packed_fp4[4] & 0x0F];
    output[10] = fp4_e2m1_lookup[(packed_fp4[5] >> 4) & 0x0F];
    output[11] = fp4_e2m1_lookup[packed_fp4[5] & 0x0F];
    output[12] = fp4_e2m1_lookup[(packed_fp4[6] >> 4) & 0x0F];
    output[13] = fp4_e2m1_lookup[packed_fp4[6] & 0x0F];
    output[14] = fp4_e2m1_lookup[(packed_fp4[7] >> 4) & 0x0F];
    output[15] = fp4_e2m1_lookup[packed_fp4[7] & 0x0F];
}
