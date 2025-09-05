#include "fp4_conversion.h"
#include <cmath>
#include <immintrin.h>
#include <algorithm>

// FP4 E2M1 lookup table
const float fp4_e2m1_lookup[16] = {
    0.0f,                    // 0000: +0
    0.010416666666666666f,   // 0001: smallest positive value
    0.16666666666666666f,    // 0010: 1/6
    0.25f,                   // 0011: 1/4
    0.333333333333333f,      // 0100: 1/3
    0.5f,                    // 0101: 1/2
    0.6666666666666f,        // 0110: 2/3
    1.0f,                    // 0111: 1.0 (maximum positive value)
    -0.0f,                   // 1000: -0
    -0.010416666666666666f,  // 1001: smallest negative value
    -0.16666666666666666f,   // 1010: -1/6
    -0.25f,                  // 1011: -1/4
    -0.333333333333333f,     // 1100: -1/3
    -0.5f,                   // 1101: -1/2
    -0.6666666666666f,       // 1110: -2/3
    -1.0f                    // 1111: -1.0 (maximum negative value)
};

/**
 * Convert FP4 E2M1 format to float using mathematical computation
 */
float fp4_to_float(uint8_t fp4_val) {
    float sign = (fp4_val & 0b1000) == 8 ? -1.0f : 1.0f;
    
    if ((fp4_val & 0b0100) == 4) {
        if ((fp4_val & 0b0010) == 2) {
            if ((fp4_val & 0b0001) == 1) {
                return 1.0f * sign;
            } else {
                return 0.6666666666666666f * sign;
            }
        } else if ((fp4_val & 0b0001) == 1) {
            return 0.5f * sign;
        } else {
            return 0.3333333333333333f * sign;
        }
    } else if ((fp4_val & 0b0010) == 2) {
        if ((fp4_val & 0b0001) == 1) {
            return 0.25f * sign;
        } else {
            return 0.16666666666666666f * sign;
        }
    } else if ((fp4_val & 0b0001) == 1) {
        return 0.010416666666666666f * sign;
    } else {
        return 0.00000000f * sign;
    }
}
