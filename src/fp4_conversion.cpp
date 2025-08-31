#include "fp4_conversion.h"
#include <cmath>

// Lookup table for fast FP4 E2M1 to float conversion
const float fp4_e2m1_lookup[16] = {
    0.0f,     // 0000: +0
    0.5f,     // 0001: +subnormal
    1.0f,     // 0010: +1.0 * 2^0
    1.5f,     // 0011: +1.5 * 2^0  
    2.0f,     // 0100: +1.0 * 2^1
    3.0f,     // 0101: +1.5 * 2^1
    INFINITY, // 0110: +inf
    NAN,      // 0111: nan
    -0.0f,    // 1000: -0
    -0.5f,    // 1001: -subnormal
    -1.0f,    // 1010: -1.0 * 2^0
    -1.5f,    // 1011: -1.5 * 2^0
    -2.0f,    // 1100: -1.0 * 2^1
    -3.0f,    // 1101: -1.5 * 2^1
    -INFINITY,// 1110: -inf
    NAN       // 1111: nan
};

/**
 * Convert FP4 E2M1 format to float using mathematical computation
 */
float fp4_to_float(uint8_t fp4_val) {
    uint8_t sign = (fp4_val >> 3) & 0x1;
    uint8_t exp = (fp4_val >> 1) & 0x3;
    uint8_t mantissa = fp4_val & 0x1;
    
    if (exp == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            float value = 0.5f;
            return sign ? -value : value;
        }
    } else if (exp == 3) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    } else {
        // Normal numbers
        int actual_exp = exp - 1;
        float mantissa_val = 1.0f + mantissa * 0.5f;
        float value = mantissa_val * powf(2.0f, actual_exp);
        return sign ? -value : value;
    }
}
