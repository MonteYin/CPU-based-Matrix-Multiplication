#pragma once

#include <cstdint>
#include <cmath>

using fp4_t = uint8_t;

// Lookup table for fast FP4 E2M1 to float conversion
extern const float fp4_e2m1_lookup[16];
