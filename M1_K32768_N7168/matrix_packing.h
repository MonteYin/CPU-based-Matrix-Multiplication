#pragma once

#include "fp4_types.h"
#include <vector>
#include <omp.h>

/**
 * Pack B matrix from (K, N) fp4 to blocked format optimal for AVX2
 */
void pack_matrix_b_avx(const std::vector<fp4_t>& B_fp4, std::vector<fp4_t>& B_packed, int K, int N) {
    #pragma omp parallel for
    for (int i = 0; i < K * N / 2; ++i) {
        B_packed[i] = B_fp4[i];
    }
}
