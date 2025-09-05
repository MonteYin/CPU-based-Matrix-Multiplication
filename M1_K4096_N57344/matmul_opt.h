#pragma once

#include "fp4_types.h"
#include <vector>

void matmul_opt(
    const std::vector<float>& A,
    const std::vector<fp4_t>& B_packed,
    std::vector<float>& C,
    int K,
    int N
);

void matmul_opt_float(
    const std::vector<float>& A,
    const std::vector<float>& B_float,
    std::vector<float>& C,
    int K,
    int N
);
