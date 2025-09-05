#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <numeric>
#include <cmath>

#include "fp4_types.h"
#include "fp4_conversion.h"
#include "matrix_packing.h"
#include "matmul_opt.h"
#include "benchmark_utils.h"

/**
 * Naive matrix multiplication reference implementation
 */
void matmul_naive(const std::vector<float>& A, const std::vector<fp4_t>& B_fp4, 
                  std::vector<float>& C, int K, int N) {
    std::fill(C.begin(), C.end(), 0.0f);
    
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            // Extract FP4 value from packed format
            const fp4_t* B_ptr = &B_fp4[j * K / 2 + k / 2];
            fp4_t packed_val = *B_ptr;
            uint8_t fp4_val = (k % 2 == 0) ? ((packed_val >> 4) & 0x0F) : (packed_val & 0x0F);
            float B_val = fp4_to_float_fast(fp4_val);
            
            C[j] += A[k] * B_val;
        }
    }
}

int main() {
    
    unsigned int seed = 42;
    
    const int M = 1;
    const int K = 4096;
    const int N = 6000;

    int num_threads = 4;
    omp_set_num_threads(num_threads); 
    // std::cout << "Using " << num_threads << " OpenMP threads." << std::endl;
    
    // std::cout << "Initializing data with random values..."<< std::endl;
    
    std::vector<float> A(M * K);
    std::vector<fp4_t> B_fp4(K * N / 2);
    std::vector<float> B_float(K * N);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) {
        A[i] = dist(gen);
    }

    // Initialize B matrix in column-major order
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; k += 2) {
            uint8_t fp4_val1, fp4_val2;
            do {
                fp4_val1 = gen() % 16;
            } while (fp4_val1 == 6 || fp4_val1 == 7 || fp4_val1 == 14 || fp4_val1 == 15);
            
            do {
                fp4_val2 = gen() % 16;
            } while (fp4_val2 == 6 || fp4_val2 == 7 || fp4_val2 == 14 || fp4_val2 == 15);
            
            int idx = j * K / 2 + k / 2;
            B_fp4[idx] = (fp4_val1 << 4) | fp4_val2;
            
            B_float[j * K + k] = fp4_to_float_fast(fp4_val1);
            B_float[j * K + k + 1] = fp4_to_float_fast(fp4_val2);
        }
    }

    // Data packing
    std::cout << "\nPacking data..." << std::endl;
    auto pack_start = std::chrono::high_resolution_clock::now();
    std::vector<fp4_t> B_packed(K * N / 2);
    pack_matrix_b_avx(B_fp4, B_packed, K, N);
    auto pack_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> pack_time = pack_end - pack_start;
    std::cout << "Data packing time: " << pack_time.count() << " ms" << std::endl;

    // Naive implementation
    std::cout << "\nRunning Naive implementation..." << std::endl;
    std::vector<float> C_naive(M * N);
    auto naive_start = std::chrono::high_resolution_clock::now();
    matmul_naive(A, B_fp4, C_naive, K, N);
    auto naive_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> naive_time = naive_end - naive_start;
    std::cout << "Naive time: " << naive_time.count() << " ms" << std::endl;
    print_performance(M, K, N, naive_time);

    // Optimized implementation FP4
    std::cout << "Running Optimized implementation (FP4)..." << std::endl;
    std::vector<float> C_opt(M * N);
    auto opt_start = std::chrono::high_resolution_clock::now();
    matmul_opt(A, B_packed, C_opt, K, N);
    auto opt_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> opt_time = opt_end - opt_start;
    std::cout << "Optimized (FP4) time: " << opt_time.count() << " ms" << std::endl;
    std::cout << "Total Optimized path time (including packing): " << pack_time.count() + opt_time.count() << " ms" << std::endl;
    print_performance(M, K, N, opt_time);

    // Float Version
    std::cout << "Running Optimized implementation (Float)..." << std::endl;
    std::vector<float> C_opt_float(M * N);
    auto opt_float_start = std::chrono::high_resolution_clock::now();
    matmul_opt_float(A, B_float, C_opt_float, K, N);
    auto opt_float_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> opt_float_time = opt_float_end - opt_float_start;
    std::cout << "Optimized (Float) time: " << opt_float_time.count() << " ms" << std::endl;
    print_performance(M, K, N, opt_float_time);
    
    // std::cout << "\n--- Numerical Accuracy Verification ---" << std::endl;
    std::cout << "FP4 vs Reference Naive:" << std::endl;
    analyze_differences(C_opt, C_naive);

    std::cout << "Float vs Reference Naive:" << std::endl;
    analyze_differences(C_opt_float, C_naive);
    
    std::vector<double> fp4_times;
    for (int i = 0; i < 1000; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_opt(A, B_packed, C_opt, K, N);
        auto end = std::chrono::high_resolution_clock::now();
        fp4_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(fp4_times.begin(), fp4_times.end());
    double avg_fp4_time = std::accumulate(fp4_times.begin() + 2, fp4_times.end() - 2, 0.0) / 996;
    
    std::vector<double> float_times;
    for (int i = 0; i < 1000; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul_opt_float(A, B_float, C_opt_float, K, N);
        auto end = std::chrono::high_resolution_clock::now();
        float_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(float_times.begin(), float_times.end());
    double avg_float_time = std::accumulate(float_times.begin() + 2, float_times.end() - 2, 0.0) / 996;
    
    std::cout << "FP4 Optimized Average: " << std::fixed << std::setprecision(3) << avg_fp4_time << " ms";
    std::cout << " (" << std::fixed << std::setprecision(3) << (M * K * N * 2.0) / (avg_fp4_time * 1e-3) / 1e9 << " GFLOPS)" << std::endl;
    
    std::cout << "Float Optimized Average: " << std::fixed << std::setprecision(3) << avg_float_time << " ms";
    std::cout << " (" << std::fixed << std::setprecision(3) << (M * K * N * 2.0) / (avg_float_time * 1e-3) / 1e9 << " GFLOPS)" << std::endl;

    return 0;
}
