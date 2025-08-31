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
    std::cout << "Using " << num_threads << " OpenMP threads." << std::endl;
    
    std::cout << "Initializing data with random values..."<< std::endl;
    
    std::vector<float> A(M * K);
    std::vector<fp4_t> B_fp4(K * N / 2);

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

    // Optimized implementation
    std::cout << "\nRunning Optimized implementation..." << std::endl;
    std::vector<float> C_opt(M * N);
    auto opt_start = std::chrono::high_resolution_clock::now();
    matmul_opt(A, B_packed, C_opt, K, N);
    auto opt_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> opt_time = opt_end - opt_start;
    std::cout << "Optimized time: " << opt_time.count() << " ms" << std::endl;
    std::cout << "Total Optimized path time (including packing): " << pack_time.count() + opt_time.count() << " ms" << std::endl;
    print_performance(M, K, N, opt_time);

    std::cout << "\n--- Numerical Accuracy Verification ---" << std::endl;
    std::cout << "Optimized vs Reference Naive:" << std::endl;
    analyze_differences(C_opt, C_naive);

    std::vector<double> times_naive, times_opt;
    
    for (int iter = 0; iter < 100; ++iter) {

        auto start = std::chrono::high_resolution_clock::now();
        matmul_naive(A, B_fp4, C_naive, K, N);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times_naive.push_back(time_ms);
        
        start = std::chrono::high_resolution_clock::now();
        matmul_opt(A, B_packed, C_opt, K, N);
        end = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times_opt.push_back(time_ms);
        
        // std::cout << "Iteration " << (iter + 1) << " - Naive: " 
        //           << std::fixed << std::setprecision(3) << time_ms 
        //           << " ms, Optimized: " << time_ms << " ms" << std::endl;
    }

    std::sort(times_naive.begin(), times_naive.end());
    std::sort(times_opt.begin(), times_opt.end());
    
    auto calculate_trimmed_average = [](const std::vector<double>& times, const std::string& label) -> double {
        if (times.size() <= 20) {
            std::cout << label << ": Using all " << times.size() << " samples (no trimming applied)" << std::endl;
            return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        }
        
        auto begin_it = times.begin() + 10;
        auto end_it = times.end() - 10;
        int count = end_it - begin_it;
        
        return std::accumulate(begin_it, end_it, 0.0) / count;
    };
    
    double avg_naive = calculate_trimmed_average(times_naive, "Naive implementation");
    double avg_opt = calculate_trimmed_average(times_opt, "Optimized implementation");

    std::cout << "\n--- Final Performance Summary ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Naive Reference - Average: " << avg_naive << " ms (" 
              << calculate_gflops(M, K, N, avg_naive) << " GFLOPS)" << std::endl;
    
    std::cout << "Optimized Average: " << avg_opt << " ms (" 
              << calculate_gflops(M, K, N, avg_opt) << " GFLOPS)" << std::endl;

    return 0;
}
