#include "matmul_opt.h"
#include "fp4_conversion.h"
#include "avx_utils.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>

void matmul_opt(
    const std::vector<float>& A,
    const std::vector<fp4_t>& B_packed,
    std::vector<float>& C,
    int K,
    int N) {

    const int row_block_size = 512;
    const int col_block_size = 32;
    
    std::fill(C.begin(), C.end(), 0.0f);
    
    #pragma omp parallel num_threads(4)
    {
        // Thread-private accumulator
        std::vector<float> C_private(N, 0.0f);
        
        // Preload lookup table into AVX registers for fast FP4 dequantization
        const __m256 lookup_low = _mm256_loadu_ps(&fp4_e2m1_lookup[0]);
        const __m256 lookup_high = _mm256_loadu_ps(&fp4_e2m1_lookup[8]);
        
        #pragma omp for schedule(static)
        for (int rb_start = 0; rb_start < K; rb_start += row_block_size) {
            int rb_end = std::min(rb_start + row_block_size, K);
            int current_row_block_size = rb_end - rb_start;
            
            for (int cb_start = 0; cb_start < N; cb_start += col_block_size) {
                int cb_end = std::min(cb_start + col_block_size, N);
                int current_col_block_size = cb_end - cb_start;
                
                // Process each column in the current block
                for (int c_local = 0; c_local < current_col_block_size; ++c_local) {
                    int c_global = cb_start + c_local;
                    const fp4_t* B_col_ptr = &B_packed[c_global * K / 2];

                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    __m256 acc4 = _mm256_setzero_ps();
                    
                    int k_local = 0;
                    
                    for (; k_local <= current_row_block_size - 32; k_local += 32) {
                        int k_global = rb_start + k_local;
                        
                        if (k_local + 32 < current_row_block_size) {
                            _mm_prefetch((const char*)(&A[k_global + 32]), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B_col_ptr + (k_global + 32) / 2), _MM_HINT_T0);
                        }
                        
                        // First batch: 0-15
                        __m256 a_vals1 = _mm256_loadu_ps(&A[k_global]);
                        __m256 a_vals2 = _mm256_loadu_ps(&A[k_global + 8]);
                        
                        __m256 b_dequant1, b_dequant2;
                        fp4_to_float_avx(B_col_ptr, k_global, lookup_low, lookup_high, b_dequant1, b_dequant2);
                        
                        acc1 = _mm256_fmadd_ps(a_vals1, b_dequant1, acc1);
                        acc2 = _mm256_fmadd_ps(a_vals2, b_dequant2, acc2);
                        
                        // Second batch: 16-31
                        __m256 a_vals3 = _mm256_loadu_ps(&A[k_global + 16]);
                        __m256 a_vals4 = _mm256_loadu_ps(&A[k_global + 24]);
                        
                        __m256 b_dequant3, b_dequant4;
                        fp4_to_float_avx(B_col_ptr, k_global + 16, lookup_low, lookup_high, b_dequant3, b_dequant4);
                        
                        acc3 = _mm256_fmadd_ps(a_vals3, b_dequant3, acc3);
                        acc4 = _mm256_fmadd_ps(a_vals4, b_dequant4, acc4);
                    }
                    
                    // 16 elements
                    for (; k_local <= current_row_block_size - 16; k_local += 16) {
                        int k_global = rb_start + k_local;
                        
                        __m256 a1 = _mm256_loadu_ps(&A[k_global]);
                        __m256 a2 = _mm256_loadu_ps(&A[k_global + 8]);
                        
                        __m256 b1, b2;
                        fp4_to_float_avx(B_col_ptr, k_global, lookup_low, lookup_high, b1, b2);
                        
                        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
                        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
                    }
                    
                    // Scalar processing for remaining elements
                    float partial_sum = 0.0f;
                    for (; k_local < current_row_block_size; ++k_local) {
                        int k_global = rb_start + k_local;
                        
                        const fp4_t* B_src_ptr = B_col_ptr + k_global / 2;
                        fp4_t packed_val = *B_src_ptr;
                        uint8_t fp4_val = (k_global % 2 == 0) ? (packed_val >> 4) : (packed_val & 0x0F);
                        float B_float = fp4_e2m1_lookup[fp4_val];
                        
                        partial_sum += A[k_global] * B_float;
                    }
                    
                    // reduction
                    __m256 acc_pair1 = _mm256_add_ps(acc1, acc2);
                    __m256 acc_pair2 = _mm256_add_ps(acc3, acc4);
                    __m256 final_acc = _mm256_add_ps(acc_pair1, acc_pair2);
                    
                    float sum = horizontal_sum_avx(final_acc) + partial_sum;
                    
                    C_private[c_global] += sum;
                }
            }
        }
        
        // Merge private results into final output
        #pragma omp critical
        {
            for (int i = 0; i < N; ++i) {
                C[i] += C_private[i];
            }
        }
    }
}

/**
 * FP32
 */
void matmul_opt_float(const std::vector<float>& A, const std::vector<float>& B_float,
                      std::vector<float>& C, int K, int N) {

    const int row_block_size = 1024;
    const int col_block_size = 32;
    
    std::fill(C.begin(), C.end(), 0.0f);
    
    #pragma omp parallel num_threads(4)
    {
        // Thread-private accumulator
        std::vector<float> C_private(N, 0.0f);
        
        #pragma omp for schedule(static)
        for (int rb_start = 0; rb_start < K; rb_start += row_block_size) {
            int rb_end = std::min(rb_start + row_block_size, K);
            int current_row_block_size = rb_end - rb_start;
            
            for (int cb_start = 0; cb_start < N; cb_start += col_block_size) {
                int cb_end = std::min(cb_start + col_block_size, N);
                int current_col_block_size = cb_end - cb_start;
                
                // Process each column
                for (int c_local = 0; c_local < current_col_block_size; ++c_local) {
                    int c_global = cb_start + c_local;
                    
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    __m256 acc4 = _mm256_setzero_ps();
                    __m256 acc5 = _mm256_setzero_ps();
                    __m256 acc6 = _mm256_setzero_ps();
                    __m256 acc7 = _mm256_setzero_ps();
                    __m256 acc8 = _mm256_setzero_ps();
                    
                    int k_local = 0;
                    
                    for (; k_local <= current_row_block_size - 128; k_local += 128) {
                        // Prefetch data
                        if (k_local + 128 < current_row_block_size) {
                            _mm_prefetch((const char*)(&A[rb_start + k_local + 128]), _MM_HINT_T0);
                            _mm_prefetch((const char*)(&B_float[c_global * K + rb_start + k_local + 128]), _MM_HINT_T0);
                        }
                        
                        int global_k = rb_start + k_local;
                        
                        // 8 groups FMA computation
                        
                        // G1
                        __m256 A1_1 = _mm256_loadu_ps(&A[global_k]);
                        __m256 A1_2 = _mm256_loadu_ps(&A[global_k + 8]);
                        __m256 B1_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k]);
                        __m256 B1_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 8]);
                        acc1 = _mm256_fmadd_ps(A1_1, B1_1, acc1);
                        acc1 = _mm256_fmadd_ps(A1_2, B1_2, acc1);
                        
                        // G2
                        __m256 A2_1 = _mm256_loadu_ps(&A[global_k + 16]);
                        __m256 A2_2 = _mm256_loadu_ps(&A[global_k + 24]);
                        __m256 B2_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 16]);
                        __m256 B2_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 24]);
                        acc2 = _mm256_fmadd_ps(A2_1, B2_1, acc2);
                        acc2 = _mm256_fmadd_ps(A2_2, B2_2, acc2);
                        
                        // G3
                        __m256 A3_1 = _mm256_loadu_ps(&A[global_k + 32]);
                        __m256 A3_2 = _mm256_loadu_ps(&A[global_k + 40]);
                        __m256 B3_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 32]);
                        __m256 B3_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 40]);
                        acc3 = _mm256_fmadd_ps(A3_1, B3_1, acc3);
                        acc3 = _mm256_fmadd_ps(A3_2, B3_2, acc3);
                        
                        // G4
                        __m256 A4_1 = _mm256_loadu_ps(&A[global_k + 48]);
                        __m256 A4_2 = _mm256_loadu_ps(&A[global_k + 56]);
                        __m256 B4_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 48]);
                        __m256 B4_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 56]);
                        acc4 = _mm256_fmadd_ps(A4_1, B4_1, acc4);
                        acc4 = _mm256_fmadd_ps(A4_2, B4_2, acc4);
                        
                        // G5
                        __m256 A5_1 = _mm256_loadu_ps(&A[global_k + 64]);
                        __m256 A5_2 = _mm256_loadu_ps(&A[global_k + 72]);
                        __m256 B5_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 64]);
                        __m256 B5_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 72]);
                        acc5 = _mm256_fmadd_ps(A5_1, B5_1, acc5);
                        acc5 = _mm256_fmadd_ps(A5_2, B5_2, acc5);
                        
                        // G6
                        __m256 A6_1 = _mm256_loadu_ps(&A[global_k + 80]);
                        __m256 A6_2 = _mm256_loadu_ps(&A[global_k + 88]);
                        __m256 B6_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 80]);
                        __m256 B6_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 88]);
                        acc6 = _mm256_fmadd_ps(A6_1, B6_1, acc6);
                        acc6 = _mm256_fmadd_ps(A6_2, B6_2, acc6);
                        
                        // G7
                        __m256 A7_1 = _mm256_loadu_ps(&A[global_k + 96]);
                        __m256 A7_2 = _mm256_loadu_ps(&A[global_k + 104]);
                        __m256 B7_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 96]);
                        __m256 B7_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 104]);
                        acc7 = _mm256_fmadd_ps(A7_1, B7_1, acc7);
                        acc7 = _mm256_fmadd_ps(A7_2, B7_2, acc7);
                        
                        // G8
                        __m256 A8_1 = _mm256_loadu_ps(&A[global_k + 112]);
                        __m256 A8_2 = _mm256_loadu_ps(&A[global_k + 120]);
                        __m256 B8_1 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 112]);
                        __m256 B8_2 = _mm256_loadu_ps(&B_float[c_global * K + global_k + 120]);
                        acc8 = _mm256_fmadd_ps(A8_1, B8_1, acc8);
                        acc8 = _mm256_fmadd_ps(A8_2, B8_2, acc8);
                    }
                    
                    // Process remaining elements
                    float partial_sum = 0.0f;
                    for (; k_local < current_row_block_size; ++k_local) {
                        partial_sum += A[rb_start + k_local] * B_float[c_global * K + rb_start + k_local];
                    }
                    
                    // reduction
                    __m256 final_acc = _mm256_add_ps(acc1, acc2);
                    final_acc = _mm256_add_ps(final_acc, acc3);
                    final_acc = _mm256_add_ps(final_acc, acc4);
                    final_acc = _mm256_add_ps(final_acc, acc5);
                    final_acc = _mm256_add_ps(final_acc, acc6);
                    final_acc = _mm256_add_ps(final_acc, acc7);
                    final_acc = _mm256_add_ps(final_acc, acc8);
                    
                    float sum = horizontal_sum_avx(final_acc) + partial_sum;
                    
                    C_private[c_global] += sum;
                }
            }
        }
        
        // merge private results
        #pragma omp critical
        {
            for (int i = 0; i < N; ++i) {
                C[i] += C_private[i];
            }
        }
    }
}