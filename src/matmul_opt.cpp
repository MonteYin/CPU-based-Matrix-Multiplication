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

    const int col_block_size = 256;
    const int row_block_size = 512;
    
    std::fill(C.begin(), C.end(), 0.0f);
    
    #pragma omp parallel for schedule(dynamic) num_threads(4)
    for (int cb_start = 0; cb_start < N; cb_start += col_block_size) {
        int cb_end = std::min(cb_start + col_block_size, N);
        
        for (int rb_start = 0; rb_start < K; rb_start += row_block_size) {
            int rb_end = std::min(rb_start + row_block_size, K);
            
            // Inner blocking
            for (int ci = cb_start; ci < cb_end; ci += 32) {
                int ci_end = std::min(ci + 32, cb_end);
                
                // Process each column in the current block
                for (int c = ci; c < ci_end; ++c) {
                    // Initialize acc
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    const fp4_t* B_ptr = &B_packed[c * K / 2];
                    
                    // Prefetch next column data
                    if (c + 1 < ci_end) {
                        _mm_prefetch((const char*)(&B_packed[(c + 1) * K / 2 + rb_start/2]), _MM_HINT_T0);
                    }
                    
                    int k = rb_start;
                    
                    for (; k <= rb_end - 32; k += 32) {
                        _mm_prefetch((const char*)(&A[k + 32]), _MM_HINT_T0);
                        _mm_prefetch((const char*)(&B_ptr[(k + 32)/2]), _MM_HINT_T0);
                        
                        __m256 A_vec1 = _mm256_loadu_ps(&A[k]);
                        __m256 A_vec2 = _mm256_loadu_ps(&A[k + 8]);
                        
                        alignas(32) float B_vals1[16];
                        fp4_batch_to_float_unrolled(&B_ptr[k/2], B_vals1);
                        
                        __m256 B_vec1 = _mm256_loadu_ps(&B_vals1[0]);
                        __m256 B_vec2 = _mm256_loadu_ps(&B_vals1[8]);
                        
                        acc1 = _mm256_fmadd_ps(A_vec1, B_vec1, acc1);
                        acc1 = _mm256_fmadd_ps(A_vec2, B_vec2, acc1);
                        
                        __m256 A_vec3 = _mm256_loadu_ps(&A[k + 16]);
                        __m256 A_vec4 = _mm256_loadu_ps(&A[k + 24]);
                        
                        alignas(32) float B_vals2[16];
                        fp4_batch_to_float_unrolled(&B_ptr[(k+16)/2], B_vals2);
                        
                        __m256 B_vec3 = _mm256_loadu_ps(&B_vals2[0]);
                        __m256 B_vec4 = _mm256_loadu_ps(&B_vals2[8]);
                        
                        acc2 = _mm256_fmadd_ps(A_vec3, B_vec3, acc2);
                        acc2 = _mm256_fmadd_ps(A_vec4, B_vec4, acc2);
                    }
                    
                    // Handle remaining elements
                    for (; k <= rb_end - 16; k += 16) {
                        __m256 A_vec1 = _mm256_loadu_ps(&A[k]);
                        __m256 A_vec2 = _mm256_loadu_ps(&A[k + 8]);
                        
                        alignas(32) float B_vals[16];
                        fp4_batch_to_float_unrolled(&B_ptr[k/2], B_vals);
                        
                        __m256 B_vec1 = _mm256_loadu_ps(&B_vals[0]);
                        __m256 B_vec2 = _mm256_loadu_ps(&B_vals[8]);
                        
                        acc1 = _mm256_fmadd_ps(A_vec1, B_vec1, acc1);
                        acc1 = _mm256_fmadd_ps(A_vec2, B_vec2, acc1);
                    }
                    
                    // Combine acc
                    __m256 total_acc = _mm256_add_ps(acc1, acc2);
                    float sum = horizontal_sum_avx(total_acc);
                    
                    for (; k < rb_end; ++k) {
                        fp4_t packed_val = B_ptr[k/2];
                        uint8_t fp4_val = (k % 2 == 0) ? ((packed_val >> 4) & 0x0F) : (packed_val & 0x0F);
                        float B_val = fp4_to_float_fast(fp4_val);
                        sum += A[k] * B_val;
                    }
                    
                    C[c] += sum;
                }
            }
        }
    }
}
