#pragma once

#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

/**
 * Print performance metrics for matrix multiplication
 */
void print_performance(int M, int K, int N, const std::chrono::duration<double, std::milli>& duration) {
    long long total_flops = 2LL * M * K * N;
    double duration_sec = duration.count() / 1000.0;
    
    double gflops = (static_cast<double>(total_flops) / duration_sec) / 1e9;
    double tflops = gflops / 1000.0;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;
}

/**
 * Calculate GFLOPS performance metric
 */
double calculate_gflops(int M, int K, int N, double time_ms) {
    long long total_flops = 2LL * M * K * N;
    double duration_sec = time_ms / 1000.0;
    if (duration_sec == 0) {
        return 0.0;
    }
    return (static_cast<double>(total_flops) / duration_sec) / 1e9;
}

/**
 * Analyze numerical differences between results
 */
void analyze_differences(const std::vector<float>& C_test, const std::vector<float>& C_ref) {
    if (C_test.size() != C_ref.size()) {
        std::cerr << "Cannot compare vectors of different sizes." << std::endl;
        return;
    }

    int num_elements = C_test.size();
    double sum_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double sum_bias_err = 0.0;

    for (int i = 0; i < num_elements; ++i) {
        float diff = C_test[i] - C_ref[i];
        float abs_diff = std::fabs(diff);

        sum_abs_err += abs_diff;
        sum_sq_err += static_cast<double>(diff) * diff;
        sum_bias_err += diff;
    }

    double mae = sum_abs_err / num_elements;
    double rmse = std::sqrt(sum_sq_err / num_elements);
    double mbe = sum_bias_err / num_elements;

    std::cout << "\n--- Numerical Difference Analysis ---" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Mean Absolute Error (MAE): " << mae << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << rmse << std::endl;
    std::cout << "Mean Bias Error (MBE): " << mbe << std::endl;
    std::cout << std::fixed << std::setprecision(3);
}
