# CPU-based Matrix Multiplication

A highly optimized C++ implementation for matrix multiplication: `C(1, 6000) = A(1, 4096) @ B_T(4096, 6000)` with A in FP32 and B in FP4 format.

## Key Optimizations

- **Multi-threading**: OpenMP parallelization across output elements
- **SIMD Vectorization**: AVX2 with FMA instructions for accelerated computation
- **Optimized FP4 Conversion**: Batch dequantization of 16 values at once
- **Memory Layout**: Column-major packing to eliminate cache misses
- **Prefetching**: Prefetching hide memory latency

## Performance Results

| Implementation | Execution Time | GFLOPS | Speedup |
|:--------------:|:--------------:|:------:|:-------:|
| **Naive**      | 27.18 ms       | 1.81   | 1.0x    |
| **Optimized**  | 4.29 ms        | 11.45  | **6.36x** |

*Tested on AMD EPYC 7763 (2 cores, 4 threads) with 16GB RAM*

The optimized implementation maintains high numerical accuracy compared to the reference naive implementation:

|  Metric  |      Value     |
|:--------:|:--------------:|
| **MAE**  | `4.637187e-05` |
| **RMSE** | `6.559217e-05` |
| **MBE**  | `4.234413e-08` |

## Quick Start

```bash
# Compile the project
make
# Run benchmarks
make run
# Clean build files
make clean
```