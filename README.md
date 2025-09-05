# CPU-based Matrix Multiplication

A highly optimized C++ implementation for matrix multiplication: `C(1, 6000) = A(1, 4096) @ B_T(4096, 6000)` with A in FP32 and B in FP4 format.

## Key Optimizations

- **Multi-threading**: OpenMP parallelization across output elements
- **SIMD Vectorization**: AVX2 with FMA instructions for accelerated computation
- **Optimized FP4 Conversion**: SIMD-accelerated batch dequantization with lookup tables
- **Memory Layout**: Column-major packing to eliminate cache misses
- **Prefetching**: Memory prefetching to hide latency

## Performance Results

### K=4096, N=6000

| Implementation | Execution Time |
|:--------------:|:--------------:|
| **torch.float32** | 3.10 ms|
| **torch.float16** | 1.89 ms|
| **torch.bfloat16** | 2.57 ms|
| **Optimized FP4**  | 3.234 ms|
| **Optimized FP32**  | 3.103 ms|

### K=4096*8, N=7168

| Implementation | Execution Time |
|:--------------:|:--------------:|
| **torch.float32** | 36.25 ms|
| **torch.float16** | 20.08 ms|
| **torch.bfloat16** | 22.97 ms|
| **Optimized FP4**  | 37.454 ms|
| **Optimized FP32**  | 41.659 ms|

### K=4096, N=7168*8

| Implementation | Execution Time |
|:--------------:|:--------------:|
| **torch.float32** | 31.41 ms|
| **torch.float16** | 17.89 ms|
| **torch.bfloat16** | 29.74 ms|
| **Optimized FP4**  | 41.726 ms|
| **Optimized FP32**  | 30.610 ms|

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