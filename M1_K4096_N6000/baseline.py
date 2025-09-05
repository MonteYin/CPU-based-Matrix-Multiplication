import torch, time
 
def perf_test(dtype):
  A = torch.randn(1, 4096, dtype=dtype).cpu()
  B = torch.randn(6000, 4096, dtype=dtype).cpu()
  gemm = lambda: A @ B.t()
  steps = 100
  T_start = time.perf_counter()
  for i in range(steps): gemm()
  T_stop = time.perf_counter()
  print(f'Gemv time for {dtype}: {(T_stop - T_start) / steps * 1000: .2f} (ms)')

perf_test(torch.float32)
perf_test(torch.float16)
perf_test(torch.bfloat16)

# print(torch.__config__.show()) 