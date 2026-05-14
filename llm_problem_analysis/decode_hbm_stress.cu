// decode_hbm_stress.cu
//
// Emulate decode-side HBM bandwidth pressure.
// Example:
//   nvcc -O3 -std=c++17 decode_hbm_stress.cu -o decode_hbm_stress
//   ./decode_hbm_stress 1 32768 100000
//
// Args:
//   argv[1] = GPU id, e.g., D GPU
//   argv[2] = working-set size in MiB
//   argv[3] = iterations

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err__));                                     \
      std::exit(1);                                                           \
    }                                                                        \
  } while (0)

__global__ void init_kernel(float4 *a, float4 *b, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += stride) {
    float x = static_cast<float>(i & 1023) * 0.001f;
    a[i] = make_float4(x, x + 1.0f, x + 2.0f, x + 3.0f);
    b[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

// Streaming kernel: read a, read b, write b.
// This creates large HBM traffic and avoids being optimized away.
__global__ void decode_hbm_kernel(const float4 *__restrict__ a,
                                  float4 *__restrict__ b,
                                  size_t n,
                                  float alpha) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += stride) {
    float4 x = a[i];
    float4 y = b[i];

    y.x = y.x + alpha * x.x;
    y.y = y.y + alpha * x.y;
    y.z = y.z + alpha * x.z;
    y.w = y.w + alpha * x.w;

    b[i] = y;
  }
}

__global__ void checksum_kernel(const float4 *b, size_t n, float *out) {
  __shared__ float smem[256];

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t lane = threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  float sum = 0.0f;
  for (size_t i = tid; i < n; i += stride) {
    float4 v = b[i];
    sum += v.x + v.y + v.z + v.w;
  }

  smem[lane] = sum;
  __syncthreads();

  for (int offset = 128; offset > 0; offset >>= 1) {
    if (lane < offset) {
      smem[lane] += smem[lane + offset];
    }
    __syncthreads();
  }

  if (lane == 0) {
    atomicAdd(out, smem[0]);
  }
}

static size_t mib_to_bytes(size_t mib) {
  return mib * 1024ULL * 1024ULL;
}

int main(int argc, char **argv) {
  int gpu = 1;
  size_t working_set_mib = 32768;
  int iters = 100000;

  if (argc >= 2) gpu = std::atoi(argv[1]);
  if (argc >= 3) working_set_mib = std::strtoull(argv[2], nullptr, 10);
  if (argc >= 4) iters = std::atoi(argv[3]);

  CHECK_CUDA(cudaSetDevice(gpu));

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu));

  printf("gpu=%d name=%s working_set=%zu MiB iters=%d\n",
         gpu, prop.name, working_set_mib, iters);

  // Split working set across two arrays.
  size_t total_bytes = mib_to_bytes(working_set_mib);
  size_t array_bytes = total_bytes / 2;
  size_t n = array_bytes / sizeof(float4);

  float4 *a = nullptr;
  float4 *b = nullptr;
  float *checksum = nullptr;

  CHECK_CUDA(cudaMalloc(&a, n * sizeof(float4)));
  CHECK_CUDA(cudaMalloc(&b, n * sizeof(float4)));
  CHECK_CUDA(cudaMalloc(&checksum, sizeof(float)));
  CHECK_CUDA(cudaMemset(checksum, 0, sizeof(float)));

  int threads = 256;
  int blocks = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, decode_hbm_kernel, threads, 0));
  blocks *= prop.multiProcessorCount;

  printf("launch blocks=%d threads=%d\n", blocks, threads);

  init_kernel<<<blocks, threads>>>(a, b, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Warmup.
  for (int i = 0; i < 16; ++i) {
    decode_hbm_kernel<<<blocks, threads>>>(a, b, n, 0.001f);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iters; ++i) {
    decode_hbm_kernel<<<blocks, threads>>>(a, b, n, 0.001f);

    // Occasionally synchronize to make progress visible and avoid huge queues.
    if ((i & 255) == 255) {
      CHECK_CUDA(cudaDeviceSynchronize());
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  auto t1 = std::chrono::high_resolution_clock::now();

  // One iteration roughly does:
  //   read a: array_bytes
  //   read b: array_bytes
  //   write b: array_bytes
  // So about 3 * array_bytes of HBM traffic.
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double bytes_per_iter = 3.0 * static_cast<double>(array_bytes);
  double total_gib = bytes_per_iter * iters / (1024.0 * 1024.0 * 1024.0);
  double gib_s = total_gib / sec;

  checksum_kernel<<<blocks, threads>>>(b, n, checksum);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  float host_sum = 0.0f;
  CHECK_CUDA(cudaMemcpy(&host_sum, checksum, sizeof(float), cudaMemcpyDeviceToHost));

  printf("HBM stress done: approx %.2f GiB in %.3f s = %.2f GiB/s\n",
         total_gib, sec, gib_s);
  printf("checksum=%f\n", host_sum);

  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaFree(b));
  CHECK_CUDA(cudaFree(checksum));

  return 0;
}
