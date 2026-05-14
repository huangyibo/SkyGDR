// kv_nvlink_transfer.cu
//
// Emulate P -> D KV-cache transfer over NVLink/NVSwitch.
// Example:
//   nvcc -O3 -std=c++17 kv_nvlink_transfer.cu -o kv_nvlink_transfer
//   ./kv_nvlink_transfer 0 1 1024 10000
//
// Args:
//   argv[1] = source/P GPU id
//   argv[2] = destination/D GPU id
//   argv[3] = transfer size in MiB
//   argv[4] = iterations

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err__));                                     \
      std::exit(1);                                                           \
    }                                                                        \
  } while (0)

__global__ void init_kernel(float *p, size_t n, float value) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += stride) {
    p[i] = value + static_cast<float>(i & 1023) * 0.001f;
  }
}

static size_t mib_to_bytes(size_t mib) {
  return mib * 1024ULL * 1024ULL;
}

int main(int argc, char **argv) {
  int src_gpu = 0;
  int dst_gpu = 1;
  size_t size_mib = 1024;
  int iters = 10000;

  if (argc >= 2) src_gpu = std::atoi(argv[1]);
  if (argc >= 3) dst_gpu = std::atoi(argv[2]);
  if (argc >= 4) size_mib = std::strtoull(argv[3], nullptr, 10);
  if (argc >= 5) iters = std::atoi(argv[4]);

  int dev_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&dev_count));
  if (src_gpu >= dev_count || dst_gpu >= dev_count || src_gpu == dst_gpu) {
    fprintf(stderr, "Invalid GPU ids. dev_count=%d src=%d dst=%d\n",
            dev_count, src_gpu, dst_gpu);
    return 1;
  }

  int can_access_src_to_dst = 0;
  int can_access_dst_to_src = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_src_to_dst, src_gpu, dst_gpu));
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_dst_to_src, dst_gpu, src_gpu));

  printf("src_gpu=%d dst_gpu=%d size=%zu MiB iters=%d\n",
         src_gpu, dst_gpu, size_mib, iters);
  printf("peer_access src->dst=%d dst->src=%d\n",
         can_access_src_to_dst, can_access_dst_to_src);

  const size_t bytes = mib_to_bytes(size_mib);
  const size_t elems = bytes / sizeof(float);

  float *src = nullptr;
  float *dst = nullptr;

  CHECK_CUDA(cudaSetDevice(src_gpu));
  CHECK_CUDA(cudaMalloc(&src, bytes));

  if (can_access_src_to_dst) {
    cudaError_t e = cudaDeviceEnablePeerAccess(dst_gpu, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      CHECK_CUDA(e);
    }
  }

  CHECK_CUDA(cudaSetDevice(dst_gpu));
  CHECK_CUDA(cudaMalloc(&dst, bytes));

  if (can_access_dst_to_src) {
    cudaError_t e = cudaDeviceEnablePeerAccess(src_gpu, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      CHECK_CUDA(e);
    }
  }

  CHECK_CUDA(cudaSetDevice(src_gpu));
  init_kernel<<<256, 256>>>(src, elems, 1.0f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaSetDevice(dst_gpu));
  init_kernel<<<256, 256>>>(dst, elems, 0.0f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaStream_t stream;
  CHECK_CUDA(cudaSetDevice(dst_gpu));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Warmup.
  for (int i = 0; i < 16; ++i) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst, dst_gpu, src, src_gpu, bytes, stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iters; ++i) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst, dst_gpu, src, src_gpu, bytes, stream));

    // Occasionally synchronize to avoid unbounded queueing.
    if ((i & 255) == 255) {
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double total_gib = static_cast<double>(bytes) * iters / (1024.0 * 1024.0 * 1024.0);
  double gib_s = total_gib / sec;

  printf("KV transfer done: %.2f GiB in %.3f s = %.2f GiB/s\n",
         total_gib, sec, gib_s);

  CHECK_CUDA(cudaStreamDestroy(stream));

  CHECK_CUDA(cudaSetDevice(src_gpu));
  CHECK_CUDA(cudaFree(src));

  CHECK_CUDA(cudaSetDevice(dst_gpu));
  CHECK_CUDA(cudaFree(dst));

  return 0;
}
