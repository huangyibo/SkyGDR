// gpu_be_memhog_minimal_task.cu: — launch in a separate process on the server.
// It repeatedly streams through a big device array (≥ a few GB) using many blocks
// to drive HBM toward saturation.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void stream_readwrite(float *__restrict__ a, float *__restrict__ b, size_t N, int iters)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int t = 0; t < iters; ++t)
    {
        for (size_t idx = i; idx < N; idx += gridDim.x * blockDim.x)
        {
            float x = a[idx];
            // light arithmetic to force writeback
            b[idx] = x * 1.0000001f + 0.0000003f;
        }
    }
}

int main(int argc, char **argv)
{
    size_t gb = (argc > 1) ? atoll(argv[1]) : 16; // default 16GB working set if available
    int iters = (argc > 2) ? atoi(argv[2]) : 100;
    size_t N = (gb * (size_t)1024 * 1024 * 1024) / sizeof(float);

    float *a, *b;
    cudaMalloc(&a, N * sizeof(float));
    cudaMalloc(&b, N * sizeof(float));
    cudaMemset(a, 1, N * sizeof(float));
    cudaMemset(b, 0, N * sizeof(float));

    int blocks = 2048; // large to drive occupancy
    int threads = 256;
    while (true)
    {
        stream_readwrite<<<blocks, threads>>>(a, b, N, iters);
        cudaDeviceSynchronize(); // keep steady pressure
    }
}
