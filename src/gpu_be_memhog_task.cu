// gpu_be_memhog_task.cu
//
// Purpose: generate sustained pressure on HBM (A100/A800) to study contention.
// Patterns supported:
//   --op=read    : read-only scan of a[] (reduces write traffic; still hits HBM)
//   --op=write   : write-only scan into b[] (forces HBM writes; minimal reads)
//   --op=rw      : read a[], light FMA, write b[] (balanced R/W; default)
//   --op=copy    : copy a[] -> b[] (memcpy-like; often saturates BW)
//
// Duration-based: run for --seconds (default 60s). Also supports --iters.
// Grid control: --blocks, --threads, --streams. Working set: --gb.
//
// Build (SM80): nvcc gpu_be_memhog_task.cu -O3 -std=c++14 -o gpu_be_memhog_task -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
//
// Run example (16GB working set for 60s):
//   ./gpu_be_memhog_task --gb=16 --seconds=60 --op=rw --blocks=2048 --threads=256
//
// Stop with Ctrl-C if running longer than expected.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#define CUDA_CHECK(cmd)                                                                           \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = (cmd);                                                                    \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

enum OpKind
{
    OP_RW,
    OP_READ,
    OP_WRITE,
    OP_COPY
};

__device__ double g_sink = 0.0; // to prevent DCE (read path)

// Grid-stride helpers
template <typename T>
__device__ __forceinline__ T ld_nc(const T *p)
{
    // plain load is fine; using __ldg may bypass L1 but isn't necessary
    return *p;
}

// Read-only: accumulate into a sink
__global__ void k_read_only(const float *__restrict__ a, size_t N, int iters)
{
    const size_t idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    double acc = 0.0;
    for (int t = 0; t < iters; ++t)
    {
        for (size_t i = idx0; i < N; i += stride)
        {
            float x = ld_nc(a + i);
            acc += (double)x;
        }
    }
    // reduce a bit so compiler can't drop it
    atomicAdd(&g_sink, acc);
}

// Write-only: write a pattern; touch b[] only (forces HBM writes)
__global__ void k_write_only(float *__restrict__ b, size_t N, int iters)
{
    const size_t idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    float v = 1.234567f + (float)idx0 * 1e-6f;
    for (int t = 0; t < iters; ++t)
    {
        for (size_t i = idx0; i < N; i += stride)
        {
            b[i] = v;
            v = v * 1.0000001f + 0.0000003f; // tiny arithmetic to keep ALUs alive
        }
    }
}

// Read+Write: read a[], FMA, write b[]
__global__ void k_read_write(const float *__restrict__ a,
                             float *__restrict__ b,
                             size_t N, int iters)
{
    const size_t idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (int t = 0; t < iters; ++t)
    {
        for (size_t i = idx0; i < N; i += stride)
        {
            float x = ld_nc(a + i);
            // light compute to avoid being purely LD/ST (keeps writeback deterministic)
            x = x * 1.0000001f + 0.0000003f;
            b[i] = x;
        }
    }
}

// Copy with vectorized float4 to increase memory throughput
__global__ void k_copy_vec4(const float4 *__restrict__ a,
                            float4 *__restrict__ b,
                            size_t N4, int iters)
{
    const size_t idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (int t = 0; t < iters; ++t)
    {
        for (size_t i = idx0; i < N4; i += stride)
        {
            b[i] = a[i];
        }
    }
}

// Copy: a[] -> b[] (memcpy-like)
__global__ void k_copy(const float *__restrict__ a,
                       float *__restrict__ b,
                       size_t N, int iters)
{
    const size_t idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (int t = 0; t < iters; ++t)
    {
        for (size_t i = idx0; i < N; i += stride)
        {
            b[i] = a[i];
        }
    }
}

struct Args
{
    int device = 0;
    double gb = 16.0; // working set
    int blocks = 2048;
    int threads = 256;
    int iters = 100;  // per kernel launch loop-count
    int seconds = 60; // wall-time to run (overrides iters loop if >0)
    OpKind op = OP_RW;
    int streams = 1;  // concurrent streams to increase MLP
    int vec = 1;      // 1 = scalar, 4 = vectorized float4 (copy op only)
};

static OpKind parse_op(const char *s)
{
    if (!s)
        return OP_RW;
    if (!strcmp(s, "rw"))
        return OP_RW;
    if (!strcmp(s, "read"))
        return OP_READ;
    if (!strcmp(s, "write"))
        return OP_WRITE;
    if (!strcmp(s, "copy"))
        return OP_COPY;
    return OP_RW;
}

static void parse_args(int argc, char **argv, Args &a)
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--device=", 9))
            a.device = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--gb=", 5))
            a.gb = atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--blocks=", 9))
            a.blocks = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--threads=", 10))
            a.threads = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--iters=", 8))
            a.iters = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--seconds=", 10))
            a.seconds = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--op=", 5))
            a.op = parse_op(argv[i] + 5);
        else if (!strncmp(argv[i], "--streams=", 10))
            a.streams = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--vec=", 6))
            a.vec = atoi(argv[i] + 6);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
        {
            printf("Usage: %s [--device=N] [--gb=16] [--blocks=2048] [--threads=256]\n"
                   "           [--iters=100] [--seconds=60] [--op=rw|read|write|copy]\n"
                   "           [--streams=1] [--vec=1|4] (vec=4 only for op=copy)\n",
                   argv[0]);
            exit(0);
        }
    }
}

int main(int argc, char **argv)
{
    Args args;
    parse_args(argc, argv, args);
    CUDA_CHECK(cudaSetDevice(args.device));

    // Working set: 2 arrays for ops that write; 1 array for read-only
    size_t elems = (size_t)(args.gb * (1024.0 * 1024.0 * 1024.0) / sizeof(float));
    if (elems == 0)
        elems = 1;

    float *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, elems * sizeof(float)));
    if (args.op != OP_READ)
        CUDA_CHECK(cudaMalloc(&b, elems * sizeof(float)));

    // Initialize
    CUDA_CHECK(cudaMemset(a, 0, elems * sizeof(float)));
    if (b)
        CUDA_CHECK(cudaMemset(b, 0, elems * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Basic sanity on launch config
    if (args.blocks <= 0)
        args.blocks = 2048;
    if (args.threads <= 0)
        args.threads = 256;
    if (args.threads > 1024)
        args.threads = 1024;
    if (args.streams <= 0)
        args.streams = 1;
    if (args.vec != 1 && args.vec != 4)
        args.vec = 1;

    // Time-bounded loop
    auto t_start = std::chrono::high_resolution_clock::now();
    int launches = 0;

    // Warmup
    switch (args.op)
    {
    case OP_RW:
        k_read_write<<<args.blocks, args.threads>>>(a, b, elems, 1);
        break;
    case OP_READ:
        k_read_only<<<args.blocks, args.threads>>>(a, elems, 1);
        break;
    case OP_WRITE:
        k_write_only<<<args.blocks, args.threads>>>(b, elems, 1);
        break;
    case OP_COPY:
        if (args.vec == 4 && (elems % 4 == 0))
        {
            size_t n4 = elems / 4;
            k_copy_vec4<<<args.blocks, args.threads>>>(
                reinterpret_cast<const float4 *>(a),
                reinterpret_cast<float4 *>(b),
                n4, 1);
        }
        else
        {
            k_copy<<<args.blocks, args.threads>>>(a, b, elems, 1);
        }
        break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measured loop: run until seconds elapse (if seconds>0), otherwise run fixed count
    double elapsed_s = 0.0;
    size_t bytes_per_iter = 0;
    switch (args.op)
    {
    case OP_RW:
        bytes_per_iter = elems * sizeof(float) * 2;
        break; // read + write
    case OP_READ:
        bytes_per_iter = elems * sizeof(float) * 1;
        break;
    case OP_WRITE:
        bytes_per_iter = elems * sizeof(float) * 1;
        break; // write stream
    case OP_COPY:
        bytes_per_iter = elems * sizeof(float) * 2;
        break; // read + write
    }

    // Use CUDA events to measure GPU-time per launch (gives you GB/s per pass)
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    double total_gpu_ms = 0.0;
    std::vector<cudaStream_t> streams(args.streams);
    for (int i = 0; i < args.streams; ++i)
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

    while (true)
    {
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int si = 0; si < args.streams; ++si)
        {
            switch (args.op)
            {
            case OP_RW:
                k_read_write<<<args.blocks, args.threads, 0, streams[si]>>>(a, b, elems, args.iters);
                break;
            case OP_READ:
                k_read_only<<<args.blocks, args.threads, 0, streams[si]>>>(a, elems, args.iters);
                break;
            case OP_WRITE:
                k_write_only<<<args.blocks, args.threads, 0, streams[si]>>>(b, elems, args.iters);
                break;
            case OP_COPY:
                if (args.vec == 4 && (elems % 4 == 0))
                {
                    size_t n4 = elems / 4;
                    k_copy_vec4<<<args.blocks, args.threads, 0, streams[si]>>>(
                        reinterpret_cast<const float4 *>(a),
                        reinterpret_cast<float4 *>(b),
                        n4, args.iters);
                }
                else
                {
                    k_copy<<<args.blocks, args.threads, 0, streams[si]>>>(a, b, elems, args.iters);
                }
                break;
            }
        }
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        total_gpu_ms += ms;
        launches++;

        auto now = std::chrono::high_resolution_clock::now();
        elapsed_s = std::chrono::duration<double>(now - t_start).count();
        if (args.seconds > 0 && elapsed_s >= args.seconds)
            break;
        if (args.seconds <= 0 && launches >= 1)
            break; // single measured launch if seconds<=0
    }

    // Report
    double bytes_total = (double)bytes_per_iter * (double)args.iters * (double)launches;
    double gb_total = bytes_total / (1024.0 * 1024.0 * 1024.0);
    double gbps = (gb_total) / (total_gpu_ms / 1000.0);
    const char *opname = (args.op == OP_RW ? "rw" : args.op == OP_READ ? "read"
                                                : args.op == OP_WRITE  ? "write"
                                                                       : "copy");

    printf("[memhog] op=%s gb=%.2f elems=%zu blocks=%d threads=%d iters/launch=%d launches=%d streams=%d vec=%d\n",
           opname, args.gb, elems, args.blocks, args.threads, args.iters, launches, args.streams, args.vec);
    printf("[memhog] GPU-time total=%.3f ms, processed=%.2f GiB, avg BW=%.2f GiB/s\n",
           total_gpu_ms, gb_total, gbps);
    printf("[memhog] elapsed wall=%.2f s\n", elapsed_s);

    // Keep device alive a moment in case profiler attaches
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < args.streams; ++i)
        cudaStreamDestroy(streams[i]);
    return 0;
}
