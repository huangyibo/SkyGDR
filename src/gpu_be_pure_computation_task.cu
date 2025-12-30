// gpu_be_pure_computation_task.cu
// Build (SM80): nvcc gpu_be_pure_computation_task.cu -O3 -std=c++14 -o gpu_be_pure_computation_task -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
// How to run (L2 cache pressure): ./gpu_be_pure_computation_task --op=L2_PRESSURE=0.5:60
// How to run (L1 cache or no pressure): ./gpu_be_pure_computation_task --op=Y/N
// How to run (global memory pressure): ./gpu_be_pure_computation_task --op=HBM

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>

// Define the size of the arrays within a single thread block
#define BLOCK_ARRAY_SIZE 1024

// consistent global memory pressure
__global__ void memory_pressure_kernel(const float * __restrict__ input, float * __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        output[i] = input[i];
    }
}

// --- Your original L1-heavy shared-memory kernel unchanged ---
__global__ void l1_cache_loop_kernel(float *dummy_output)
{
    __shared__ float sh_A[BLOCK_ARRAY_SIZE];
    __shared__ float sh_B[BLOCK_ARRAY_SIZE];
    __shared__ float sh_C[BLOCK_ARRAY_SIZE];

    int tid = threadIdx.x;

    while (1)
    {
        if (tid < BLOCK_ARRAY_SIZE)
        {
            sh_A[tid] = (float)tid;
            sh_B[tid] = (float)(BLOCK_ARRAY_SIZE - tid);
        }
        __syncthreads();

        if (tid < BLOCK_ARRAY_SIZE)
        {
            sh_C[tid] = sh_A[tid] + sh_B[tid];
        }
        __syncthreads();

        if (tid == 0)
        {
            *dummy_output += sh_C[0];
        }
    }
}

// --- Helper L2 access kernels ---

// Kernel used to perform an initial preload (touch) of the region: this will go to HBM
__global__ void preload_touch_kernel(float *buf, size_t elements)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Simple write to ensure pages are resident and cached/allocated.
    for (size_t i = tid; i < elements; i += stride)
    {
        // write a non-zero pattern so memory is actually touched
        buf[i] = (float)(i & 0xFF);
    }
}

// Kernel that accesses only the persisted window repeatedly for sustain iterations.
// It does not touch outside 'elements' region so it should hit L2 (if persisted) rather than HBM.
__global__ void l2_persist_worker(float *buf, size_t elements, unsigned long long iterations)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float acc = 0.0f;

    // A light-weight loop that repeatedly streams through the persisted region
    for (unsigned long long it = 0; it < iterations; ++it)
    {
        // Strided walk; this only reads within 'elements' so should not go to HBM after persisting
        for (size_t i = tid; i < elements; i += stride)
        {
            acc += buf[i];
        }

        // A tiny barrier: use threadfence_block to keep things visible inside the block
        // (not required, but helps avoid compiler/hardware reordering in some cases)
        __syncthreads();
    }

    // Prevent the compiler from optimizing away accesses.
    if (tid == 0)
        buf[0] = acc;
}

// Utility: print CUDA error and exit if failure
static void checkCuda(cudaError_t st, const char *msg)
{
    if (st != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR (%s): %s\n", msg, cudaGetErrorString(st));
        exit(-1);
    }
}

// ------------------ New function: run_l2_persisting_pressure ------------------
//
// target_fraction: fraction of A100 L2 to occupy, range (0.0, 1.0]. Example: 0.5 => 50% of 40MB => 20MB
// sustain_seconds: how many seconds to keep sustained L2 pressure (kernel runs for this duration).
//
// Behavior:
//  1) Allocate a device buffer (up to MAX_BUFFER_BYTES).
//  2) Preload (touch) only the window once to bring it into device memory (this will hit HBM once).
//  3) Use cudaStreamSetAttribute with cudaAccessPropertyPersisting to mark the window as persisting in L2.
//  4) Launch a worker kernel that repeatedly accesses only that persisted window for sustain_seconds,
//     providing sustained L2 pressure without further HBM traffic (assuming persistence holds).
//
void run_l2_persisting_pressure(float target_fraction, int sustain_seconds)
{
    if (target_fraction <= 0.0f || target_fraction > 1.0f)
    {
        fprintf(stderr, "target_fraction must be in (0.0, 1.0]\n");
        return;
    }
    if (sustain_seconds <= 0)
    {
        fprintf(stderr, "sustain_seconds must be > 0\n");
        return;
    }

    printf("[INFO] run_l2_persisting_pressure: target_fraction=%f sustain_seconds=%d\n", target_fraction, sustain_seconds);

    // A100 approximate L2 size
    const size_t A100_L2_BYTES = 40ULL * 1024 * 1024;    // 40 MB
    const size_t MAX_BUFFER_BYTES = 64ULL * 1024 * 1024; // 64 MB buffer allocation to be safe

    // Determine the desired window size (bytes) based on fraction of L2
    size_t desired_bytes = (size_t)(A100_L2_BYTES * target_fraction);

    // Bound desired_bytes so we don't exceed our buffer allocation
    if (desired_bytes > MAX_BUFFER_BYTES)
        desired_bytes = MAX_BUFFER_BYTES;

    // Ensure at least one cache-line-aligned size (round up to multiple of sizeof(float))
    size_t window_elements = (desired_bytes + sizeof(float) - 1) / sizeof(float);
    size_t window_bytes = window_elements * sizeof(float);

    printf("[INFO] Desired window: %zu bytes (~%zu MB) -> %zu elements\n",
           window_bytes, window_bytes / (1024 * 1024), window_elements);

    // Allocate a device buffer (MAX_BUFFER_BYTES). We'll only persist the first 'window_bytes' region.
    float *d_buf = nullptr;
    checkCuda(cudaMalloc(&d_buf, MAX_BUFFER_BYTES), "cudaMalloc d_buf");
    checkCuda(cudaMemset(d_buf, 0, MAX_BUFFER_BYTES), "cudaMemset d_buf");

    // 1) One-time preload: touch the window region (this will use HBM once)
    {
        int threads = 256;
        int blocks = 108 * 8;
        size_t elems_to_touch = window_elements;
        printf("[INFO] Preloading (touching %zu elements) to bring region into HBM & cache (one-time)...\n", elems_to_touch);
        preload_touch_kernel<<<blocks, threads>>>(d_buf, elems_to_touch);
        checkCuda(cudaGetLastError(), "preload_touch_kernel launch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize preload");
    }

    // 2) Configure stream access policy window to mark base_ptr..base_ptr+window_bytes as persisting in L2
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Prepare accessPolicyWindow attribute
    // Use cudaStreamAttrValue union to set accessPolicyWindow
    // Note: this struct and field names exist in supported CUDA versions.
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));

    // fill accessPolicyWindow
    attr.accessPolicyWindow.base_ptr = (void *)d_buf;
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    // Apply to stream: from this point accesses on that stream will use the policy window
    cudaError_t st = cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &attr);

    if (st != cudaSuccess)
    {
        fprintf(stderr, "ERROR: cudaStreamSetAttribute failed: %s\n", cudaGetErrorString(st));
        // Cleanup and exit the function
        cudaStreamDestroy(stream);
        cudaFree(d_buf);
        return;
    }
    printf("[INFO] Access policy window set: persisting first %zu bytes in L2 on the stream.\n", window_bytes);

    // 3) Launch the worker kernel that repeatedly accesses only the window region
    // Compute iterations roughly based on sustain_seconds:
    // We don't have precise wall-clock per-iteration cost here; choose an iterations count that
    // results in a long-running kernel. To be robust, compute iterations such that the inner loop
    // will iterate many times; we also periodically synchronize to check time (we won't here).
    int threads = 256;
    int blocks = 108 * 8;

    // We choose a conservative iterations value: each iteration performs a full sweep of the window.
    // To approximate sustained_seconds, pick iterations = sustain_seconds * 1000 (tunable).
    // If you want longer/shorter, adjust multiplier.
    unsigned long long iterations = (unsigned long long)sustain_seconds * 1000ULL;

    printf("[INFO] Launching l2_persist_worker for %llu iterations (approx %d seconds)...\n", iterations, sustain_seconds);

    // Launch on the configured stream so access policy applies
    l2_persist_worker<<<blocks, threads, 0, stream>>>(d_buf, window_elements, iterations);
    checkCuda(cudaGetLastError(), "l2_persist_worker launch");

    // Wait for completion (sustained period)
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize worker");

    // 4) Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_buf);

    printf("[INFO] run_l2_persisting_pressure finished.\n");
}

// ------------------ End of L2 persist function ------------------

// Main program: parse flags and run selected mode
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s --op=Y|N|L2_PRESSURE|HBM\n", argv[0]);
        fprintf(stderr, "  --op=Y         : run original L1 cache heavy kernel (infinite loop)\n");
        fprintf(stderr, "  --op=N         : idle (do nothing)\n");
        fprintf(stderr, "  --op=L2_PRESSURE=<fraction>:<seconds>  : e.g. --op=L2_PRESSURE=0.5:30 (50%% of L2 for 30s)\n");
        fprintf(stderr, "  --op=HBM       : keep pressure on global memory\n");
        return -1;
    }

    int op_flag = 0; // 0: idle, 1: L1 cache heavy computation, 2: L2 pressure

    // parse simple flags
    if (strcmp(argv[1], "--op=Y") == 0)
    {
        op_flag = 1;
    }
    else if (strcmp(argv[1], "--op=N") == 0)
    {
        op_flag = 0;
    }
    else if (strncmp(argv[1], "--op=L2_PRESSURE=", 16) == 0)
    {
        op_flag = 2;
    }
    else if (strncmp(argv[1], "--op=HBM", 9) == 0)
    {
        op_flag = 3;
    }
    else
    {
        fprintf(stderr, "Invalid operation flag. Use --op=Y or --op=N or --op=L2_PRESSURE=...\n");
        return -1;
    }

    // 1. Allocate a small dummy buffer on the device for the optional global write
    float *d_dummy_out = nullptr;
    cudaError_t cudaStatus = cudaMalloc(&d_dummy_out, sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    cudaMemset(d_dummy_out, 0, sizeof(float));

    // 2. Set up the launch configuration
    int num_blocks = 108 * 8; // multiple blocks per SM
    int threads_per_block = 1024;

    // Handle each op
    if (op_flag == 1)
    {
        // run L1 kernel (infinite)
        printf("[INFO] Launching L1 cache loop kernel...\n");
        l1_cache_loop_kernel<<<num_blocks, threads_per_block>>>(d_dummy_out);
        cudaDeviceSynchronize();
    }
    else if (op_flag == 2)
    {
        // parse argument --op=L2_PRESSURE=<fraction>:<seconds>
        // default values
        float fraction = 0.5f;
        int seconds = 30;

        // parse provided value after '='
        const char *arg = argv[1] + 16; // skip "--op=L2_PRESSURE="
        // expected format fraction:seconds
        float f = 0.0f;
        int s = 0;
        if (sscanf(arg, "%f:%d", &f, &s) >= 1)
        {
            if (f > 0.0f)
                fraction = f;
            if (s > 0)
                seconds = s;
        }
        printf("[INFO] Starting L2 pressure with fraction=%f for %d seconds\n", fraction, seconds);
        run_l2_persisting_pressure(fraction, seconds);
    }
    else if (op_flag == 3)
    {
        // run global memory pressure kernel (infinite)
        printf("[INFO] Launching global memory pressure kernel...\n");
        // Allocate large buffers for memory pressure
        size_t n = 1 << 28;
        size_t size = n * sizeof(float);
        float *d_input = nullptr;
        float *d_output = nullptr;

        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        // Initialize input buffer
        cudaMemset(d_input, 1, size);

        int threads_per_block = 256;
        int num_blocks = (n + threads_per_block - 1) / threads_per_block;

        while (true) {
            memory_pressure_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, n);
            cudaDeviceSynchronize();
        }

        // Cleanup pressure buffers
        cudaFree(d_input);
        cudaFree(d_output);
    }
    else
    {
        printf("[INFO] Idle mode (--op=N). Nothing launched.\n");
    }

    // Cleanup
    cudaFree(d_dummy_out);

    return 0;
}

