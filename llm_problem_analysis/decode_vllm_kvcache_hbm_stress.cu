// decode_vllm_kvcache_hbm_stress.cu
//
// vLLM-like HBM-sensitive decode KV-cache benchmark.
//
// Goal:
//   Emulate the dominant memory behavior of an optimized vLLM decoder node:
//     - many active decode sequences,
//     - repeated reads of historical K/V cache,
//     - tiled/paged KV-cache scanning,
//     - vectorized HBM loads,
//     - small partial writes,
//     - append new K/V for generated tokens.
//
// This is not a numerically faithful attention implementation.
// It is a hardware-behavior emulator for HBM-sensitive decoder execution.
//
// Build:
//   nvcc -O3 -std=c++17 decode_vllm_kvcache_hbm_stress.cu \
//     -o decode_vllm_kvcache_hbm_stress
//
// Example:
//   ./decode_vllm_kvcache_hbm_stress \
//     --gpu 1 \
//     --layers 80 \
//     --kv-heads 8 \
//     --head-dim 128 \
//     --context 32768 \
//     --gen 128 \
//     --batch 16 \
//     --tile-tokens 128

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err__));                                     \
      std::exit(1);                                                           \
    }                                                                        \
  } while (0)

struct Options {
  int gpu = 0;

  int layers = 80;
  int kv_heads = 8;
  int head_dim = 128;

  int context_tokens = 32768;
  int gen_tokens = 128;

  // Number of concurrent active decode sequences.
  //
  // Real vLLM decode becomes HBM-sensitive under continuous batching.
  // This benchmark emulates that by scanning the KV cache for multiple
  // active decode sequences per generated token.
  int batch = 16;

  // Number of historical tokens scanned by one CUDA block.
  //
  // This plays a similar role to a KV-cache page/tile. Smaller tiles create
  // more blocks and more parallelism. 64/128/256 are good values to try.
  int tile_tokens = 128;

  int threads = 256;
  int report_every = 16;
};

static void usage(const char *prog) {
  printf(
      "Usage: %s [options]\n"
      "\n"
      "Options:\n"
      "  --gpu N              GPU id. Default: 0\n"
      "  --layers N           Transformer layers. Default: 80\n"
      "  --kv-heads N         KV heads. Default: 8\n"
      "  --head-dim N         Head dimension. Must be multiple of 4. Default: 128\n"
      "  --context N          Initial context tokens. Default: 32768\n"
      "  --gen N              Generated tokens. Default: 128\n"
      "  --batch N            Active decode sequences. Default: 16\n"
      "  --tile-tokens N      Tokens per KV tile/block. Default: 128\n"
      "  --threads N          Threads per block. Default: 256\n"
      "  --report-every N     Progress interval. Default: 16\n"
      "  --help               Show this message\n",
      prog);
}

static Options parse_args(int argc, char **argv) {
  Options opt;

  auto need_arg = [&](int &i, const char *name) -> const char * {
    if (i + 1 >= argc) {
      fprintf(stderr, "Missing value for %s\n", name);
      std::exit(1);
    }
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--gpu") == 0) {
      opt.gpu = std::atoi(need_arg(i, "--gpu"));
    } else if (strcmp(argv[i], "--layers") == 0) {
      opt.layers = std::atoi(need_arg(i, "--layers"));
    } else if (strcmp(argv[i], "--kv-heads") == 0) {
      opt.kv_heads = std::atoi(need_arg(i, "--kv-heads"));
    } else if (strcmp(argv[i], "--head-dim") == 0) {
      opt.head_dim = std::atoi(need_arg(i, "--head-dim"));
    } else if (strcmp(argv[i], "--context") == 0) {
      opt.context_tokens = std::atoi(need_arg(i, "--context"));
    } else if (strcmp(argv[i], "--gen") == 0) {
      opt.gen_tokens = std::atoi(need_arg(i, "--gen"));
    } else if (strcmp(argv[i], "--batch") == 0) {
      opt.batch = std::atoi(need_arg(i, "--batch"));
    } else if (strcmp(argv[i], "--tile-tokens") == 0) {
      opt.tile_tokens = std::atoi(need_arg(i, "--tile-tokens"));
    } else if (strcmp(argv[i], "--threads") == 0) {
      opt.threads = std::atoi(need_arg(i, "--threads"));
    } else if (strcmp(argv[i], "--report-every") == 0) {
      opt.report_every = std::atoi(need_arg(i, "--report-every"));
    } else if (strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opt.layers <= 0 || opt.kv_heads <= 0 || opt.head_dim <= 0 ||
      opt.context_tokens <= 0 || opt.gen_tokens <= 0 || opt.batch <= 0 ||
      opt.tile_tokens <= 0 || opt.threads <= 0) {
    fprintf(stderr, "Invalid non-positive argument.\n");
    std::exit(1);
  }

  if (opt.head_dim % 4 != 0) {
    fprintf(stderr, "--head-dim must be a multiple of 4 for float4 loads.\n");
    std::exit(1);
  }

  return opt;
}

__host__ __device__ static inline size_t kv_vec_index(int layer,
                                                      int token,
                                                      int head,
                                                      int vec,
                                                      int max_tokens,
                                                      int kv_heads,
                                                      int vecs_per_head) {
  // Layout:
  //   kv[layer][token][head][vec4]
  size_t idx = static_cast<size_t>(layer);
  idx = idx * static_cast<size_t>(max_tokens) + static_cast<size_t>(token);
  idx = idx * static_cast<size_t>(kv_heads) + static_cast<size_t>(head);
  idx = idx * static_cast<size_t>(vecs_per_head) + static_cast<size_t>(vec);
  return idx;
}

__global__ void init_kv_cache(float4 *k_cache,
                              float4 *v_cache,
                              int layers,
                              int max_tokens,
                              int kv_heads,
                              int vecs_per_head) {
  size_t total = static_cast<size_t>(layers) * max_tokens *
                 kv_heads * vecs_per_head;

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < total; i += stride) {
    float x = static_cast<float>((i * 1315423911ULL) & 0xffff) / 65536.0f;
    k_cache[i] = make_float4(x, x + 0.1f, x + 0.2f, x + 0.3f);
    v_cache[i] = make_float4(x + 0.4f, x + 0.5f, x + 0.6f, x + 0.7f);
  }
}

// vLLM-like decode KV-cache scan.
//
// Grid:
//   blockIdx.x = layer
//   blockIdx.y = kv_head
//   blockIdx.z = batch_id * num_tiles + tile_id
//
// Each block scans one tile of historical K/V cache for one active decode
// sequence. The kernel performs vectorized reads of K and V from HBM and
// writes one small partial result.
//
// This preserves the dominant HBM behavior of decode attention:
//   read historical K/V cache heavily;
//   write only small partials.
__global__ void vllm_decode_kv_scan_kernel(const float4 *__restrict__ k_cache,
                                           const float4 *__restrict__ v_cache,
                                           float *__restrict__ partials,
                                           int layers,
                                           int max_tokens,
                                           int kv_heads,
                                           int vecs_per_head,
                                           int seq_len,
                                           int tile_tokens,
                                           int num_tiles,
                                           int batch) {
  int layer = blockIdx.x;
  int head = blockIdx.y;

  int z = blockIdx.z;
  int tile_id = z % num_tiles;
  int batch_id = z / num_tiles;

  if (layer >= layers || head >= kv_heads || batch_id >= batch) {
    return;
  }

  int token_start = tile_id * tile_tokens;
  int token_end = min(seq_len, token_start + tile_tokens);

  float local = 0.0f;

  int total_vecs = (token_end - token_start) * vecs_per_head;

  for (int linear = threadIdx.x; linear < total_vecs; linear += blockDim.x) {
    int rel_tok = linear / vecs_per_head;
    int vec = linear % vecs_per_head;
    int token = token_start + rel_tok;

    size_t idx = kv_vec_index(layer,
                              token,
                              head,
                              vec,
                              max_tokens,
                              kv_heads,
                              vecs_per_head);

    float4 k = k_cache[idx];
    float4 v = v_cache[idx];

    // Light arithmetic to prevent dead-code elimination.
    //
    // Real decode attention performs dot products and weighted V reductions.
    // Here we keep arithmetic intentionally light so HBM dominates.
    float s0 = k.x * 0.125f + k.y * 0.25f + k.z * 0.5f + k.w;
    float s1 = v.x * 0.125f + v.y * 0.25f + v.z * 0.5f + v.w;

    local += s0 + s1 + static_cast<float>(batch_id) * 0.00001f;
  }

  extern __shared__ float smem[];
  int tid = threadIdx.x;
  smem[tid] = local;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      smem[tid] += smem[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    size_t out_idx =
        (((static_cast<size_t>(layer) * kv_heads + head) * batch + batch_id) *
         num_tiles) +
        tile_id;

    partials[out_idx] = smem[0];
  }
}

// Append newly generated K/V for one token.
//
// Real vLLM writes new K/V for the current token into the KV cache.
// This write traffic is small compared with historical K/V reads.
__global__ void append_new_kv_kernel(float4 *k_cache,
                                     float4 *v_cache,
                                     const float *__restrict__ partials,
                                     int layers,
                                     int max_tokens,
                                     int kv_heads,
                                     int vecs_per_head,
                                     int num_tiles,
                                     int batch,
                                     int new_token_pos,
                                     int step) {
  size_t total = static_cast<size_t>(layers) * kv_heads * vecs_per_head;

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t linear = tid; linear < total; linear += stride) {
    int vec = linear % vecs_per_head;
    int head = (linear / vecs_per_head) % kv_heads;
    int layer = linear / (vecs_per_head * kv_heads);

    size_t pidx =
        (((static_cast<size_t>(layer) * kv_heads + head) * batch + 0) *
         num_tiles);

    float p = partials[pidx];

    float base = 0.000001f * p +
                 static_cast<float>((linear + step * 9973) & 0xff) * 0.001f;

    float4 k = make_float4(base, base + 0.1f, base + 0.2f, base + 0.3f);
    float4 v = make_float4(base + 0.4f, base + 0.5f, base + 0.6f, base + 0.7f);

    size_t idx = kv_vec_index(layer,
                              new_token_pos,
                              head,
                              vec,
                              max_tokens,
                              kv_heads,
                              vecs_per_head);

    k_cache[idx] = k;
    v_cache[idx] = v;
  }
}

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);

  CHECK_CUDA(cudaSetDevice(opt.gpu));

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, opt.gpu));

  int vecs_per_head = opt.head_dim / 4;
  int max_tokens = opt.context_tokens + opt.gen_tokens;
  int max_num_tiles = (max_tokens + opt.tile_tokens - 1) / opt.tile_tokens;

  size_t kv_vecs = static_cast<size_t>(opt.layers) * max_tokens *
                   opt.kv_heads * vecs_per_head;

  size_t kv_bytes_one = kv_vecs * sizeof(float4);
  size_t kv_bytes_total = 2 * kv_bytes_one;

  size_t partial_elems = static_cast<size_t>(opt.layers) * opt.kv_heads *
                         opt.batch * max_num_tiles;

  size_t partial_bytes = partial_elems * sizeof(float);

  printf("GPU: %d %s\n", opt.gpu, prop.name);
  printf("layers=%d kv_heads=%d head_dim=%d vecs_per_head=%d\n",
         opt.layers, opt.kv_heads, opt.head_dim, vecs_per_head);
  printf("context=%d gen=%d max_tokens=%d\n",
         opt.context_tokens, opt.gen_tokens, max_tokens);
  printf("batch=%d tile_tokens=%d max_num_tiles=%d\n",
         opt.batch, opt.tile_tokens, max_num_tiles);
  printf("KV cache: %.2f GiB total K+V, fp32/float4 storage\n",
         static_cast<double>(kv_bytes_total) /
             (1024.0 * 1024.0 * 1024.0));
  printf("partials: %.2f MiB\n",
         static_cast<double>(partial_bytes) / (1024.0 * 1024.0));

  float4 *k_cache = nullptr;
  float4 *v_cache = nullptr;
  float *partials = nullptr;

  CHECK_CUDA(cudaMalloc(&k_cache, kv_bytes_one));
  CHECK_CUDA(cudaMalloc(&v_cache, kv_bytes_one));
  CHECK_CUDA(cudaMalloc(&partials, partial_bytes));

  int init_blocks = 4096;
  int threads = opt.threads;

  init_kv_cache<<<init_blocks, threads>>>(k_cache,
                                          v_cache,
                                          opt.layers,
                                          max_tokens,
                                          opt.kv_heads,
                                          vecs_per_head);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemset(partials, 0, partial_bytes));
  CHECK_CUDA(cudaDeviceSynchronize());

  printf("Initialization done.\n");

  long double total_kv_read_bytes = 0.0L;
  long double total_kv_write_bytes = 0.0L;
  long double total_partial_write_bytes = 0.0L;

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < opt.gen_tokens; ++step) {
    int seq_len = opt.context_tokens + step;
    int num_tiles = (seq_len + opt.tile_tokens - 1) / opt.tile_tokens;

    dim3 grid(opt.layers, opt.kv_heads, num_tiles * opt.batch);
    size_t shmem = static_cast<size_t>(threads) * sizeof(float);

    vllm_decode_kv_scan_kernel<<<grid, threads, shmem>>>(k_cache,
                                                         v_cache,
                                                         partials,
                                                         opt.layers,
                                                         max_tokens,
                                                         opt.kv_heads,
                                                         vecs_per_head,
                                                         seq_len,
                                                         opt.tile_tokens,
                                                         num_tiles,
                                                         opt.batch);
    CHECK_CUDA(cudaGetLastError());

    int append_blocks = 1024;
    append_new_kv_kernel<<<append_blocks, threads>>>(k_cache,
                                                     v_cache,
                                                     partials,
                                                     opt.layers,
                                                     max_tokens,
                                                     opt.kv_heads,
                                                     vecs_per_head,
                                                     num_tiles,
                                                     opt.batch,
                                                     seq_len,
                                                     step);
    CHECK_CUDA(cudaGetLastError());

    // Approximate HBM traffic:
    //
    // Each active decode sequence reads K and V for all historical tokens.
    // K/V are stored as float4, i.e., FP32 storage here.
    long double per_seq_kv_read =
        static_cast<long double>(seq_len) *
        opt.layers *
        opt.kv_heads *
        vecs_per_head *
        2.0L *
        sizeof(float4);

    total_kv_read_bytes +=
        static_cast<long double>(opt.batch) * per_seq_kv_read;

    // Append one K and one V vector for one generated token.
    long double append_write =
        static_cast<long double>(opt.layers) *
        opt.kv_heads *
        vecs_per_head *
        2.0L *
        sizeof(float4);

    total_kv_write_bytes += append_write;

    long double partial_write =
        static_cast<long double>(opt.layers) *
        opt.kv_heads *
        opt.batch *
        num_tiles *
        sizeof(float);

    total_partial_write_bytes += partial_write;

    if (opt.report_every > 0 &&
        ((step + 1) % opt.report_every == 0 ||
         step + 1 == opt.gen_tokens)) {
      CHECK_CUDA(cudaDeviceSynchronize());
      printf("generated %d / %d tokens, seq_len=%d, num_tiles=%d\n",
             step + 1, opt.gen_tokens, seq_len + 1, num_tiles);
      fflush(stdout);
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  auto t1 = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(t1 - t0).count();

  long double total_bytes =
      total_kv_read_bytes + total_kv_write_bytes + total_partial_write_bytes;

  double total_gib =
      static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);

  double kv_read_gib =
      static_cast<double>(total_kv_read_bytes) /
      (1024.0 * 1024.0 * 1024.0);

  double kv_write_gib =
      static_cast<double>(total_kv_write_bytes) /
      (1024.0 * 1024.0 * 1024.0);

  double partial_gib =
      static_cast<double>(total_partial_write_bytes) /
      (1024.0 * 1024.0 * 1024.0);

  printf("\nDone.\n");
  printf("Time: %.3f s\n", sec);
  printf("Generated tokens: %d\n", opt.gen_tokens);
  printf("Active decode batch: %d\n", opt.batch);
  printf("Throughput: %.2f tokens/s\n", opt.gen_tokens / sec);
  printf("Effective sequence-throughput: %.2f seq-tokens/s\n",
         static_cast<double>(opt.gen_tokens * opt.batch) / sec);
  printf("Approx KV read traffic:       %.2f GiB\n", kv_read_gib);
  printf("Approx KV append write:       %.2f GiB\n", kv_write_gib);
  printf("Approx partial write traffic: %.2f GiB\n", partial_gib);
  printf("Approx total traffic:         %.2f GiB\n", total_gib);
  printf("Approx bandwidth:             %.2f GiB/s\n", total_gib / sec);

  CHECK_CUDA(cudaFree(k_cache));
  CHECK_CUDA(cudaFree(v_cache));
  CHECK_CUDA(cudaFree(partials));

  return 0;
}
