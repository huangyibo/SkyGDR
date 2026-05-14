// decode_transformer_kvcache_stress.cu
//
// Transformer-decoder-style HBM stress benchmark.
// It allocates a large KV cache in GPU memory and repeatedly generates
// tokens. Each decode step scans historical K/V cache for each layer/head.
//
// This is not a numerically faithful LLM implementation. It is a resource
// emulator for GPU HBM access patterns during decode.
//
// Build:
//   nvcc -O3 -std=c++17 decode_transformer_kvcache_stress.cu \
//     -o decode_kvcache
//
// Example:
//   ./decode_kvcache --gpu 1 --layers 32 --kv-heads 8 --head-dim 128 \
//     --context 8192 --gen 256 --elem fp32
//
// FP32 mode is used by default for simplicity and correctness.
// FP16 mode is not implemented in this minimal version.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

  int layers = 32;
  int kv_heads = 8;
  int head_dim = 128;

  int context_tokens = 8192;
  int gen_tokens = 256;

  int ffn_dim = 4096;

  // Number of CUDA thread blocks per FFN pass.
  int ffn_blocks = 512;
  int threads = 256;

  // Print progress every N generated tokens.
  int report_every = 32;
};

static void usage(const char *prog) {
  printf(
      "Usage: %s [options]\n"
      "\n"
      "Options:\n"
      "  --gpu N              GPU id. Default: 0\n"
      "  --layers N           Transformer layers. Default: 32\n"
      "  --kv-heads N         KV heads. Default: 8\n"
      "  --head-dim N         Head dimension. Default: 128\n"
      "  --context N          Initial context tokens. Default: 8192\n"
      "  --gen N              Number of generated tokens. Default: 256\n"
      "  --ffn-dim N          FFN working-set dimension. Default: 4096\n"
      "  --ffn-blocks N       FFN blocks. Default: 512\n"
      "  --threads N          CUDA threads per block. Default: 256\n"
      "  --report-every N     Progress interval. Default: 32\n"
      "  --help               Show help\n",
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
    } else if (strcmp(argv[i], "--ffn-dim") == 0) {
      opt.ffn_dim = std::atoi(need_arg(i, "--ffn-dim"));
    } else if (strcmp(argv[i], "--ffn-blocks") == 0) {
      opt.ffn_blocks = std::atoi(need_arg(i, "--ffn-blocks"));
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
      opt.context_tokens <= 0 || opt.gen_tokens <= 0 || opt.ffn_dim <= 0 ||
      opt.ffn_blocks <= 0 || opt.threads <= 0) {
    fprintf(stderr, "Invalid non-positive argument.\n");
    std::exit(1);
  }

  return opt;
}

static size_t kv_index(int layer,
                       int token,
                       int head,
                       int dim,
                       int max_tokens,
                       int kv_heads,
                       int head_dim) {
  // Layout:
  //   kv[layer][token][head][dim]
  size_t idx = layer;
  idx = idx * max_tokens + token;
  idx = idx * kv_heads + head;
  idx = idx * head_dim + dim;
  return idx;
}

__global__ void init_kv_cache(float *k_cache,
                              float *v_cache,
                              int layers,
                              int max_tokens,
                              int kv_heads,
                              int head_dim) {
  size_t total = static_cast<size_t>(layers) * max_tokens *
                 kv_heads * head_dim;

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < total; i += stride) {
    float x = static_cast<float>((i * 1315423911ULL) & 0xffff) / 65536.0f;
    k_cache[i] = x * 0.01f;
    v_cache[i] = x * 0.02f;
  }
}

__global__ void init_query_and_ffn(float *query,
                                   float *attn_out,
                                   float *ffn_a,
                                   float *ffn_b,
                                   int layers,
                                   int kv_heads,
                                   int head_dim,
                                   int ffn_dim) {
  size_t q_total = static_cast<size_t>(layers) * kv_heads * head_dim;
  size_t f_total = static_cast<size_t>(layers) * ffn_dim;

  size_t total = max(q_total, f_total);
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < total; i += stride) {
    if (i < q_total) {
      float x = static_cast<float>((i * 2654435761ULL) & 0xffff) / 65536.0f;
      query[i] = x * 0.01f;
      attn_out[i] = 0.0f;
    }

    if (i < f_total) {
      float y = static_cast<float>((i * 11400714819323198485ULL) & 0xffff) /
                65536.0f;
      ffn_a[i] = y;
      ffn_b[i] = 0.0f;
    }
  }
}

__device__ float block_reduce_sum(float value) {
  extern __shared__ float smem[];

  int tid = threadIdx.x;
  smem[tid] = value;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      smem[tid] += smem[tid + offset];
    }
    __syncthreads();
  }

  return smem[0];
}

// One block handles one (layer, kv_head).
//
// For each generated token, this kernel scans all previous tokens:
//   1. Reads K cache to compute attention scores.
//   2. Reads K again and V cache to compute weighted V.
//   3. Writes an attention output vector.
//
// This intentionally creates HBM traffic similar to decode attention.
__global__ void decode_attention_kernel(const float *__restrict__ k_cache,
                                        const float *__restrict__ v_cache,
                                        const float *__restrict__ query,
                                        float *__restrict__ attn_out,
                                        int layers,
                                        int max_tokens,
                                        int kv_heads,
                                        int head_dim,
                                        int seq_len) {
  int pair = blockIdx.x;
  int layer = pair / kv_heads;
  int head = pair % kv_heads;

  if (layer >= layers) {
    return;
  }

  int tid = threadIdx.x;

  const float scale = rsqrtf(static_cast<float>(head_dim));

  // Query base: query[layer][head][dim]
  size_t q_base = (static_cast<size_t>(layer) * kv_heads + head) * head_dim;

  // First pass: compute max attention logit for numerical stability.
  float local_max = -3.402823466e+38F;

  for (int tok = 0; tok < seq_len; ++tok) {
    float partial = 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x) {
      size_t idx = kv_index(layer, tok, head, d,
                            max_tokens, kv_heads, head_dim);
      partial += query[q_base + d] * k_cache[idx];
    }

    float dot = block_reduce_sum(partial) * scale;

    if (tid == 0) {
      local_max = fmaxf(local_max, dot);
    }
    __syncthreads();

    local_max = __shfl_sync(0xffffffff, local_max, 0);
  }

  // Second pass: compute softmax denominator.
  float denom = 0.0f;

  for (int tok = 0; tok < seq_len; ++tok) {
    float partial = 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x) {
      size_t idx = kv_index(layer, tok, head, d,
                            max_tokens, kv_heads, head_dim);
      partial += query[q_base + d] * k_cache[idx];
    }

    float dot = block_reduce_sum(partial) * scale;

    if (tid == 0) {
      denom += expf(dot - local_max);
    }
    __syncthreads();

    denom = __shfl_sync(0xffffffff, denom, 0);
  }

  // Third pass: weighted sum over V cache.
  //
  // Threads compute different dimensions of the output vector. This pass reads
  // K for weights and V for values. Real kernels fuse/optimize this heavily,
  // but the HBM access pattern is the key part for this benchmark.
  for (int out_d = tid; out_d < head_dim; out_d += blockDim.x) {
    float acc = 0.0f;

    for (int tok = 0; tok < seq_len; ++tok) {
      float partial = 0.0f;

      for (int d = 0; d < head_dim; ++d) {
        size_t k_idx = kv_index(layer, tok, head, d,
                                max_tokens, kv_heads, head_dim);
        partial += query[q_base + d] * k_cache[k_idx];
      }

      float weight = expf(partial * scale - local_max) / denom;

      size_t v_idx = kv_index(layer, tok, head, out_d,
                              max_tokens, kv_heads, head_dim);
      acc += weight * v_cache[v_idx];
    }

    attn_out[q_base + out_d] = acc;
  }
}

// FFN-style HBM stress.
//
// Real decode also reads model weights heavily. This kernel approximates
// the weight/activation memory traffic after attention.
__global__ void ffn_hbm_kernel(const float *__restrict__ attn_out,
                               float *__restrict__ ffn_a,
                               float *__restrict__ ffn_b,
                               int layers,
                               int kv_heads,
                               int head_dim,
                               int ffn_dim,
                               int step) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  size_t total = static_cast<size_t>(layers) * ffn_dim;
  int hidden_dim = kv_heads * head_dim;

  for (size_t i = tid; i < total; i += stride) {
    int layer = i / ffn_dim;
    int j = i % ffn_dim;

    int h0 = j % hidden_dim;
    int h1 = (j * 17 + step) % hidden_dim;

    size_t a0 = static_cast<size_t>(layer) * hidden_dim + h0;
    size_t a1 = static_cast<size_t>(layer) * hidden_dim + h1;

    float x = attn_out[a0] + 0.5f * attn_out[a1];
    float y = ffn_a[i];

    // Cheap nonlinear-ish math to prevent compiler elimination.
    y = y * 1.0001f + x;
    y = y / (1.0f + fabsf(y) * 0.00001f);

    ffn_b[i] = y;
  }
}

// Append synthetic K/V for the newly generated token.
// This makes seq_len grow over time, like real autoregressive decoding.
__global__ void append_generated_kv_kernel(float *k_cache,
                                           float *v_cache,
                                           const float *attn_out,
                                           int layers,
                                           int max_tokens,
                                           int kv_heads,
                                           int head_dim,
                                           int new_token_pos,
                                           int step) {
  size_t total = static_cast<size_t>(layers) * kv_heads * head_dim;

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t linear = tid; linear < total; linear += stride) {
    int d = linear % head_dim;
    int head = (linear / head_dim) % kv_heads;
    int layer = linear / (head_dim * kv_heads);

    size_t out_idx = (static_cast<size_t>(layer) * kv_heads + head) *
                     head_dim + d;

    float x = attn_out[out_idx];
    float noise = static_cast<float>((linear + step * 9973) & 0xff) * 0.0001f;

    size_t kv_idx = kv_index(layer, new_token_pos, head, d,
                             max_tokens, kv_heads, head_dim);

    k_cache[kv_idx] = x * 0.01f + noise;
    v_cache[kv_idx] = x * 0.02f + noise;
  }
}

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);

  CHECK_CUDA(cudaSetDevice(opt.gpu));

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, opt.gpu));

  int max_tokens = opt.context_tokens + opt.gen_tokens;

  size_t kv_elems = static_cast<size_t>(opt.layers) * max_tokens *
                    opt.kv_heads * opt.head_dim;

  size_t kv_bytes_one = kv_elems * sizeof(float);
  size_t kv_bytes_total = 2 * kv_bytes_one;

  size_t hidden_elems = static_cast<size_t>(opt.layers) *
                        opt.kv_heads * opt.head_dim;

  size_t ffn_elems = static_cast<size_t>(opt.layers) * opt.ffn_dim;

  printf("GPU: %d %s\n", opt.gpu, prop.name);
  printf("layers=%d kv_heads=%d head_dim=%d\n",
         opt.layers, opt.kv_heads, opt.head_dim);
  printf("context=%d gen=%d max_tokens=%d\n",
         opt.context_tokens, opt.gen_tokens, max_tokens);
  printf("KV cache: %.2f GiB total K+V, fp32\n",
         static_cast<double>(kv_bytes_total) /
         (1024.0 * 1024.0 * 1024.0));
  printf("FFN working set: %.2f MiB per array\n",
         static_cast<double>(ffn_elems * sizeof(float)) /
         (1024.0 * 1024.0));

  float *k_cache = nullptr;
  float *v_cache = nullptr;
  float *query = nullptr;
  float *attn_out = nullptr;
  float *ffn_a = nullptr;
  float *ffn_b = nullptr;

  CHECK_CUDA(cudaMalloc(&k_cache, kv_bytes_one));
  CHECK_CUDA(cudaMalloc(&v_cache, kv_bytes_one));
  CHECK_CUDA(cudaMalloc(&query, hidden_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&attn_out, hidden_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&ffn_a, ffn_elems * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&ffn_b, ffn_elems * sizeof(float)));

  int init_blocks = 1024;

  init_kv_cache<<<init_blocks, opt.threads>>>(k_cache,
                                              v_cache,
                                              opt.layers,
                                              max_tokens,
                                              opt.kv_heads,
                                              opt.head_dim);
  CHECK_CUDA(cudaGetLastError());

  init_query_and_ffn<<<init_blocks, opt.threads>>>(query,
                                                   attn_out,
                                                   ffn_a,
                                                   ffn_b,
                                                   opt.layers,
                                                   opt.kv_heads,
                                                   opt.head_dim,
                                                   opt.ffn_dim);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  int attention_blocks = opt.layers * opt.kv_heads;
  size_t shared_bytes = opt.threads * sizeof(float);

  printf("attention blocks=%d threads=%d shared=%zu bytes\n",
         attention_blocks, opt.threads, shared_bytes);

  auto t0 = std::chrono::high_resolution_clock::now();

  long double total_kv_read_bytes = 0.0L;
  long double total_kv_write_bytes = 0.0L;
  long double total_ffn_bytes = 0.0L;

  for (int step = 0; step < opt.gen_tokens; ++step) {
    int seq_len = opt.context_tokens + step;

    decode_attention_kernel<<<attention_blocks,
                              opt.threads,
                              shared_bytes>>>(k_cache,
                                              v_cache,
                                              query,
                                              attn_out,
                                              opt.layers,
                                              max_tokens,
                                              opt.kv_heads,
                                              opt.head_dim,
                                              seq_len);
    CHECK_CUDA(cudaGetLastError());

    ffn_hbm_kernel<<<opt.ffn_blocks, opt.threads>>>(attn_out,
                                                    ffn_a,
                                                    ffn_b,
                                                    opt.layers,
                                                    opt.kv_heads,
                                                    opt.head_dim,
                                                    opt.ffn_dim,
                                                    step);
    CHECK_CUDA(cudaGetLastError());

    append_generated_kv_kernel<<<init_blocks, opt.threads>>>(k_cache,
                                                             v_cache,
                                                             attn_out,
                                                             opt.layers,
                                                             max_tokens,
                                                             opt.kv_heads,
                                                             opt.head_dim,
                                                             seq_len,
                                                             step);
    CHECK_CUDA(cudaGetLastError());

    // Approximate minimum KV traffic:
    //   attention pass reads K twice and V once.
    //
    // This underestimates this naive kernel because the third pass recomputes
    // dot products in a less optimized way. For paper-quality accounting,
    // profile with Nsight Compute hardware counters.
    long double per_token_kv = static_cast<long double>(opt.layers) *
                               opt.kv_heads * opt.head_dim *
                               sizeof(float);

    total_kv_read_bytes += 3.0L * seq_len * per_token_kv;

    // Append one K and one V vector for the newly generated token.
    total_kv_write_bytes += 2.0L * per_token_kv;

    // FFN approximation:
    //   read ffn_a, write ffn_b, read a small attention vector.
    total_ffn_bytes += 2.0L * static_cast<long double>(ffn_elems) *
                       sizeof(float);

    if (opt.report_every > 0 &&
        ((step + 1) % opt.report_every == 0 || step + 1 == opt.gen_tokens)) {
      CHECK_CUDA(cudaDeviceSynchronize());
      printf("generated %d / %d tokens, current seq_len=%d\n",
             step + 1, opt.gen_tokens, seq_len + 1);
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());

  auto t1 = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(t1 - t0).count();

  long double total_bytes =
      total_kv_read_bytes + total_kv_write_bytes + total_ffn_bytes;

  double total_gib = static_cast<double>(total_bytes) /
                     (1024.0 * 1024.0 * 1024.0);

  double kv_read_gib = static_cast<double>(total_kv_read_bytes) /
                       (1024.0 * 1024.0 * 1024.0);

  double kv_write_gib = static_cast<double>(total_kv_write_bytes) /
                        (1024.0 * 1024.0 * 1024.0);

  double ffn_gib = static_cast<double>(total_ffn_bytes) /
                   (1024.0 * 1024.0 * 1024.0);

  printf("\nDone.\n");
  printf("Time: %.3f s\n", sec);
  printf("Generated tokens: %d\n", opt.gen_tokens);
  printf("Throughput: %.2f tokens/s\n", opt.gen_tokens / sec);
  printf("Approx KV read traffic:  %.2f GiB\n", kv_read_gib);
  printf("Approx KV write traffic: %.2f GiB\n", kv_write_gib);
  printf("Approx FFN traffic:      %.2f GiB\n", ffn_gib);
  printf("Approx total traffic:    %.2f GiB\n", total_gib);
  printf("Approx bandwidth:        %.2f GiB/s\n", total_gib / sec);

  CHECK_CUDA(cudaFree(k_cache));
  CHECK_CUDA(cudaFree(v_cache));
  CHECK_CUDA(cudaFree(query));
  CHECK_CUDA(cudaFree(attn_out));
  CHECK_CUDA(cudaFree(ffn_a));
  CHECK_CUDA(cudaFree(ffn_b));

  return 0;
}
