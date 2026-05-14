// kv_nvlink_transfer_trace.cu
//
// Emulate P -> D KV-cache transfer over NVLink/NVSwitch using a
// trace-inspired KV-cache size distribution.
//
// Default distribution is based on Inferact/codex_swebenchpro_traces:
//   cached tokens per LLM call:
//     p50 = 60,928
//     p90 = 112,512
//     p99 = 162,048
//
// Example:
//   nvcc -O3 -std=c++17 kv_nvlink_transfer_trace.cu -o kv_trace
//
//   # Llama-style default: 80 layers, 8 KV heads, head dim 128, fp16.
//   ./kv_trace --src 0 --dst 1 --iters 10000
//
//   # Emulate only newly computed KV, not cached-prefix KV.
//   ./kv_trace --src 0 --dst 1 --dist computed --iters 10000
//
//   # Smaller model, e.g., 32 layers, 8 KV heads, head_dim 128.
//   ./kv_trace --src 0 --dst 1 --layers 32 --kv-heads 8 --head-dim 128
//
// KV bytes formula:
//   bytes = tokens * layers * kv_heads * head_dim * 2(K,V) * elem_bytes

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

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
  int src_gpu = 0;
  int dst_gpu = 1;
  int iters = 10000;

  // Model KV geometry.
  int layers = 80;
  int kv_heads = 8;
  int head_dim = 128;
  int elem_bytes = 2;  // fp16/bf16

  // Distribution: cached, input, or computed.
  std::string dist = "cached";

  // Synthetic tail cap. If 0, inferred from distribution.
  size_t max_tokens = 0;

  // Queue sync interval.
  int sync_every = 256;

  // RNG seed.
  uint64_t seed = 1;
};

struct QuantileDist {
  size_t p50;
  size_t p90;
  size_t p99;
  size_t max_v;
};

// Inferact/codex_swebenchpro_traces aggregate statistics.
//
// Per-call input tokens:
//   p50 63,917; p90 114,888; p99 166,322
//
// Per-call cached tokens:
//   p50 60,928; p90 112,512; p99 162,048
//
// Per-call computed tokens:
//   p50 758; p90 8,736; p99 53,323
static QuantileDist get_dist(const std::string &name, size_t max_override) {
  QuantileDist d{};

  if (name == "input") {
    d = {63917, 114888, 166322, 200000};
  } else if (name == "cached") {
    d = {60928, 112512, 162048, 200000};
  } else if (name == "computed") {
    d = {758, 8736, 53323, 70000};
  } else {
    fprintf(stderr, "Unknown dist=%s. Use input|cached|computed.\n",
            name.c_str());
    std::exit(1);
  }

  if (max_override > 0) {
    d.max_v = max_override;
  }

  d.max_v = std::max(d.max_v, d.p99);
  return d;
}

static void usage(const char *prog) {
  printf(
      "Usage: %s [options]\n"
      "\n"
      "Options:\n"
      "  --src N             Source/P GPU id. Default: 0\n"
      "  --dst N             Destination/D GPU id. Default: 1\n"
      "  --iters N           Number of transfers. Default: 10000\n"
      "  --dist NAME         input|cached|computed. Default: cached\n"
      "  --layers N          Transformer layers. Default: 80\n"
      "  --kv-heads N        Number of KV heads. Default: 8\n"
      "  --head-dim N        KV head dimension. Default: 128\n"
      "  --elem-bytes N      KV element bytes. fp16/bf16=2. Default: 2\n"
      "  --max-tokens N      Override synthetic tail max tokens.\n"
      "  --sync-every N      Stream sync interval. Default: 256\n"
      "  --seed N            RNG seed. Default: 1\n"
      "  --help              Print this message.\n",
      prog);
}

static Options parse_args(int argc, char **argv) {
  Options opt;

  for (int i = 1; i < argc; ++i) {
    auto need_arg = [&](const char *name) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", name);
        std::exit(1);
      }
      return argv[++i];
    };

    if (strcmp(argv[i], "--src") == 0) {
      opt.src_gpu = std::atoi(need_arg("--src"));
    } else if (strcmp(argv[i], "--dst") == 0) {
      opt.dst_gpu = std::atoi(need_arg("--dst"));
    } else if (strcmp(argv[i], "--iters") == 0) {
      opt.iters = std::atoi(need_arg("--iters"));
    } else if (strcmp(argv[i], "--dist") == 0) {
      opt.dist = need_arg("--dist");
    } else if (strcmp(argv[i], "--layers") == 0) {
      opt.layers = std::atoi(need_arg("--layers"));
    } else if (strcmp(argv[i], "--kv-heads") == 0) {
      opt.kv_heads = std::atoi(need_arg("--kv-heads"));
    } else if (strcmp(argv[i], "--head-dim") == 0) {
      opt.head_dim = std::atoi(need_arg("--head-dim"));
    } else if (strcmp(argv[i], "--elem-bytes") == 0) {
      opt.elem_bytes = std::atoi(need_arg("--elem-bytes"));
    } else if (strcmp(argv[i], "--max-tokens") == 0) {
      opt.max_tokens = std::strtoull(need_arg("--max-tokens"), nullptr, 10);
    } else if (strcmp(argv[i], "--sync-every") == 0) {
      opt.sync_every = std::atoi(need_arg("--sync-every"));
    } else if (strcmp(argv[i], "--seed") == 0) {
      opt.seed = std::strtoull(need_arg("--seed"), nullptr, 10);
    } else if (strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(0);
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(1);
    }
  }

  if (opt.iters <= 0 || opt.layers <= 0 || opt.kv_heads <= 0 ||
      opt.head_dim <= 0 || opt.elem_bytes <= 0 || opt.sync_every <= 0) {
    fprintf(stderr, "Invalid non-positive argument.\n");
    std::exit(1);
  }

  return opt;
}

static size_t align_up(size_t x, size_t align) {
  return ((x + align - 1) / align) * align;
}

static size_t kv_bytes_for_tokens(size_t tokens, const Options &opt) {
  // tokens * layers * kv_heads * head_dim * 2(K,V) * elem_bytes
  long double bytes = static_cast<long double>(tokens);
  bytes *= static_cast<long double>(opt.layers);
  bytes *= static_cast<long double>(opt.kv_heads);
  bytes *= static_cast<long double>(opt.head_dim);
  bytes *= 2.0L;
  bytes *= static_cast<long double>(opt.elem_bytes);

  size_t out = static_cast<size_t>(bytes);

  // Use 256-byte alignment for copy size.
  return align_up(out, 256);
}

// Sample from a piecewise log-uniform distribution constrained by
// p50/p90/p99/max. This preserves the heavy-tail nature without needing
// the full raw trace file.
//
// Probability intervals:
//   [small, p50]  : 50%
//   [p50, p90]   : 40%
//   [p90, p99]   : 9%
//   [p99, max]   : 1%
static size_t sample_tokens(const QuantileDist &d, std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  double u = u01(rng);

  size_t lo = 1;
  size_t hi = d.p50;

  if (u < 0.50) {
    lo = std::max<size_t>(1, d.p50 / 8);
    hi = d.p50;
  } else if (u < 0.90) {
    lo = d.p50;
    hi = d.p90;
  } else if (u < 0.99) {
    lo = d.p90;
    hi = d.p99;
  } else {
    lo = d.p99;
    hi = d.max_v;
  }

  lo = std::max<size_t>(1, lo);
  hi = std::max(hi, lo);

  // Log-uniform interpolation avoids over-concentrating near the upper bound.
  double log_lo = std::log(static_cast<double>(lo));
  double log_hi = std::log(static_cast<double>(hi));
  double x = std::exp(log_lo + (log_hi - log_lo) * u01(rng));

  return std::max<size_t>(1, static_cast<size_t>(x));
}

__global__ void init_kernel(unsigned char *p, size_t n, unsigned char value) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += stride) {
    p[i] = static_cast<unsigned char>(value + (i & 0xff));
  }
}

static void print_percentiles(std::vector<size_t> v, const char *name) {
  std::sort(v.begin(), v.end());

  auto getp = [&](double p) -> size_t {
    if (v.empty()) return 0;
    size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
    return v[idx];
  };

  printf("%s: p50=%zu p90=%zu p99=%zu max=%zu\n",
         name, getp(0.50), getp(0.90), getp(0.99), v.back());
}

int main(int argc, char **argv) {
  Options opt = parse_args(argc, argv);
  QuantileDist qdist = get_dist(opt.dist, opt.max_tokens);

  int dev_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&dev_count));

  if (opt.src_gpu >= dev_count || opt.dst_gpu >= dev_count ||
      opt.src_gpu == opt.dst_gpu) {
    fprintf(stderr, "Invalid GPU ids. dev_count=%d src=%d dst=%d\n",
            dev_count, opt.src_gpu, opt.dst_gpu);
    return 1;
  }

  int can_src_to_dst = 0;
  int can_dst_to_src = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can_src_to_dst,
                                     opt.src_gpu,
                                     opt.dst_gpu));
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can_dst_to_src,
                                     opt.dst_gpu,
                                     opt.src_gpu));

  printf("src_gpu=%d dst_gpu=%d iters=%d dist=%s\n",
         opt.src_gpu, opt.dst_gpu, opt.iters, opt.dist.c_str());
  printf("peer_access src->dst=%d dst->src=%d\n",
         can_src_to_dst, can_dst_to_src);
  printf("KV geometry: layers=%d kv_heads=%d head_dim=%d elem_bytes=%d\n",
         opt.layers, opt.kv_heads, opt.head_dim, opt.elem_bytes);
  printf("Trace quantiles: p50=%zu p90=%zu p99=%zu max=%zu tokens\n",
         qdist.p50, qdist.p90, qdist.p99, qdist.max_v);

  std::mt19937_64 rng(opt.seed);
  std::vector<size_t> token_samples;
  std::vector<size_t> byte_samples;

  token_samples.reserve(opt.iters);
  byte_samples.reserve(opt.iters);

  size_t max_bytes = 0;
  long double total_bytes_planned = 0.0L;

  for (int i = 0; i < opt.iters; ++i) {
    size_t tokens = sample_tokens(qdist, rng);
    size_t bytes = kv_bytes_for_tokens(tokens, opt);

    token_samples.push_back(tokens);
    byte_samples.push_back(bytes);

    max_bytes = std::max(max_bytes, bytes);
    total_bytes_planned += static_cast<long double>(bytes);
  }

  print_percentiles(token_samples, "sampled tokens");
  print_percentiles(byte_samples, "sampled bytes");

  printf("max transfer allocation: %.2f GiB\n",
         static_cast<double>(max_bytes) / (1024.0 * 1024.0 * 1024.0));

  unsigned char *src = nullptr;
  unsigned char *dst = nullptr;

  CHECK_CUDA(cudaSetDevice(opt.src_gpu));
  CHECK_CUDA(cudaMalloc(&src, max_bytes));

  if (can_src_to_dst) {
    cudaError_t e = cudaDeviceEnablePeerAccess(opt.dst_gpu, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      CHECK_CUDA(e);
    }
  }

  CHECK_CUDA(cudaSetDevice(opt.dst_gpu));
  CHECK_CUDA(cudaMalloc(&dst, max_bytes));

  if (can_dst_to_src) {
    cudaError_t e = cudaDeviceEnablePeerAccess(opt.src_gpu, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      CHECK_CUDA(e);
    }
  }

  CHECK_CUDA(cudaSetDevice(opt.src_gpu));
  init_kernel<<<256, 256>>>(src, max_bytes, 1);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaSetDevice(opt.dst_gpu));
  init_kernel<<<256, 256>>>(dst, max_bytes, 0);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaStream_t stream;
  CHECK_CUDA(cudaSetDevice(opt.dst_gpu));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Warmup with median-sized transfer.
  size_t warmup_bytes = kv_bytes_for_tokens(qdist.p50, opt);
  warmup_bytes = std::min(warmup_bytes, max_bytes);

  for (int i = 0; i < 16; ++i) {
    CHECK_CUDA(cudaMemcpyPeerAsync(dst,
                                   opt.dst_gpu,
                                   src,
                                   opt.src_gpu,
                                   warmup_bytes,
                                   stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < opt.iters; ++i) {
    size_t bytes = byte_samples[i];

    CHECK_CUDA(cudaMemcpyPeerAsync(dst,
                                   opt.dst_gpu,
                                   src,
                                   opt.src_gpu,
                                   bytes,
                                   stream));

    if ((i + 1) % opt.sync_every == 0) {
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto t1 = std::chrono::high_resolution_clock::now();

  double sec = std::chrono::duration<double>(t1 - t0).count();
  double total_gib =
      static_cast<double>(total_bytes_planned) / (1024.0 * 1024.0 * 1024.0);
  double gib_s = total_gib / sec;

  printf("Trace-shaped KV transfer done:\n");
  printf("  transfers: %d\n", opt.iters);
  printf("  total:     %.2f GiB\n", total_gib);
  printf("  time:      %.3f s\n", sec);
  printf("  bw:        %.2f GiB/s\n", gib_s);

  CHECK_CUDA(cudaStreamDestroy(stream));

  CHECK_CUDA(cudaSetDevice(opt.src_gpu));
  CHECK_CUDA(cudaFree(src));

  CHECK_CUDA(cudaSetDevice(opt.dst_gpu));
  CHECK_CUDA(cudaFree(dst));

  return 0;
}
