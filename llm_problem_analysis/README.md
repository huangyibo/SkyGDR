# NVLink + Decode HBM/KV-Cache Contention Benchmarks

This repository contains four CUDA microbenchmarks for studying GPU-memory
contention in LLM serving, especially prefill/decode disaggregation.

The target scenario is:

```text
P GPU / prefill GPU
    |
    | KV-cache transfer over NVLink / NVSwitch
    v
D GPU / decode GPU
    +
D GPU concurrently runs decode kernels that read existing KV cache from HBM
```

The key question is:

```text
Does P->D KV-cache transfer interfere with decode-side HBM bandwidth?
```

The four programs are:

```text
1. kv_nvlink_transfer.cu
   Fixed-size GPU-to-GPU KV transfer emulator.

2. decode_hbm_stress.cu
   Simple synthetic decode-side HBM bandwidth stressor.

3. kv_nvlink_transfer_trace.cu
   Trace-shaped GPU-to-GPU KV transfer emulator.

4. decode_transformer_kvcache_stress.cu
   Transformer-decoder-style KV-cache HBM stressor.
```

---

## 1. Prerequisites

### CUDA

Check that CUDA is available:

```bash
which nvcc
nvcc --version
```

If `nvcc` is not in `PATH`, try:

```bash
ls -l /usr/local/cuda*
find /usr/local -name nvcc 2>/dev/null
```

Then build with an explicit compiler path:

```bash
make NVCC=/usr/local/cuda/bin/nvcc
```

### GPU topology

Check visible GPUs:

```bash
nvidia-smi -L
```

Check topology:

```bash
nvidia-smi topo -m
```

A GPU pair connected by NVLink/NVSwitch should show `NV#`, for example:

```text
        GPU0    GPU1
GPU0     X      NV18
GPU1    NV18     X
```

For these benchmarks, choose one GPU as the prefill/source GPU and another as
the decode/destination GPU.

Example:

```text
P GPU = GPU0
D GPU = GPU1
```

If the topology shows `PIX`, `PXB`, `PHB`, `NODE`, or `SYS` instead of `NV#`,
then that pair is not directly connected by NVLink.

---

## 2. Build

Build all programs:

```bash
make
```

Build for A100:

```bash
make CUDA_ARCH=sm_80
```

Build for H100/H200:

```bash
make CUDA_ARCH=sm_90
```

Use a specific CUDA compiler:

```bash
make NVCC=/usr/local/cuda/bin/nvcc
```

Clean generated files:

```bash
make clean
```

---

## 3. Program: `kv_nvlink_transfer`

### Source file

```text
kv_nvlink_transfer.cu
```

### Purpose

`kv_nvlink_transfer` emulates fixed-size P->D KV-cache movement between two
GPUs using `cudaMemcpyPeerAsync()`.

It is useful for measuring raw GPU-to-GPU transfer bandwidth between two GPUs,
especially when they are connected by NVLink/NVSwitch.

This program always transfers the same number of MiB per iteration.

### Command format

```bash
./kv_nvlink_transfer <src_gpu> <dst_gpu> <size_mib> <iters>
```

Arguments:

```text
src_gpu    Source / prefill GPU id
dst_gpu    Destination / decode GPU id
size_mib   Transfer size per iteration, in MiB
iters      Number of transfer iterations
```

### Example

```bash
./kv_nvlink_transfer 0 1 1024 10000
```

This means:

```text
source GPU      = GPU0
destination GPU = GPU1
transfer size   = 1024 MiB per copy
iterations      = 10000
```

### Expected output

Example:

```text
src_gpu=0 dst_gpu=1 size=1024 MiB iters=10000
peer_access src->dst=1 dst->src=1
KV transfer done: 10000.00 GiB in 60.123 s = 166.32 GiB/s
```

Important fields:

```text
peer_access src->dst=1
```

means CUDA peer access is available between the two GPUs.

```text
166.32 GiB/s
```

is the measured P->D transfer bandwidth.

### Suggested sweep

```bash
./kv_nvlink_transfer 0 1 64   10000
./kv_nvlink_transfer 0 1 256  10000
./kv_nvlink_transfer 0 1 1024 10000
./kv_nvlink_transfer 0 1 4096 1000
```

---

## 4. Program: `decode_hbm_stress`

### Source file

```text
decode_hbm_stress.cu
```

### Purpose

`decode_hbm_stress` is a simple HBM bandwidth stressor.

It allocates large GPU buffers and repeatedly streams through them. This
creates heavy HBM read/write traffic on the decode GPU.

This is a simple synthetic stress test. It is not transformer-aware and does
not explicitly model KV-cache attention.

### Command format

```bash
./decode_hbm_stress <gpu> <working_set_mib> <iters>
```

Arguments:

```text
gpu              GPU id to run on
working_set_mib  Total working-set size in MiB
iters            Number of kernel iterations
```

### Example

```bash
./decode_hbm_stress 1 32768 100000
```

This means:

```text
GPU             = GPU1
working set     = 32768 MiB = 32 GiB
iterations      = 100000
```

### Expected output

Example:

```text
gpu=1 name=NVIDIA A100-SXM4-80GB working_set=32768 MiB iters=100000
launch blocks=864 threads=256
HBM stress done: approx 4800000.00 GiB in 2600.000 s = 1846.15 GiB/s
checksum=123456.000000
```

Important field:

```text
HBM stress done: ... = 1846.15 GiB/s
```

This is an estimated HBM bandwidth based on the program's memory-access model.

### Suggested sweep

```bash
./decode_hbm_stress 1 4096  100000
./decode_hbm_stress 1 8192  100000
./decode_hbm_stress 1 16384 100000
./decode_hbm_stress 1 32768 100000
```

---

## 5. Program: `kv_nvlink_transfer_trace`

### Source file

```text
kv_nvlink_transfer_trace.cu
```

### Purpose

`kv_nvlink_transfer_trace` emulates P->D KV-cache movement with variable
transfer sizes.

Unlike `kv_nvlink_transfer`, this program does not use a fixed copy size.
Instead, it samples token counts from a trace-inspired long-context LLM
workload distribution and converts tokens into KV-cache bytes.

This is useful for studying realistic heavy-tailed KV movement.

### KV-size model

The default model geometry is:

```text
layers     = 80
kv_heads   = 8
head_dim   = 128
elem_bytes = 2
```

The KV size per token is:

```text
layers * kv_heads * head_dim * 2(K,V) * elem_bytes
```

With the default values:

```text
80 * 8 * 128 * 2 * 2 bytes
= 327,680 bytes/token
= 320 KiB/token
```

So the transfer size is:

```text
transfer_bytes = token_count * 320 KiB
```

### Modes

The program supports three distribution modes:

```text
input
cached
computed
```

#### `input`

Transfers KV corresponding to the total input context.

This is an upper-bound mode.

#### `cached`

Transfers KV corresponding to cached/reused prefix tokens.

This is often the most relevant mode for KV-cache reuse and P/D
disaggregation experiments.

#### `computed`

Transfers KV corresponding to newly computed / uncached prefill tokens.

This is usually smaller than `cached`, but the tail can still be large.

### Command format

```bash
./kv_nvlink_transfer_trace [options]
```

Options:

```text
--src N             Source / prefill GPU id. Default: 0
--dst N             Destination / decode GPU id. Default: 1
--iters N           Number of transfers. Default: 10000
--dist NAME         input | cached | computed. Default: cached
--layers N          Transformer layers. Default: 80
--kv-heads N        Number of KV heads. Default: 8
--head-dim N        KV head dimension. Default: 128
--elem-bytes N      KV element bytes. fp16/bf16=2. Default: 2
--max-tokens N      Override synthetic tail max tokens
--sync-every N      Stream sync interval. Default: 256
--seed N            RNG seed. Default: 1
--help              Print help
```

### Example: cached-prefix KV transfer

```bash
./kv_nvlink_transfer_trace \
  --src 0 \
  --dst 1 \
  --dist cached \
  --iters 1000
```

### Example: newly computed KV transfer

```bash
./kv_nvlink_transfer_trace \
  --src 0 \
  --dst 1 \
  --dist computed \
  --iters 10000
```

### Example: smaller model geometry

```bash
./kv_nvlink_transfer_trace \
  --src 0 \
  --dst 1 \
  --dist computed \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --iters 10000
```

### Expected output

Example:

```text
src_gpu=0 dst_gpu=1 iters=1000 dist=cached
peer_access src->dst=1 dst->src=1
KV geometry: layers=80 kv_heads=8 head_dim=128 elem_bytes=2
Trace quantiles: p50=60928 p90=112512 p99=162048 max=200000 tokens
sampled tokens: p50=60421 p90=111832 p99=160900 max=198000
sampled bytes: p50=19798753280 p90=36645068800 p99=52723712000 max=64880640000
max transfer allocation: 60.42 GiB
Trace-shaped KV transfer done:
  transfers: 1000
  total:     21000.00 GiB
  time:      130.000 s
  bw:        161.54 GiB/s
```

Important fields:

```text
sampled tokens
sampled bytes
max transfer allocation
bw
```

### Approximate default transfer sizes

With the default model geometry:

```text
80 layers, 8 KV heads, 128 head dim, 2-byte element
```

the transfer sizes are approximately:

```text
input mode:
  P50 ≈ 19.51 GiB
  P90 ≈ 35.06 GiB
  P99 ≈ 50.76 GiB

cached mode:
  P50 ≈ 18.59 GiB
  P90 ≈ 34.34 GiB
  P99 ≈ 49.45 GiB

computed mode:
  P50 ≈ 0.23 GiB
  P90 ≈ 2.67 GiB
  P99 ≈ 16.27 GiB
```

For another model, scale linearly:

```text
new_size =
  old_size *
  (layers / 80) *
  (kv_heads / 8) *
  (head_dim / 128) *
  (elem_bytes / 2)
```

---

## 6. Program: `decode_transformer_kvcache_stress`

### Source file

```text
decode_transformer_kvcache_stress.cu
```

### Purpose

`decode_transformer_kvcache_stress` is a transformer-decoder-style HBM stress
benchmark.

It allocates an existing KV cache on the decode GPU and repeatedly generates
tokens. For each generated token, it scans historical K/V cache and appends
new K/V entries.

This emulates the HBM access pattern of autoregressive decode more closely
than `decode_hbm_stress`.

### Emulated decode loop

The program approximates:

```text
for each generated token:
  for each transformer layer:
    read historical K cache
    read historical V cache
    compute attention-like reduction
    run FFN-like HBM pass
    append new K/V for the generated token
```

This benchmark is not a numerically faithful LLM implementation. It is a
resource-pressure emulator.

### Command format

```bash
./decode_transformer_kvcache_stress [options]
```

Options:

```text
--gpu N              GPU id. Default: 0
--layers N           Transformer layers. Default: 32
--kv-heads N         KV heads. Default: 8
--head-dim N         Head dimension. Default: 128
--context N          Initial context tokens. Default: 8192
--gen N              Number of generated tokens. Default: 256
--ffn-dim N          FFN working-set dimension. Default: 4096
--ffn-blocks N       FFN blocks. Default: 512
--threads N          CUDA threads per block. Default: 256
--report-every N     Progress interval. Default: 32
--help               Show help
```

### Example: moderate run

```bash
./decode_transformer_kvcache_stress \
  --gpu 1 \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --context 8192 \
  --gen 256
```

### Example: heavier HBM pressure

```bash
./decode_transformer_kvcache_stress \
  --gpu 1 \
  --layers 40 \
  --kv-heads 8 \
  --head-dim 128 \
  --context 16384 \
  --gen 256 \
  --ffn-dim 8192
```

### Example: large-model-like KV cache

```bash
./decode_transformer_kvcache_stress \
  --gpu 1 \
  --layers 80 \
  --kv-heads 8 \
  --head-dim 128 \
  --context 32768 \
  --gen 128
```

Be careful: this program currently stores KV cache in FP32 for simplicity.
That means its KV-cache allocation is about 2x larger than FP16/BF16 KV cache.

### KV-cache memory estimate

The program allocates both K and V:

```text
KV bytes =
  layers * (context + gen) * kv_heads * head_dim * 2(K,V) * 4 bytes
```

Example:

```text
layers   = 32
context  = 8192
gen      = 256
kv_heads = 8
head_dim = 128

KV bytes =
  32 * 8448 * 8 * 128 * 2 * 4
≈ 2.06 GiB
```

For:

```text
layers   = 80
context  = 32768
gen      = 128
kv_heads = 8
head_dim = 128
```

the allocation is roughly:

```text
≈ 20.1 GiB
```

### Expected output

Example:

```text
GPU: 1 NVIDIA A100-SXM4-80GB
layers=32 kv_heads=8 head_dim=128
context=8192 gen=256 max_tokens=8448
KV cache: 2.06 GiB total K+V, fp32
FFN working set: 0.50 MiB per array
attention blocks=256 threads=256 shared=1024 bytes
generated 32 / 256 tokens, current seq_len=8224
...
Done.
Time: 12.345 s
Generated tokens: 256
Throughput: 20.74 tokens/s
Approx KV read traffic: 194.00 GiB
Approx KV write traffic: 0.06 GiB
Approx FFN traffic: 0.25 GiB
Approx total traffic: 194.31 GiB
Approx bandwidth: 15.74 GiB/s
```

Important fields:

```text
Throughput
Approx KV read traffic
Approx total traffic
Approx bandwidth
```

### Suggested context-length sweep

```bash
./decode_transformer_kvcache_stress \
  --gpu 1 --layers 32 --kv-heads 8 --head-dim 128 \
  --context 4096 --gen 256

./decode_transformer_kvcache_stress \
  --gpu 1 --layers 32 --kv-heads 8 --head-dim 128 \
  --context 8192 --gen 256

./decode_transformer_kvcache_stress \
  --gpu 1 --layers 32 --kv-heads 8 --head-dim 128 \
  --context 16384 --gen 256

./decode_transformer_kvcache_stress \
  --gpu 1 --layers 32 --kv-heads 8 --head-dim 128 \
  --context 32768 --gen 128
```

---

## 7. Quick Start: Fixed-Size Transfer + Simple HBM Stress

Assume:

```text
P GPU = GPU0
D GPU = GPU1
```

### Step 1: Build

```bash
make CUDA_ARCH=sm_80
```

Use `sm_90` for H100/H200.

### Step 2: Run fixed-size KV transfer alone

```bash
./kv_nvlink_transfer 0 1 1024 10000 > kv_fixed_alone.log 2>&1
```

### Step 3: Run simple HBM stress alone

```bash
./decode_hbm_stress 1 32768 100000 > decode_hbm_alone.log 2>&1
```

### Step 4: Run both together

```bash
./decode_hbm_stress 1 32768 100000 > decode_hbm_overlap.log 2>&1 &
DECODE_PID=$!

./kv_nvlink_transfer 0 1 1024 10000 > kv_fixed_overlap.log 2>&1

wait $DECODE_PID
```

### Step 5: Compare

```bash
cat kv_fixed_alone.log
cat kv_fixed_overlap.log

cat decode_hbm_alone.log
cat decode_hbm_overlap.log
```

Look for:

```text
KV transfer bandwidth alone
KV transfer bandwidth during overlap
Decode HBM bandwidth alone
Decode HBM bandwidth during overlap
```

---

## 8. Quick Start: Trace-Shaped Transfer + Transformer-Style Decode

This is the more realistic experiment.

Assume:

```text
P GPU = GPU0
D GPU = GPU1
```

### Step 1: Run transformer-style decode alone

```bash
./decode_transformer_kvcache_stress \
  --gpu 1 \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --context 8192 \
  --gen 512 \
  > decode_kvcache_alone.log 2>&1
```

### Step 2: Run trace-shaped transfer alone

Use the same model geometry as the decode benchmark:

```bash
./kv_nvlink_transfer_trace \
  --src 0 \
  --dst 1 \
  --dist computed \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --iters 10000 \
  > kv_trace_alone.log 2>&1
```

### Step 3: Run both together

```bash
./decode_transformer_kvcache_stress \
  --gpu 1 \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --context 8192 \
  --gen 512 \
  > decode_kvcache_overlap.log 2>&1 &
DECODE_PID=$!

./kv_nvlink_transfer_trace \
  --src 0 \
  --dst 1 \
  --dist computed \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --iters 10000 \
  > kv_trace_overlap.log 2>&1

wait $DECODE_PID
```

### Step 4: Compare

```bash
cat decode_kvcache_alone.log
cat decode_kvcache_overlap.log

cat kv_trace_alone.log
cat kv_trace_overlap.log
```

Important metrics:

```text
decode tokens/s
decode approximate bandwidth
KV transfer bandwidth
total runtime
```

---

## 9. Recommended Experiments

### Experiment A: Does fixed-size KV transfer hurt decode?

```bash
for SIZE in 64 256 1024 4096; do
  ./decode_hbm_stress 1 32768 100000 > decode_hbm_${SIZE}m.log 2>&1 &
  PID=$!

  ./kv_nvlink_transfer 0 1 ${SIZE} 10000 > kv_fixed_${SIZE}m.log 2>&1

  wait $PID
done
```

### Experiment B: Which trace mode hurts decode most?

```bash
for DIST in input cached computed; do
  ./decode_transformer_kvcache_stress \
    --gpu 1 \
    --layers 32 \
    --kv-heads 8 \
    --head-dim 128 \
    --context 8192 \
    --gen 512 \
    > decode_${DIST}.log 2>&1 &
  PID=$!

  ./kv_nvlink_transfer_trace \
    --src 0 \
    --dst 1 \
    --dist ${DIST} \
    --layers 32 \
    --kv-heads 8 \
    --head-dim 128 \
    --iters 1000 \
    > kv_${DIST}.log 2>&1

  wait $PID
done
```

For `computed`, you may want more iterations:

```bash
./kv_nvlink_transfer_trace \
  --src 0 \
  --dst 1 \
  --dist computed \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --iters 10000
```

### Experiment C: Does longer context increase sensitivity?

```bash
for CTX in 4096 8192 16384 32768; do
  ./decode_transformer_kvcache_stress \
    --gpu 1 \
    --layers 32 \
    --kv-heads 8 \
    --head-dim 128 \
    --context ${CTX} \
    --gen 256 \
    > decode_ctx_${CTX}_alone.log 2>&1

  ./decode_transformer_kvcache_stress \
    --gpu 1 \
    --layers 32 \
    --kv-heads 8 \
    --head-dim 128 \
    --context ${CTX} \
    --gen 256 \
    > decode_ctx_${CTX}_overlap.log 2>&1 &
  PID=$!

  ./kv_nvlink_transfer_trace \
    --src 0 \
    --dst 1 \
    --dist computed \
    --layers 32 \
    --kv-heads 8 \
    --head-dim 128 \
    --iters 10000 \
    > kv_ctx_${CTX}_overlap.log 2>&1

  wait $PID
done
```

---

## 10. What to Report

For each experiment, report:

```text
1. KV transfer bandwidth alone
2. KV transfer bandwidth during decode overlap
3. Decode tokens/s alone
4. Decode tokens/s during KV-transfer overlap
5. Decode slowdown
6. KV-transfer bandwidth drop
```

Definitions:

```text
decode_slowdown =
  decode_time_overlap / decode_time_alone

decode_throughput_ratio =
  decode_tokens_per_second_overlap / decode_tokens_per_second_alone

kv_bandwidth_ratio =
  kv_bandwidth_overlap / kv_bandwidth_alone
```

A simple table format:

```text
Workload                 KV BW   Decode tok/s   Decode slowdown
----------------------   -----   ------------   ---------------
decode alone             N/A     100%           1.00x
KV transfer alone        100%    N/A            N/A
decode + fixed KV        82%     91%            1.10x
decode + trace computed  88%     94%            1.06x
decode + trace cached    70%     80%            1.25x
```

---

## 11. Monitoring

Basic GPU monitoring:

```bash
nvidia-smi dmon -s pucvmt
```

Topology:

```bash
nvidia-smi topo -m
```

Peer-to-peer capability:

```bash
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
nvidia-smi topo -p2p n
```

NVLink status:

```bash
nvidia-smi nvlink --status
```

CUDA profiling:

```bash
nsys profile -t cuda,nvtx -o overlap_test \
  ./decode_transformer_kvcache_stress \
    --gpu 1 \
    --context 8192 \
    --gen 128
```

Nsight Compute:

```bash
ncu --set full \
  ./decode_transformer_kvcache_stress \
    --gpu 1 \
    --context 8192 \
    --gen 32
```

---

## 12. Slurm Usage

On a Slurm cluster, first allocate an interactive GPU session.

Example:

```bash
srun -p <gpu_partition> --gres=gpu:2 --pty bash
```

or:

```bash
srun -p <gpu_partition> --gpus=2 --pty bash
```

Then check topology:

```bash
nvidia-smi topo -m
```

Build:

```bash
make CUDA_ARCH=sm_80
```

Run benchmarks inside the allocated session.

If your cluster requires `salloc`:

```bash
salloc -p <gpu_partition> --gres=gpu:2
srun --pty bash
```

---

## 13. Common Issues

### `nvcc: command not found`

Find CUDA:

```bash
ls -l /usr/local/cuda*
find /usr/local -name nvcc 2>/dev/null
```

Build with explicit path:

```bash
make NVCC=/usr/local/cuda/bin/nvcc
```

### CUDA architecture error

If compilation fails because of unsupported architecture, specify the correct
architecture manually.

A100:

```bash
make clean
make CUDA_ARCH=sm_80
```

H100/H200:

```bash
make clean
make CUDA_ARCH=sm_90
```

### Out of memory

Reduce one or more parameters:

```text
--layers
--context
--gen
--kv-heads
--head-dim
--ffn-dim
```

Example smaller run:

```bash
./decode_transformer_kvcache_stress \
  --gpu 1 \
  --layers 16 \
  --kv-heads 8 \
  --head-dim 128 \
  --context 4096 \
  --gen 128
```

For `kv_nvlink_transfer_trace`, use a smaller model geometry:

```bash
./kv_nvlink_transfer_trace \
  --src 0 \
  --dst 1 \
  --dist cached \
  --layers 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --iters 1000
```

### No NVLink between selected GPUs

Run:

```bash
nvidia-smi topo -m
```

Pick a pair with `NV#`.

For example:

```text
GPU0 <-> GPU1 = NV18
```

Then use:

```bash
./kv_nvlink_transfer 0 1 1024 10000
```

### Peer access is disabled

If output shows:

```text
peer_access src->dst=0
```

then CUDA peer access is not available between the selected GPUs. Choose
another GPU pair or check cluster/container GPU visibility.

### Program runs too long

Reduce:

```text
iters
gen
context
working_set_mib
```

Examples:

```bash
./kv_nvlink_transfer 0 1 1024 1000

./decode_hbm_stress 1 8192 10000

./decode_transformer_kvcache_stress \
  --gpu 1 \
  --context 4096 \
  --gen 64
```

---

## 14. Interpretation Guide

### If KV bandwidth drops during decode

This suggests decode-side HBM activity can interfere with incoming P->D KV
movement.

Possible reason:

```text
NVLink writes land in D GPU memory while decode kernels read/write HBM.
Both consume D GPU memory-system bandwidth.
```

### If decode tokens/s drops during KV transfer

This suggests P->D KV transfer interferes with decode execution.

Possible reason:

```text
Decode attention scans historical KV cache from HBM.
Concurrent KV transfer adds HBM write/read pressure.
```

### If neither drops much

Possible explanations:

```text
1. Decode benchmark is compute-bound, not HBM-bound.
2. KV transfer is not saturating NVLink or D-GPU HBM.
3. Copy engine and SM kernels overlap well on this GPU.
4. Context length is too short.
5. Transfer sizes are too small.
```

Try increasing:

```text
--context
--layers
--kv-heads
--head-dim
transfer size
iters
```

### If only KV transfer drops

The decode kernel may be consuming enough memory bandwidth to slow incoming
copies, while decode still has sufficient compute/memory slack.

### If only decode drops

The transfer may consume enough destination HBM bandwidth to slow decode, while
NVLink still has enough link-level bandwidth.

---

## 15. Limitations

These are microbenchmarks, not full LLM inference engines.

They do not fully model:

```text
vLLM PagedAttention
FlashAttention
CUDA graphs
tensor parallelism
pipeline parallelism
real model weights
quantization
scheduler behavior
request batching
prefix-cache block tables
RDMA across physical nodes
NIXL / Mooncake / Dynamo runtime details
```

They are intended to isolate and demonstrate the hardware-level contention
between:

```text
P->D KV-cache transfer
and
D-GPU decode-side HBM access
```

For inter-node P/D disaggregation, the transfer path is usually:

```text
GPU memory -> PCIe -> NIC -> network -> NIC -> PCIe -> GPU memory
```

rather than NVLink/NVSwitch. In that case, replace the NVLink transfer
benchmark with a GPUDirect RDMA benchmark.

---

## 16. Suggested Directory Layout

```text
.
├── Makefile
├── README.md
├── kv_nvlink_transfer.cu
├── decode_hbm_stress.cu
├── kv_nvlink_transfer_trace.cu
└── decode_transformer_kvcache_stress.cu
```

Build:

```bash
make
```

Run:

```bash
./kv_nvlink_transfer 0 1 1024 10000
./decode_hbm_stress 1 32768 100000
./kv_nvlink_transfer_trace --src 0 --dst 1 --dist cached --iters 1000
./decode_transformer_kvcache_stress --gpu 1 --context 8192 --gen 256
```
