# Motivation benchmarks for GPUDirect RDMA scheduling

We want to motivate the "nosiy neighbors" problem of colocating differnet GPU workloads together
with the new trend of cloud GPU workloads and GPU resource sharing.

Suppose we colocate a network-intensive GPU compute task and a GPU-memory-intensive GPU compute task
into the same GPU device, these two task compete for the GPU L2 cache and memory bandwidth without
careful scheduling, eventually resulting in non-neligible performance interference with each other.
For example, suppose the former task is latency-sensitive, its latency and throughput (RPS) will
be significantly impacted by the background GPU memory-intesive task. We intend to validate this
GPU "nosiy neighbor" problem with solid data evidence.


## Experiment setup

### Workloads setting

**1. RDMA-based latency-sensitive GPU task (RdmaNet):**

- RDMA server (GPU): expose RDMA-registered GPU device memory into client side via out-of-band TCP channel,
and then do nothing.

- RDMA client (CPU): After getting remote GPU memory, it will generate the requests of remote RDMA Write or
Read with specific message size, and summarize the latency distribution table after executing for a duration
or a given number of iterations. 
    
    - Message size (Bytes): 8, 128, 1024, 4096, 8192
    - Op type: RDMA Write or Read

- *Implementation:* `gpu_server.cu` for server, `cpu_client.cc` for client. Their usage is as follows:

    ```bash
    # Server
    $ ./gpu_server -h
    Usage: ./gpu_server <ib_dev> <msg_bytes> <tcp_port> <port> <gid_idx>

    # Client
    $ ./cpu_client -h
    Usage: ./cpu_client <server_ip> <tcp_port> <ib_dev> <iters> <bytes> <op=write|read> <port> <gid_idx>
    ```

**2. GPU-memory-intensive GPU task (MemHog):**

- GPU server: issue concurrent cuda kernels/threads that repeatly read and write GPU memory to generate
various GPU memory bandwidth pressures. We control the GPU BW pressures using the number of concurrent
GPU units and the data size operated. This task could saturate the GPU memory bandwidth.

    - Data size: 1MB, 128MB, 1GB, 8GB, 16GB
    - GPU block #:
    - GPU warp/kernel #: 

- *Implementation:*  (1) `gpu_be_memhog_minimal_task.cu`, (2) `gpu_be_memhog_task.cu`.

*Note: You can quickly build the above tasks using `make all` command.*



### Testbed setup

The minimal testbed consists of one CPU server and one GPU server.
Both machines are equipped with Mellanox RDMA NIC (RoCEv2), e.g., 100Gbps CX5 RNIC.
GPU server is equipped with at least one GPU device, e.g., A100 in our case.

**Libraries required for GPU server:** Requires CUDA 11+/12+, NVIDIA driver, and nvidia-peermem loaded on the GPU host.


## Running benchmark

The metrics focused include (1) Median latency (us), (2) P99 tail latency (us), (3) message rate,
and (4) stddev (Standard deviation) of RDMA communication.

**Baseline:**
1. Only run **RdmaNet** task as foreground GPU task. Record the latency distribution table and message rate (throughput). 


2. Run **RdmaNet** task as the foreground GPU task while running **MemHog** task as the background task, and record the 
latency distribution table and message rate (throughput) of **RdmaNet** task.

**Our solution (TODO):**
The solution could consist of (1) MPS based isolation and (2) MIG based isolation.

1. **MPS based**: Run **RdmaNet** and **MemHog** in different MPS units, respectively. 

2. **MIG based**: Run **RdmaNet** and **MemHog** in different MIG units, respectively. 

We could have a better solution than MPS- and MIG-based, e.g., hierachical GPU scheduling/isolation.


