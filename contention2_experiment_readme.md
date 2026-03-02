# Contention 2 实验 README（可用于论文）

本文档说明如何利用以下组件完成可发表级别的 Contention 2 实验（PCIe 方向争用）：

- `src/tools/rdma_msgsize_sweep.py`
- `src/cpu_client.cc`
- `src/gpu_server.cu`
- `src/gpu_be_pcie_memcpy_task.cu`
- `src/tools/gpu_metrics_logger.py`

核心研究问题：

1. RDMA WRITE 与 H2D 是否存在强方向性争用。
2. RDMA READ 与 D2H 是否存在强方向性争用。
3. 对照组合（WRITE+D2H、READ+H2D）是否显著更弱。

---

## 1. 方向对应假设（论文主线）

方向匹配关系：

- `RDMA WRITE`（NIC -> GPU）与 `H2D`（CPU -> GPU）都占用“入 GPU”方向。
- `RDMA READ`（GPU -> NIC）与 `D2H`（GPU -> CPU）都占用“出 GPU”方向。

---

## 2. 组件分工

1. `gpu_server`（GPU 机器）
   - 暴露 GPU MR，提供 RDMA 目标地址与 `rkey`。

2. `cpu_client`（CPU 机器）
   - 发起 RDMA 压测，输出吞吐与分位延迟。
   - `write` 时可启用 `write_ack=1`（更严格的可见性语义）。

3. `gpu_pcie_memcpy`（GPU 机器）
   - 后台干扰源，只做单向 `h2d` 或 `d2h` 压力。

4. `gpu_metrics_logger.py`（GPU 机器）
   - 持续记录 `pcie_tx/rx_GiB_s` 与 `pcie_tx/rx_util_pct`，确认干扰强度。

5. `rdma_msgsize_sweep.py`（CPU 机器）
   - 扫消息大小，并批量导出 CSV。

---

## 3. 2x2 实验

每个前台操作都配两个后台方向，形成 4 组：

1. `WRITE + H2D`（方向匹配，预期强争用）
2. `WRITE + D2H`（方向不匹配，对照）
3. `READ + D2H`（方向匹配，预期强争用）
4. `READ + H2D`（方向不匹配，对照）

并且每个前台操作都要有 baseline（无后台）：

- `h2d + none`
- `d2h + none`

---

## 4. 前置条件与编译

前置条件：

- 两台机器网络/RDMA 连通。
- GPU 机器 CUDA + GPUDirect RDMA 环境已就绪。
- 两端参数一致：`ib_dev/port/gid_idx/mtu`。

编译：

```bash
cd src
make
```

结果目录：

```bash
mkdir -p results/contention2
```

---

## 5. 参数固定建议（先固定再扫）

建议先固定一组“主实验参数”作为 paper 主结果：

- `iters=100000`
- `qd=64`
- `span=1G`
- `pattern=random`
- `align=256`
- `mtu=1024`
- `write_ack=1`（仅 `op=write` 时建议开启）

消息大小流程建议：

1. 先用 `rdma_msgsize_sweep.py` 扫一轮 `msg_sizes`。
2. 选一个对 tail 最敏感的 `msg_size`（例如 8K/16K/64K）作为主实验固定值。
3. 论文图表主文放固定值结果，附录放全 sweep。

---

## 6. 干扰强度分层（Low/Med/High）

为了更有说服力，建议每个组合做 3 档干扰强度：

- Low：active util 30%~50%
- Med：active util 60%~80%
- High：active util 85%~95%

其中 `active util` 定义为：

- `max(pcie_tx_util_pct, pcie_rx_util_pct)`

通过调 `gpu_pcie_memcpy` 参数实现分层：

- `chunk_mb`：64/128/256
- `streams`：8/12/16
- `batch`：2/4/8
- `inflight`：4/8/16

---

## 7. 运行顺序（每个实验组都相同）

环境参数示例（按你的机器替换）：

- `GPU_IP=10.10.10.11`
- `TCP_PORT=33333`
- `IB_DEV=mlx5_0`
- `PORT=1`
- `GID_IDX=3`
- `MTU=1024`

### Step A：GPU 机器启动前台服务（终端 1）

```bash
cd src
./gpu_server mlx5_0 1G 33333 1 3 1024
```

### Step B：GPU 机器启动后台干扰（终端 2）

`D2H` 示例：

```bash
cd src
./gpu_pcie_memcpy --dir=d2h --seconds=180 \
  --chunk_mb=128 --streams=8 --batch=8 --inflight=8 \
  --pinned=1 --report_ms=1000 \
  | tee ../results/contention2/pcie_d2h_high.log
```

`H2D` 示例：

```bash
cd src
./gpu_pcie_memcpy --dir=h2d --seconds=180 \
  --chunk_mb=128 --streams=8 --batch=8 --inflight=8 \
  --pinned=1 --report_ms=1000 \
  | tee ../results/contention2/pcie_h2d_high.log
```

### Step C：GPU 机器并行启动指标 logger（终端 3）

```bash
cd src
uv run python tools/gpu_metrics_logger.py \
  --gpu 0 --interval_ms 200 \
  --out ../results/contention2/gpu_metrics_caseX.csv
```

### Step D：CPU 机器跑 RDMA（终端 4）

`READ` sweep 示例：

```bash
cd src
python tools/rdma_msgsize_sweep.py \
  --cpu_client ./cpu_client \
  --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 \
  --iters 100000 --op read --port 1 --gid_idx 3 \
  --qd 64 --span 1G --pattern random --align 256 --mtu 1024 \
  --msg_sizes 4096,8192,16384,32768,65536 \
  --sample 1 --max_samples 0 \
  --out ../results/contention2/read_caseX.csv
```

`WRITE` sweep 示例（带可见性 ACK）：

```bash
cd src
python tools/rdma_msgsize_sweep.py \
  --cpu_client ./cpu_client \
  --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 \
  --iters 100000 --op write --port 1 --gid_idx 3 \
  --qd 64 --span 1G --pattern random --align 256 --mtu 1024 \
  --msg_sizes 4096,8192,16384,32768,65536 \
  --sample 1 --max_samples 0 --write_ack 1 \
  --out ../results/contention2/write_caseX.csv
```

停止顺序：

1. 停 `rdma_msgsize_sweep.py`（或 `cpu_client`）
2. 停 `gpu_pcie_memcpy`
3. 停 `gpu_metrics_logger.py`

---

## 8. baseline 与对照命名建议

建议每次结果文件名带上 4 个维度：

- 前台：`read|write`
- 后台：`none|h2d|d2h`
- 强度：`low|med|high`
- 重复编号：`r1..rN`

示例：

- `read_none_r1.csv`
- `read_d2h_high_r1.csv`
- `write_h2d_med_r3.csv`
- `gpu_metrics_write_h2d_high_r3.csv`

---

## 9. 统计口径（论文建议）

每个条件至少重复 `N=5` 次，报告：

- `P50/P99/P999` 的均值与标准差
- `Throughput_GiB_per_s` 的均值与标准差
- 可选：95% CI

关键归一化指标：

- `TailRatio = P99_contended / P99_baseline`
- `TailRatio999 = P999_contended / P999_baseline`
- `ThrRatio = Throughput_contended / Throughput_baseline`

要证明“方向对应”成立，至少要看到：

- `TailRatio(WRITE+H2D) > TailRatio(WRITE+D2H)`
- `TailRatio(READ+D2H) > TailRatio(READ+H2D)`

---

## 10. 图表建议（paper 直接可用）

主文建议 3 张图 + 1 张表：

1. 条形图：4 组合的 `TailRatio`（按 low/med/high 分组）
2. 条形图：4 组合的 `ThrRatio`
3. 时间对齐图：`pcie util` 与 `p99` 的时间序列（选 1 组代表）
4. 表格：6 个条件（含 2 个 baseline）的 `P50/P99/P999/Throughput`

---

## 11. 常见问题

1. `cpu_client` 连接失败
   - 检查 `gpu_server` 是否监听对应 `tcp_port`，IP 是否正确。

2. `WC error` 或 `RNR`
   - 先确认 `ib_dev/port/gid_idx/mtu` 两端一致。

3. `nvcc` 与系统编译器版本冲突
   - 参考本仓库已有做法，指定 `-ccbin /usr/bin/g++-11`。

4. 利用率上不去
   - 先确认 logger 中 active util，再按 `chunk/streams/batch/inflight` 四维调参。
   - 单向 `92%~93%` 通常已接近可达上限，不必强求 100%。

