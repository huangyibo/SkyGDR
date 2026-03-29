# Contention 2 实验方案 v2（单次执行，双部分）

这份文档是 `contention2` 当前的执行手册。

配套关系如下：

- 结果摘要看 `contention2_paper_v2_report.md`
- A1 消融看 `contention2_A1_host_vs_gdr_guide.md`
- 旧的 `contention2_fourfig_experiment_report.md` 只保留为早期 `low` 档归档，不再作为当前主结果

补充说明：

- 本仓库快照里没有同时带上 `results/` 和 `paper/` 目录
- 文中这些路径仍然代表完整实验环境中的标准输出位置

本版按你的新要求重构：

1. **Part A**：固定大 RDMA 消息（`msg_size=10485760`）评估其对 H2D / D2H 的影响。  
   并把 `gpu_pcie_memcpy` 的 `--report_ms` 从 `100` 调到 **`10000`（10s）** 来看稳定带宽。
2. **Part B**：后台长期挂 H2D / D2H，扫 RDMA `msg_size` 从小到大，观察 RDMA 随 size 的变化。
3. **所有实验只跑一次**（`repeat=1`）。

---

## [Hotfix] 2026-03-04：Read 路径重跑计划（仅重跑 Read）

背景：

- 旧版 `cpu_client` 的 read 压测口径过浅（单 QP / 低在途），导致 read 结果偏低。
- write 路径数据保持有效，不需要重跑。

本次只重跑以下 3 项：

- Part A: `A4_read.csv`（`d2h_with_read`）
- Part B: `B2_read_none_sweep.csv`
- Part B: `B4_read_on_d2h_sweep.csv`

保留不变（不重跑）：

- `A2_write.csv`
- `B1_write_none_sweep.csv`
- `B3_write_on_h2d_sweep.csv`
- `A1/A3` baseline 背景 memcpy 数据

Read 统一参数（新口径）：

- `READ_QPS=8`
- `READ_QD=16`（每个 QP 的深度）
- `READ_MIN_QD=16`
- `READ_RD_ATOMIC=16`
- 总在途约为 `qps * qd = 128`

### H0. 先备份旧的 read 结果

在仓库根目录执行：

```bash
cd /home/enine/danyang/SkyGDR
bk=results/contention2_paper_v2/backup_read_before_fix_20260304
mkdir -p "$bk"/{raw,ts,metrics,summary,fig}

cp -a results/contention2_paper_v2/raw/A4_read.csv "$bk/raw/" 2>/dev/null || true
cp -a results/contention2_paper_v2/raw/B2_read_none_sweep.csv "$bk/raw/" 2>/dev/null || true
cp -a results/contention2_paper_v2/raw/B4_read_on_d2h_sweep.csv "$bk/raw/" 2>/dev/null || true
cp -a results/contention2_paper_v2/ts/A4_read_ts.csv "$bk/ts/" 2>/dev/null || true
cp -a results/contention2_paper_v2/metrics/bg_A4_d2h_with_read.log "$bk/metrics/" 2>/dev/null || true
cp -a results/contention2_paper_v2/metrics/gpu_A4_d2h_with_read.csv "$bk/metrics/" 2>/dev/null || true
cp -a results/contention2_paper_v2/metrics/bg_B4_d2h.log "$bk/metrics/" 2>/dev/null || true
cp -a results/contention2_paper_v2/metrics/gpu_B4_d2h.csv "$bk/metrics/" 2>/dev/null || true
cp -a results/contention2_paper_v2/summary/partA_rdma_metrics.csv "$bk/summary/" 2>/dev/null || true
cp -a results/contention2_paper_v2/summary/partB_size_impact.csv "$bk/summary/" 2>/dev/null || true
cp -a results/contention2_paper_v2/fig/partB_read_sweep_throughput.png "$bk/fig/" 2>/dev/null || true
cp -a results/contention2_paper_v2/fig/partB_fourcases_throughput.png "$bk/fig/" 2>/dev/null || true
cp -a results/contention2_paper_v2/fig/partB_ratio_vs_size.png "$bk/fig/" 2>/dev/null || true
```

### H1. 编译新版二进制

两台机器都执行：

```bash
cd /home/enine/danyang/SkyGDR/src
make
```

说明：必须使用包含 `--qps` 支持的新版 `cpu_client` / `rdma_msgsize_sweep.py`。

### H2. 重跑顺序（推荐）

1. 先跑 `A4`（Part A 的 read case）  
2. 再跑 `B2`（read 无后台）  
3. 最后跑 `B4`（read + D2H 后台）  
4. 跑分析脚本刷新 `summary/fig`

---

## 0. 前提与机器角色

- 无 SSH 自动化，全部手动双机执行。
- `gpu_pcie_memcpy` 只能在 GPU 机器运行。
- GPU 机器：
  - 终端 G1：`gpu_server`（常驻）
  - 终端 G2：`gpu_pcie_memcpy`（后台）
  - 终端 G3：`gpu_metrics_logger.py`（观测）
- CPU 机器：
  - 终端 C1：`rdma_msgsize_sweep.py`

---

## 1. 固定参数与目录

统一参数：

- `SERVER_IP=10.10.10.11`
- `TCP_PORT=33333`
- `IB_DEV=mlx5_0`
- `PORT=1`
- `GID_IDX=3`
- `MTU=1024`
- `ITERS=100000`
- `QPS=8`（write/read 统一）
- `QD=16`（write/read 统一）
- `READ_MIN_QD=16`（统一传参）
- `RD_ATOMIC=16`（统一传参）
- `WRITE_ACK=0`（write/read 对齐并发口径）
- `SPAN=1G`
- `PATTERN=random`
- `ALIGN=256`
- `WARMUP_ITERS=20000`
- `WARMUP_RUNS=1`
- `MSG_LARGE=10485760`
- `MSG_SWEEP=见 Part B 分桶（256B 到 100MB）`

路径：

- 根目录：`/home/enine/danyang/SkyGDR`
- 结果目录：`/home/enine/danyang/SkyGDR/results/contention2_paper_v2`

创建目录（两台机器都执行）：

```bash
mkdir -p /home/enine/danyang/SkyGDR/results/contention2_paper_v2/{raw,ts,metrics,fig,summary}
```

编译（两台机器都执行）：

```bash
cd /home/enine/danyang/SkyGDR/src
make
```

---

## 2. 常驻服务（GPU 机器）

G1（全程保持）：

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_server mlx5_0 1G 33333 1 3 1024
```

---

## 3. Part A：大流量 RDMA 对 H2D/D2H 的影响

### 3.1 设计

只做 4 个 case（都只跑 1 次）：

- A1: `h2d_baseline`（只有后台 H2D）
- A2: `h2d_with_write`（后台 H2D + 前台 RDMA write, `msg=10MB`）
- A3: `d2h_baseline`（只有后台 D2H）
- A4: `d2h_with_read`（后台 D2H + 前台 RDMA read, `msg=10MB`）

后台 memcpy 参数统一：

- `--chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000`

说明：

- baseline case：GPU 侧采样 120s（C1 可执行 `sleep 120`）。
- contended case：先起 G2/G3，再在 C1 跑 RDMA；RDMA结束后再多采 20s 再停 G2/G3。

### 3.2 A1：`h2d_baseline`

G2:

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=h2d --seconds=600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_paper_v2/metrics/bg_A1_h2d_baseline.log
```

G3:

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_paper_v2/metrics/gpu_A1_h2d_baseline.csv
```

C1（仅计时）：

```bash
sleep 120
```

然后手动在 G2/G3 `Ctrl+C` 停止。

### 3.3 A2：`h2d_with_write`

G2:

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=h2d --seconds=600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_paper_v2/metrics/bg_A2_h2d_with_write.log
```

G3:

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_paper_v2/metrics/gpu_A2_h2d_with_write.csv
```

C1:

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py \
  --cpu_client ./cpu_client \
  --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 \
  --iters 100000 --op write --port 1 --gid_idx 3 \
  --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 \
  --msg_sizes 10485760 --warmup_iters 20000 --warmup_runs 1 \
  --sample 1 --max_samples 0 --write_ack 0 --read_min_qd 16 --rd_atomic 16 \
  --ts_ms 200 --ts_out ../results/contention2_paper_v2/ts/A2_write_ts.csv \
  --out ../results/contention2_paper_v2/raw/A2_write.csv
```

RDMA 结束后 `sleep 20`，再手动停 G2/G3。

### 3.4 A3：`d2h_baseline`

G2:

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=d2h --seconds=600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_paper_v2/metrics/bg_A3_d2h_baseline.log
```

G3:

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_paper_v2/metrics/gpu_A3_d2h_baseline.csv
```

C1（仅计时）：

```bash
sleep 120
```

然后手动在 G2/G3 `Ctrl+C` 停止。

### 3.5 A4：`d2h_with_read`

G2:

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=d2h --seconds=600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_paper_v2/metrics/bg_A4_d2h_with_read.log
```

G3:

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_paper_v2/metrics/gpu_A4_d2h_with_read.csv
```

C1:

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py \
  --cpu_client ./cpu_client \
  --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 \
  --iters 100000 --op read --port 1 --gid_idx 3 \
  --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 \
  --msg_sizes 10485760 --warmup_iters 20000 --warmup_runs 1 \
  --sample 1 --max_samples 0 --read_min_qd 16 --rd_atomic 16 \
  --ts_ms 200 --ts_out ../results/contention2_paper_v2/ts/A4_read_ts.csv \
  --out ../results/contention2_paper_v2/raw/A4_read.csv
```

RDMA 结束后 `sleep 20`，再手动停 G2/G3。

### 3.6 Part A 汇总（算 H2D/D2H 降幅）

在 CPU 机器运行：

```bash
python3 - <<'PY'
import re,statistics
def mean_bw(path):
    vals=[]
    for line in open(path):
        m=re.search(r'bw_gib_s=([0-9.]+)', line)
        if m:
            vals.append(float(m.group(1)))
    vals=vals[1:] if len(vals)>1 else vals
    return statistics.mean(vals) if vals else float('nan')

h2d_base=mean_bw('results/contention2_paper_v2/metrics/bg_A1_h2d_baseline.log')
h2d_cont=mean_bw('results/contention2_paper_v2/metrics/bg_A2_h2d_with_write.log')
d2h_base=mean_bw('results/contention2_paper_v2/metrics/bg_A3_d2h_baseline.log')
d2h_cont=mean_bw('results/contention2_paper_v2/metrics/bg_A4_d2h_with_read.log')

print('H2D baseline mean GiB/s:', h2d_base)
print('H2D contended mean GiB/s:', h2d_cont)
print('H2D degradation %:', (1-h2d_cont/h2d_base)*100 if h2d_base>0 else float('nan'))
print('D2H baseline mean GiB/s:', d2h_base)
print('D2H contended mean GiB/s:', d2h_cont)
print('D2H degradation %:', (1-d2h_cont/d2h_base)*100 if d2h_base>0 else float('nan'))
PY
```

---

## 4. Part B：重做吞吐 sweep（不再关注 latency）

### 4.1 设计（256B 到 100MB）

Part B 仍是 4 个 case（各跑 1 次）：

- B1: `write_none_sweep`（无后台）
- B2: `read_none_sweep`（无后台）
- B3: `write_on_h2d_sweep`（后台 H2D 持续）
- B4: `read_on_d2h_sweep`（后台 D2H 持续）

本轮只看 throughput，不做 latency 分析。  
消息大小覆盖：`256B -> 100MB`，按 4 个分桶跑（避免大包用 100000 iters 导致总时长失控）：

- `S1=256,512,1024,2048,4096,8192,16384`
- `S2=32768,65536,131072,262144,524288,1048576,2097152`
- `S3=4194304,8388608,16777216,33554432`
- `S4=67108864,100000000`

对应迭代数（四个 case 一致）：

- `S1: iters=1000000, warmup_iters=200000`
- `S2: iters=300000, warmup_iters=50000`
- `S3: iters=60000, warmup_iters=10000`
- `S4: iters=6000, warmup_iters=1000`

write/read 参数统一口径：

- `--qps 8 --qd 16 --read_min_qd 16 --rd_atomic 16`
- write 额外要求：`--write_ack 0`（否则会串行 ACK，和 read 口径不一致）

后台持续设置（B3/B4）：

- `--seconds=3600`（保证 sweep 全程不断）
- `--report_ms=10000`

### 4.2 预设变量与清理旧文件（C1）

```bash
cd /home/enine/danyang/SkyGDR/src

S1=256,512,1024,2048,4096,8192,16384
S2=32768,65536,131072,262144,524288,1048576,2097152
S3=4194304,8388608,16777216,33554432
S4=67108864,100000000

# Throughput-only: 保留极低成本采样，避免 latency 处理开销。
LAT_SAMPLE=1000000000
LAT_MAX=1

rm -f ../results/contention2_paper_v2/raw/B1_write_none_sweep.csv
rm -f ../results/contention2_paper_v2/raw/B2_read_none_sweep.csv
rm -f ../results/contention2_paper_v2/raw/B3_write_on_h2d_sweep.csv
rm -f ../results/contention2_paper_v2/raw/B4_read_on_d2h_sweep.csv
```

### 4.3 B1：`write_none_sweep`（分桶 4 次）

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 1000000 --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S1 --warmup_iters 200000 --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B1_write_none_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 300000  --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S2 --warmup_iters 50000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B1_write_none_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 60000   --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S3 --warmup_iters 10000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B1_write_none_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 6000    --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S4 --warmup_iters 1000   --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B1_write_none_sweep.csv
```

### 4.4 B2：`read_none_sweep`（分桶 4 次）

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 1000000 --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S1 --warmup_iters 200000 --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B2_read_none_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 300000  --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S2 --warmup_iters 50000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B2_read_none_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 60000   --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S3 --warmup_iters 10000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B2_read_none_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 6000    --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S4 --warmup_iters 1000   --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B2_read_none_sweep.csv
```

### 4.5 B3：`write_on_h2d_sweep`

G2:

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=h2d --seconds=3600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_paper_v2/metrics/bg_B3_h2d.log
```

G3:

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_paper_v2/metrics/gpu_B3_h2d.csv
```

C1（分桶 4 次）：

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 1000000 --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S1 --warmup_iters 200000 --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B3_write_on_h2d_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 300000  --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S2 --warmup_iters 50000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B3_write_on_h2d_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 60000   --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S3 --warmup_iters 10000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B3_write_on_h2d_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 6000    --op write --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S4 --warmup_iters 1000   --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --write_ack 0 --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B3_write_on_h2d_sweep.csv
```

RDMA sweep 完成后再手动停 G2/G3。

### 4.6 B4：`read_on_d2h_sweep`

G2:

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=d2h --seconds=3600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_paper_v2/metrics/bg_B4_d2h.log
```

G3:

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_paper_v2/metrics/gpu_B4_d2h.csv
```

C1（分桶 4 次）：

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 1000000 --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S1 --warmup_iters 200000 --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B4_read_on_d2h_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 300000  --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S2 --warmup_iters 50000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B4_read_on_d2h_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 60000   --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S3 --warmup_iters 10000  --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B4_read_on_d2h_sweep.csv
python3 tools/rdma_msgsize_sweep.py --cpu_client ./cpu_client --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 --iters 6000    --op read --port 1 --gid_idx 3 --qd 16 --qps 8 --span 1G --pattern random --align 256 --mtu 1024 --msg_sizes $S4 --warmup_iters 1000   --warmup_runs 1 --sample $LAT_SAMPLE --max_samples $LAT_MAX --read_min_qd 16 --rd_atomic 16 --out ../results/contention2_paper_v2/raw/B4_read_on_d2h_sweep.csv
```

RDMA sweep 完成后再手动停 G2/G3。

### 4.7 Part B 汇总（throughput-only）

在 CPU 机器执行，生成新的 `summary/partB_size_impact.csv`：

```bash
python3 - <<'PY'
import csv

def load(path):
    d = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('RetCode') and int(float(r['RetCode'])) != 0:
                continue
            m = int(float(r['MsgBytes']))
            d[m] = float(r['Throughput_GiB_per_s'])
    return d

w0 = load('results/contention2_paper_v2/raw/B1_write_none_sweep.csv')
r0 = load('results/contention2_paper_v2/raw/B2_read_none_sweep.csv')
w1 = load('results/contention2_paper_v2/raw/B3_write_on_h2d_sweep.csv')
r1 = load('results/contention2_paper_v2/raw/B4_read_on_d2h_sweep.csv')

out = 'results/contention2_paper_v2/summary/partB_size_impact.csv'
all_sizes = sorted(set(w0) | set(w1) | set(r0) | set(r1))
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'msg_bytes',
        'write_none_thr', 'write_h2d_thr', 'write_thr_ratio',
        'read_none_thr', 'read_d2h_thr', 'read_thr_ratio'
    ])
    for m in all_sizes:
        wn = w0.get(m, float('nan'))
        wh = w1.get(m, float('nan'))
        rn = r0.get(m, float('nan'))
        rd = r1.get(m, float('nan'))
        wtr = (wh / wn) if wn == wn and wn > 0 and wh == wh else ''
        rtr = (rd / rn) if rn == rn and rn > 0 and rd == rd else ''
        w.writerow([
            m,
            '' if wn != wn else wn,
            '' if wh != wh else wh,
            wtr,
            '' if rn != rn else rn,
            '' if rd != rd else rd,
            rtr
        ])

print('saved:', out)
PY
```

重画图表：

```bash
cd /home/enine/danyang/SkyGDR
python3 src/tools/contention2_v2_analyze.py --base_dir results/contention2_paper_v2
```

---

## 5. 最终产物清单

- Part A：
  - `metrics/bg_A1_h2d_baseline.log`
  - `metrics/bg_A2_h2d_with_write.log`
  - `metrics/bg_A3_d2h_baseline.log`
  - `metrics/bg_A4_d2h_with_read.log`
- Part B：
  - `raw/B1_write_none_sweep.csv`
  - `raw/B2_read_none_sweep.csv`
  - `raw/B3_write_on_h2d_sweep.csv`
  - `raw/B4_read_on_d2h_sweep.csv`
  - `summary/partB_size_impact.csv`

---

## 6. 关键注意事项

- `--report_ms=10000`（10s）是本方案核心，不要用 `100`。
- `gpu_pcie_memcpy` 参数是双横线：`--chunk_mb`，不要写成 `-chunk_mb`。
- Part B 后台一定要长时运行（`--seconds=3600`）直到 RDMA sweep 完成。
- 本方案所有 case 都是 **1 次执行**，不做重复统计。
- write/read sweep 都必须带：`--qps 8 --qd 16 --read_min_qd 16 --rd_atomic 16`。
- write 必须是：`--write_ack 0`，否则不会与 read 保持同等并发口径。
- Part B 已改为 throughput-only，不再做 latency 图表分析。
- 完成 4 个 sweep 后，执行分析脚本刷新图表与汇总：

```bash
cd /home/enine/danyang/SkyGDR
python3 src/tools/contention2_v2_analyze.py --base_dir results/contention2_paper_v2
```
