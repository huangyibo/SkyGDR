# Contention 2 A1 实验指南：`Host-RDMA` vs `GPUDirect RDMA`（归因消融）

目标：把“共享瓶颈主要在 GPU-side PCIe 段”钉死。  
方法：固定同一套 RDMA 注入参数，只改变 **RDMA 目标内存类型**：

- `+GDR`：RDMA 目标是 GPU memory（现有 `gpu_server`）
- `+HostRDMA`：RDMA 目标是 CPU pinned memory（非 GDR，需 host-MR server）

最终图：一张 3 组柱状图（`H2D normalized BW`）：

- `baseline`（只有 H2D）
- `+GDR`
- `+HostRDMA`

---

## 1) 前提与变量控制

只改一个变量：`RDMA target memory`。其余必须保持一致：

- RDMA：`op=write`，`msg_size=10485760`，`iters=100000`，`qd=1`，`mtu=1024`
- 后台 memcpy：`h2d`，`--chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000`
- 网络参数：`mlx5_0 / port=1 / gid_idx=3 / tcp_port=33333`

机器角色（手动双机）：

- GPU 机器：`gpu_server` / `gpu_pcie_memcpy` / `gpu_metrics_logger.py`
- CPU 机器：`rdma_msgsize_sweep.py`

---

## 2) 目录与命名

```bash
mkdir -p /home/enine/danyang/SkyGDR/results/contention2_a1/{raw,metrics,summary,fig}
```

本文统一用：

- `baseline`：`bg_A1_h2d_baseline.log`
- `+GDR`：`bg_A1_h2d_with_gdr_write.log`
- `+HostRDMA`：`bg_A1_h2d_with_host_write.log`

---

## 3) 关键前置：Host-memory RDMA 服务端

当前仓库默认 `gpu_server` 是 GPU MR（GDR）路径。  
要做 `+HostRDMA`，你需要一个“Host pinned memory MR”服务端（例如 `gpu_server_host`）。

建议服务端命名：

- GDR：`./bin/gpu_server`
- Host：`./bin/gpu_server_host`

> 如果你现在还没有 `gpu_server_host`，先补这个二进制再跑 A1；否则无法做 `+HostRDMA` 组。

---

## 4) 三组实验（每组跑 1 次）

下面命令按你现有环境参数写死。

### A1-0: baseline（只有 H2D）

GPU 机器，终端 G2：

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=h2d --seconds=600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_a1/metrics/bg_A1_h2d_baseline.log
```

GPU 机器，终端 G3（可选，但建议）：

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_a1/metrics/gpu_A1_h2d_baseline.csv
```

维持 120s 后停掉 G2/G3（`Ctrl+C`）。

---

### A1-1: `+GDR`（H2D + RDMA write 到 GPU memory）

GPU 机器，终端 G1（先启动 GDR server）：

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_server mlx5_0 1G 33333 1 3 1024
```

GPU 机器，终端 G2：

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=h2d --seconds=600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_a1/metrics/bg_A1_h2d_with_gdr_write.log
```

GPU 机器，终端 G3（可选）：

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_a1/metrics/gpu_A1_h2d_with_gdr_write.csv
```

CPU 机器，终端 C1（等 G2 先稳定 10s 再执行）：

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py \
  --cpu_client ./cpu_client \
  --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 \
  --iters 100000 --op write --port 1 --gid_idx 3 \
  --qd 1 --span 1G --pattern random --align 256 --mtu 1024 \
  --msg_sizes 10485760 --warmup_iters 20000 --warmup_runs 1 \
  --sample 1 --max_samples 0 --write_ack 1 \
  --out ../results/contention2_a1/raw/A1_gdr_write.csv
```

RDMA 结束后再等 20s，再停 G1/G2/G3。

---

### A1-2: `+HostRDMA`（H2D + RDMA write 到 CPU pinned memory）

GPU 机器，终端 G1（启动 Host-MR server）：

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_server_host mlx5_0 1G 33333 1 3 1024
```

GPU 机器，终端 G2：

```bash
cd /home/enine/danyang/SkyGDR/src
./bin/gpu_pcie_memcpy --dir=h2d --seconds=600 --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=10000 \
  | tee ../results/contention2_a1/metrics/bg_A1_h2d_with_host_write.log
```

GPU 机器，终端 G3（可选）：

```bash
cd /home/enine/danyang/SkyGDR/src
uv run python tools/gpu_metrics_logger.py --gpu 0 --interval_ms 200 \
  --out ../results/contention2_a1/metrics/gpu_A1_h2d_with_host_write.csv
```

CPU 机器，终端 C1（其余参数与 `+GDR` 完全一致）：

```bash
cd /home/enine/danyang/SkyGDR/src
python3 tools/rdma_msgsize_sweep.py \
  --cpu_client ./cpu_client \
  --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 \
  --iters 100000 --op write --port 1 --gid_idx 3 \
  --qd 1 --span 1G --pattern random --align 256 --mtu 1024 \
  --msg_sizes 10485760 --warmup_iters 20000 --warmup_runs 1 \
  --sample 1 --max_samples 0 --write_ack 1 \
  --out ../results/contention2_a1/raw/A1_host_write.csv
```

RDMA 结束后再等 20s，再停 G1/G2/G3。

---

## 5) 汇总与画图（3 组柱状图）

在 CPU 机器（仓库根目录）执行：

```bash
cd /home/enine/danyang/SkyGDR
python3 - <<'PY'
import re, csv, statistics
from pathlib import Path
import matplotlib.pyplot as plt

base = Path("results/contention2_a1")

def mean_bw(log_path: Path):
    vals = []
    for line in log_path.read_text().splitlines():
        m = re.search(r"bw_gib_s=([0-9.]+)", line)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) > 1:
        vals = vals[1:]
    return statistics.mean(vals) if vals else float("nan")

b = mean_bw(base / "metrics" / "bg_A1_h2d_baseline.log")
g = mean_bw(base / "metrics" / "bg_A1_h2d_with_gdr_write.log")
h = mean_bw(base / "metrics" / "bg_A1_h2d_with_host_write.log")

rows = [
    ("baseline", b, 1.0),
    ("+GDR", g, g / b if b > 0 else float("nan")),
    ("+HostRDMA", h, h / b if b > 0 else float("nan")),
]

(base / "summary").mkdir(parents=True, exist_ok=True)
with (base / "summary" / "A1_h2d_norm.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "h2d_bw_gib_s", "h2d_norm_to_baseline"])
    w.writerows(rows)

labels = [r[0] for r in rows]
norms = [r[2] for r in rows]
colors = ["#4c78a8", "#f58518", "#54a24b"]

plt.figure(figsize=(6.4, 4.2))
bars = plt.bar(labels, norms, color=colors)
plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
plt.ylabel("H2D normalized BW")
plt.title("A1 Attribution Ablation: baseline vs +GDR vs +HostRDMA")
for bar, v in zip(bars, norms):
    plt.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"{v:.3f}", ha="center", va="bottom")
plt.ylim(0, max(norms) * 1.18)
plt.tight_layout()
(base / "fig").mkdir(parents=True, exist_ok=True)
plt.savefig(base / "fig" / "A1_h2d_norm_bar.png", dpi=180)
print("[ok] csv:", base / "summary" / "A1_h2d_norm.csv")
print("[ok] fig:", base / "fig" / "A1_h2d_norm_bar.png")
PY
```

---

## 6) 结果判读（你要写进 paper 的话术）

如果你观察到：

- `+GDR` 的 `H2D normalized BW` 显著低于 `+HostRDMA`
- 且 `+HostRDMA` 更接近 baseline

那么支持结论：

> 主要冲突来自 GPU-side shared PCIe segment（而非仅 CPU 内存/主机侧路径）。

---

## 7) 最小化踩坑清单

- `report_ms` 必须是 `10000`，否则瞬时抖动会污染均值。
- 三组都用同一张网卡、同一 `gid_idx/mtu/qd/msg/iters`。
- 每组都要保证 RDMA 与 H2D 至少重叠一个完整 10s 统计窗口。
- 先起背景，再起 RDMA；结束时先停 RDMA，再让背景多跑 20s。
- 若 `+HostRDMA` 组跑不起来，先确认 `gpu_server_host` 是否真的注册的是 host pinned MR。

