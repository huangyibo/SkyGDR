# 双机 AMD RDMA PD + CPU KV Offloading 运行手册

这份手册对应仓库里的新入口：

- `scripts/run_pd_dualnode_rdma.sh`
- `scripts/pd_dualnode_rdma.env.example`
- `src/tools/pd_dualnode_proxy_workload.py`

目标是把下面这条链路跑通：

- `amd0 (45.76.230.97)`：`prefiller + proxy + lmcache server`
- `amd1 (144.202.52.73)`：`decoder`
- `PDBackend`：直接走 `NIXL + UCX + RDMA`
- `CPU / shared tier`：`LocalCPUBackend + RemoteBackend(lm://45.76.230.97:65432)`

## 1. 关键默认值

- 容器镜像：`rocm/vllm-dev:nightly`
- 容器名：`pd-lmcache`
- 模型：`Qwen/Qwen3-8B`
- 服务名：`Qwen3-8B-Instruct`
- `amd0` 默认选卡：`GPU3 + bnxt_re0`
- `amd1` 默认选卡：`GPU3 + bnxt_re0`
- `UCX_TLS` 默认：`rc,cuda_copy,cuda_ipc,self,sm`
- `pd_buffer_device` 默认：`cuda`

如果要覆盖这些默认值，先在 shell 里 `source scripts/pd_dualnode_rdma.env.example`，再额外 `export` 你要改的变量。

## 2. 先在两边做 RDMA 探查

在 `amd0`：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 probe
```

在 `amd1`：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd1 probe
```

重点确认：

- 目标 `bnxt_re*` 是 `ACTIVE`
- 脚本解析出的 `netdev` 正确
- 如果 `amd1` 的 `GPU3 + bnxt_re0` 不成立，就在 shell 里覆写：
  - `export DECODER_GPU_INDEX=...`
  - `export DECODER_RDMA_DEVICE=...`

## 3. 两边先准备统一容器环境

在 `amd0`：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 prepare
```

在 `amd1`：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd1 prepare
```

这一步会：

- 起新的 `pd-lmcache` 容器
- 挂载当前仓库、`/data`、`/datasets`、`/mnt`
- 在容器里补齐 `lmcache`、`nixl`、`fastapi`、`uvicorn`、`httpx`

## 4. 启动顺序

### 4.1 `amd0` 启动 LMCache server

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 start-lmcache-server
```

### 4.2 `amd1` 启动 decoder

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd1 start-decoder
```

### 4.3 回到 `amd0` 启动 proxy 和 prefiller

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 start-proxy
bash scripts/run_pd_dualnode_rdma.sh amd0 start-prefiller
```

如果想把 `amd0` 这三项一次性起完，也可以：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 start-local-stack
```

前提是 `amd1` 的 decoder 已经先起来。

## 5. 验证

### 5.1 Smoke test

在 `amd0`：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 smoke
```

输出写到：

- `results/pd_dualnode_rdma/data/smoke_samples.csv`

### 5.2 Reuse test

在 `amd0`：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 reuse
```

输出写到：

- `results/pd_dualnode_rdma/data/reuse_samples.csv`

这一步会发：

- 第 1 轮 seed
- 后续 3 轮小 suffix 追加请求

## 6. 看状态和日志

在任意节点：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 status
bash scripts/run_pd_dualnode_rdma.sh amd1 status
```

日志位置统一在：

- `results/pd_dualnode_rdma/logs/lmcache-server.log`
- `results/pd_dualnode_rdma/logs/prefiller.log`
- `results/pd_dualnode_rdma/logs/proxy.log`
- `results/pd_dualnode_rdma/logs/decoder.log`

重点看：

- prefiller 是否出现 `LMCache hit` / `Retrieved`
- decoder 是否出现 `PDBackend` / `RemoteBackend`
- proxy 是否能持续转发
- `lmcache server` 是否保持健康

## 7. 停止

在两边分别执行：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 stop
bash scripts/run_pd_dualnode_rdma.sh amd1 stop
```

如果要连容器一起删掉：

```bash
cd ~/danyang/SkyGDR
bash scripts/run_pd_dualnode_rdma.sh amd0 destroy-container
bash scripts/run_pd_dualnode_rdma.sh amd1 destroy-container
```
