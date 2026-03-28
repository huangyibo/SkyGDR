# 单 GPU 条件下的 PD Imitation 执行手册（Qwen3-8B-Instruct 实操版）

## 0. 这份手册到底要做什么

这份手册的目标很明确：

- 在 **只有一台 A100** 的条件下，构造一套 **纯 PD imitation** 的数据采集流程
- 不强行搭建真实的 `prefill/decode disaggregation`
- 不依赖 LMCache 作为第 1 阶段的必需组件
- 直接产出一个可用于后续 case study 的逻辑 trace：
  - `pd_imitation_trace.csv`

第 1 阶段只做三件事：

1. 采 `prefill-only` 在不同 prompt 长度下的耗时
2. 采 `decode-only` 在不同 context 长度和生成长度下的耗时
3. 用目标模型的结构参数计算 `KV bytes per token`

然后把这三部分合成一个逻辑上的 PD trace。

这份 trace 是**逻辑 trace**，不是“真实 PD 系统里的网络抓包结果”。它回答的是：

- 某个 prompt 长度下，prefill 要多久
- 某个 context 长度和生成长度下，decode 要多久
- prefill 结束后，逻辑上会有多少 KV
- decode worker 理论上需要接收或持有多少 KV

## 1. 这一版写死的默认配置

为了让你可以直接复制命令，这一版把模型和关键参数固定下来。

### 1.1 模型口径

这一版使用：

- 实际下载的官方模型路径：`Qwen/Qwen3-8B`
- 对外 API 固定模型名：`Qwen3-8B-Instruct`

为什么这么做：

- Qwen 官方当前给出的文本 8B 主仓是 `Qwen/Qwen3-8B`
- 为了让你采样脚本里的 `--model` 字段更稳定、更直观，本手册在 `vllm serve` 时会固定：
  - `--served-model-name Qwen3-8B-Instruct`

这样对实验脚本来说，模型名就是：

- `Qwen3-8B-Instruct`

但底层实际加载的还是官方仓：

- `Qwen/Qwen3-8B`

### 1.2 固定 KV 参数

根据 `Qwen/Qwen3-8B` 官方 `config.json`，这一版固定使用：

- `num_hidden_layers = 36`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `dtype_bytes = 2`（`bfloat16`）

因此：

```text
KV_bytes_per_token = 2 × 36 × 8 × 128 × 2 = 147456 bytes
```

也就是：

- `144 KiB / token`

这一步后面会直接写死进 trace 生成命令，不需要你再去找 HF cache 里的 `config.json`。

### 1.3 固定服务参数

这一版默认：

- `dtype = bfloat16`
- `max-model-len = 32768`
- `gpu-memory-utilization = 0.85`
- `max-num-seqs = 4`
- `generation-config = vllm`
- endpoint 使用：
  - `/v1/completions`

为什么优先用 `completion`：

- 它不会像 `chat` endpoint 那样额外经过 chat template
- 对 `prompt_tokens` 桶控更干净
- 更适合 phase-1 的 timing collection

为什么这一版要显式加 `max-num-seqs = 4`：

- 我们希望 baseline 和 offloading 都承受相同的请求重叠压力
- 对 native CPU KV offloading 来说，想观察更明显的 CPU-GPU KV 搬运，仅靠单条串行请求通常不够
- 这里用“更长的上下文 + 更长的生成 + 受控并发”来放大两者差异

## 2. 为什么第 1 阶段不需要 LMCache

如果你当前不跑真正的 PD 分离 offloading，那么第 1 阶段并不需要 LMCache。

原因是：

- 你现在不需要真实跨机 KV 搬运
- 你只需要：
  - `prompt_tokens -> prefill_time`
  - `(context_tokens, generated_tokens) -> decode_time`
  - `KV_bytes_per_token`

这三样已经足够生成 `pd_imitation_trace.csv`。

所以 phase 1 的主线应该是：

- `vLLM native / single-GPU / prefill-decode timing extraction`

而不是：

- 先去搭一个完整的 LMCache 远端 backend

## 3. 为什么仍然保留 vLLM 原生 KV offloading 和 LMCache

这两个东西在后面仍然有用，但不是当前主流程的必需项。

### 3.1 vLLM 原生 KV offloading

如果你后续要观察单 GPU 条件下的 cache offloading，对你当前硬件最直接的选择是：

- `vLLM` 原生 KV offloading

但它在这份手册中的位置是：

- **附加实验**
- 不是 phase-1 的主流程依赖

### 3.2 LMCache

LMCache 更适合：

- remote / external backend
- 更真实的 cache movement
- 第 2 阶段的 replay 或 chunk-aware transfer 建模

所以更准确的关系是：

- **phase 1**：先用 vLLM 做 timing + 逻辑 trace
- **phase 2**：如果要做真实远端 KV 路径，再考虑 LMCache

## 4. 目录、环境变量和输出路径

下面所有命令默认在 GPU server 上执行。

先统一环境变量：

```bash
export ROOT=$HOME/SkyGDR
export BASELINE_RUN_ROOT=$ROOT/results/pd_imitation_qwen3_8b_instruct
export OFFLOAD_RUN_ROOT=$ROOT/results/pd_imitation_qwen3_8b_instruct_native_offload
export RUN_ROOT=$BASELINE_RUN_ROOT
export MODEL_PATH=Qwen/Qwen3-8B
export SERVED_MODEL=Qwen3-8B-Instruct
export API_BASE=http://127.0.0.1:8000
export PORT=8000
export DATA_ROOT=/data/danyang
export VENV_PATH=$DATA_ROOT/venvs/vllm
export UV_CACHE_DIR=$DATA_ROOT/uv-cache
export TMPDIR=$DATA_ROOT/tmp
export HF_HOME=$DATA_ROOT/hf-cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

mkdir -p $DATA_ROOT/{uv-cache,tmp,venvs,hf-cache}
mkdir -p $BASELINE_RUN_ROOT/{logs,data,summary}
mkdir -p $OFFLOAD_RUN_ROOT/{logs,data,summary}
cd $ROOT
```

这一版最终会产出：

- `$RUN_ROOT/data/prefill_prompts.jsonl`
- `$RUN_ROOT/data/decode_prompts.jsonl`
- `$RUN_ROOT/data/prefill_samples.csv`
- `$RUN_ROOT/data/decode_samples.csv`
- `$RUN_ROOT/summary/pd_imitation_trace.csv`
- `$RUN_ROOT/summary/pd_imitation_summary.json`

这里有一个很重要的约定：

- 默认 `RUN_ROOT=$BASELINE_RUN_ROOT`
- baseline 和 offloading 一定要落到两个不同目录
- 否则后面生成对照图和对照报告时会把两轮数据混在一起

## 5. 一次性环境准备

### 5.1 安装 `uv`

如果 GPU server 还没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

### 5.2 创建 Python 环境并安装依赖

`vLLM` 官方 quickstart 当前推荐：

- `Python 3.12`
- 用 `uv venv`
- 用 `uv pip install vllm --torch-backend=auto`

这里直接照这个来：

```bash
cd $ROOT
mkdir -p $DATA_ROOT/{uv-cache,tmp,venvs,hf-cache}
uv venv --python 3.12 --seed $VENV_PATH
source $VENV_PATH/bin/activate

uv pip install vllm --torch-backend=auto
uv pip install "transformers>=4.51.0" openai
```

验证：

```bash
python3 - <<'PY'
import vllm, transformers
print("vllm =", vllm.__version__)
print("transformers =", transformers.__version__)
PY
```

在当前这台机器上，已经验证过一组可用版本：

- `vllm = 0.18.0`
- `transformers = 4.57.6`

说明：

- `Qwen3` 官方模型卡明确建议使用较新的 `transformers`
- 如果 `transformers<4.51.0`，会遇到 `KeyError: 'qwen3'`
- 之所以把 cache、tmp 和 venv 都放到 `/data/danyang`，是因为这台机器的 `/home` 空间非常紧，而 `vllm + torch + 模型缓存` 都比较大，继续落到 `/home` 很容易再次安装失败

## 6. 启动 vLLM 服务

### 6.1 主流程推荐启动命令

这条命令是 phase-1 的默认主线，不启用任何 offloading。

如果你是第一次跑，请先确认：

```bash
export RUN_ROOT=$BASELINE_RUN_ROOT
```

然后再启动：

```bash
cd $ROOT
source $VENV_PATH/bin/activate

vllm serve $MODEL_PATH \
  --served-model-name $SERVED_MODEL \
  --host 127.0.0.1 \
  --port $PORT \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --generation-config vllm \
  2>&1 | tee $RUN_ROOT/logs/vllm_qwen3_8b_instruct.log
```

这条命令建议单独占一个终端，记作 `G1`。

### 6.2 为什么这样配

- `--served-model-name $SERVED_MODEL`
  - 让 API 层统一使用 `Qwen3-8B-Instruct`
- `--dtype bfloat16`
  - 官方模型为 BF16
  - A100 对 BF16 支持稳定
- `--max-model-len 32768`
  - 与 Qwen 官方文档的原生长度口径一致
- `--gpu-memory-utilization 0.85`
  - 比较保守，单卡更稳
- `--max-num-seqs 4`
  - 明确允许多条请求在服务端重叠
  - 这是为了让 baseline 和 offloading 都处在更接近“有 KV 压力”的状态
- `--generation-config vllm`
  - 避免 Hugging Face 仓库里的 `generation_config.json` 覆盖你的实验采样参数

## 7. 服务健康检查

等 `G1` 启动完成后，在另一个终端 `G2` 里执行：

### 7.1 看 `/v1/models`

```bash
curl -s $API_BASE/v1/models | python3 -m json.tool
```

你应该能看到类似：

- `id: "Qwen3-8B-Instruct"`

### 7.2 做一个最小 completion 请求

```bash
curl -s $API_BASE/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-8B-Instruct",
    "prompt": "Hello",
    "max_tokens": 1,
    "temperature": 0
  }' | python3 -m json.tool
```

如果这一步能正常返回，说明采样脚本也能工作。

## 8. 第 1A 阶段：构造“长对话”模板与 prefill prompts

### 8.1 先写一个长对话模板

这一版不再只用一句简单前缀，而是改成更像真实多轮对话的纯文本模板。

这样做的目的有两个：

- 让上下文更接近真实 assistant 场景
- 让 prompt 里天然包含更多历史轮次，从而放大 KV footprint

直接复制执行：

```bash
cat > $RUN_ROOT/data/long_dialogue_prefix.txt <<'EOF'
System: You are a careful and concise assistant helping with code, systems, and ML infrastructure questions.

User: I am profiling a distributed inference runtime and want to understand memory movement.
Assistant: Sure. We should separate prefill cost, decode cost, and cache movement so we can reason about contention clearly.

User: I also care about long-context serving and cache offloading.
Assistant: Then we should use longer dialogue-style prompts, preserve multi-turn history, and compare a baseline path with an offloading path under the same request mix.

User: Please keep track of token count, decode length, throughput, and KV size.
Assistant: Understood. We will focus on prompt length, generation length, and the implied KV footprint under the target model.

User: I want the workload to resemble a realistic debugging conversation with repeated context carry-over.
Assistant:
EOF
```

### 8.2 首轮 prefill bucket

为了更容易观察长上下文下的差异，这一版把 bucket 往长上下文集中：

- `2048`
- `4096`
- `8192`
- `16384`
- `24576`
- `28672`

每桶样本数：

- `12`

为什么不再直接用 `32768`：

- 你之前已经实际撞到过 `max-model-len=32768` 的上限
- `32768 prompt + 1 output token` 会直接报错
- `28672` 仍然足够长，但更稳，更适合 baseline/offloading 对照

### 8.3 直接复制执行

```bash
cd $ROOT
source $VENV_PATH/bin/activate

python3 src/tools/pd_build_bucket_prompts.py \
  --model_or_tokenizer $MODEL_PATH \
  --target_tokens 2048,4096,8192,16384,24576,28672 \
  --samples_per_bucket 12 \
  --prefix_file $RUN_ROOT/data/long_dialogue_prefix.txt \
  --out $RUN_ROOT/data/prefill_prompts.jsonl
```

### 8.4 快速检查

```bash
head -n 2 $RUN_ROOT/data/prefill_prompts.jsonl
wc -l $RUN_ROOT/data/prefill_prompts.jsonl
```

理论上总行数应为：

- `6 × 12 = 72`

## 9. 第 1B 阶段：采集 prefill-only 样本

### 9.1 采样策略

固定：

- `max_tokens = 1`
- `temperature = 0`
- endpoint = `completion`

这一版把：

- `total_elapsed_ms`

直接当作：

- `effective_prefill_ms`

因为 decode 只生成 `1 token`，影响很小。

### 9.2 直接复制执行

```bash
cd $ROOT
source $VENV_PATH/bin/activate

python3 src/tools/pd_collect_openai_samples.py \
  --api_base $API_BASE \
  --model $SERVED_MODEL \
  --mode prefill \
  --input_jsonl $RUN_ROOT/data/prefill_prompts.jsonl \
  --endpoint completion \
  --parallel_requests 4 \
  --out_csv $RUN_ROOT/data/prefill_samples.csv
```

这里的 `--parallel_requests 4` 很重要：

- 它会同时发出 4 条请求
- 对 baseline 和 offloading 都使用同样的并发
- 这样才能让两边承受相近的请求重叠压力
- 如果你发现 baseline 出现明显失败，再把 baseline 和 offloading **一起**降到 `2`

### 9.3 快速检查

```bash
head -n 5 $RUN_ROOT/data/prefill_samples.csv
tail -n 5 $RUN_ROOT/data/prefill_samples.csv
```

你重点看这几列：

- `prompt_tokens`
- `elapsed_ms`
- `effective_prefill_ms`
- `prefill_tps`
- `usage_prompt_tokens`
- `usage_completion_tokens`
- `http_status`
- `error`

### 9.4 验收标准

最低要求：

- 每个 bucket 都有成功样本
- `http_status` 主要为 `200`
- `error` 为空
- 平均延迟随 `prompt_tokens` 增大而上升

## 10. 第 1C 阶段：构造 decode prompts

### 10.1 首轮 context bucket

固定为：

- `2048`
- `4096`
- `8192`
- `16384`
- `24576`
- `28672`

每桶样本数：

- `12`

### 10.2 直接复制执行

```bash
cd $ROOT
source $VENV_PATH/bin/activate

python3 src/tools/pd_build_bucket_prompts.py \
  --model_or_tokenizer $MODEL_PATH \
  --target_tokens 2048,4096,8192,16384,24576,28672 \
  --samples_per_bucket 12 \
  --prefix_file $RUN_ROOT/data/long_dialogue_prefix.txt \
  --out $RUN_ROOT/data/decode_prompts.jsonl
```

## 11. 第 1D 阶段：采集 decode-only 样本

### 11.1 首轮 generation bucket

固定为：

- `128`
- `256`
- `512`

### 11.2 采样口径

这一版用最粗粒度、最稳定的估计方式：

- 不扣掉 prefill 时间
- 直接使用：
  - `decode_ms_per_token = elapsed_ms / generated_tokens`

这不是纯净的真实 decode kernel 时间，但足够支撑第一版 PD imitation。

为什么把 generation 往大调：

- `g=32` 和 `g=64` 更容易被固定开销污染
- `g=128/256/512` 更接近 steady-state decode
- 生成越长，decode 期间累计持有和增长的 KV 也越多，更容易放大 offloading 路径的效果

### 11.3 直接复制执行

```bash
cd $ROOT
source $VENV_PATH/bin/activate

python3 src/tools/pd_collect_openai_samples.py \
  --api_base $API_BASE \
  --model $SERVED_MODEL \
  --mode decode \
  --input_jsonl $RUN_ROOT/data/decode_prompts.jsonl \
  --generated_tokens 128,256,512 \
  --endpoint completion \
  --parallel_requests 4 \
  --out_csv $RUN_ROOT/data/decode_samples.csv
```

### 11.4 快速检查

```bash
head -n 5 $RUN_ROOT/data/decode_samples.csv
tail -n 5 $RUN_ROOT/data/decode_samples.csv
```

重点看：

- `context_tokens`
- `generated_tokens`
- `elapsed_ms`
- `decode_ms_per_token`
- `decode_tps`
- `usage_prompt_tokens`
- `usage_completion_tokens`

## 12. 第 1E 阶段：生成逻辑 PD trace

这一版直接把 `Qwen3-8B-Instruct` 的 KV 参数写死，所以不需要你再去找模型配置文件。

### 12.1 直接复制执行

```bash
cd $ROOT
source $VENV_PATH/bin/activate

python3 src/tools/pd_imitation_trace.py \
  --prefill_csv $RUN_ROOT/data/prefill_samples.csv \
  --decode_csv $RUN_ROOT/data/decode_samples.csv \
  --num_layers 36 \
  --num_kv_heads 8 \
  --head_dim 128 \
  --dtype_bytes 2 \
  --chunk_size_tokens 256 \
  --out_csv $RUN_ROOT/summary/pd_imitation_trace.csv \
  --summary_out $RUN_ROOT/summary/pd_imitation_summary.json
```

### 12.2 输出解释

主要输出：

- `$RUN_ROOT/summary/pd_imitation_trace.csv`
- `$RUN_ROOT/summary/pd_imitation_summary.json`

其中 `pd_imitation_trace.csv` 至少会告诉你：

- `prefill_time_ms`
- `decode_time_ms`
- `kv_bytes_per_token`
- `prefill_kv_bytes`
- `decode_required_kv_bytes`
- `chunked_prefill_kv_bytes`
- `chunked_decode_kv_bytes`

### 12.3 快速检查

```bash
head -n 10 $RUN_ROOT/summary/pd_imitation_trace.csv
cat $RUN_ROOT/summary/pd_imitation_summary.json
```

## 13. 如果你想顺手观察 vLLM 原生 KV offloading

这不是 phase-1 主流程，但非常适合拿来做第二组对照。

推荐顺序是：

1. 先跑完整的 baseline
2. 再开 native CPU KV offloading
3. 用完全相同的 prompt buckets 和采样脚本重跑一轮
4. 最后生成 baseline vs offloading 的对照报告

### 13.1 先切到 offloading 结果目录

```bash
export RUN_ROOT=$OFFLOAD_RUN_ROOT
mkdir -p $RUN_ROOT/{logs,data,summary}
```

### 13.2 改服务启动命令

把第 6 节里的启动命令改成：

```bash
cd $ROOT
source $VENV_PATH/bin/activate

vllm serve $MODEL_PATH \
  --served-model-name $SERVED_MODEL \
  --host 127.0.0.1 \
  --port $PORT \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --generation-config vllm \
  --kv-offloading-size 32 \
  --kv-offloading-backend native \
  --disable-hybrid-kv-cache-manager \
  2>&1 | tee $RUN_ROOT/logs/vllm_qwen3_8b_instruct_native_kv_offload.log
```

### 13.3 这意味着什么

- `--kv-offloading-size 32`
  - 给 CPU 侧 KV offloading buffer 分配 `32 GiB`
  - 这不是 vLLM 官方给出的统一推荐值，而是为了更容易观察 offloading 效果而选的“更激进”的实验值
  - 对当前 `Qwen3-8B` + 长上下文 workload，这个值更容易让 CPU-GPU 间的 KV 搬运变得可观
- `--kv-offloading-backend native`
  - 使用 vLLM 原生 CPU KV offloading
- `--disable-hybrid-kv-cache-manager`
  - 在 vLLM `0.18.x` 上，`OffloadingConnector` 与默认的 hybrid KV cache manager（HMA）不兼容；不加这一行会直接报：`Connector OffloadingConnector does not support HMA`

这个实验适合回答：

- 开启 native KV offloading 后，单 GPU 上的请求时延和吞吐怎么变
- 在更长对话、更长 context、更长 generation、受控并发下，CPU-GPU KV 搬运是否开始显著影响 decode / prefill

但它不改变这份手册的主目标：

- 当前主目标仍然是生成逻辑上的 `pd_imitation_trace.csv`

### 13.4 重新跑同一套采样流程

切到 `RUN_ROOT=$OFFLOAD_RUN_ROOT` 之后，重新执行下面这些节：

- 第 8 节：构造 prefill prompts
- 第 9 节：采集 prefill-only
- 第 10 节：构造 decode prompts
- 第 11 节：采集 decode-only
- 第 12 节：生成逻辑 PD trace

这样你会得到第二组完整产物：

- `$OFFLOAD_RUN_ROOT/data/prefill_samples.csv`
- `$OFFLOAD_RUN_ROOT/data/decode_samples.csv`
- `$OFFLOAD_RUN_ROOT/summary/pd_imitation_trace.csv`
- `$OFFLOAD_RUN_ROOT/summary/pd_imitation_summary.json`

### 13.5 生成 baseline vs offloading 对照报告

baseline 跑完、offloading 也跑完之后，可以直接生成一份对照报告：

```bash
cd $ROOT
source $VENV_PATH/bin/activate

python3 src/tools/pd_imitation_report.py \
  --results_dir $BASELINE_RUN_ROOT \
  --compare_results_dir $OFFLOAD_RUN_ROOT \
  --base_label baseline \
  --compare_label native_offload
```

这条命令会在 baseline 目录下额外生成：

- `$BASELINE_RUN_ROOT/summary/pd_imitation_compare_report.md`
- `$BASELINE_RUN_ROOT/fig/compare_prefill_latency.svg`
- `$BASELINE_RUN_ROOT/fig/compare_decode_g512_mspt.svg`（与第 11 节最大 `generated_tokens` 桶一致；若你改小 generation bucket，文件名会随报告脚本变化）

这份对照报告的重点是：

- baseline 和 native CPU offloading 的 prefill latency 是否出现分叉
- 当前最大 generation bucket（本手册为 `g=512`）这个更接近 steady-state 的 decode proxy 在不同 context 下是否明显变差

## 14. 最低验收标准

当下面这些都满足时，说明第 1 阶段完成：

- `$RUN_ROOT/data/prefill_samples.csv` 已生成
- `$RUN_ROOT/data/decode_samples.csv` 已生成
- `$RUN_ROOT/summary/pd_imitation_trace.csv` 已生成
- `$RUN_ROOT/summary/pd_imitation_summary.json` 已生成

并且满足这些合理性检查：

- prefill 平均延迟随 `prompt_tokens` 增加而上升
- decode 的 `ms/token` 会随 `context_tokens` 增大而变差
- `prefill_kv_bytes` 与 `prompt_tokens` 线性相关
- `decode_required_kv_bytes` 与 `context_tokens` 线性相关

如果你进一步做了 offloading 对照，还应额外满足：

- `$OFFLOAD_RUN_ROOT/summary/pd_imitation_trace.csv` 已生成
- `$BASELINE_RUN_ROOT/summary/pd_imitation_compare_report.md` 已生成
- baseline 和 offloading 没有写到同一个目录里

## 15. 第 2 阶段再做什么

如果第 1 阶段稳定，下一步再考虑：

- chunk-aware correction
- 更真实的 prompt 长度分布
- 更真实的请求到达过程
- CPU-only server 作为 replay sink
- vLLM 原生 KV offloading 对照实验
- LMCache 远端 KV 搬运

第 2 阶段应该消费 `pd_imitation_trace.csv`，而不是替换掉第 1 阶段的数据采集路径。
