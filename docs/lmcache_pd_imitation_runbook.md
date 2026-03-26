# LMCache-Informed PD Imitation Runbook

## Overview

This runbook describes how to collect the minimum data needed to build a
`prefill/decode disaggregation (PD)` imitation trace when only one server has a
GPU (`1 x A100`) and the other server is CPU-only.

The key design choice is to **not** force a full LMCache PD deployment. Instead,
phase 1 measures:

- `prefill-only` timing and throughput as a function of prompt length
- `decode-only` timing and throughput as a function of context length and
  generation length
- `KV bytes per token` from the target model configuration

Then phase 1 converts those measurements into a logical
`pd_imitation_trace.csv`.

This is the right first step when:

- you cannot run a faithful `2-GPU` PD deployment yet
- you still want a data-driven estimate of how much KV would be produced by
  prefill and consumed by decode
- you want to feed a later PCIe/cache-offload case study with realistic
  prompt-length and decode-length distributions

The CPU-only server is intentionally excluded from phase 1. It becomes relevant
only in phase 2, when replaying the logical trace as a real networked flow.

## Why LMCache still matters

The final phase-1 collection path does not depend on running LMCache as a remote
backend. However, LMCache still informs the design:

- it motivates the `PD` setting and cache movement workflow
- it suggests later chunk-based modeling
- it provides a realistic direction for phase 2 replay or remote KV validation

In phase 1, the goal is simpler: isolate the model-side timing behavior and
derive the logical KV volume that a PD system would need to move.

## Outputs

Phase 1 should produce four files:

- `prefill_prompts.jsonl`
- `decode_prompts.jsonl`
- `prefill_samples.csv`
- `decode_samples.csv`

Then the offline aggregation step should produce:

- `pd_imitation_trace.csv`
- optional summary file, e.g. `pd_imitation_summary.json`

## Repository tools

This repository includes three helper scripts for this workflow:

- `src/tools/pd_build_bucket_prompts.py`
  - generate synthetic prompts with token lengths matched to requested buckets
- `src/tools/pd_collect_openai_samples.py`
  - call a vLLM OpenAI-compatible endpoint and collect timing samples
- `src/tools/pd_imitation_trace.py`
  - combine prefill/decode sample CSVs and model KV parameters into a logical PD
    trace

## Environment assumptions

- GPU server
  - runs vLLM
  - has the target model and tokenizer available
  - exposes an OpenAI-compatible endpoint, usually `http://127.0.0.1:8000`
- CPU-only server
  - not used in phase 1
- local laptop / repo copy
  - can be used to prepare scripts and inspect CSVs

## Phase 1A: Build synthetic prefill prompts

### Goal

Construct prompt files where token length is the controlled variable.

### Default buckets

Use these prompt-length buckets for the first run:

- `512`
- `1024`
- `2048`
- `4096`
- `8192`
- `16384`
- `32768`

Default samples per bucket:

- `20`

### Command template

Run on the GPU server or any machine that has the target tokenizer:

```bash
cd /home/enine/danyang/SkyGDR
python3 src/tools/pd_build_bucket_prompts.py \
  --model_or_tokenizer Qwen/Qwen3-8B-Instruct \
  --target_tokens 512,1024,2048,4096,8192,16384,32768 \
  --samples_per_bucket 20 \
  --out prefill_prompts.jsonl
```

### Output format

Each JSONL row contains:

- `sample_id`
- `target_tokens`
- `prompt_tokens`
- `prompt_text`

The script searches for a prompt whose tokenizer length matches the target
bucket exactly. If exact matching is impossible for a bucket/sample pair, the
script exits with an error instead of silently drifting.

## Phase 1B: Collect prefill-only samples

### Goal

Measure prompt processing cost with decode minimized.

### Request settings

Use:

- `max_tokens = 1`
- `temperature = 0`
- deterministic settings wherever possible

This keeps almost all of the latency in the prefill path.

### Command template

```bash
cd /home/enine/danyang/SkyGDR
python3 src/tools/pd_collect_openai_samples.py \
  --api_base http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B-Instruct \
  --mode prefill \
  --input_jsonl prefill_prompts.jsonl \
  --endpoint completion \
  --out_csv prefill_samples.csv
```

### Output fields

The CSV contains:

- `sample_id`
- `mode`
- `prompt_tokens`
- `context_tokens`
- `generated_tokens`
- `submit_ts_unix_ms`
- `finish_ts_unix_ms`
- `elapsed_ms`
- `effective_prefill_ms`
- `decode_ms_per_token`
- `decode_tps`
- `prefill_tps`
- `response_id`
- `usage_prompt_tokens`
- `usage_completion_tokens`
- `http_status`
- `error`

For `prefill` mode:

- `context_tokens = prompt_tokens`
- `generated_tokens = 1`
- `effective_prefill_ms = elapsed_ms`
- `prefill_tps = prompt_tokens / effective_prefill_ms`

For first-pass PD imitation, prefer the `completion` endpoint over `chat`:

- `completion` keeps the raw prompt unchanged
- `chat` may add hidden chat-template tokens on the server side
- the extra `usage_*` columns are there to verify whether server-side token
  accounting matches the intended bucket

### Acceptance check

For each bucket:

- at least `20` successful rows
- low variance inside the same bucket
- monotonic increase of mean latency with prompt length

## Phase 1C: Build decode prompt set

### Goal

Prepare requests that isolate decode behavior under controlled context length.

### Default context buckets

- `512`
- `1024`
- `2048`
- `4096`
- `8192`
- `16384`

Default samples per bucket:

- `20`

### Command template

```bash
cd /home/enine/danyang/SkyGDR
python3 src/tools/pd_build_bucket_prompts.py \
  --model_or_tokenizer Qwen/Qwen3-8B-Instruct \
  --target_tokens 512,1024,2048,4096,8192,16384 \
  --samples_per_bucket 20 \
  --out decode_prompts.jsonl
```

## Phase 1D: Collect decode-only samples

### Goal

Measure average decode cost per token under controlled context length and
generation length.

### Default generation buckets

- `32`
- `64`
- `128`
- `256`

### Command template

```bash
cd /home/enine/danyang/SkyGDR
python3 src/tools/pd_collect_openai_samples.py \
  --api_base http://127.0.0.1:8000 \
  --model Qwen/Qwen3-8B-Instruct \
  --mode decode \
  --input_jsonl decode_prompts.jsonl \
  --generated_tokens 32,64,128,256 \
  --endpoint completion \
  --out_csv decode_samples.csv
```

### Output fields

For `decode` mode:

- `context_tokens` comes from the prompt bucket
- `generated_tokens` is one of the configured decode buckets
- `decode_ms_per_token = elapsed_ms / generated_tokens`
- `decode_tps = generated_tokens / elapsed_ms`

This first pass intentionally uses the coarse estimate:

- do **not** subtract prefill time from decode time yet
- use full end-to-end request time divided by generated tokens

That is enough for first-pass PD imitation.

## Phase 1E: Extract KV bytes per token

### Goal

Compute the logical KV size generated by prefill and required by decode.

### Required parameters

The target model must provide or imply:

- `num_hidden_layers`
- `num_key_value_heads`
- `head_dim`
- `dtype_bytes`

### Formula

Use:

```text
KV_bytes_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

where:

- `2` accounts for `K` and `V`
- `dtype_bytes` is typically:
  - `2` for `bfloat16` / `float16`
  - `1` for `fp8`

### Notes

- phase 1 assumes a single-device logical KV view
- no tensor-parallel or sharded correction is applied by default
- chunking is modeled later as an additional derived quantity

## Phase 1F: Generate the logical PD trace

### Goal

Combine the sample CSVs and model KV parameters into a reusable logical trace.

### Command template

If you have a Hugging Face `config.json` locally:

```bash
cd /home/enine/danyang/SkyGDR
python3 src/tools/pd_imitation_trace.py \
  --prefill_csv prefill_samples.csv \
  --decode_csv decode_samples.csv \
  --model_config /path/to/config.json \
  --chunk_size_tokens 256 \
  --out_csv pd_imitation_trace.csv \
  --summary_out pd_imitation_summary.json
```

Or pass parameters manually:

```bash
python3 src/tools/pd_imitation_trace.py \
  --prefill_csv prefill_samples.csv \
  --decode_csv decode_samples.csv \
  --num_layers 32 \
  --num_kv_heads 8 \
  --head_dim 128 \
  --dtype_bytes 2 \
  --chunk_size_tokens 256 \
  --out_csv pd_imitation_trace.csv
```

### Matching rule

The script pairs:

- `prefill prompt bucket P`
- `decode context bucket C = P`
- each decode generation bucket `G`

That means the first-pass trace assumes:

- decode starts with a context length equal to the prefilling prompt length

### Output fields

`pd_imitation_trace.csv` contains:

- `req_id`
- `prompt_tokens`
- `context_tokens`
- `generated_tokens`
- `prefill_time_ms`
- `prefill_time_std_ms`
- `decode_time_ms`
- `decode_time_std_ms`
- `kv_bytes_per_token`
- `prefill_kv_bytes`
- `decode_required_kv_bytes`
- `chunk_size_tokens`
- `chunked_prefill_kv_bytes`
- `chunked_decode_kv_bytes`

## Interpretation

This trace is **logical**, not a direct measurement of network traffic. Each row
answers:

- how long prefill takes for a prompt of size `P`
- how long decode takes for context `P` and generation length `G`
- how much KV logically exists after prefill
- how much KV a decode worker would need to receive or hold

That is exactly the information needed to drive a later:

- PD replay experiment
- PCIe/cache-offload imitation
- synthetic network/KV transfer generator

## Why the CPU-only server is not used in phase 1

The CPU-only server is unnecessary in this first step because phase 1 does not
attempt to create real cross-machine KV movement.

Phase 1 only needs:

- prompt-length to prefill-time mapping
- context/generation-length to decode-time mapping
- model-derived KV volume

Those three ingredients already determine a logical PD trace.

The CPU-only server becomes useful later if you want to:

- replay the logical trace as real network traffic
- emulate KV movement over RPC/storage
- validate chunk-size or remote-backend effects

## Validation checklist

Phase 1 is successful if:

- `prefill_samples.csv` exists and has successful rows for every prompt bucket
- `decode_samples.csv` exists and has successful rows for every
  `(context_tokens, generated_tokens)` pair
- `kv_bytes_per_token` can be computed from model config or manual parameters
- `pd_imitation_trace.csv` is generated without missing bucket matches

Sanity checks:

- prefill mean latency increases with prompt length
- decode `ms/token` worsens as context length grows
- `prefill_kv_bytes` is linear in `prompt_tokens`
- `decode_required_kv_bytes` is linear in `context_tokens`

## Optional second phase

Once phase 1 is stable, extend toward a more realistic PD experiment by adding:

- chunk-aware correction
- real prompt-length distributions
- real request arrival processes
- CPU-only server replay sink
- LMCache-backed remote KV movement

That second phase should consume `pd_imitation_trace.csv` rather than replace the
phase-1 collection path.
