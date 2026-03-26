#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def to_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def mean(vals):
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def stdev(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(var)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def load_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def load_model_config(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def infer_head_dim(cfg: dict):
    if "head_dim" in cfg:
        return int(cfg["head_dim"])
    hidden_size = cfg.get("hidden_size") or cfg.get("n_embd")
    num_attention_heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    if hidden_size and num_attention_heads:
        return int(hidden_size) // int(num_attention_heads)
    raise SystemExit("could not infer head_dim from model config")


def infer_num_layers(cfg: dict):
    if "num_hidden_layers" in cfg:
        return int(cfg["num_hidden_layers"])
    if "n_layer" in cfg:
        return int(cfg["n_layer"])
    raise SystemExit("could not infer num_layers from model config")


def infer_num_kv_heads(cfg: dict):
    if "num_key_value_heads" in cfg:
        return int(cfg["num_key_value_heads"])
    if "n_head_kv" in cfg:
        return int(cfg["n_head_kv"])
    if "multi_query" in cfg and cfg["multi_query"]:
        return 1
    if "num_attention_heads" in cfg:
        return int(cfg["num_attention_heads"])
    if "n_head" in cfg:
        return int(cfg["n_head"])
    raise SystemExit("could not infer num_kv_heads from model config")


def infer_dtype_bytes(cfg: dict, explicit_dtype: str = ""):
    if explicit_dtype:
        dt = explicit_dtype.lower()
    else:
        dt = str(cfg.get("torch_dtype", "")).lower()
    if dt in ("float16", "half", "fp16", "bfloat16", "bf16"):
        return 2
    if dt in ("float32", "fp32"):
        return 4
    if dt in ("fp8", "float8_e4m3fn", "float8_e5m2"):
        return 1
    raise SystemExit(
        "could not infer dtype_bytes; pass --dtype_bytes explicitly or provide a model config with torch_dtype"
    )


def derive_kv_params(args):
    if args.model_config:
        cfg = load_model_config(Path(args.model_config))
        num_layers = args.num_layers or infer_num_layers(cfg)
        num_kv_heads = args.num_kv_heads or infer_num_kv_heads(cfg)
        head_dim = args.head_dim or infer_head_dim(cfg)
        dtype_bytes = args.dtype_bytes or infer_dtype_bytes(cfg, args.dtype)
        return num_layers, num_kv_heads, head_dim, dtype_bytes
    required = [args.num_layers, args.num_kv_heads, args.head_dim, args.dtype_bytes]
    if any(v is None for v in required):
        raise SystemExit(
            "provide either --model_config or the full set of "
            "--num_layers --num_kv_heads --head_dim --dtype_bytes"
        )
    return args.num_layers, args.num_kv_heads, args.head_dim, args.dtype_bytes


def aggregate_prefill(rows):
    agg = {}
    for row in rows:
        if row.get("mode") != "prefill":
            continue
        if row.get("error"):
            continue
        prompt_tokens = to_int(row.get("prompt_tokens"))
        t = to_float(row.get("effective_prefill_ms"))
        if math.isnan(t):
            continue
        agg.setdefault(prompt_tokens, []).append(t)
    return agg


def aggregate_decode(rows):
    agg = {}
    for row in rows:
        if row.get("mode") != "decode":
            continue
        if row.get("error"):
            continue
        context_tokens = to_int(row.get("context_tokens"))
        generated_tokens = to_int(row.get("generated_tokens"))
        elapsed_ms = to_float(row.get("elapsed_ms"))
        if math.isnan(elapsed_ms):
            continue
        agg.setdefault((context_tokens, generated_tokens), []).append(elapsed_ms)
    return agg


def parse_args():
    ap = argparse.ArgumentParser(description="Build a logical PD imitation trace from prefill/decode sample CSVs.")
    ap.add_argument("--prefill_csv", required=True)
    ap.add_argument("--decode_csv", required=True)
    ap.add_argument("--model_config", default="", help="optional HF config.json path")
    ap.add_argument("--num_layers", type=int)
    ap.add_argument("--num_kv_heads", type=int)
    ap.add_argument("--head_dim", type=int)
    ap.add_argument("--dtype_bytes", type=int)
    ap.add_argument("--dtype", default="", help="optional dtype hint, e.g. bf16/fp16")
    ap.add_argument("--chunk_size_tokens", type=int, default=256)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--summary_out", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    num_layers, num_kv_heads, head_dim, dtype_bytes = derive_kv_params(args)
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes

    prefill_rows = load_csv_rows(Path(args.prefill_csv))
    decode_rows = load_csv_rows(Path(args.decode_csv))
    prefill_agg = aggregate_prefill(prefill_rows)
    decode_agg = aggregate_decode(decode_rows)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trace_rows = []
    for prompt_tokens in sorted(prefill_agg):
        if not prefill_agg[prompt_tokens]:
            continue
        prefill_mean = mean(prefill_agg[prompt_tokens])
        prefill_std = stdev(prefill_agg[prompt_tokens])
        decode_keys = sorted(k for k in decode_agg if k[0] == prompt_tokens)
        if not decode_keys:
            continue
        for context_tokens, generated_tokens in decode_keys:
            decode_mean = mean(decode_agg[(context_tokens, generated_tokens)])
            decode_std = stdev(decode_agg[(context_tokens, generated_tokens)])
            prefill_kv_bytes = prompt_tokens * kv_bytes_per_token
            decode_required_kv_bytes = context_tokens * kv_bytes_per_token
            chunked_prefill_tokens = ceil_div(prompt_tokens, args.chunk_size_tokens) * args.chunk_size_tokens
            chunked_decode_tokens = ceil_div(context_tokens, args.chunk_size_tokens) * args.chunk_size_tokens
            trace_rows.append({
                "req_id": f"p{prompt_tokens}_g{generated_tokens}",
                "prompt_tokens": prompt_tokens,
                "context_tokens": context_tokens,
                "generated_tokens": generated_tokens,
                "prefill_time_ms": prefill_mean,
                "prefill_time_std_ms": prefill_std,
                "decode_time_ms": decode_mean,
                "decode_time_std_ms": decode_std,
                "kv_bytes_per_token": kv_bytes_per_token,
                "prefill_kv_bytes": prefill_kv_bytes,
                "decode_required_kv_bytes": decode_required_kv_bytes,
                "chunk_size_tokens": args.chunk_size_tokens,
                "chunked_prefill_kv_bytes": chunked_prefill_tokens * kv_bytes_per_token,
                "chunked_decode_kv_bytes": chunked_decode_tokens * kv_bytes_per_token,
            })

    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "req_id",
            "prompt_tokens",
            "context_tokens",
            "generated_tokens",
            "prefill_time_ms",
            "prefill_time_std_ms",
            "decode_time_ms",
            "decode_time_std_ms",
            "kv_bytes_per_token",
            "prefill_kv_bytes",
            "decode_required_kv_bytes",
            "chunk_size_tokens",
            "chunked_prefill_kv_bytes",
            "chunked_decode_kv_bytes",
        ])
        for row in trace_rows:
            writer.writerow([
                row["req_id"],
                row["prompt_tokens"],
                row["context_tokens"],
                row["generated_tokens"],
                f"{row['prefill_time_ms']:.6f}",
                f"{row['prefill_time_std_ms']:.6f}",
                f"{row['decode_time_ms']:.6f}",
                f"{row['decode_time_std_ms']:.6f}",
                row["kv_bytes_per_token"],
                row["prefill_kv_bytes"],
                row["decode_required_kv_bytes"],
                row["chunk_size_tokens"],
                row["chunked_prefill_kv_bytes"],
                row["chunked_decode_kv_bytes"],
            ])

    if args.summary_out:
        summary = {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype_bytes": dtype_bytes,
            "kv_bytes_per_token": kv_bytes_per_token,
            "prefill_buckets": sorted(prefill_agg.keys()),
            "decode_pairs": sorted([list(k) for k in decode_agg.keys()]),
            "trace_rows": len(trace_rows),
        }
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

    print(f"wrote {out_path}")
    if args.summary_out:
        print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
