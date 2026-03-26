#!/usr/bin/env python3

import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_int_list(text: str):
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("empty integer list")
    return vals


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def post_json(url: str, payload: dict, timeout_s: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def build_payload(args, row, generated_tokens: int):
    if args.endpoint == "chat":
        return {
            "model": args.model,
            "messages": [{"role": "user", "content": row["prompt_text"]}],
            "max_tokens": generated_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
    return {
        "model": args.model,
        "prompt": row["prompt_text"],
        "max_tokens": generated_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }


def endpoint_url(api_base: str, endpoint: str):
    base = api_base.rstrip("/")
    if endpoint == "chat":
        return f"{base}/v1/chat/completions"
    return f"{base}/v1/completions"


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return num / den


def parse_args():
    ap = argparse.ArgumentParser(description="Collect prefill/decode samples from a vLLM OpenAI-compatible endpoint.")
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["prefill", "decode"], required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--generated_tokens", default="32,64,128,256", help="decode mode only")
    ap.add_argument("--endpoint", choices=["chat", "completion"], default="chat")
    ap.add_argument("--timeout_s", type=float, default=600.0)
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(Path(args.input_jsonl))
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = endpoint_url(args.api_base, args.endpoint)
    decode_buckets = parse_int_list(args.generated_tokens) if args.mode == "decode" else [1]

    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "sample_id",
            "mode",
            "prompt_tokens",
            "context_tokens",
            "generated_tokens",
            "submit_ts_unix_ms",
            "finish_ts_unix_ms",
            "elapsed_ms",
            "effective_prefill_ms",
            "decode_ms_per_token",
            "decode_tps",
            "prefill_tps",
            "response_id",
            "usage_prompt_tokens",
            "usage_completion_tokens",
            "http_status",
            "error",
        ])

        for row in rows:
            prompt_tokens = int(row["prompt_tokens"])
            for generated_tokens in decode_buckets:
                sample_id = row["sample_id"] if args.mode == "prefill" else f"{row['sample_id']}_g{generated_tokens}"
                payload = build_payload(args, row, generated_tokens)
                submit_ts = int(time.time() * 1000)
                status = 0
                error = ""
                try:
                    status, resp = post_json(url, payload, args.timeout_s)
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    error = exc.read().decode("utf-8", errors="replace")
                    resp = {}
                except Exception as exc:
                    error = str(exc)
                    resp = {}
                finish_ts = int(time.time() * 1000)
                elapsed_ms = finish_ts - submit_ts
                response_id = ""
                usage_prompt_tokens = ""
                usage_completion_tokens = ""
                if isinstance(resp, dict):
                    response_id = resp.get("id", "")
                    usage = resp.get("usage") or {}
                    if isinstance(usage, dict):
                        usage_prompt_tokens = usage.get("prompt_tokens", "")
                        usage_completion_tokens = usage.get("completion_tokens", "")

                if args.mode == "prefill":
                    effective_prefill_ms = float(elapsed_ms)
                    prefill_tps = safe_div(prompt_tokens * 1000.0, effective_prefill_ms)
                    decode_ms_per_token = float("nan")
                    decode_tps = float("nan")
                    context_tokens = prompt_tokens
                else:
                    effective_prefill_ms = float("nan")
                    prefill_tps = float("nan")
                    decode_ms_per_token = safe_div(float(elapsed_ms), generated_tokens)
                    decode_tps = safe_div(generated_tokens * 1000.0, float(elapsed_ms))
                    context_tokens = prompt_tokens

                if status >= 400 and not error:
                    error = json.dumps(resp, ensure_ascii=False)

                writer.writerow([
                    sample_id,
                    args.mode,
                    prompt_tokens,
                    context_tokens,
                    generated_tokens,
                    submit_ts,
                    finish_ts,
                    elapsed_ms,
                    "" if math.isnan(effective_prefill_ms) else f"{effective_prefill_ms:.6f}",
                    "" if math.isnan(decode_ms_per_token) else f"{decode_ms_per_token:.6f}",
                    "" if math.isnan(decode_tps) else f"{decode_tps:.6f}",
                    "" if math.isnan(prefill_tps) else f"{prefill_tps:.6f}",
                    response_id,
                    usage_prompt_tokens,
                    usage_completion_tokens,
                    status,
                    error,
                ])
                fp.flush()
                print(f"[{args.mode}] sample_id={sample_id} status={status} elapsed_ms={elapsed_ms}")

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
