#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


LM_METRICS = {
    "lmcache_requested_tokens": [
        "lmcache:num_requested_tokens",
        "lmcache:num_requested_tokens_total",
        "lmcache_num_requested_tokens",
        "lmcache_num_requested_tokens_total",
    ],
    "lmcache_hit_tokens": [
        "lmcache:num_hit_tokens",
        "lmcache:num_hit_tokens_total",
        "lmcache_num_hit_tokens",
        "lmcache_num_hit_tokens_total",
    ],
    "lmcache_vllm_hit_tokens": [
        "lmcache:num_vllm_hit_tokens",
        "lmcache:num_vllm_hit_tokens_total",
        "lmcache_num_vllm_hit_tokens",
        "lmcache_num_vllm_hit_tokens_total",
    ],
    "lmcache_remote_read_bytes": [
        "lmcache:num_remote_read_bytes",
        "lmcache:num_remote_read_bytes_total",
        "lmcache_num_remote_read_bytes",
        "lmcache_num_remote_read_bytes_total",
    ],
    "lmcache_remote_write_bytes": [
        "lmcache:num_remote_write_bytes",
        "lmcache:num_remote_write_bytes_total",
        "lmcache_num_remote_write_bytes",
        "lmcache_num_remote_write_bytes_total",
    ],
    "lmcache_remote_read_requests": [
        "lmcache:num_remote_read_requests",
        "lmcache:num_remote_read_requests_total",
        "lmcache_num_remote_read_requests",
        "lmcache_num_remote_read_requests_total",
    ],
    "lmcache_remote_write_requests": [
        "lmcache:num_remote_write_requests",
        "lmcache:num_remote_write_requests_total",
        "lmcache_num_remote_write_requests",
        "lmcache_num_remote_write_requests_total",
    ],
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completion_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/completions"


def metrics_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/metrics"


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


def get_text(url: str, timeout_s: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8")


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    parsed = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        value = parts[1]
        try:
            parsed[name] = float(value)
        except Exception:
            continue
    return parsed


def extract_lm_metrics(parsed: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, aliases in LM_METRICS.items():
        value = 0.0
        for alias in aliases:
            if alias in parsed:
                value = parsed[alias]
                break
        out[key] = value
    return out


def fetch_lm_metrics(url: str, timeout_s: float) -> tuple[dict[str, float], str]:
    try:
        text = get_text(url, timeout_s)
    except Exception as exc:
        return {key: 0.0 for key in LM_METRICS}, str(exc)
    return extract_lm_metrics(parse_prometheus_metrics(text)), ""


def diff_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    out = {}
    for key in LM_METRICS:
        out[key] = max(0.0, after.get(key, 0.0) - before.get(key, 0.0))
    return out


def build_payload(model: str, row: dict, ignore_eos: bool) -> dict:
    payload = {
        "model": model,
        "prompt": row["prompt_text"],
        "max_tokens": int(row["max_tokens"]),
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    return payload


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Run an external prefix-cache workload sequentially against a vLLM OpenAI-compatible endpoint, "
            "and attribute LMCache remote read/write counters to each request."
        )
    )
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--timeout_s", type=float, default=900.0)
    ap.add_argument("--metrics_timeout_s", type=float, default=30.0)
    ap.add_argument("--post_request_settle_ms", type=int, default=1200)
    ap.add_argument("--sleep_between_requests_ms", type=int, default=250)
    ap.add_argument("--ignore_eos", action="store_true")
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(Path(args.input_jsonl))
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req_url = completion_url(args.api_base)
    prom_url = metrics_url(args.api_base)

    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "request_id",
                "phase",
                "session_id",
                "turn_id",
                "prompt_tokens",
                "reused_prefix_tokens_est",
                "appended_tokens_est",
                "reuse_ratio_est",
                "expected_external_hit",
                "max_tokens",
                "submit_ts_unix_ms",
                "response_finish_ts_unix_ms",
                "post_metrics_ts_unix_ms",
                "elapsed_ms",
                "response_id",
                "usage_prompt_tokens",
                "usage_completion_tokens",
                "http_status",
                "error",
                "metrics_error",
                "lmcache_requested_tokens",
                "lmcache_hit_tokens",
                "lmcache_vllm_hit_tokens",
                "lmcache_remote_read_bytes",
                "lmcache_remote_write_bytes",
                "lmcache_remote_read_requests",
                "lmcache_remote_write_requests",
                "lmcache_hit_ratio",
                "lmcache_remote_read_GiB",
                "lmcache_remote_write_GiB",
            ],
        )
        writer.writeheader()

        for row in rows:
            print(
                f"[{row['phase']}] request_id={row['request_id']} "
                f"prompt_tokens={row['prompt_tokens']} reused={row['reused_prefix_tokens_est']}"
            )
            before_metrics, before_err = fetch_lm_metrics(prom_url, args.metrics_timeout_s)

            submit_ts = int(time.time() * 1000)
            status = 0
            error = ""
            try:
                status, resp = post_json(
                    req_url,
                    build_payload(args.model, row, args.ignore_eos),
                    args.timeout_s,
                )
            except urllib.error.HTTPError as exc:
                status = exc.code
                error = exc.read().decode("utf-8", errors="replace")
                resp = {}
            except Exception as exc:
                error = str(exc)
                resp = {}
            response_finish_ts = int(time.time() * 1000)

            if args.post_request_settle_ms > 0:
                time.sleep(args.post_request_settle_ms / 1000.0)
            after_metrics, after_err = fetch_lm_metrics(prom_url, args.metrics_timeout_s)
            post_metrics_ts = int(time.time() * 1000)

            usage = resp.get("usage") or {}
            response_id = resp.get("id", "") if isinstance(resp, dict) else ""
            diff = diff_metrics(before_metrics, after_metrics)
            requested_tokens = diff["lmcache_requested_tokens"]
            hit_tokens = diff["lmcache_hit_tokens"]

            out_row = {
                "request_id": row["request_id"],
                "phase": row["phase"],
                "session_id": row["session_id"],
                "turn_id": int(row["turn_id"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "reused_prefix_tokens_est": int(row["reused_prefix_tokens_est"]),
                "appended_tokens_est": int(row["appended_tokens_est"]),
                "reuse_ratio_est": float(row["reuse_ratio_est"]),
                "expected_external_hit": int(row["expected_external_hit"]),
                "max_tokens": int(row["max_tokens"]),
                "submit_ts_unix_ms": submit_ts,
                "response_finish_ts_unix_ms": response_finish_ts,
                "post_metrics_ts_unix_ms": post_metrics_ts,
                "elapsed_ms": response_finish_ts - submit_ts,
                "response_id": response_id,
                "usage_prompt_tokens": usage.get("prompt_tokens", ""),
                "usage_completion_tokens": usage.get("completion_tokens", ""),
                "http_status": status,
                "error": error,
                "metrics_error": "; ".join(x for x in [before_err, after_err] if x),
                "lmcache_requested_tokens": int(round(requested_tokens)),
                "lmcache_hit_tokens": int(round(hit_tokens)),
                "lmcache_vllm_hit_tokens": int(round(diff["lmcache_vllm_hit_tokens"])),
                "lmcache_remote_read_bytes": int(round(diff["lmcache_remote_read_bytes"])),
                "lmcache_remote_write_bytes": int(round(diff["lmcache_remote_write_bytes"])),
                "lmcache_remote_read_requests": int(round(diff["lmcache_remote_read_requests"])),
                "lmcache_remote_write_requests": int(round(diff["lmcache_remote_write_requests"])),
                "lmcache_hit_ratio": (hit_tokens / requested_tokens) if requested_tokens > 0 else 0.0,
                "lmcache_remote_read_GiB": diff["lmcache_remote_read_bytes"] / (1024.0 ** 3),
                "lmcache_remote_write_GiB": diff["lmcache_remote_write_bytes"] / (1024.0 ** 3),
            }
            writer.writerow(out_row)
            fp.flush()
            print(
                f"[{row['phase']}] request_id={row['request_id']} "
                f"status={status} elapsed_ms={out_row['elapsed_ms']} "
                f"remote_read_GiB={out_row['lmcache_remote_read_GiB']:.3f} "
                f"remote_write_GiB={out_row['lmcache_remote_write_GiB']:.3f} "
                f"hit_ratio={out_row['lmcache_hit_ratio']:.3f}"
            )

            if args.sleep_between_requests_ms > 0:
                time.sleep(args.sleep_between_requests_ms / 1000.0)

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
