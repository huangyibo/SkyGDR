#!/usr/bin/env python3

import argparse
import concurrent.futures
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def endpoint_url(api_base: str) -> str:
    return f"{api_base.rstrip('/')}/v1/completions"


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


def build_payload(model: str, row: dict, ignore_eos: bool):
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


def run_one_sample(url: str, model: str, timeout_s: float, row: dict, ignore_eos: bool):
    payload = build_payload(model, row, ignore_eos)
    submit_ts = int(time.time() * 1000)
    status = 0
    error = ""
    try:
        status, resp = post_json(url, payload, timeout_s)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", errors="replace")
        resp = {}
    except Exception as exc:
        error = str(exc)
        resp = {}
    finish_ts = int(time.time() * 1000)

    usage = resp.get("usage") or {}
    response_id = resp.get("id", "") if isinstance(resp, dict) else ""

    out_row = {
        "request_id": row["request_id"],
        "launch_group": int(row["launch_group"]),
        "phase": row["phase"],
        "session_id": row["session_id"],
        "turn_id": int(row["turn_id"]),
        "prompt_tokens": int(row["prompt_tokens"]),
        "reused_prefix_tokens_est": int(row["reused_prefix_tokens_est"]),
        "appended_tokens_est": int(row["appended_tokens_est"]),
        "reuse_ratio_est": float(row["reuse_ratio_est"]),
        "expected_restore": int(row["expected_restore"]),
        "max_tokens": int(row["max_tokens"]),
        "submit_ts_unix_ms": submit_ts,
        "finish_ts_unix_ms": finish_ts,
        "elapsed_ms": finish_ts - submit_ts,
        "response_id": response_id,
        "usage_prompt_tokens": usage.get("prompt_tokens", ""),
        "usage_completion_tokens": usage.get("completion_tokens", ""),
        "http_status": status,
        "error": error,
    }
    return out_row


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run a restore-focused multi-turn workload against a vLLM OpenAI-compatible endpoint."
    )
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--timeout_s", type=float, default=900.0)
    ap.add_argument("--sleep_between_groups_ms", type=int, default=250)
    ap.add_argument("--ignore_eos", action="store_true")
    ap.add_argument("--out_csv", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(Path(args.input_jsonl))
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = endpoint_url(args.api_base)

    launch_groups = {}
    for row in rows:
        launch_groups.setdefault(int(row["launch_group"]), []).append(row)

    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "request_id",
                "launch_group",
                "phase",
                "session_id",
                "turn_id",
                "prompt_tokens",
                "reused_prefix_tokens_est",
                "appended_tokens_est",
                "reuse_ratio_est",
                "expected_restore",
                "max_tokens",
                "submit_ts_unix_ms",
                "finish_ts_unix_ms",
                "elapsed_ms",
                "response_id",
                "usage_prompt_tokens",
                "usage_completion_tokens",
                "http_status",
                "error",
            ],
        )
        writer.writeheader()

        for group_id in sorted(launch_groups):
            group_rows = launch_groups[group_id]
            phase = group_rows[0]["phase"]
            print(f"[group {group_id}] phase={phase} requests={len(group_rows)}")
            if len(group_rows) == 1:
                result = run_one_sample(url, args.model, args.timeout_s, group_rows[0], args.ignore_eos)
                writer.writerow(result)
                fp.flush()
                print(
                    f"[{phase}] request_id={result['request_id']} "
                    f"status={result['http_status']} elapsed_ms={result['elapsed_ms']}"
                )
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(group_rows)) as ex:
                    futs = [
                        ex.submit(run_one_sample, url, args.model, args.timeout_s, row, args.ignore_eos)
                        for row in group_rows
                    ]
                    for fut in concurrent.futures.as_completed(futs):
                        result = fut.result()
                        writer.writerow(result)
                        fp.flush()
                        print(
                            f"[{phase}] request_id={result['request_id']} "
                            f"status={result['http_status']} elapsed_ms={result['elapsed_ms']}"
                        )

            if args.sleep_between_groups_ms > 0:
                time.sleep(args.sleep_between_groups_ms / 1000.0)

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
