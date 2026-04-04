#!/usr/bin/env python3
"""Send smoke or reuse requests through the PD proxy and record timings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def build_seed_prompt(repetitions: int) -> str:
    base = (
        "You are validating dual-node PD with LMCache on AMD ROCm. "
        "Keep the answer short and deterministic. "
        "This paragraph is intentionally repeated to create a long reusable prefix.\n"
    )
    return base * repetitions


def build_appendix(turn_idx: int, repetitions: int) -> str:
    line = (
        f"\nTurn {turn_idx}: extend the same session with a tiny suffix. "
        "Focus on preserving the previous prefix exactly.\n"
    )
    return line * repetitions


def request_completion(api_base: str, payload: dict, timeout_secs: int) -> tuple[dict, float]:
    endpoint = f"{api_base.rstrip('/')}/v1/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach {endpoint}: {exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return json.loads(raw), elapsed_ms


def send_turn(
    api_base: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_secs: int,
) -> tuple[dict, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    return request_completion(api_base, payload, timeout_secs)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "turn",
        "prompt_chars",
        "elapsed_ms",
        "completion_tokens",
        "finish_reason",
        "text_preview",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_smoke(args: argparse.Namespace) -> list[dict]:
    prompt = build_seed_prompt(args.prompt_repetitions)
    response, elapsed_ms = send_turn(
        args.api_base,
        args.model,
        prompt,
        args.max_tokens,
        args.timeout_secs,
    )
    choice = response["choices"][0]
    return [
        {
            "mode": "smoke",
            "turn": 0,
            "prompt_chars": len(prompt),
            "elapsed_ms": round(elapsed_ms, 3),
            "completion_tokens": response.get("usage", {}).get("completion_tokens", ""),
            "finish_reason": choice.get("finish_reason", ""),
            "text_preview": choice.get("text", "").strip().replace("\n", " ")[:160],
        }
    ]


def run_reuse(args: argparse.Namespace) -> list[dict]:
    prompt = build_seed_prompt(args.prompt_repetitions)
    rows: list[dict] = []
    total_turns = args.append_turns + 1

    for turn in range(total_turns):
        response, elapsed_ms = send_turn(
            args.api_base,
            args.model,
            prompt,
            args.max_tokens,
            args.timeout_secs,
        )
        choice = response["choices"][0]
        rows.append(
            {
                "mode": "reuse",
                "turn": turn,
                "prompt_chars": len(prompt),
                "elapsed_ms": round(elapsed_ms, 3),
                "completion_tokens": response.get("usage", {}).get("completion_tokens", ""),
                "finish_reason": choice.get("finish_reason", ""),
                "text_preview": choice.get("text", "").strip().replace("\n", " ")[:160],
            }
        )
        prompt += build_appendix(turn + 1, args.append_repetitions)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("smoke", "reuse"), required=True)
    parser.add_argument("--prompt_repetitions", type=int, default=384)
    parser.add_argument("--append_turns", type=int, default=3)
    parser.add_argument("--append_repetitions", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--timeout_secs", type=int, default=600)
    parser.add_argument("--out_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.mode == "smoke":
            rows = run_smoke(args)
        else:
            rows = run_reuse(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    out_path = Path(args.out_csv)
    write_rows(out_path, rows)

    print(f"wrote {len(rows)} rows to {out_path}")
    for row in rows:
        print(
            f"turn={row['turn']} prompt_chars={row['prompt_chars']} "
            f"elapsed_ms={row['elapsed_ms']} finish_reason={row['finish_reason']}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
