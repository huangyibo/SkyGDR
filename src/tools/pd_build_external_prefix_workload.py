#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BASE_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "vector", "matrix", "tensor", "cache", "token", "prompt", "decode",
    "prefill", "bandwidth", "latency", "fabric", "memory", "pipeline",
    "agent", "terminal", "environment", "feedback", "trajectory", "prefix",
    "reasoning", "session", "restore", "reuse", "external", "shared",
    "storage", "remote", "command", "tool", "context", "planner",
]


def parse_int_list(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("empty integer list")
    return vals


def load_tokenizer(model_or_tokenizer: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "transformers is required for pd_build_external_prefix_workload.py. "
            "Install it in the target environment before running this script."
        ) from exc
    return AutoTokenizer.from_pretrained(model_or_tokenizer, trust_remote_code=True)


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def build_seed_text(prefix: str, min_tokens: int, seed: int) -> str:
    rnd = random.Random(seed)
    words = []
    if prefix.strip():
        words.append(prefix.strip())
    words.extend([f"seed_{seed}", f"target_{min_tokens}"])
    for _ in range(max(min_tokens * 4, 4096)):
        words.append(BASE_WORDS[rnd.randrange(len(BASE_WORDS))])
    return " ".join(words)


def build_token_corpus(tokenizer, prefix: str, min_tokens: int, seed: int) -> list[int]:
    seed_text = build_seed_text(prefix, min_tokens, seed)
    ids = encode(tokenizer, seed_text)
    if len(ids) < min_tokens:
        raise RuntimeError(
            f"seed corpus too short for min_tokens={min_tokens}; got only {len(ids)} tokens"
        )
    return ids


def find_exact_prefix_text(tokenizer, corpus_ids: list[int], target_tokens: int, prev_text: str) -> str:
    candidate_ends = [target_tokens]
    for delta in range(1, 97):
        candidate_ends.append(target_tokens + delta)
        candidate_ends.append(target_tokens - delta)

    seen = set()
    for end in candidate_ends:
        if end in seen or end <= 0 or end > len(corpus_ids):
            continue
        seen.add(end)
        text = decode(tokenizer, corpus_ids[:end]).rstrip()
        if prev_text and not text.startswith(prev_text):
            continue
        if len(encode(tokenizer, text)) == target_tokens:
            return text

    raise RuntimeError(
        f"failed to construct exact prefix text for target_tokens={target_tokens}"
    )


def load_text_arg(inline_text: str, path_text: str) -> str:
    if path_text:
        return Path(path_text).read_text(encoding="utf-8")
    return inline_text


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Build a multi-turn workload for external prefix-cache imitation. "
            "Turn 0 seeds a long prefix into LMCache; later turns append a small, chunk-aligned suffix."
        )
    )
    ap.add_argument("--model_or_tokenizer", required=True)
    ap.add_argument("--seed_prompt_tokens", type=int, default=24832)
    ap.add_argument("--append_tokens", default="256,256,256,256,256")
    ap.add_argument("--num_turns", type=int, default=6)
    ap.add_argument("--decode_tokens", type=int, default=16)
    ap.add_argument("--max_prompt_tokens", type=int, default=32768)
    ap.add_argument("--chunk_size_tokens", type=int, default=256)
    ap.add_argument(
        "--session_prefix",
        default=(
            "System: You are assisting a long-running coding agent that acts over many turns. "
            "Each new turn appends only a small amount of tool output or user feedback, while the prior "
            "history must remain available for reasoning."
        ),
    )
    ap.add_argument("--session_prefix_file", default="")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_or_tokenizer)

    append_targets = parse_int_list(args.append_tokens)
    if len(append_targets) < max(0, args.num_turns - 1):
        append_targets.extend([append_targets[-1]] * (args.num_turns - 1 - len(append_targets)))
    append_targets = append_targets[: max(0, args.num_turns - 1)]

    session_prefix = load_text_arg(args.session_prefix, args.session_prefix_file)
    total_targets = [args.seed_prompt_tokens]
    for append_tokens in append_targets:
        total_targets.append(total_targets[-1] + append_tokens)
    total_targets = total_targets[: args.num_turns]

    if not total_targets:
        raise SystemExit("num_turns must be >= 1")
    if total_targets[-1] > args.max_prompt_tokens:
        raise SystemExit(
            f"final prompt target {total_targets[-1]} exceeds max_prompt_tokens={args.max_prompt_tokens}"
        )
    for total in total_targets:
        if total % args.chunk_size_tokens != 0:
            raise SystemExit(
                f"target prompt tokens {total} is not aligned to chunk_size_tokens={args.chunk_size_tokens}; "
                "this workflow intentionally uses chunk-aligned turns to maximize external prefix-cache hits"
            )

    corpus_ids = build_token_corpus(tokenizer, session_prefix, total_targets[-1] + 2048, args.seed)

    rows = []
    prev_text = ""
    prev_prompt_tokens = 0
    for turn_id, target_prompt_tokens in enumerate(total_targets):
        prompt_text = find_exact_prefix_text(tokenizer, corpus_ids, target_prompt_tokens, prev_text)
        actual_prompt_tokens = len(encode(tokenizer, prompt_text))
        if actual_prompt_tokens != target_prompt_tokens:
            raise SystemExit(
                f"constructed prompt for turn {turn_id} has {actual_prompt_tokens} tokens, "
                f"expected {target_prompt_tokens}"
            )

        phase = "seed" if turn_id == 0 else "reuse"
        appended_tokens = actual_prompt_tokens - prev_prompt_tokens
        reused_prefix_tokens = prev_prompt_tokens if turn_id > 0 else 0
        reuse_ratio = (reused_prefix_tokens / actual_prompt_tokens) if actual_prompt_tokens > 0 else 0.0

        rows.append(
            {
                "request_id": f"turn_{turn_id:03d}_{phase}",
                "phase": phase,
                "session_id": "main_session",
                "turn_id": turn_id,
                "prompt_tokens": actual_prompt_tokens,
                "reused_prefix_tokens_est": reused_prefix_tokens,
                "appended_tokens_est": appended_tokens,
                "reuse_ratio_est": reuse_ratio,
                "expected_external_hit": 1 if turn_id > 0 else 0,
                "max_tokens": args.decode_tokens,
                "prompt_text": prompt_text,
            }
        )
        prev_text = prompt_text
        prev_prompt_tokens = actual_prompt_tokens

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
