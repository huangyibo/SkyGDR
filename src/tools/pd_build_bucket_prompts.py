#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path


BASE_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "vector", "matrix", "tensor", "cache", "token", "prompt", "decode",
    "prefill", "bandwidth", "latency", "fabric", "memory", "pipeline",
]


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


def load_tokenizer(model_or_tokenizer: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "transformers is required for pd_build_bucket_prompts.py. "
            "Install it in the target environment before running this script."
        ) from exc
    return AutoTokenizer.from_pretrained(model_or_tokenizer, trust_remote_code=True)


def encode(tokenizer, text: str):
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def build_seed_text(bucket: int, sample_idx: int, prefix: str):
    rnd = random.Random((bucket << 16) ^ sample_idx)
    words = [prefix.strip(), f"bucket_{bucket}", f"sample_{sample_idx}"]
    for _ in range(max(bucket * 4, 2048)):
        words.append(BASE_WORDS[rnd.randrange(len(BASE_WORDS))])
    return " ".join(w for w in words if w)


def find_exact_prompt(tokenizer, target_tokens: int, prefix: str, sample_idx: int) -> str:
    seed_text = build_seed_text(target_tokens, sample_idx, prefix)
    ids = encode(tokenizer, seed_text)
    if len(ids) < target_tokens:
        raise RuntimeError(
            f"seed corpus too short for target_tokens={target_tokens}; got only {len(ids)} tokens"
        )

    lower = max(1, target_tokens - 64)
    upper = min(len(ids), target_tokens + 64)
    for end in range(lower, upper + 1):
        text = decode(tokenizer, ids[:end]).strip()
        actual = len(encode(tokenizer, text))
        if actual == target_tokens:
            return text

    for start in range(0, min(128, max(0, len(ids) - target_tokens))):
        sub = ids[start:start + target_tokens]
        if len(sub) < target_tokens:
            break
        text = decode(tokenizer, sub).strip()
        actual = len(encode(tokenizer, text))
        if actual == target_tokens:
            return text

    raise RuntimeError(
        f"failed to construct exact prompt for target_tokens={target_tokens}, sample_idx={sample_idx}"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Build synthetic prompts that match requested token buckets.")
    ap.add_argument("--model_or_tokenizer", required=True, help="HF model or tokenizer path/name")
    ap.add_argument("--target_tokens", required=True, help="comma-separated token buckets, e.g. 512,1024,2048")
    ap.add_argument("--samples_per_bucket", type=int, default=20)
    ap.add_argument("--prefix", default="You are a helpful assistant.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True, help="output JSONL path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    tokenizer = load_tokenizer(args.model_or_tokenizer)
    buckets = parse_int_list(args.target_tokens)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fp:
        for bucket in buckets:
            for sample_idx in range(args.samples_per_bucket):
                sample_id = f"tok{bucket}_s{sample_idx:03d}"
                prompt_text = find_exact_prompt(tokenizer, bucket, args.prefix, sample_idx)
                prompt_tokens = len(encode(tokenizer, prompt_text))
                if prompt_tokens != bucket:
                    raise SystemExit(
                        f"internal error: prompt length drifted for {sample_id}: expected {bucket}, got {prompt_tokens}"
                    )
                row = {
                    "sample_id": sample_id,
                    "target_tokens": bucket,
                    "prompt_tokens": prompt_tokens,
                    "prompt_text": prompt_text,
                }
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
