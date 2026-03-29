#!/usr/bin/env python3

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
    "reasoning", "session", "restore", "reuse", "offload", "turn",
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
            "transformers is required for pd_build_trajectory_workload.py. "
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


def build_seed_text(target_tokens: int, sample_idx: int, prefix: str) -> str:
    rnd = random.Random((target_tokens << 16) ^ sample_idx)
    words = [prefix.strip(), f"target_{target_tokens}", f"sample_{sample_idx}"]
    for _ in range(max(target_tokens * 4, 4096)):
        words.append(BASE_WORDS[rnd.randrange(len(BASE_WORDS))])
    return " ".join(w for w in words if w)


def find_exact_text(tokenizer, target_tokens: int, prefix: str, sample_idx: int) -> str:
    seed_text = build_seed_text(target_tokens, sample_idx, prefix)
    ids = encode(tokenizer, seed_text)
    if len(ids) < target_tokens:
        raise RuntimeError(
            f"seed corpus too short for target_tokens={target_tokens}; got only {len(ids)} tokens"
        )

    lower = max(1, target_tokens - 96)
    upper = min(len(ids), target_tokens + 96)
    for end in range(lower, upper + 1):
        text = decode(tokenizer, ids[:end]).strip()
        if len(encode(tokenizer, text)) == target_tokens:
            return text

    for start in range(0, min(192, max(0, len(ids) - target_tokens))):
        sub = ids[start:start + target_tokens]
        if len(sub) < target_tokens:
            break
        text = decode(tokenizer, sub).strip()
        if len(encode(tokenizer, text)) == target_tokens:
            return text

    raise RuntimeError(
        f"failed to construct exact text for target_tokens={target_tokens}, sample_idx={sample_idx}"
    )


def load_text_arg(inline_text: str, path_text: str) -> str:
    if path_text:
        return Path(path_text).read_text(encoding="utf-8")
    return inline_text


def append_block(prev_text: str, block_text: str) -> str:
    prev = prev_text.rstrip()
    block = block_text.strip()
    if not prev:
        return block
    return prev + "\n\n" + block


def ensure_within_limit(tokenizer, text: str, max_prompt_tokens: int) -> int:
    prompt_tokens = len(encode(tokenizer, text))
    if prompt_tokens > max_prompt_tokens:
        raise SystemExit(
            f"constructed prompt exceeds max_prompt_tokens={max_prompt_tokens}: got {prompt_tokens}"
        )
    return prompt_tokens


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build a restore-focused trajectory workload with high prefix reuse and pressure bursts."
    )
    ap.add_argument("--model_or_tokenizer", required=True)
    ap.add_argument("--base_prefix_tokens", type=int, default=24576)
    ap.add_argument("--append_tokens", default="256,256,256,256,256,256")
    ap.add_argument("--num_turns", type=int, default=6)
    ap.add_argument("--main_decode_tokens", type=int, default=32)
    ap.add_argument("--pressure_prompt_tokens", type=int, default=28672)
    ap.add_argument("--pressure_burst_size", type=int, default=8)
    ap.add_argument("--pressure_rounds_per_turn", type=int, default=2)
    ap.add_argument("--pressure_decode_tokens", type=int, default=1)
    ap.add_argument("--max_prompt_tokens", type=int, default=32768)
    ap.add_argument(
        "--session_prefix",
        default=(
            "System: You are assisting an autonomous coding agent working over a long session. "
            "The agent repeatedly reasons over prior context, terminal outputs, and tool feedback."
        ),
    )
    ap.add_argument("--session_prefix_file", default="")
    ap.add_argument(
        "--pressure_prefix",
        default=(
            "System: You are processing a large unrelated debugging transcript. "
            "This request exists to create memory pressure and compete for prefix-cache residency."
        ),
    )
    ap.add_argument("--pressure_prefix_file", default="")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    tokenizer = load_tokenizer(args.model_or_tokenizer)

    append_targets = parse_int_list(args.append_tokens)
    if len(append_targets) < args.num_turns:
        append_targets.extend([append_targets[-1]] * (args.num_turns - len(append_targets)))
    append_targets = append_targets[:args.num_turns]

    session_prefix = load_text_arg(args.session_prefix, args.session_prefix_file)
    pressure_prefix = load_text_arg(args.pressure_prefix, args.pressure_prefix_file)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_prefix_text = find_exact_text(tokenizer, args.base_prefix_tokens, session_prefix, args.seed)
    append_blocks = [
        find_exact_text(tokenizer, append_targets[i], session_prefix, args.seed + 1000 + i)
        for i in range(args.num_turns)
    ]

    num_pressure_prompts = max(0, args.num_turns - 1) * args.pressure_rounds_per_turn * args.pressure_burst_size
    pressure_prompts = [
        find_exact_text(
            tokenizer,
            args.pressure_prompt_tokens,
            pressure_prefix,
            args.seed + 2000 + i,
        )
        for i in range(num_pressure_prompts)
    ]

    rows = []
    launch_group = 0
    pressure_idx = 0

    current_prompt = append_block(base_prefix_text, append_blocks[0])
    current_prompt_tokens = ensure_within_limit(tokenizer, current_prompt, args.max_prompt_tokens)
    rows.append(
        {
            "request_id": "main_turn_000",
            "launch_group": launch_group,
            "phase": "warmup",
            "session_id": "main_session",
            "turn_id": 0,
            "prompt_tokens": current_prompt_tokens,
            "reused_prefix_tokens_est": 0,
            "appended_tokens_est": current_prompt_tokens,
            "reuse_ratio_est": 0.0,
            "expected_restore": 0,
            "max_tokens": args.main_decode_tokens,
            "prompt_text": current_prompt,
        }
    )

    for turn_id in range(1, args.num_turns):
        for burst_idx in range(args.pressure_rounds_per_turn):
            launch_group += 1
            for req_idx in range(args.pressure_burst_size):
                prompt_text = pressure_prompts[pressure_idx]
                pressure_idx += 1
                prompt_tokens = ensure_within_limit(tokenizer, prompt_text, args.max_prompt_tokens)
                rows.append(
                    {
                        "request_id": f"pressure_t{turn_id:03d}_b{burst_idx:02d}_r{req_idx:02d}",
                        "launch_group": launch_group,
                        "phase": "pressure",
                        "session_id": f"pressure_t{turn_id:03d}_b{burst_idx:02d}",
                        "turn_id": turn_id,
                        "prompt_tokens": prompt_tokens,
                        "reused_prefix_tokens_est": 0,
                        "appended_tokens_est": prompt_tokens,
                        "reuse_ratio_est": 0.0,
                        "expected_restore": 0,
                        "max_tokens": args.pressure_decode_tokens,
                        "prompt_text": prompt_text,
                    }
                )

        launch_group += 1
        prev_prompt = current_prompt
        current_prompt = append_block(prev_prompt, append_blocks[turn_id])
        prompt_tokens = ensure_within_limit(tokenizer, current_prompt, args.max_prompt_tokens)
        reused_prefix_tokens = len(encode(tokenizer, prev_prompt))
        appended_tokens = max(0, prompt_tokens - reused_prefix_tokens)
        reuse_ratio = (reused_prefix_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0
        rows.append(
            {
                "request_id": f"main_turn_{turn_id:03d}",
                "launch_group": launch_group,
                "phase": "reuse",
                "session_id": "main_session",
                "turn_id": turn_id,
                "prompt_tokens": prompt_tokens,
                "reused_prefix_tokens_est": reused_prefix_tokens,
                "appended_tokens_est": appended_tokens,
                "reuse_ratio_est": round(reuse_ratio, 6),
                "expected_restore": 1,
                "max_tokens": args.main_decode_tokens,
                "prompt_text": current_prompt,
            }
        )

    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}")
    print(
        "summary: "
        f"turns={args.num_turns} "
        f"pressure_groups={(args.num_turns - 1) * args.pressure_rounds_per_turn} "
        f"pressure_requests={num_pressure_prompts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
