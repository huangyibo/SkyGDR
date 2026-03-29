#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_tokenizer(model_or_tokenizer: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "transformers is required for pd_build_external_prefix_workload.py. "
            "Install it in the target environment before running this script."
        ) from exc
    return AutoTokenizer.from_pretrained(model_or_tokenizer, trust_remote_code=True)


def load_terminalbench_split(dataset_name: str, split: str, cache_dir: str, streaming: bool):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(
            "datasets is required for Terminal-Bench-driven workload generation. "
            "Install it with `uv pip install datasets` in the target environment."
        ) from exc
    kwargs = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if streaming:
        kwargs["streaming"] = True
    return load_dataset(dataset_name, **kwargs)


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def parse_append_tokens(text: str, reuse_turns: int) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        vals = [256]
    if len(vals) < reuse_turns:
        vals.extend([vals[-1]] * (reuse_turns - len(vals)))
    return vals[:reuse_turns]


def normalize_steps(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            return [{"src": "raw", "msg": text}]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return [{"src": "raw", "msg": str(value)}]


def stringify_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def render_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return stringify_scalar(value)
    if isinstance(value, list):
        parts = [render_value(v) for v in value]
        parts = [p for p in parts if p]
        return "\n".join(parts)
    if isinstance(value, dict):
        ordered_keys = [
            "src", "role", "speaker", "type",
            "msg", "content", "text", "query", "instruction",
            "command", "cmd", "cwd",
            "tool", "tool_name", "tools", "tool_calls",
            "obs", "observation", "output", "stdout", "stderr", "result",
            "returncode", "exit_code", "status",
        ]
        lines = []
        used = set()
        for key in ordered_keys:
            if key not in value:
                continue
            used.add(key)
            rendered = render_value(value[key])
            if rendered:
                lines.append(f"{key}: {rendered}")
        for key in sorted(value.keys()):
            if key in used:
                continue
            rendered = render_value(value[key])
            if rendered:
                lines.append(f"{key}: {rendered}")
        return "\n".join(lines)
    return str(value).strip()


def render_step(step: object) -> str:
    if isinstance(step, str):
        return step.strip()
    if not isinstance(step, dict):
        return stringify_scalar(step)
    actor = (
        step.get("src")
        or step.get("role")
        or step.get("speaker")
        or step.get("type")
        or "step"
    )
    body_keys = [
        "msg", "content", "text", "query", "instruction",
        "command", "cmd",
        "tools", "tool_calls", "tool", "tool_name",
        "obs", "observation", "output", "stdout", "stderr", "result",
        "cwd", "returncode", "exit_code", "status",
    ]
    body_parts = []
    used = {"src", "role", "speaker", "type"}
    for key in body_keys:
        if key not in step:
            continue
        used.add(key)
        rendered = render_value(step[key])
        if rendered:
            body_parts.append(f"{key}: {rendered}")
    for key in sorted(step.keys()):
        if key in used:
            continue
        rendered = render_value(step[key])
        if rendered:
            body_parts.append(f"{key}: {rendered}")
    body = "\n".join(body_parts).strip()
    if body:
        return f"{str(actor).upper()}\n{body}"
    return str(actor).upper()


def extract_steps_from_row(row: dict) -> list:
    for key in [
        "steps",
        "trajectory",
        "messages",
        "history",
        "events",
        "trace",
        "conversation",
    ]:
        if key in row:
            steps = normalize_steps(row[key])
            if steps:
                return steps
    return []


def row_name(row: dict, idx: int) -> str:
    for key in [
        "name",
        "task_name",
        "trial_name",
        "instance_id",
        "trial_id",
        "id",
        "problem_id",
    ]:
        value = row.get(key)
        if value:
            return str(value)
    return f"row_{idx:05d}"


def row_id(row: dict, idx: int) -> str:
    for key in [
        "id",
        "trial_id",
        "instance_id",
        "problem_id",
        "name",
        "task_name",
        "trial_name",
    ]:
        value = row.get(key)
        if value:
            return str(value)
    return f"row_{idx:05d}"


def build_transcript(row: dict) -> tuple[list[str], str]:
    steps = extract_steps_from_row(row)
    rendered_steps = [render_step(step) for step in steps]
    rendered_steps = [s for s in rendered_steps if s.strip()]
    transcript = "\n\n".join(rendered_steps).strip()
    return rendered_steps, transcript


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


def select_candidates(dataset, tokenizer, required_tokens: int, min_steps: int, max_rows: int) -> list[dict]:
    candidates = []
    for idx, row in enumerate(dataset):
        if idx >= max_rows:
            break
        steps, transcript = build_transcript(row)
        if len(steps) < min_steps or not transcript:
            continue
        token_ids = encode(tokenizer, transcript)
        if len(token_ids) < required_tokens:
            continue
        candidates.append(
            {
                "row_index": idx,
                "row_id": row_id(row, idx),
                "row_name": row_name(row, idx),
                "num_steps": len(steps),
                "token_ids": token_ids,
                "total_tokens": len(token_ids),
                "transcript": transcript,
            }
        )
    candidates.sort(key=lambda x: (x["total_tokens"], x["num_steps"]), reverse=True)
    return candidates


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Build a multi-session external prefix-cache workload from real Terminal-Bench 2.0 trajectories. "
            "Seed requests prime the external cache; later reuse rounds replay the same real trajectory prefixes "
            "with only a small chunk-aligned suffix appended per turn."
        )
    )
    ap.add_argument("--model_or_tokenizer", required=True)
    ap.add_argument("--dataset_name", default="yoonholee/terminalbench-trajectories")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset_cache_dir", default="")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max_rows_to_scan", type=int, default=400)
    ap.add_argument("--num_sessions", type=int, default=4)
    ap.add_argument("--reuse_turns_per_session", type=int, default=5)
    ap.add_argument("--seed_prompt_tokens", type=int, default=24576)
    ap.add_argument("--append_tokens", default="256")
    ap.add_argument("--decode_tokens", type=int, default=16)
    ap.add_argument("--max_prompt_tokens", type=int, default=32768)
    ap.add_argument("--chunk_size_tokens", type=int, default=256)
    ap.add_argument("--min_steps_per_trajectory", type=int, default=12)
    ap.add_argument("--selected_rows_out", default="")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_or_tokenizer)

    append_targets = parse_append_tokens(args.append_tokens, args.reuse_turns_per_session)
    total_targets = [args.seed_prompt_tokens]
    for append_tokens in append_targets:
        total_targets.append(total_targets[-1] + append_tokens)

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

    dataset = load_terminalbench_split(
        args.dataset_name,
        args.split,
        args.dataset_cache_dir,
        args.streaming,
    )
    candidates = select_candidates(
        dataset,
        tokenizer,
        total_targets[-1],
        args.min_steps_per_trajectory,
        args.max_rows_to_scan,
    )
    if len(candidates) < args.num_sessions:
        raise SystemExit(
            f"only found {len(candidates)} Terminal-Bench trajectories with enough content; "
            f"need {args.num_sessions}. Try increasing --max_rows_to_scan or lowering seed/turn targets."
        )

    selected = candidates[: args.num_sessions]
    rows = []
    selected_rows = []
    for sess_idx, candidate in enumerate(selected):
        session_id = f"tb_session_{sess_idx:02d}"
        prev_text = ""
        prev_prompt_tokens = 0
        for turn_id, target_prompt_tokens in enumerate(total_targets):
            prompt_text = find_exact_prefix_text(
                tokenizer,
                candidate["token_ids"],
                target_prompt_tokens,
                prev_text,
            )
            actual_prompt_tokens = len(encode(tokenizer, prompt_text))
            if actual_prompt_tokens != target_prompt_tokens:
                raise SystemExit(
                    f"constructed prompt for session {session_id} turn {turn_id} has {actual_prompt_tokens} tokens, "
                    f"expected {target_prompt_tokens}"
                )

            phase = "seed" if turn_id == 0 else "reuse"
            appended_tokens = actual_prompt_tokens - prev_prompt_tokens
            reused_prefix_tokens = prev_prompt_tokens if turn_id > 0 else 0
            reuse_ratio = (reused_prefix_tokens / actual_prompt_tokens) if actual_prompt_tokens > 0 else 0.0
            dispatch_group = (
                f"seed_{session_id}"
                if phase == "seed"
                else f"reuse_round_{turn_id:03d}"
            )

            rows.append(
                {
                    "request_id": f"{session_id}_turn_{turn_id:03d}_{phase}",
                    "phase": phase,
                    "dispatch_group": dispatch_group,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "prompt_tokens": actual_prompt_tokens,
                    "reused_prefix_tokens_est": reused_prefix_tokens,
                    "appended_tokens_est": appended_tokens,
                    "reuse_ratio_est": reuse_ratio,
                    "expected_external_hit": 1 if turn_id > 0 else 0,
                    "max_tokens": args.decode_tokens,
                    "source_dataset": args.dataset_name,
                    "source_row_index": candidate["row_index"],
                    "source_row_id": candidate["row_id"],
                    "source_row_name": candidate["row_name"],
                    "source_num_steps": candidate["num_steps"],
                    "source_total_tokens": candidate["total_tokens"],
                    "prompt_text": prompt_text,
                }
            )
            prev_text = prompt_text
            prev_prompt_tokens = actual_prompt_tokens

        selected_rows.append(
            {
                "session_id": session_id,
                "source_dataset": args.dataset_name,
                "source_row_index": candidate["row_index"],
                "source_row_id": candidate["row_id"],
                "source_row_name": candidate["row_name"],
                "source_num_steps": candidate["num_steps"],
                "source_total_tokens": candidate["total_tokens"],
            }
        )

    dispatch_group_sizes = {}
    for row in rows:
        dispatch_group_sizes[row["dispatch_group"]] = dispatch_group_sizes.get(row["dispatch_group"], 0) + 1
    for row in rows:
        row["dispatch_group_size"] = dispatch_group_sizes[row["dispatch_group"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.selected_rows_out:
        selected_path = Path(args.selected_rows_out)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        with selected_path.open("w", encoding="utf-8") as fp:
            for row in selected_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"wrote {out_path} with {len(rows)} requests, "
        f"{args.num_sessions} sessions, {args.reuse_turns_per_session} reuse rounds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
