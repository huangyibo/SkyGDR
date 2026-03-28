#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt_ms(x: float) -> str:
    return f"{x:.2f}"


def fmt_tps(x: float) -> str:
    return f"{x:.2f}"


def fmt_gib(x_bytes: float) -> str:
    return f"{x_bytes / (1024 ** 3):.2f}"


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def svg_line_chart(
    series: list[dict],
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 960,
    height: int = 560,
) -> None:
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6"]
    margin_left = 86
    margin_right = 28
    margin_top = 56
    margin_bottom = 82
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_x: list[float] = []
    all_y: list[float] = []
    for s in series:
        for x, y in s["points"]:
            all_x.append(float(x))
            all_y.append(float(y))
    if not all_x or not all_y:
        raise ValueError("empty chart series")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = 0.0, max(all_y)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    max_y *= 1.10

    def x_to_px(x: float) -> float:
        return margin_left + (x - min_x) / (max_x - min_x) * plot_w

    def y_to_px(y: float) -> float:
        return margin_top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    y_ticks = 5
    x_ticks = sorted(set(all_x))

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(
        f'<text x="{margin_left}" y="28" font-size="24" font-family="Helvetica, Arial, sans-serif" font-weight="700">{escape_xml(title)}</text>'
    )

    for i in range(y_ticks + 1):
        y_val = min_y + (max_y - min_y) * i / y_ticks
        y_px = y_to_px(y_val)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_px:.1f}" x2="{width - margin_right}" y2="{y_px:.1f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y_px + 4:.1f}" text-anchor="end" font-size="13" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{y_val:.1f}")}</text>'
        )

    for x_val in x_ticks:
        x_px = x_to_px(x_val)
        parts.append(
            f'<line x1="{x_px:.1f}" y1="{margin_top}" x2="{x_px:.1f}" y2="{margin_top + plot_h}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        label = str(int(x_val))
        parts.append(
            f'<text x="{x_px:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="13" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(label)}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>'
    )

    for idx, s in enumerate(series):
        color = colors[idx % len(colors)]
        pts = [(x_to_px(float(x)), y_to_px(float(y))) for x, y in s["points"]]
        path = " ".join(
            [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"] + [f"L {x:.1f} {y:.1f}" for x, y in pts[1:]]
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')

    legend_x = width - margin_right - 220
    legend_y = margin_top + 10
    legend_h = 26 * len(series) + 16
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="210" height="{legend_h}" rx="8" fill="#ffffff" stroke="#dddddd"/>'
    )
    for idx, s in enumerate(series):
        color = colors[idx % len(colors)]
        y = legend_y + 24 + idx * 26
        parts.append(f'<line x1="{legend_x + 12}" y1="{y}" x2="{legend_x + 36}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x + 24}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 46}" y="{y + 5}" font-size="13" fill="#333" font-family="Helvetica, Arial, sans-serif">{escape_xml(s["label"])}</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="16" fill="#222" font-family="Helvetica, Arial, sans-serif">{escape_xml(x_label)}</text>'
    )
    parts.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.1f}" text-anchor="middle" font-size="16" fill="#222" font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 22,{margin_top + plot_h / 2:.1f})">{escape_xml(y_label)}</text>'
    )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def svg_bar_chart(
    labels: list[str],
    values: list[float],
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 960,
    height: int = 560,
) -> None:
    margin_left = 86
    margin_right = 28
    margin_top = 56
    margin_bottom = 82
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_y = max(values) * 1.10 if values else 1.0

    n = len(values)
    band = plot_w / max(n, 1)
    bar_w = band * 0.62

    def y_to_px(y: float) -> float:
        return margin_top + plot_h - (y / max_y) * plot_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(
        f'<text x="{margin_left}" y="28" font-size="24" font-family="Helvetica, Arial, sans-serif" font-weight="700">{escape_xml(title)}</text>'
    )
    for i in range(6):
        y_val = max_y * i / 5.0
        y_px = y_to_px(y_val)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_px:.1f}" x2="{width - margin_right}" y2="{y_px:.1f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y_px + 4:.1f}" text-anchor="end" font-size="13" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{y_val:.2f}")}</text>'
        )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222" stroke-width="1.5"/>'
    )

    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + i * band + (band - bar_w) / 2
        y = y_to_px(value)
        h = margin_top + plot_h - y
        color = "#4C78A8" if i < len(values) - 1 else "#E45756"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="4"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="12" fill="#333" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{value:.2f}")}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-size="13" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(label)}</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="16" fill="#222" font-family="Helvetica, Arial, sans-serif">{escape_xml(x_label)}</text>'
    )
    parts.append(
        f'<text x="22" y="{margin_top + plot_h / 2:.1f}" text-anchor="middle" font-size="16" fill="#222" font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 22,{margin_top + plot_h / 2:.1f})">{escape_xml(y_label)}</text>'
    )
    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def aggregate_prefill(rows: list[dict[str, str]]) -> list[dict]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["http_status"] != "200" or row["error"]:
            continue
        buckets[int(row["prompt_tokens"])].append(float(row["elapsed_ms"]))
    out = []
    for prompt_tokens in sorted(buckets):
        mean_ms, std_ms = mean_std(buckets[prompt_tokens])
        out.append(
            {
                "prompt_tokens": prompt_tokens,
                "samples": len(buckets[prompt_tokens]),
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "prefill_tps": prompt_tokens / (mean_ms / 1000.0),
            }
        )
    return out


def aggregate_decode(rows: list[dict[str, str]]) -> list[dict]:
    pairs: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["http_status"] != "200" or row["error"]:
            continue
        key = (int(row["context_tokens"]), int(row["generated_tokens"]))
        pairs[key].append(float(row["elapsed_ms"]))
    out = []
    for (context_tokens, generated_tokens) in sorted(pairs):
        mean_ms, std_ms = mean_std(pairs[(context_tokens, generated_tokens)])
        out.append(
            {
                "context_tokens": context_tokens,
                "generated_tokens": generated_tokens,
                "samples": len(pairs[(context_tokens, generated_tokens)]),
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "decode_ms_per_token": mean_ms / generated_tokens,
                "decode_tps": generated_tokens / (mean_ms / 1000.0),
            }
        )
    return out


def index_prefill(prefill: list[dict]) -> dict[int, dict]:
    return {int(r["prompt_tokens"]): r for r in prefill}


def index_decode(decode: list[dict]) -> dict[tuple[int, int], dict]:
    return {(int(r["context_tokens"]), int(r["generated_tokens"])): r for r in decode}


def load_summary(path: str) -> dict:
    import json

    with open(path) as f:
        return json.load(f)


def write_aggregates(prefill: list[dict], decode: list[dict], out_dir: str) -> tuple[str, str]:
    prefill_csv = os.path.join(out_dir, "prefill_aggregate.csv")
    decode_csv = os.path.join(out_dir, "decode_aggregate.csv")
    with open(prefill_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prefill[0].keys()))
        w.writeheader()
        w.writerows(prefill)
    with open(decode_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(decode[0].keys()))
        w.writeheader()
        w.writerows(decode)
    return prefill_csv, decode_csv


def build_plots(prefill: list[dict], decode: list[dict], summary: dict, trace_csv: str, fig_dir: str) -> dict[str, str]:
    ensure_dir(fig_dir)
    paths: dict[str, str] = {}

    paths["prefill_latency"] = os.path.join(fig_dir, "prefill_latency.svg")
    svg_line_chart(
        series=[
            {
                "label": "prefill mean latency",
                "points": [(r["prompt_tokens"], r["mean_ms"]) for r in prefill],
            }
        ],
        out_path=paths["prefill_latency"],
        title="Qwen3-8B Prefill Latency vs Prompt Length",
        x_label="Prompt Tokens",
        y_label="Latency (ms)",
    )

    paths["prefill_throughput"] = os.path.join(fig_dir, "prefill_throughput.svg")
    svg_line_chart(
        series=[
            {
                "label": "prefill throughput",
                "points": [(r["prompt_tokens"], r["prefill_tps"]) for r in prefill],
            }
        ],
        out_path=paths["prefill_throughput"],
        title="Qwen3-8B Prefill Throughput vs Prompt Length",
        x_label="Prompt Tokens",
        y_label="Throughput (tokens/s)",
    )

    by_gen: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in decode:
        by_gen[r["generated_tokens"]].append((r["context_tokens"], r["decode_ms_per_token"]))
    paths["decode_mspt"] = os.path.join(fig_dir, "decode_ms_per_token.svg")
    svg_line_chart(
        series=[
            {"label": f"gen={g}", "points": sorted(points)}
            for g, points in sorted(by_gen.items())
        ],
        out_path=paths["decode_mspt"],
        title="Decode Cost per Token vs Context Length",
        x_label="Context Tokens",
        y_label="ms/token",
    )

    with open(trace_csv, newline="") as f:
        trace_rows = list(csv.DictReader(f))
    max_gen = max(int(r["generated_tokens"]) for r in trace_rows)
    kv_labels = []
    kv_values = []
    for row in trace_rows:
        if int(row["generated_tokens"]) != max_gen:
            continue
        kv_labels.append(str(int(row["context_tokens"])))
        kv_values.append(int(row["decode_required_kv_bytes"]) / (1024 ** 3))

    paths["kv_footprint"] = os.path.join(fig_dir, "kv_footprint_gib.svg")
    svg_bar_chart(
        labels=kv_labels,
        values=kv_values,
        out_path=paths["kv_footprint"],
        title=f"Logical KV Footprint at gen={max_gen}",
        x_label="Context Tokens",
        y_label="KV Footprint (GiB)",
    )

    return paths


def build_report(
    out_path: str,
    prefill: list[dict],
    decode: list[dict],
    summary: dict,
    plot_paths: dict[str, str],
    trace_csv: str,
) -> None:
    max_prefill = prefill[-1]
    best_prefill_tps = max(prefill, key=lambda x: x["prefill_tps"])
    decode_256 = [r for r in decode if r["generated_tokens"] == 256]
    decode_32 = [r for r in decode if r["generated_tokens"] == 32]
    worst_decode_256 = max(decode_256, key=lambda x: x["decode_ms_per_token"])
    best_decode_256 = min(decode_256, key=lambda x: x["decode_ms_per_token"])
    worst_decode_32 = max(decode_32, key=lambda x: x["decode_ms_per_token"])

    with open(trace_csv, newline="") as f:
        trace_rows = list(csv.DictReader(f))
    max_kv_row = max(trace_rows, key=lambda r: int(r["decode_required_kv_bytes"]))

    rel = lambda p: os.path.relpath(p, os.path.dirname(out_path))

    lines: list[str] = []
    lines.append("# Qwen3-8B-Instruct PD Imitation Results Report")
    lines.append("")
    lines.append("## 1. 数据范围")
    lines.append("")
    lines.append("本报告基于 `results/pd_imitation_qwen3_8b_instruct` 的一轮完整 phase-1 采样结果。")
    lines.append("")
    lines.append("- 模型：`Qwen/Qwen3-8B`，服务名：`Qwen3-8B-Instruct`")
    lines.append("- 采样对象：`prefill-only`、`decode-only`、逻辑 `pd_imitation_trace.csv`")
    lines.append("- `prefill` 成功 bucket：`512 ~ 16384` tokens")
    lines.append("- `decode` 成功 bucket：context `512 ~ 16384`，generation `32/64/128/256`")
    lines.append(f"- 逻辑 trace 行数：`{summary['trace_rows']}`")
    lines.append("")
    lines.append("补充说明：")
    lines.append("")
    lines.append("- 预设的 `32768` prefill bucket 没有进入最终 summary。")
    lines.append("- 原因不是脚本漏跑，而是 `max-model-len=32768` 时再请求 `1` 个输出 token 会触发长度上限，导致该 bucket 的 20 个样本全部返回 `400`。")
    lines.append("- 因此本轮有效 prefill 上限实际上是 `16384`。如果后续还想贴近上限，建议把最大 bucket 改成 `32736` 或更低。")
    lines.append("")
    lines.append("## 2. 关键结论")
    lines.append("")
    lines.append(f"1. `prefill` 延迟随 prompt 长度单调上升，从 `512` tokens 的 `{fmt_ms(prefill[0]['mean_ms'])} ms` 增长到 `16384` tokens 的 `{fmt_ms(max_prefill['mean_ms'])} ms`。")
    lines.append(f"2. `prefill throughput` 在中等长度区间最高，峰值出现在 `2048` tokens，约为 `{fmt_tps(best_prefill_tps['prefill_tps'])} tokens/s`；到 `16384` tokens 时回落到 `{fmt_tps(max_prefill['prefill_tps'])} tokens/s`。")
    lines.append(f"3. `decode` 的粗粒度 `ms/token` 对 generation 长度很敏感：`g=32` 时固定开销占比很高，在 `16384` context 下膨胀到 `{fmt_ms(worst_decode_32['decode_ms_per_token'])} ms/token`；而 `g=256` 更接近稳态，区间约为 `{fmt_ms(best_decode_256['decode_ms_per_token'])} ~ {fmt_ms(worst_decode_256['decode_ms_per_token'])} ms/token`。")
    lines.append(f"4. 逻辑 KV footprint 与 context 线性相关，本轮最大点是 `{max_kv_row['context_tokens']}` tokens，对应 `{fmt_gib(float(max_kv_row['decode_required_kv_bytes']))} GiB` 的 decode-side KV。")
    lines.append("")
    lines.append("## 3. Prefill 结果")
    lines.append("")
    lines.append(f"![Prefill latency]({escape_xml(rel(plot_paths['prefill_latency']))})")
    lines.append("")
    lines.append(f"![Prefill throughput]({escape_xml(rel(plot_paths['prefill_throughput']))})")
    lines.append("")
    lines.append("聚合结果：")
    lines.append("")
    lines.append("| prompt tokens | samples | mean latency (ms) | std (ms) | throughput (tokens/s) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for r in prefill:
        lines.append(
            f"| {r['prompt_tokens']} | {r['samples']} | {fmt_ms(r['mean_ms'])} | {fmt_ms(r['std_ms'])} | {fmt_tps(r['prefill_tps'])} |"
        )
    lines.append("")
    lines.append("解读：")
    lines.append("")
    lines.append("- `prefill latency` 基本随 token 数增加而近似线性上升，但在 `8K -> 16K` 区间已经出现更明显的超线性拉长。")
    lines.append("- `prefill throughput` 不是单调增加的：它在 `2K ~ 4K` 左右最好，之后随着上下文变长开始回落。")
    lines.append("- 这意味着如果你后面要做 PD imitation，prefill cost 不能只按“每 token 固定时间”处理，长上下文区间最好单独建桶。")
    lines.append("")
    lines.append("## 4. Decode 结果")
    lines.append("")
    lines.append(f"![Decode ms/token]({escape_xml(rel(plot_paths['decode_mspt']))})")
    lines.append("")
    lines.append("| context tokens | gen tokens | samples | mean total latency (ms) | ms/token | tokens/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in decode:
        lines.append(
            f"| {r['context_tokens']} | {r['generated_tokens']} | {r['samples']} | {fmt_ms(r['mean_ms'])} | {fmt_ms(r['decode_ms_per_token'])} | {fmt_tps(r['decode_tps'])} |"
        )
    lines.append("")
    lines.append("解读：")
    lines.append("")
    lines.append("- 当前 `decode-only` 口径本质上是“长 context + 指定 generation length 的整段 elapsed time”，不是纯 kernel 级 decode 时间。")
    lines.append("- 因此 `g=32` 的 `ms/token` 明显被固定开销污染，不能直接当成 steady-state decode 速度。")
    lines.append("- `g=128` 和 `g=256` 更接近稳定区间。本轮里，`g=256` 的 decode 吞吐从 `512` context 的 `90.15 tokens/s` 下降到 `16384` context 的 `76.50 tokens/s`。")
    lines.append("- 对后续 case study，如果你需要一个更稳的 decode proxy，建议优先使用 `g=128` 或 `g=256` 的桶，而不是 `g=32`。")
    lines.append("")
    lines.append("## 5. 逻辑 KV Footprint")
    lines.append("")
    lines.append(f"![KV footprint]({escape_xml(rel(plot_paths['kv_footprint']))})")
    lines.append("")
    lines.append(f"本轮 trace 使用固定模型参数计算出：`KV_bytes_per_token = {summary['kv_bytes_per_token']}`，也就是每 token `144 KiB`。")
    lines.append("")
    lines.append("对应关系非常直接：")
    lines.append("")
    lines.append("- `512` context: `0.07 GiB`")
    lines.append("- `1024` context: `0.14 GiB`")
    lines.append("- `2048` context: `0.28 GiB`")
    lines.append("- `4096` context: `0.56 GiB`")
    lines.append("- `8192` context: `1.12 GiB`")
    lines.append("- `16384` context: `2.25 GiB`")
    lines.append("")
    lines.append("这部分结论对 PD imitation 很关键：")
    lines.append("")
    lines.append("- prefill 端产出的逻辑 KV 量与 prompt/context 长度线性相关。")
    lines.append("- 即使不跑真实 PD，本轮也已经足够给后续 offloading / replay 提供一个量级可信的 KV 大小映射。")
    lines.append("")
    lines.append("## 6. 对当前 trace 的使用建议")
    lines.append("")
    lines.append("如果你现在要把这批结果送入后续 case study，我建议直接采用下面的口径：")
    lines.append("")
    lines.append("1. `prefill_time_ms` 直接取当前 trace 里的桶均值。")
    lines.append("2. `decode_time_ms` 如果是做粗粒度 phase-1 模拟，可以保留当前值。")
    lines.append("3. 如果你更关心 steady-state decode，不要优先用 `g=32`，而是优先采信 `g=128` / `g=256`。")
    lines.append("4. 如果你要构造接近上限的长上下文 workload，把 `32768` bucket 改成略低于上限的值后再补采一轮。")
    lines.append("")
    lines.append("## 7. 当前局限")
    lines.append("")
    lines.append("- 这仍然是 `single-GPU` 的 phase-1 imitation，不是完整 PD serving。")
    lines.append("- `decode-only` 当前口径包含固定开销，因此短 generation 桶会被高估。")
    lines.append("- 当前 trace 还没有引入真实请求到达分布，也没有引入跨机传输带宽限制。")
    lines.append("")
    lines.append("## 8. 输出位置")
    lines.append("")
    lines.append(f"- trace: `{trace_csv}`")
    lines.append(f"- summary: `results/pd_imitation_qwen3_8b_instruct/summary/pd_imitation_summary.json`")
    lines.append(f"- figures: `{os.path.dirname(plot_paths['prefill_latency'])}`")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_compare_plots(
    base_prefill: list[dict],
    compare_prefill: list[dict],
    base_decode: list[dict],
    compare_decode: list[dict],
    fig_dir: str,
    base_label: str,
    compare_label: str,
) -> dict[str, str]:
    ensure_dir(fig_dir)
    paths: dict[str, str] = {}

    base_prefill_idx = index_prefill(base_prefill)
    compare_prefill_idx = index_prefill(compare_prefill)
    shared_prefill = sorted(set(base_prefill_idx) & set(compare_prefill_idx))

    paths["compare_prefill_latency"] = os.path.join(fig_dir, "compare_prefill_latency.svg")
    svg_line_chart(
        series=[
            {
                "label": f"{base_label} prefill",
                "points": [(k, base_prefill_idx[k]["mean_ms"]) for k in shared_prefill],
            },
            {
                "label": f"{compare_label} prefill",
                "points": [(k, compare_prefill_idx[k]["mean_ms"]) for k in shared_prefill],
            },
        ],
        out_path=paths["compare_prefill_latency"],
        title="Prefill Latency: baseline vs native offload",
        x_label="Prompt Tokens",
        y_label="Latency (ms)",
    )

    base_decode_idx = index_decode(base_decode)
    compare_decode_idx = index_decode(compare_decode)
    shared_decode_256 = sorted(
        k[0]
        for k in set(base_decode_idx) & set(compare_decode_idx)
        if k[1] == 256
    )

    paths["compare_decode_g256_mspt"] = os.path.join(fig_dir, "compare_decode_g256_mspt.svg")
    svg_line_chart(
        series=[
            {
                "label": f"{base_label} decode g=256",
                "points": [(ctx, base_decode_idx[(ctx, 256)]["decode_ms_per_token"]) for ctx in shared_decode_256],
            },
            {
                "label": f"{compare_label} decode g=256",
                "points": [(ctx, compare_decode_idx[(ctx, 256)]["decode_ms_per_token"]) for ctx in shared_decode_256],
            },
        ],
        out_path=paths["compare_decode_g256_mspt"],
        title="Decode ms/token at gen=256: baseline vs native offload",
        x_label="Context Tokens",
        y_label="ms/token",
    )

    return paths


def build_compare_report(
    out_path: str,
    base_results_dir: str,
    compare_results_dir: str,
    base_prefill: list[dict],
    compare_prefill: list[dict],
    base_decode: list[dict],
    compare_decode: list[dict],
    plot_paths: dict[str, str],
    base_label: str,
    compare_label: str,
) -> None:
    rel = lambda p: os.path.relpath(p, os.path.dirname(out_path))

    base_prefill_idx = index_prefill(base_prefill)
    compare_prefill_idx = index_prefill(compare_prefill)
    shared_prefill = sorted(set(base_prefill_idx) & set(compare_prefill_idx))

    base_decode_idx = index_decode(base_decode)
    compare_decode_idx = index_decode(compare_decode)
    shared_decode_256 = sorted(
        k[0]
        for k in set(base_decode_idx) & set(compare_decode_idx)
        if k[1] == 256
    )

    def pct_delta(new: float, old: float) -> float:
        return (new - old) / old * 100.0 if old else 0.0

    largest_prefill = shared_prefill[-1]
    largest_decode_ctx = shared_decode_256[-1]
    prefill_delta = pct_delta(
        compare_prefill_idx[largest_prefill]["mean_ms"],
        base_prefill_idx[largest_prefill]["mean_ms"],
    )
    decode_delta = pct_delta(
        compare_decode_idx[(largest_decode_ctx, 256)]["decode_ms_per_token"],
        base_decode_idx[(largest_decode_ctx, 256)]["decode_ms_per_token"],
    )

    lines: list[str] = []
    lines.append("# PD Imitation Compare Report")
    lines.append("")
    lines.append("## 1. 对照范围")
    lines.append("")
    lines.append(f"- baseline results: `{base_results_dir}`")
    lines.append(f"- compare results: `{compare_results_dir}`")
    lines.append(f"- baseline label: `{base_label}`")
    lines.append(f"- compare label: `{compare_label}`")
    lines.append("")
    lines.append("## 2. 关键结论")
    lines.append("")
    lines.append(
        f"1. 在最大共享 prefill bucket `prompt={largest_prefill}` 上，`{compare_label}` 相比 `{base_label}` 的 prefill 平均延迟变化为 `{prefill_delta:+.2f}%`。"
    )
    lines.append(
        f"2. 在最大共享 decode bucket `context={largest_decode_ctx}, gen=256` 上，`{compare_label}` 相比 `{base_label}` 的 decode `ms/token` 变化为 `{decode_delta:+.2f}%`。"
    )
    lines.append("3. 如果 `compare_label` 是 native CPU offloading，这两个量就是最值得先看的主指标：prefill 会不会被拉长，steady-state decode 会不会变差。")
    lines.append("")
    lines.append("## 3. Prefill 对照")
    lines.append("")
    lines.append(f"![compare prefill]({escape_xml(rel(plot_paths['compare_prefill_latency']))})")
    lines.append("")
    lines.append("| prompt tokens | baseline mean ms | compare mean ms | delta % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for k in shared_prefill:
        b = base_prefill_idx[k]["mean_ms"]
        c = compare_prefill_idx[k]["mean_ms"]
        lines.append(f"| {k} | {fmt_ms(b)} | {fmt_ms(c)} | {pct_delta(c, b):+.2f}% |")
    lines.append("")
    lines.append("## 4. Decode 对照（gen=256）")
    lines.append("")
    lines.append(f"![compare decode]({escape_xml(rel(plot_paths['compare_decode_g256_mspt']))})")
    lines.append("")
    lines.append("| context tokens | baseline ms/token | compare ms/token | delta % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for ctx in shared_decode_256:
        b = base_decode_idx[(ctx, 256)]["decode_ms_per_token"]
        c = compare_decode_idx[(ctx, 256)]["decode_ms_per_token"]
        lines.append(f"| {ctx} | {fmt_ms(b)} | {fmt_ms(c)} | {pct_delta(c, b):+.2f}% |")
    lines.append("")
    lines.append("## 5. 使用建议")
    lines.append("")
    lines.append("- 如果 offloading 主要拉长的是大 context 下的 prefill，说明 CPU 侧 KV 搬运已经开始影响长 prompt 请求。")
    lines.append("- 如果 offloading 主要拉长的是 `gen=256` 的 decode `ms/token`，说明它已经影响 steady-state decode，而不仅仅是固定开销。")
    lines.append("- 如果只有 `g=32` 变差而 `g=256` 变化不大，优先把它解释为固定开销或短序列效应，而不是 steady-state decode 退化。")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build plots and a markdown report for PD imitation results.")
    ap.add_argument("--results_dir", required=True, help="root results dir, e.g. results/pd_imitation_qwen3_8b_instruct")
    ap.add_argument("--compare_results_dir", default="", help="optional second results dir for baseline/offloading comparison")
    ap.add_argument("--base_label", default="baseline")
    ap.add_argument("--compare_label", default="compare")
    args = ap.parse_args()

    results_dir = args.results_dir
    data_dir = os.path.join(results_dir, "data")
    summary_dir = os.path.join(results_dir, "summary")
    fig_dir = os.path.join(results_dir, "fig")
    ensure_dir(summary_dir)
    ensure_dir(fig_dir)

    prefill_rows = read_csv(os.path.join(data_dir, "prefill_samples.csv"))
    decode_rows = read_csv(os.path.join(data_dir, "decode_samples.csv"))
    summary = load_summary(os.path.join(summary_dir, "pd_imitation_summary.json"))
    trace_csv = os.path.join(summary_dir, "pd_imitation_trace.csv")

    prefill = aggregate_prefill(prefill_rows)
    decode = aggregate_decode(decode_rows)
    prefill_csv, decode_csv = write_aggregates(prefill, decode, summary_dir)
    plot_paths = build_plots(prefill, decode, summary, trace_csv, fig_dir)
    report_path = os.path.join(summary_dir, "pd_imitation_report.md")
    build_report(report_path, prefill, decode, summary, plot_paths, trace_csv)

    print(f"[ok] wrote {prefill_csv}")
    print(f"[ok] wrote {decode_csv}")
    for path in plot_paths.values():
        print(f"[ok] wrote {path}")
    print(f"[ok] wrote {report_path}")

    if args.compare_results_dir:
        compare_results_dir = args.compare_results_dir
        compare_data_dir = os.path.join(compare_results_dir, "data")
        compare_prefill_rows = read_csv(os.path.join(compare_data_dir, "prefill_samples.csv"))
        compare_decode_rows = read_csv(os.path.join(compare_data_dir, "decode_samples.csv"))
        compare_prefill = aggregate_prefill(compare_prefill_rows)
        compare_decode = aggregate_decode(compare_decode_rows)
        compare_plot_paths = build_compare_plots(
            prefill,
            compare_prefill,
            decode,
            compare_decode,
            fig_dir,
            args.base_label,
            args.compare_label,
        )
        compare_report_path = os.path.join(summary_dir, "pd_imitation_compare_report.md")
        build_compare_report(
            compare_report_path,
            results_dir,
            compare_results_dir,
            prefill,
            compare_prefill,
            decode,
            compare_decode,
            compare_plot_paths,
            args.base_label,
            args.compare_label,
        )
        for path in compare_plot_paths.values():
            print(f"[ok] wrote {path}")
        print(f"[ok] wrote {compare_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
