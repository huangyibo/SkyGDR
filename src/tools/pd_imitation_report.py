#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def fmt(x: float) -> str:
    return f"{x:.3f}"


def to_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except Exception:
        return default


def unique_group_rows(request_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {}
    for row in request_rows:
        key = row.get("dispatch_group") or row["request_id"]
        if key not in seen:
            seen[key] = row
    return list(seen.values())


def build_report(results_dir: str, out_path: str) -> None:
    summary_dir = os.path.join(results_dir, "summary")
    request_rows = read_csv(os.path.join(summary_dir, "request_pcie_summary.csv"))
    with open(os.path.join(summary_dir, "pcie_timeline_summary.json")) as f:
        pcie_summary = json.load(f)

    seed = [r for r in request_rows if r["phase"] == "seed"]
    reuse = [r for r in request_rows if r["phase"] == "reuse"]
    group_rows = unique_group_rows(request_rows)
    seed_groups = [r for r in group_rows if r["phase"] == "seed"]
    reuse_groups = [r for r in group_rows if r["phase"] == "reuse"]

    reuse_ratios = [to_float(r["reuse_ratio_est"]) for r in reuse]
    reuse_rx_peaks = [to_float(r["peak_rx_GiB_s"]) for r in reuse]
    reuse_tx_peaks = [to_float(r["peak_tx_GiB_s"]) for r in reuse]
    seed_tx_peaks = [to_float(r["peak_tx_GiB_s"]) for r in seed]

    reuse_group_reads = [to_float(r.get("group_lmcache_remote_read_GiB", r.get("lmcache_remote_read_GiB", "-1")), default=-1.0) for r in reuse_groups]
    reuse_group_reads = [v for v in reuse_group_reads if v >= 0]
    reuse_group_hit_ratios = [to_float(r.get("group_lmcache_hit_ratio", r.get("lmcache_hit_ratio", "-1")), default=-1.0) for r in reuse_groups]
    reuse_group_hit_ratios = [v for v in reuse_group_hit_ratios if v >= 0]
    seed_group_writes = [to_float(r.get("group_lmcache_remote_write_GiB", r.get("lmcache_remote_write_GiB", "-1")), default=-1.0) for r in seed_groups]
    seed_group_writes = [v for v in seed_group_writes if v >= 0]

    rel = lambda p: os.path.relpath(p, os.path.dirname(out_path))
    total_svg = os.path.join(summary_dir, "pcie_timeline.svg")
    tx_svg = os.path.join(summary_dir, "pcie_tx_timeline.svg")
    rx_svg = os.path.join(summary_dir, "pcie_rx_timeline.svg")
    zoom_svg = os.path.join(summary_dir, "pcie_request_zooms.svg")

    max_group = max((int(r.get("dispatch_group_size") or 1) for r in request_rows), default=1)
    best_reuse = max(reuse, key=lambda r: to_float(r["peak_rx_GiB_s"])) if reuse else None

    lines = []
    lines.append("# External Prefix-Cache Imitation Report")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("这份结果面向的是：")
    lines.append("")
    lines.append("- 使用真实 Terminal-Bench 2.0 trajectories 构造多 session agent workload")
    lines.append("- 先用 `seed` requests 把长历史前缀写入 LMCache external/shared cache")
    lines.append("- 再按 `reuse_round_*` 并发发出多个高复用 turn，把 aggregate prefill load 尽量打满")
    lines.append("- 直接看 prefill 侧 external read 和 PCIe RX/H2D 是否被持续抬高")
    lines.append("")
    lines.append("## 2. 工作负载概况")
    lines.append("")
    lines.append(f"- requests: `{len(request_rows)}`")
    lines.append(f"- dispatch groups: `{len(group_rows)}`")
    lines.append(f"- max concurrent requests per group: `{max_group}`")
    lines.append(f"- seed requests: `{len(seed)}`")
    lines.append(f"- reuse requests: `{len(reuse)}`")
    if reuse:
        lines.append(f"- mean text-side reuse ratio: `{mean(reuse_ratios) * 100.0:.2f}%`")
        lines.append(f"- mean request peak RX: `{fmt(mean(reuse_rx_peaks))} GiB/s`")
        lines.append(f"- mean request peak TX: `{fmt(mean(reuse_tx_peaks))} GiB/s`")
    if reuse_groups:
        lines.append(f"- mean reuse-group remote read: `{fmt(mean(reuse_group_reads))} GiB/group`")
        lines.append(f"- total reuse-group remote read: `{fmt(sum(reuse_group_reads))} GiB`")
        lines.append(f"- mean reuse-group LMCache hit ratio: `{mean(reuse_group_hit_ratios) * 100.0:.2f}%`")
    if seed_groups:
        lines.append(f"- total seed-group remote write: `{fmt(sum(seed_group_writes))} GiB`")
    lines.append("")
    lines.append("说明：")
    lines.append("")
    lines.append("- 当前 `lmcache_*` 的请求级字段不再作为主真值；并发模式下，LMCache metrics 以 `dispatch_group` 为归因单位。")
    lines.append("- 因此最重要的是 `group_lmcache_remote_read_GiB`、`group_lmcache_hit_ratio` 和 PCIe RX 图。")
    lines.append("")
    lines.append("## 3. 全局 PCIe 图")
    lines.append("")
    lines.append(f"![total timeline]({rel(total_svg)})")
    lines.append("")
    lines.append(f"![tx timeline]({rel(tx_svg)})")
    lines.append("")
    lines.append(f"![rx timeline]({rel(rx_svg)})")
    lines.append("")
    lines.append(f"![request zooms]({rel(zoom_svg)})")
    lines.append("")
    lines.append("## 4. 分阶段统计")
    lines.append("")
    lines.append("| phase | duration (s) | total transfer (GiB) | avg TX GiB/s | avg RX GiB/s | peak TX GiB/s | peak RX GiB/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for phase in ["seed", "reuse"]:
        st = pcie_summary["phases"].get(phase)
        if not st:
            continue
        lines.append(
            f"| {phase} | {st['duration_s']:.3f} | {st['total_transfer_GiB']:.3f} | {st['avg_tx_GiB_s']:.3f} | {st['avg_rx_GiB_s']:.3f} | {st['peak_tx_GiB_s']:.3f} | {st['peak_rx_GiB_s']:.3f} |"
        )
    lines.append("")
    lines.append("## 5. dispatch-group 级摘要")
    lines.append("")
    lines.append("| dispatch_group | phase | size | group read GiB | group write GiB | group hit ratio |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in group_rows:
        lines.append(
            f"| {row.get('dispatch_group', row['request_id'])} | {row['phase']} | {row.get('dispatch_group_size', '1')} | "
            f"{to_float(row.get('group_lmcache_remote_read_GiB', row.get('lmcache_remote_read_GiB', '-1'))):.3f} | "
            f"{to_float(row.get('group_lmcache_remote_write_GiB', row.get('lmcache_remote_write_GiB', '-1'))):.3f} | "
            f"{to_float(row.get('group_lmcache_hit_ratio', row.get('lmcache_hit_ratio', '-1'))) * 100.0:.2f}% |"
        )
    lines.append("")
    lines.append("## 6. 请求级摘要")
    lines.append("")
    lines.append("| request_id | phase | session | turn | prompt tokens | reuse ratio | peak RX GiB/s | peak TX GiB/s | elapsed ms |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in request_rows:
        lines.append(
            f"| {row['request_id']} | {row['phase']} | {row['session_id']} | {row['turn_id']} | "
            f"{row['prompt_tokens']} | {to_float(row['reuse_ratio_est']) * 100.0:.2f}% | "
            f"{to_float(row['peak_rx_GiB_s']):.3f} | {to_float(row['peak_tx_GiB_s']):.3f} | {to_float(row['elapsed_ms']):.2f} |"
        )
    lines.append("")
    lines.append("## 7. 如何解读")
    lines.append("")
    lines.append("- `seed` 阶段看的是首轮长上下文如何写入 external cache，因此更容易偏向 `TX / remote write`。")
    lines.append("- `reuse` 阶段看的是多 session 并发 prefill 如何从 external cache 拉历史 KV，因此更应该看 `group_lmcache_remote_read_GiB` 与 `RX`。")
    lines.append("- 如果你要判断 prefill 是否被持续打满，优先看 `reuse` 的 aggregate RX，而不是单个请求的短 burst。")
    if best_reuse:
        lines.append("")
        lines.append(
            f"当前请求级最强的 RX 峰出现在 `{best_reuse['request_id']}`，"
            f"`peak RX = {to_float(best_reuse['peak_rx_GiB_s']):.3f} GiB/s`。"
        )
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an external prefix-cache imitation report from one results directory.")
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    summary_dir = os.path.join(args.results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    out_path = os.path.join(summary_dir, "pd_imitation_report.md")
    build_report(args.results_dir, out_path)
    print(f"[ok] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
