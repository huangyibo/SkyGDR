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


def build_report(results_dir: str, out_path: str) -> None:
    summary_dir = os.path.join(results_dir, "summary")
    request_rows = read_csv(os.path.join(summary_dir, "request_pcie_summary.csv"))
    with open(os.path.join(summary_dir, "pcie_timeline_summary.json")) as f:
        pcie_summary = json.load(f)

    seed = [r for r in request_rows if r["phase"] == "seed"]
    reuse = [r for r in request_rows if r["phase"] == "reuse"]

    reuse_hit_ratios = [float(r["lmcache_hit_ratio"]) for r in reuse]
    reuse_remote_reads = [float(r["lmcache_remote_read_GiB"]) for r in reuse]
    reuse_remote_writes = [float(r["lmcache_remote_write_GiB"]) for r in reuse]
    reuse_rx_peaks = [float(r["peak_rx_GiB_s"]) for r in reuse]
    reuse_tx_peaks = [float(r["peak_tx_GiB_s"]) for r in reuse]
    reuse_ratios = [float(r["reuse_ratio_est"]) for r in reuse]

    seed_remote_writes = [float(r["lmcache_remote_write_GiB"]) for r in seed]
    seed_tx_peaks = [float(r["peak_tx_GiB_s"]) for r in seed]

    best_reuse_read = max(reuse, key=lambda r: float(r["lmcache_remote_read_GiB"])) if reuse else None

    rel = lambda p: os.path.relpath(p, os.path.dirname(out_path))
    total_svg = os.path.join(summary_dir, "pcie_timeline.svg")
    tx_svg = os.path.join(summary_dir, "pcie_tx_timeline.svg")
    rx_svg = os.path.join(summary_dir, "pcie_rx_timeline.svg")

    lines = []
    lines.append("# External Prefix-Cache Imitation Report")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("这份结果专门针对：")
    lines.append("")
    lines.append("- 单 GPU 条件下的 external/shared prefix-cache imitation")
    lines.append("- 首轮长 prompt 先写入 LMCache 外部后端")
    lines.append("- 后续每轮只追加少量 chunk-aligned token")
    lines.append("- 观察 prefill 是否从外部 prefix cache 读回大部分历史 KV")
    lines.append("")
    lines.append("当前主流程显式关闭了 vLLM 自己的 GPU prefix caching，所以重复前缀不会被 GPU 本地缓存遮住；")
    lines.append("reuse turn 的主命中路径应该来自 LMCache 外部后端。")
    lines.append("")
    lines.append("## 2. 关键结果")
    lines.append("")
    lines.append(f"- seed requests: `{len(seed)}`")
    lines.append(f"- reuse requests: `{len(reuse)}`")
    if reuse:
        lines.append(f"- mean estimated text-side reuse ratio: `{mean(reuse_ratios) * 100.0:.2f}%`")
        lines.append(f"- mean LMCache hit ratio: `{mean(reuse_hit_ratios) * 100.0:.2f}%`")
        lines.append(f"- total reuse remote read volume: `{fmt(sum(reuse_remote_reads))} GiB`")
        lines.append(f"- total reuse remote write volume: `{fmt(sum(reuse_remote_writes))} GiB`")
        lines.append(f"- mean reuse remote read volume: `{fmt(mean(reuse_remote_reads))} GiB/request`")
        lines.append(f"- mean reuse peak RX: `{fmt(mean(reuse_rx_peaks))} GiB/s`")
        lines.append(f"- mean reuse peak TX: `{fmt(mean(reuse_tx_peaks))} GiB/s`")
    if seed:
        lines.append(f"- seed remote write volume: `{fmt(sum(seed_remote_writes))} GiB`")
        lines.append(f"- mean seed peak TX: `{fmt(mean(seed_tx_peaks))} GiB/s`")
    lines.append("")
    if best_reuse_read:
        lines.append("最值得先看的 reuse turn：")
        lines.append("")
        lines.append(
            f"- request `{best_reuse_read['request_id']}`: "
            f"remote read `{fmt(float(best_reuse_read['lmcache_remote_read_GiB']))} GiB`, "
            f"hit ratio `{float(best_reuse_read['lmcache_hit_ratio']) * 100.0:.2f}%`, "
            f"peak RX `{fmt(float(best_reuse_read['peak_rx_GiB_s']))} GiB/s`, "
            f"peak TX `{fmt(float(best_reuse_read['peak_tx_GiB_s']))} GiB/s`"
        )
        lines.append("")
    lines.append("## 3. 全局 PCIe 图")
    lines.append("")
    lines.append(f"![total timeline]({rel(total_svg)})")
    lines.append("")
    lines.append(f"![tx timeline]({rel(tx_svg)})")
    lines.append("")
    lines.append(f"![rx timeline]({rel(rx_svg)})")
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
    lines.append("## 5. turn 级摘要")
    lines.append("")
    lines.append("| request_id | phase | turn | prompt tokens | reuse ratio | LMCache hit ratio | remote read GiB | remote write GiB | peak RX GiB/s | peak TX GiB/s | elapsed ms |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in request_rows:
        lines.append(
            f"| {row['request_id']} | {row['phase']} | {row['turn_id']} | {row['prompt_tokens']} | "
            f"{float(row['reuse_ratio_est']) * 100.0:.2f}% | {float(row['lmcache_hit_ratio']) * 100.0:.2f}% | "
            f"{float(row['lmcache_remote_read_GiB']):.3f} | {float(row['lmcache_remote_write_GiB']):.3f} | "
            f"{float(row['peak_rx_GiB_s']):.3f} | {float(row['peak_tx_GiB_s']):.3f} | {float(row['elapsed_ms']):.2f} |"
        )
    lines.append("")
    lines.append("## 6. 如何解读")
    lines.append("")
    lines.append("- `seed` 阶段的关键动作是把首轮长 prompt 的 KV 写入外部 prefix cache，所以它通常更偏向 `remote write + TX`。")
    lines.append("- `reuse` 阶段的关键动作是从外部 prefix cache 读回历史 KV，只对新增 suffix 做新的 prefill，所以它应该更值得看 `remote read + RX`。")
    lines.append("- 如果 `LMCache hit ratio` 已经很高，但 `peak RX` 仍不高，通常说明外部读回被平滑摊开了，而不是命中没发生。")
    lines.append("- 如果 `LMCache remote write GiB` 在 reuse turn 仍然明显偏大，往往表示每轮新增 suffix 太长，或者还在写入很多非复用块。")
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
