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
    data_dir = os.path.join(results_dir, "data")
    summary_dir = os.path.join(results_dir, "summary")
    request_rows = read_csv(os.path.join(data_dir, "trajectory_samples.csv"))
    pcie_rows = read_csv(os.path.join(summary_dir, "request_pcie_summary.csv"))
    with open(os.path.join(summary_dir, "pcie_timeline_summary.json")) as f:
        pcie_summary = json.load(f)

    req_idx = {r["request_id"]: r for r in request_rows}
    joined = []
    for row in pcie_rows:
        base = req_idx.get(row["request_id"], {})
        joined.append({**base, **row})

    warmup = [r for r in joined if r["phase"] == "warmup"]
    pressure = [r for r in joined if r["phase"] == "pressure"]
    reuse = [r for r in joined if r["phase"] == "reuse"]

    reuse_rx_peaks = [float(r["peak_rx_GiB_s"]) for r in reuse]
    reuse_tx_peaks = [float(r["peak_tx_GiB_s"]) for r in reuse]
    pressure_tx_peaks = [float(r["peak_tx_GiB_s"]) for r in pressure]
    reuse_ratios = [float(r["reuse_ratio_est"]) for r in reuse]

    best_reuse_rx = max(reuse, key=lambda r: float(r["peak_rx_GiB_s"])) if reuse else None

    rel = lambda p: os.path.relpath(p, os.path.dirname(out_path))
    total_svg = os.path.join(summary_dir, "pcie_timeline.svg")
    tx_svg = os.path.join(summary_dir, "pcie_tx_timeline.svg")
    rx_svg = os.path.join(summary_dir, "pcie_rx_timeline.svg")

    lines = []
    lines.append("# Prefill-Restore Imitation Report")
    lines.append("")
    lines.append("## 1. 实验目标")
    lines.append("")
    lines.append("这份结果不再关注旧的 `prefill-only / decode-only / logical trace` 主流程，而是专门针对：")
    lines.append("")
    lines.append("- 多轮 trajectory")
    lines.append("- 高 prefix reuse")
    lines.append("- 压力请求触发 eviction")
    lines.append("- 观察下一轮 prefill 是否出现明显 RX/H2D restore")
    lines.append("")
    lines.append("## 2. 关键结果")
    lines.append("")
    lines.append(f"- warmup requests: `{len(warmup)}`")
    lines.append(f"- pressure requests: `{len(pressure)}`")
    lines.append(f"- reuse requests: `{len(reuse)}`")
    if reuse:
        lines.append(f"- mean reuse ratio over reuse turns: `{mean(reuse_ratios) * 100.0:.2f}%`")
        lines.append(f"- mean reuse peak RX: `{fmt(mean(reuse_rx_peaks))} GiB/s`")
        lines.append(f"- mean reuse peak TX: `{fmt(mean(reuse_tx_peaks))} GiB/s`")
    if pressure:
        lines.append(f"- mean pressure peak TX: `{fmt(mean(pressure_tx_peaks))} GiB/s`")
    lines.append("")
    if best_reuse_rx:
        lines.append("最值得先看的 reuse turn：")
        lines.append("")
        lines.append(
            f"- request `{best_reuse_rx['request_id']}`: "
            f"peak RX `{fmt(float(best_reuse_rx['peak_rx_GiB_s']))} GiB/s`, "
            f"peak TX `{fmt(float(best_reuse_rx['peak_tx_GiB_s']))} GiB/s`, "
            f"reuse ratio `{float(best_reuse_rx['reuse_ratio_est']) * 100.0:.2f}%`"
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
    for phase in ["warmup", "pressure", "reuse"]:
        st = pcie_summary["phases"].get(phase)
        if not st:
            continue
        lines.append(
            f"| {phase} | {st['duration_s']:.3f} | {st['total_transfer_GiB']:.3f} | {st['avg_tx_GiB_s']:.3f} | {st['avg_rx_GiB_s']:.3f} | {st['peak_tx_GiB_s']:.3f} | {st['peak_rx_GiB_s']:.3f} |"
        )
    lines.append("")
    lines.append("## 5. reuse 请求级摘要")
    lines.append("")
    lines.append("| request_id | turn | prompt tokens | reuse ratio | elapsed ms | tx total GiB | rx total GiB | peak tx GiB/s | peak rx GiB/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in reuse:
        lines.append(
            f"| {row['request_id']} | {row['turn_id']} | {row['prompt_tokens']} | {float(row['reuse_ratio_est']) * 100.0:.2f}% | {float(row['elapsed_ms']):.2f} | {float(row['tx_total_GiB']):.3f} | {float(row['rx_total_GiB']):.3f} | {float(row['peak_tx_GiB_s']):.3f} | {float(row['peak_rx_GiB_s']):.3f} |"
        )
    lines.append("")
    lines.append("## 6. 如何解读")
    lines.append("")
    lines.append("- `pressure` 阶段的主指标是 `peak TX` 和 `tx total`，它反映 eviction 压力是否足够强。")
    lines.append("- `reuse` 阶段的主指标是 `peak RX` 和 `rx total`，它反映 prefill restore 是否真的被打出来。")
    lines.append("- 如果 `reuse ratio` 很高但 `reuse peak RX` 仍然很低，最常见的解释是历史 prefix 还留在 GPU，或者 restore 被更细粒度地摊平了。")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a restore-focused imitation report from one results directory.")
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
