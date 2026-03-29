#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Iterable


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_float(v, default=float("nan")):
    try:
        return float(v)
    except Exception:
        return default


def to_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def load_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_metrics(path: str) -> list[dict]:
    rows = load_csv(path)
    out = []
    for r in rows:
        out.append(
            {
                "ts_unix_ms": to_int(r.get("ts_unix_ms")),
                "t_ms": to_int(r.get("t_ms")),
                "pcie_tx_GiB_s": to_float(r.get("pcie_tx_GiB_s")),
                "pcie_rx_GiB_s": to_float(r.get("pcie_rx_GiB_s")),
                "pcie_total_GiB_s": to_float(r.get("pcie_total_GiB_s")),
                "pcie_tx_util_pct": to_float(r.get("pcie_tx_util_pct")),
                "pcie_rx_util_pct": to_float(r.get("pcie_rx_util_pct")),
                "pcie_total_util_pct": to_float(r.get("pcie_total_util_pct")),
                "pcie_tx_cum_GiB": to_float(r.get("pcie_tx_cum_GiB")),
                "pcie_rx_cum_GiB": to_float(r.get("pcie_rx_cum_GiB")),
                "pcie_total_cum_GiB": to_float(r.get("pcie_total_cum_GiB")),
                "gpu_mem_used_GiB": to_float(r.get("gpu_mem_used_GiB")),
                "gpu_mem_used_pct": to_float(r.get("gpu_mem_used_pct")),
                "util_gpu_pct": to_float(r.get("util_gpu_pct")),
                "util_mem_pct": to_float(r.get("util_mem_pct")),
                "cpu_util_pct": to_float(r.get("cpu_util_pct")),
                "cpu_mem_used_GiB": to_float(r.get("cpu_mem_used_GiB")),
                "cpu_mem_used_pct": to_float(r.get("cpu_mem_used_pct")),
                "pcie_link_ref_GiB_s": to_float(r.get("pcie_link_ref_GiB_s")),
            }
        )
    return out


def load_request_window(prefill_csv: str, decode_csv: str) -> dict[str, int]:
    starts = []
    ends = []
    prefill_rows = load_csv(prefill_csv) if prefill_csv else []
    decode_rows = load_csv(decode_csv) if decode_csv else []

    def window(rows: list[dict[str, str]]) -> tuple[int, int] | None:
        ok = [r for r in rows if r.get("http_status") == "200" and not r.get("error")]
        if not ok:
            return None
        return (
            min(to_int(r["submit_ts_unix_ms"]) for r in ok),
            max(to_int(r["finish_ts_unix_ms"]) for r in ok),
        )

    pw = window(prefill_rows)
    dw = window(decode_rows)
    if pw:
        starts.append(pw[0])
        ends.append(pw[1])
    if dw:
        starts.append(dw[0])
        ends.append(dw[1])
    if not starts:
        raise SystemExit("no successful prefill/decode rows found")

    return {
        "prefill_start_unix_ms": pw[0] if pw else 0,
        "prefill_end_unix_ms": pw[1] if pw else 0,
        "decode_start_unix_ms": dw[0] if dw else 0,
        "decode_end_unix_ms": dw[1] if dw else 0,
        "window_start_unix_ms": min(starts),
        "window_end_unix_ms": max(ends),
    }


def select_rows(rows: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    out = [r for r in rows if r["ts_unix_ms"] >= start_ms and r["ts_unix_ms"] <= end_ms]
    if len(out) >= 2:
        return out
    return rows


def summarize_window(rows: list[dict], start_ms: int, end_ms: int) -> dict:
    sel = select_rows(rows, start_ms, end_ms)
    if not sel:
        return {}
    duration_s = max(0.0, (end_ms - start_ms) / 1000.0)
    first = sel[0]
    last = sel[-1]
    tx_total = max(0.0, last["pcie_tx_cum_GiB"] - first["pcie_tx_cum_GiB"])
    rx_total = max(0.0, last["pcie_rx_cum_GiB"] - first["pcie_rx_cum_GiB"])
    total_total = max(0.0, last["pcie_total_cum_GiB"] - first["pcie_total_cum_GiB"])
    peak_tx = max(sel, key=lambda r: r["pcie_tx_GiB_s"])
    peak_rx = max(sel, key=lambda r: r["pcie_rx_GiB_s"])
    peak_total = max(sel, key=lambda r: r["pcie_total_GiB_s"])
    return {
        "start_unix_ms": start_ms,
        "end_unix_ms": end_ms,
        "duration_s": duration_s,
        "num_samples": len(sel),
        "tx_total_GiB": tx_total,
        "rx_total_GiB": rx_total,
        "total_transfer_GiB": total_total,
        "avg_tx_GiB_s": (tx_total / duration_s) if duration_s > 0 else 0.0,
        "avg_rx_GiB_s": (rx_total / duration_s) if duration_s > 0 else 0.0,
        "avg_total_GiB_s": (total_total / duration_s) if duration_s > 0 else 0.0,
        "peak_tx_GiB_s": peak_tx["pcie_tx_GiB_s"],
        "peak_tx_t_ms": peak_tx["ts_unix_ms"] - start_ms,
        "peak_rx_GiB_s": peak_rx["pcie_rx_GiB_s"],
        "peak_rx_t_ms": peak_rx["ts_unix_ms"] - start_ms,
        "peak_total_GiB_s": peak_total["pcie_total_GiB_s"],
        "peak_total_t_ms": peak_total["ts_unix_ms"] - start_ms,
        "link_ref_GiB_s": first["pcie_link_ref_GiB_s"],
    }


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _window_rect(start_ms: int, end_ms: int, base_ms: int, x_fn, y: float, h: float, color: str, label: str) -> list[str]:
    if end_ms <= start_ms:
        return []
    x1 = x_fn((start_ms - base_ms) / 1000.0)
    x2 = x_fn((end_ms - base_ms) / 1000.0)
    if x2 <= x1:
        return []
    return [
        f'<rect x="{x1:.1f}" y="{y:.1f}" width="{(x2 - x1):.1f}" height="{h:.1f}" fill="{color}" opacity="0.10"/>',
        f'<text x="{(x1 + x2)/2:.1f}" y="{y + 16:.1f}" text-anchor="middle" font-size="12" fill="{color}" font-family="Helvetica, Arial, sans-serif">{escape_xml(label)}</text>',
    ]


def build_svg(rows: list[dict], windows: dict[str, int], out_svg: str, title: str) -> None:
    if not rows:
        raise SystemExit("no metrics rows for svg")

    width = 1100
    height = 920
    margin_left = 86
    margin_right = 24
    margin_top = 44
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    panel_h = 210
    gap = 34

    base_ts = windows["window_start_unix_ms"]
    t_s = [(r["ts_unix_ms"] - base_ts) / 1000.0 for r in rows]
    max_t = max(t_s) if t_s else 1.0
    if max_t <= 0:
        max_t = 1.0

    def x_fn(x: float) -> float:
        return margin_left + (x / max_t) * plot_w

    def y_fn_factory(top: float, max_y: float):
        def y_fn(y: float) -> float:
            return top + panel_h - (y / max_y) * panel_h if max_y > 0 else top + panel_h
        return y_fn

    max_bw = max(max(r["pcie_tx_GiB_s"], r["pcie_rx_GiB_s"], r["pcie_total_GiB_s"]) for r in rows) * 1.10
    max_bw = max(max_bw, 1.0)
    max_cum = max(r["pcie_total_cum_GiB"] for r in rows)
    min_tx_cum = rows[0]["pcie_tx_cum_GiB"]
    min_rx_cum = rows[0]["pcie_rx_cum_GiB"]
    min_total_cum = rows[0]["pcie_total_cum_GiB"]
    max_cum = max(
        r["pcie_tx_cum_GiB"] - min_tx_cum for r in rows
        if not math.isnan(r["pcie_tx_cum_GiB"])
    )
    max_cum = max(
        max_cum,
        max(r["pcie_rx_cum_GiB"] - min_rx_cum for r in rows if not math.isnan(r["pcie_rx_cum_GiB"])),
        max(r["pcie_total_cum_GiB"] - min_total_cum for r in rows if not math.isnan(r["pcie_total_cum_GiB"])),
        1.0,
    ) * 1.10
    max_misc = max(
        max(r["gpu_mem_used_GiB"] for r in rows if not math.isnan(r["gpu_mem_used_GiB"])),
        1.0,
    ) * 1.10

    y_bw = y_fn_factory(margin_top + 24, max_bw)
    y_cum = y_fn_factory(margin_top + 24 + panel_h + gap, max_cum)
    y_misc = y_fn_factory(margin_top + 24 + (panel_h + gap) * 2, max_misc)

    panels = [
        {"top": margin_top + 24, "y_fn": y_bw, "max_y": max_bw, "ylabel": "PCIe GiB/s"},
        {"top": margin_top + 24 + panel_h + gap, "y_fn": y_cum, "max_y": max_cum, "ylabel": "Cum GiB"},
        {"top": margin_top + 24 + (panel_h + gap) * 2, "y_fn": y_misc, "max_y": max_misc, "ylabel": "GPU mem GiB"},
    ]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="{margin_left}" y="26" font-size="24" font-weight="700" font-family="Helvetica, Arial, sans-serif">{escape_xml(title)}</text>')

    for panel in panels:
        top = panel["top"]
        y_fn = panel["y_fn"]
        max_y = panel["max_y"]
        for i in range(6):
            y_val = max_y * i / 5.0
            y_px = y_fn(y_val)
            parts.append(f'<line x1="{margin_left}" y1="{y_px:.1f}" x2="{width - margin_right}" y2="{y_px:.1f}" stroke="#ececec" stroke-width="1"/>')
            parts.append(f'<text x="{margin_left - 10}" y="{y_px + 4:.1f}" text-anchor="end" font-size="12" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{y_val:.1f}")}</text>')
        parts.append(f'<line x1="{margin_left}" y1="{top + panel_h:.1f}" x2="{width - margin_right}" y2="{top + panel_h:.1f}" stroke="#222" stroke-width="1.4"/>')
        parts.append(f'<line x1="{margin_left}" y1="{top:.1f}" x2="{margin_left}" y2="{top + panel_h:.1f}" stroke="#222" stroke-width="1.4"/>')
        parts.append(
            f'<text x="24" y="{top + panel_h/2:.1f}" text-anchor="middle" font-size="15" fill="#222" font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 24,{top + panel_h/2:.1f})">{escape_xml(panel["ylabel"])}</text>'
        )
        parts.extend(_window_rect(windows["prefill_start_unix_ms"], windows["prefill_end_unix_ms"], base_ts, x_fn, top, panel_h, "#4C78A8", "prefill"))
        parts.extend(_window_rect(windows["decode_start_unix_ms"], windows["decode_end_unix_ms"], base_ts, x_fn, top, panel_h, "#E45756", "decode"))

    # x ticks
    for i in range(7):
        tx = max_t * i / 6.0
        x = x_fn(tx)
        parts.append(f'<line x1="{x:.1f}" y1="{panels[-1]["top"]:.1f}" x2="{x:.1f}" y2="{panels[-1]["top"] + panel_h:.1f}" stroke="#f2f2f2" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-size="12" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{tx:.1f}")}</text>')
    parts.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height - 2}" text-anchor="middle" font-size="16" fill="#222" font-family="Helvetica, Arial, sans-serif">time since first request submit (s)</text>')

    def add_line(points: Iterable[tuple[float, float]], color: str, width_px: float = 2.4) -> None:
        pts = list(points)
        if not pts:
            return
        path = " ".join([f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"] + [f"L {x:.1f} {y:.1f}" for x, y in pts[1:]])
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width_px}" stroke-linejoin="round" stroke-linecap="round"/>')

    # panel 1
    add_line([(x_fn(t), y_bw(r["pcie_tx_GiB_s"])) for t, r in zip(t_s, rows)], "#4C78A8")
    add_line([(x_fn(t), y_bw(r["pcie_rx_GiB_s"])) for t, r in zip(t_s, rows)], "#F58518")
    add_line([(x_fn(t), y_bw(r["pcie_total_GiB_s"])) for t, r in zip(t_s, rows)], "#54A24B", 2.8)
    # panel 2
    add_line([(x_fn(t), y_cum(r["pcie_tx_cum_GiB"] - min_tx_cum)) for t, r in zip(t_s, rows)], "#4C78A8")
    add_line([(x_fn(t), y_cum(r["pcie_rx_cum_GiB"] - min_rx_cum)) for t, r in zip(t_s, rows)], "#F58518")
    add_line([(x_fn(t), y_cum(r["pcie_total_cum_GiB"] - min_total_cum)) for t, r in zip(t_s, rows)], "#54A24B", 2.8)
    # panel 3
    add_line([(x_fn(t), y_misc(r["gpu_mem_used_GiB"])) for t, r in zip(t_s, rows)], "#B279A2", 2.8)

    legend = [
        ("tx", "#4C78A8"),
        ("rx", "#F58518"),
        ("tx+rx", "#54A24B"),
        ("gpu_mem_used", "#B279A2"),
    ]
    lx = width - 220
    ly = 44
    parts.append(f'<rect x="{lx}" y="{ly}" width="190" height="114" rx="8" fill="#fff" stroke="#ddd"/>')
    for i, (label, color) in enumerate(legend):
        yy = ly + 22 + i * 22
        parts.append(f'<line x1="{lx + 12}" y1="{yy}" x2="{lx + 36}" y2="{yy}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 46}" y="{yy + 4}" font-size="13" fill="#333" font-family="Helvetica, Arial, sans-serif">{escape_xml(label)}</text>')

    parts.append("</svg>")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_window_csv(rows: list[dict], windows: dict[str, int], out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ts_unix_ms",
            "t_since_start_s",
            "phase",
            "pcie_tx_GiB_s",
            "pcie_rx_GiB_s",
            "pcie_total_GiB_s",
            "pcie_tx_cum_GiB",
            "pcie_rx_cum_GiB",
            "pcie_total_cum_GiB",
            "pcie_tx_util_pct",
            "pcie_rx_util_pct",
            "pcie_total_util_pct",
            "gpu_mem_used_GiB",
            "util_gpu_pct",
            "cpu_util_pct",
        ])
        base_ts = windows["window_start_unix_ms"]
        for r in rows:
            if r["ts_unix_ms"] < base_ts or r["ts_unix_ms"] > windows["window_end_unix_ms"]:
                continue
            phase = "idle"
            if windows["prefill_start_unix_ms"] and r["ts_unix_ms"] >= windows["prefill_start_unix_ms"] and r["ts_unix_ms"] <= windows["prefill_end_unix_ms"]:
                phase = "prefill"
            elif windows["decode_start_unix_ms"] and r["ts_unix_ms"] >= windows["decode_start_unix_ms"] and r["ts_unix_ms"] <= windows["decode_end_unix_ms"]:
                phase = "decode"
            w.writerow([
                r["ts_unix_ms"],
                f"{(r['ts_unix_ms'] - base_ts) / 1000.0:.6f}",
                phase,
                f"{r['pcie_tx_GiB_s']:.6f}",
                f"{r['pcie_rx_GiB_s']:.6f}",
                f"{r['pcie_total_GiB_s']:.6f}",
                f"{r['pcie_tx_cum_GiB']:.6f}",
                f"{r['pcie_rx_cum_GiB']:.6f}",
                f"{r['pcie_total_cum_GiB']:.6f}",
                f"{r['pcie_tx_util_pct']:.6f}",
                f"{r['pcie_rx_util_pct']:.6f}",
                f"{r['pcie_total_util_pct']:.6f}",
                f"{r['gpu_mem_used_GiB']:.6f}",
                f"{r['util_gpu_pct']:.6f}",
                f"{r['cpu_util_pct']:.6f}",
            ])


def write_summary_md(out_md: str, run_label: str, windows: dict[str, int], full_stats: dict, prefill_stats: dict, decode_stats: dict, svg_path: str) -> None:
    rel_svg = os.path.relpath(svg_path, os.path.dirname(out_md))
    lines = []
    lines.append(f"# PCIe Offload Observation Report: {run_label}")
    lines.append("")
    lines.append("## 1. 观测范围")
    lines.append("")
    lines.append(f"- full window start: `{windows['window_start_unix_ms']}`")
    lines.append(f"- full window end: `{windows['window_end_unix_ms']}`")
    lines.append(f"- prefill window: `{windows['prefill_start_unix_ms']} -> {windows['prefill_end_unix_ms']}`")
    lines.append(f"- decode window: `{windows['decode_start_unix_ms']} -> {windows['decode_end_unix_ms']}`")
    lines.append("")
    lines.append("这里记录的是 GPU 侧 NVML 提供的 PCIe TX/RX moving-average throughput。")
    lines.append("")
    lines.append("- 最稳妥的主指标是 `pcie_total_GiB_s = tx + rx`。")
    lines.append("- 在很多平台上，host->GPU 的 restore/offload-return 往往更容易体现在 GPU PCIe RX 上，但方向语义最好结合实测一起判断。")
    lines.append("")
    lines.append("## 2. 时序图")
    lines.append("")
    lines.append(f"![pcie timeline]({escape_xml(rel_svg)})")
    lines.append("")
    lines.append("## 3. 全窗口统计")
    lines.append("")
    lines.append(f"- duration: `{full_stats['duration_s']:.3f} s`")
    lines.append(f"- total TX volume: `{full_stats['tx_total_GiB']:.3f} GiB`")
    lines.append(f"- total RX volume: `{full_stats['rx_total_GiB']:.3f} GiB`")
    lines.append(f"- total bidirectional volume: `{full_stats['total_transfer_GiB']:.3f} GiB`")
    lines.append(f"- avg TX bandwidth: `{full_stats['avg_tx_GiB_s']:.3f} GiB/s`")
    lines.append(f"- avg RX bandwidth: `{full_stats['avg_rx_GiB_s']:.3f} GiB/s`")
    lines.append(f"- avg bidirectional bandwidth: `{full_stats['avg_total_GiB_s']:.3f} GiB/s`")
    lines.append(f"- peak TX bandwidth: `{full_stats['peak_tx_GiB_s']:.3f} GiB/s` at `{full_stats['peak_tx_t_ms'] / 1000.0:.3f} s`")
    lines.append(f"- peak RX bandwidth: `{full_stats['peak_rx_GiB_s']:.3f} GiB/s` at `{full_stats['peak_rx_t_ms'] / 1000.0:.3f} s`")
    lines.append(f"- peak bidirectional bandwidth: `{full_stats['peak_total_GiB_s']:.3f} GiB/s` at `{full_stats['peak_total_t_ms'] / 1000.0:.3f} s`")
    lines.append("")
    lines.append("## 4. 分阶段统计")
    lines.append("")
    lines.append("| phase | duration (s) | tx total (GiB) | rx total (GiB) | total (GiB) | avg total GiB/s | peak total GiB/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, st in [("prefill", prefill_stats), ("decode", decode_stats), ("full", full_stats)]:
        lines.append(
            f"| {name} | {st['duration_s']:.3f} | {st['tx_total_GiB']:.3f} | {st['rx_total_GiB']:.3f} | {st['total_transfer_GiB']:.3f} | {st['avg_total_GiB_s']:.3f} | {st['peak_total_GiB_s']:.3f} |"
        )
    lines.append("")
    lines.append("## 5. 解读建议")
    lines.append("")
    lines.append("- 如果你要抓“offloading 开始到最后 restore”的总体量，优先看 `full` 和 `decode` 的 `total_transfer_GiB`。")
    lines.append("- 如果你要抓最容易体现 restore 的瞬时冲击，优先看 `decode` 窗口内的 `peak RX` 和 `peak total`。")
    lines.append("- 如果 `total_transfer_GiB` 已经不小，但时间指标仍几乎不变，说明当前 offloading 流量可能存在，但还不足以成为端到端瓶颈。")
    lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze GPU PCIe metrics for PD offload runs.")
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--prefill_csv", required=True)
    ap.add_argument("--decode_csv", required=True)
    ap.add_argument("--run_label", default="offload_run")
    ap.add_argument("--out_svg", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    metrics_rows = load_metrics(args.metrics_csv)
    windows = load_request_window(args.prefill_csv, args.decode_csv)
    selected_rows = select_rows(metrics_rows, windows["window_start_unix_ms"], windows["window_end_unix_ms"])
    full_stats = summarize_window(metrics_rows, windows["window_start_unix_ms"], windows["window_end_unix_ms"])
    prefill_stats = summarize_window(metrics_rows, windows["prefill_start_unix_ms"], windows["prefill_end_unix_ms"])
    decode_stats = summarize_window(metrics_rows, windows["decode_start_unix_ms"], windows["decode_end_unix_ms"])

    ensure_dir(os.path.dirname(args.out_svg) or ".")
    build_svg(selected_rows, windows, args.out_svg, f"PCIe Offload Timeline: {args.run_label}")
    write_window_csv(metrics_rows, windows, args.out_csv)
    write_summary_md(args.out_md, args.run_label, windows, full_stats, prefill_stats, decode_stats, args.out_svg)
    if args.out_json:
        ensure_dir(os.path.dirname(args.out_json) or ".")
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_label": args.run_label,
                    "windows": windows,
                    "full": full_stats,
                    "prefill": prefill_stats,
                    "decode": decode_stats,
                },
                f,
                indent=2,
            )

    print(f"[ok] wrote {args.out_svg}")
    print(f"[ok] wrote {args.out_md}")
    print(f"[ok] wrote {args.out_csv}")
    if args.out_json:
        print(f"[ok] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
