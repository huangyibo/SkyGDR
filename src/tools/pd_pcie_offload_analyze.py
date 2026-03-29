#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict


PHASE_COLORS = {
    "seed": "#4C78A8",
    "reuse": "#F58518",
}


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


def load_requests(path: str) -> list[dict]:
    rows = load_csv(path)
    out = []
    for r in rows:
        if r.get("http_status") != "200" or r.get("error"):
            continue
        out.append(
            {
                "request_id": r["request_id"],
                "phase": r["phase"],
                "session_id": r["session_id"],
                "turn_id": to_int(r["turn_id"]),
                "prompt_tokens": to_int(r["prompt_tokens"]),
                "reused_prefix_tokens_est": to_int(r["reused_prefix_tokens_est"]),
                "appended_tokens_est": to_int(r["appended_tokens_est"]),
                "reuse_ratio_est": to_float(r["reuse_ratio_est"]),
                "expected_external_hit": to_int(r["expected_external_hit"]),
                "max_tokens": to_int(r["max_tokens"]),
                "submit_ts_unix_ms": to_int(r["submit_ts_unix_ms"]),
                "response_finish_ts_unix_ms": to_int(r["response_finish_ts_unix_ms"]),
                "post_metrics_ts_unix_ms": to_int(r["post_metrics_ts_unix_ms"]),
                "elapsed_ms": to_float(r["elapsed_ms"]),
                "usage_prompt_tokens": to_int(r["usage_prompt_tokens"], default=-1),
                "usage_completion_tokens": to_int(r["usage_completion_tokens"], default=-1),
                "lmcache_requested_tokens": to_int(r.get("lmcache_requested_tokens"), default=-1),
                "lmcache_hit_tokens": to_int(r.get("lmcache_hit_tokens"), default=-1),
                "lmcache_vllm_hit_tokens": to_int(r.get("lmcache_vllm_hit_tokens"), default=-1),
                "lmcache_remote_read_bytes": to_int(r.get("lmcache_remote_read_bytes"), default=-1),
                "lmcache_remote_write_bytes": to_int(r.get("lmcache_remote_write_bytes"), default=-1),
                "lmcache_remote_read_requests": to_int(r.get("lmcache_remote_read_requests"), default=-1),
                "lmcache_remote_write_requests": to_int(r.get("lmcache_remote_write_requests"), default=-1),
                "lmcache_hit_ratio": to_float(r.get("lmcache_hit_ratio"), default=0.0),
                "lmcache_remote_read_GiB": to_float(r.get("lmcache_remote_read_GiB"), default=0.0),
                "lmcache_remote_write_GiB": to_float(r.get("lmcache_remote_write_GiB"), default=0.0),
            }
        )
    if not out:
        raise SystemExit("no successful request rows found")
    return out


def load_windows(requests: list[dict]) -> dict[str, int]:
    return {
        "window_start_unix_ms": min(r["submit_ts_unix_ms"] for r in requests),
        "window_end_unix_ms": max(r["post_metrics_ts_unix_ms"] for r in requests),
    }


def build_request_spans(requests: list[dict]) -> list[dict]:
    return [
        {
            "request_id": r["request_id"],
            "phase": r["phase"],
            "turn_id": r["turn_id"],
            "start_unix_ms": r["submit_ts_unix_ms"],
            "end_unix_ms": r["post_metrics_ts_unix_ms"],
        }
        for r in requests
    ]


def select_rows(rows: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    return [r for r in rows if start_ms <= r["ts_unix_ms"] <= end_ms]


def summarize_single_interval(rows: list[dict], start_ms: int, end_ms: int) -> dict:
    sel = select_rows(rows, start_ms, end_ms)
    if not sel:
        return {}
    first = sel[0]
    last = sel[-1]
    return {
        "duration_s": max(0.0, (end_ms - start_ms) / 1000.0),
        "num_samples": len(sel),
        "tx_total_GiB": max(0.0, last["pcie_tx_cum_GiB"] - first["pcie_tx_cum_GiB"]),
        "rx_total_GiB": max(0.0, last["pcie_rx_cum_GiB"] - first["pcie_rx_cum_GiB"]),
        "total_transfer_GiB": max(0.0, last["pcie_total_cum_GiB"] - first["pcie_total_cum_GiB"]),
        "peak_tx_GiB_s": max((r["pcie_tx_GiB_s"] for r in sel), default=0.0),
        "peak_rx_GiB_s": max((r["pcie_rx_GiB_s"] for r in sel), default=0.0),
        "peak_total_GiB_s": max((r["pcie_total_GiB_s"] for r in sel), default=0.0),
        "peak_tx_row": max(sel, key=lambda r: r["pcie_tx_GiB_s"]),
        "peak_rx_row": max(sel, key=lambda r: r["pcie_rx_GiB_s"]),
        "peak_total_row": max(sel, key=lambda r: r["pcie_total_GiB_s"]),
        "link_ref_GiB_s": first["pcie_link_ref_GiB_s"],
    }


def summarize_intervals(rows: list[dict], intervals: list[tuple[int, int]]) -> dict:
    if not intervals:
        return {}
    stats = [summarize_single_interval(rows, start, end) for start, end in intervals]
    stats = [s for s in stats if s]
    if not stats:
        return {}
    duration_s = sum(s["duration_s"] for s in stats)
    tx_total = sum(s["tx_total_GiB"] for s in stats)
    rx_total = sum(s["rx_total_GiB"] for s in stats)
    total_transfer = sum(s["total_transfer_GiB"] for s in stats)
    peak_tx_stat = max(stats, key=lambda s: s["peak_tx_GiB_s"])
    peak_rx_stat = max(stats, key=lambda s: s["peak_rx_GiB_s"])
    peak_total_stat = max(stats, key=lambda s: s["peak_total_GiB_s"])
    return {
        "duration_s": duration_s,
        "num_samples": sum(s["num_samples"] for s in stats),
        "tx_total_GiB": tx_total,
        "rx_total_GiB": rx_total,
        "total_transfer_GiB": total_transfer,
        "avg_tx_GiB_s": (tx_total / duration_s) if duration_s > 0 else 0.0,
        "avg_rx_GiB_s": (rx_total / duration_s) if duration_s > 0 else 0.0,
        "avg_total_GiB_s": (total_transfer / duration_s) if duration_s > 0 else 0.0,
        "peak_tx_GiB_s": peak_tx_stat["peak_tx_GiB_s"],
        "peak_rx_GiB_s": peak_rx_stat["peak_rx_GiB_s"],
        "peak_total_GiB_s": peak_total_stat["peak_total_GiB_s"],
        "peak_tx_t_ms": peak_tx_stat["peak_tx_row"]["ts_unix_ms"] - intervals[0][0],
        "peak_rx_t_ms": peak_rx_stat["peak_rx_row"]["ts_unix_ms"] - intervals[0][0],
        "peak_total_t_ms": peak_total_stat["peak_total_row"]["ts_unix_ms"] - intervals[0][0],
        "link_ref_GiB_s": stats[0]["link_ref_GiB_s"],
    }


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def build_main_svg(rows: list[dict], spans: list[dict], windows: dict[str, int], out_svg: str, title: str) -> None:
    width = 1100
    height = 820
    margin_left = 86
    margin_right = 24
    margin_top = 44
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    panel_h = 180
    gap = 32

    base_ts = windows["window_start_unix_ms"]
    t_s = [(r["ts_unix_ms"] - base_ts) / 1000.0 for r in rows]
    max_t = max(t_s) if t_s else 1.0
    if max_t <= 0:
        max_t = 1.0

    max_total = max((r["pcie_total_GiB_s"] for r in rows), default=1.0) * 1.10
    max_tx = max((r["pcie_tx_GiB_s"] for r in rows), default=1.0) * 1.10
    max_rx = max((r["pcie_rx_GiB_s"] for r in rows), default=1.0) * 1.10
    max_total = max(max_total, 1.0)
    max_tx = max(max_tx, 1.0)
    max_rx = max(max_rx, 1.0)

    def x_fn(x: float) -> float:
        return margin_left + (x / max_t) * plot_w

    panels = [
        ("pcie_total_GiB_s", "Total GiB/s", max_total, "#54A24B"),
        ("pcie_tx_GiB_s", "TX GiB/s", max_tx, "#4C78A8"),
        ("pcie_rx_GiB_s", "RX GiB/s", max_rx, "#F58518"),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{margin_left}" y="26" font-size="24" font-weight="700" font-family="Helvetica, Arial, sans-serif">{escape_xml(title)}</text>',
    ]

    for idx, (key, ylabel, max_y, color) in enumerate(panels):
        top = margin_top + 24 + idx * (panel_h + gap)

        def y_fn(y: float, _top=top, _max=max_y):
            return _top + panel_h - (y / _max) * panel_h if _max > 0 else _top + panel_h

        for i in range(6):
            y_val = max_y * i / 5.0
            y_px = y_fn(y_val)
            parts.append(f'<line x1="{margin_left}" y1="{y_px:.1f}" x2="{width - margin_right}" y2="{y_px:.1f}" stroke="#ececec" stroke-width="1"/>')
            parts.append(f'<text x="{margin_left - 10}" y="{y_px + 4:.1f}" text-anchor="end" font-size="12" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{y_val:.1f}")}</text>')
        parts.append(f'<line x1="{margin_left}" y1="{top + panel_h:.1f}" x2="{width - margin_right}" y2="{top + panel_h:.1f}" stroke="#222" stroke-width="1.4"/>')
        parts.append(f'<line x1="{margin_left}" y1="{top:.1f}" x2="{margin_left}" y2="{top + panel_h:.1f}" stroke="#222" stroke-width="1.4"/>')
        parts.append(
            f'<text x="24" y="{top + panel_h/2:.1f}" text-anchor="middle" font-size="15" fill="#222" font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 24,{top + panel_h/2:.1f})">{escape_xml(ylabel)}</text>'
        )

        for span in spans:
            x1 = x_fn((span["start_unix_ms"] - base_ts) / 1000.0)
            x2 = x_fn((span["end_unix_ms"] - base_ts) / 1000.0)
            color_span = PHASE_COLORS.get(span["phase"], "#bbbbbb")
            parts.append(
                f'<rect x="{x1:.1f}" y="{top:.1f}" width="{max(0.5, x2 - x1):.1f}" height="{panel_h:.1f}" fill="{color_span}" opacity="0.08"/>'
            )

        pts = [(x_fn(t), y_fn(r[key])) for t, r in zip(t_s, rows)]
        if pts:
            path = " ".join([f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"] + [f"L {x:.1f} {y:.1f}" for x, y in pts[1:]])
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>')

    bottom_top = margin_top + 24 + 2 * (panel_h + gap)
    for i in range(7):
        tx = max_t * i / 6.0
        x = x_fn(tx)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top + 24:.1f}" x2="{x:.1f}" y2="{bottom_top + panel_h:.1f}" stroke="#f2f2f2" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-size="12" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{tx:.1f}")}</text>')
    parts.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height - 2}" text-anchor="middle" font-size="16" fill="#222" font-family="Helvetica, Arial, sans-serif">time since first request submit (s)</text>')

    legend_x = width - 250
    legend_y = 44
    legend_items = [("seed", PHASE_COLORS["seed"]), ("reuse", PHASE_COLORS["reuse"])]
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="220" height="70" rx="8" fill="#fff" stroke="#ddd"/>')
    for idx, (label, color) in enumerate(legend_items):
        yy = legend_y + 22 + idx * 22
        parts.append(f'<rect x="{legend_x + 12}" y="{yy - 8}" width="16" height="12" fill="{color}" opacity="0.35"/>')
        parts.append(f'<text x="{legend_x + 40}" y="{yy + 2}" font-size="13" fill="#333" font-family="Helvetica, Arial, sans-serif">{escape_xml(label)}</text>')

    parts.append("</svg>")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def build_direction_svg(rows: list[dict], spans: list[dict], windows: dict[str, int], out_svg: str, title: str, key: str, color: str, ylabel: str) -> None:
    width = 1100
    height = 420
    margin_left = 86
    margin_right = 24
    margin_top = 44
    margin_bottom = 54
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    base_ts = windows["window_start_unix_ms"]
    t_s = [(r["ts_unix_ms"] - base_ts) / 1000.0 for r in rows]
    max_t = max(t_s) if t_s else 1.0
    if max_t <= 0:
        max_t = 1.0

    max_y = max((r[key] for r in rows if not math.isnan(r[key])), default=1.0) * 1.10
    max_y = max(max_y, 1.0)

    def x_fn(x: float) -> float:
        return margin_left + (x / max_t) * plot_w

    def y_fn(y: float) -> float:
        return margin_top + plot_h - (y / max_y) * plot_h if max_y > 0 else margin_top + plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{margin_left}" y="26" font-size="24" font-weight="700" font-family="Helvetica, Arial, sans-serif">{escape_xml(title)}</text>',
    ]

    for i in range(6):
        y_val = max_y * i / 5.0
        y_px = y_fn(y_val)
        parts.append(f'<line x1="{margin_left}" y1="{y_px:.1f}" x2="{width - margin_right}" y2="{y_px:.1f}" stroke="#ececec" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y_px + 4:.1f}" text-anchor="end" font-size="12" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{y_val:.1f}")}</text>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h:.1f}" x2="{width - margin_right}" y2="{margin_top + plot_h:.1f}" stroke="#222" stroke-width="1.4"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top:.1f}" x2="{margin_left}" y2="{margin_top + plot_h:.1f}" stroke="#222" stroke-width="1.4"/>')
    parts.append(
        f'<text x="24" y="{margin_top + plot_h/2:.1f}" text-anchor="middle" font-size="15" fill="#222" font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 24,{margin_top + plot_h/2:.1f})">{escape_xml(ylabel)}</text>'
    )

    for span in spans:
        x1 = x_fn((span["start_unix_ms"] - base_ts) / 1000.0)
        x2 = x_fn((span["end_unix_ms"] - base_ts) / 1000.0)
        color_span = PHASE_COLORS.get(span["phase"], "#bbbbbb")
        parts.append(
            f'<rect x="{x1:.1f}" y="{margin_top:.1f}" width="{max(0.5, x2 - x1):.1f}" height="{plot_h:.1f}" fill="{color_span}" opacity="0.08"/>'
        )

    pts = [(x_fn(t), y_fn(r[key])) for t, r in zip(t_s, rows)]
    if pts:
        path = " ".join([f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"] + [f"L {x:.1f} {y:.1f}" for x, y in pts[1:]])
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>')

    for i in range(7):
        tx = max_t * i / 6.0
        x = x_fn(tx)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top:.1f}" x2="{x:.1f}" y2="{margin_top + plot_h:.1f}" stroke="#f2f2f2" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-size="12" fill="#444" font-family="Helvetica, Arial, sans-serif">{escape_xml(f"{tx:.1f}")}</text>')
    parts.append(f'<text x="{margin_left + plot_w/2:.1f}" y="{height - 2}" text-anchor="middle" font-size="16" fill="#222" font-family="Helvetica, Arial, sans-serif">time since first request submit (s)</text>')

    parts.append("</svg>")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_window_csv(rows: list[dict], spans: list[dict], windows: dict[str, int], out_csv: str) -> None:
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
            if r["ts_unix_ms"] < windows["window_start_unix_ms"] or r["ts_unix_ms"] > windows["window_end_unix_ms"]:
                continue
            active_phases = [
                span["phase"]
                for span in spans
                if span["start_unix_ms"] <= r["ts_unix_ms"] <= span["end_unix_ms"]
            ]
            phase = "+".join(sorted(set(active_phases))) if active_phases else "idle"
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


def write_request_csv(metrics_rows: list[dict], requests: list[dict], out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "request_id",
            "phase",
            "session_id",
            "turn_id",
            "prompt_tokens",
            "reused_prefix_tokens_est",
            "appended_tokens_est",
            "reuse_ratio_est",
            "expected_external_hit",
            "elapsed_ms",
            "usage_prompt_tokens",
            "usage_completion_tokens",
            "lmcache_requested_tokens",
            "lmcache_hit_tokens",
            "lmcache_vllm_hit_tokens",
            "lmcache_hit_ratio",
            "lmcache_remote_read_GiB",
            "lmcache_remote_write_GiB",
            "tx_total_GiB",
            "rx_total_GiB",
            "total_transfer_GiB",
            "peak_tx_GiB_s",
            "peak_rx_GiB_s",
            "peak_total_GiB_s",
        ])
        for r in requests:
            st = summarize_single_interval(metrics_rows, r["submit_ts_unix_ms"], r["post_metrics_ts_unix_ms"])
            if not st:
                continue
            w.writerow([
                r["request_id"],
                r["phase"],
                r["session_id"],
                r["turn_id"],
                r["prompt_tokens"],
                r["reused_prefix_tokens_est"],
                r["appended_tokens_est"],
                f"{r['reuse_ratio_est']:.6f}",
                r["expected_external_hit"],
                f"{r['elapsed_ms']:.6f}",
                r["usage_prompt_tokens"],
                r["usage_completion_tokens"],
                r["lmcache_requested_tokens"],
                r["lmcache_hit_tokens"],
                r["lmcache_vllm_hit_tokens"],
                f"{r['lmcache_hit_ratio']:.6f}",
                f"{r['lmcache_remote_read_GiB']:.6f}",
                f"{r['lmcache_remote_write_GiB']:.6f}",
                f"{st['tx_total_GiB']:.6f}",
                f"{st['rx_total_GiB']:.6f}",
                f"{st['total_transfer_GiB']:.6f}",
                f"{st['peak_tx_GiB_s']:.6f}",
                f"{st['peak_rx_GiB_s']:.6f}",
                f"{st['peak_total_GiB_s']:.6f}",
            ])


def write_summary_md(
    out_md: str,
    run_label: str,
    windows: dict[str, int],
    full_stats: dict,
    phase_stats: dict[str, dict],
    spans: list[dict],
    request_summary_csv: str,
    total_svg: str,
    tx_svg: str,
    rx_svg: str,
) -> None:
    rel_total = os.path.relpath(total_svg, os.path.dirname(out_md))
    rel_tx = os.path.relpath(tx_svg, os.path.dirname(out_md))
    rel_rx = os.path.relpath(rx_svg, os.path.dirname(out_md))
    rel_req = os.path.relpath(request_summary_csv, os.path.dirname(out_md))

    lines = []
    lines.append(f"# External Prefix-Cache PCIe Report: {run_label}")
    lines.append("")
    lines.append("## 1. 观测范围")
    lines.append("")
    lines.append(f"- full window start: `{windows['window_start_unix_ms']}`")
    lines.append(f"- full window end: `{windows['window_end_unix_ms']}`")
    lines.append(f"- requests: `{len(spans)}`")
    lines.append("")
    lines.append("本报告针对的是：")
    lines.append("")
    lines.append("- `seed`：首轮长 prompt，把完整前缀写入 external/shared prefix cache")
    lines.append("- `reuse`：后续各轮只追加少量新 token，观察 prefill 是否从外部 cache 读回大部分历史 KV")
    lines.append("")
    lines.append("## 2. 时序图")
    lines.append("")
    lines.append(f"![pcie timeline]({escape_xml(rel_total)})")
    lines.append("")
    lines.append("### TX")
    lines.append("")
    lines.append(f"![pcie tx timeline]({escape_xml(rel_tx)})")
    lines.append("")
    lines.append("### RX")
    lines.append("")
    lines.append(f"![pcie rx timeline]({escape_xml(rel_rx)})")
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
    lines.append(f"- peak TX bandwidth: `{full_stats['peak_tx_GiB_s']:.3f} GiB/s`")
    lines.append(f"- peak RX bandwidth: `{full_stats['peak_rx_GiB_s']:.3f} GiB/s`")
    lines.append(f"- peak total bandwidth: `{full_stats['peak_total_GiB_s']:.3f} GiB/s`")
    lines.append("")
    lines.append("## 4. 分阶段统计")
    lines.append("")
    lines.append("| phase | duration (s) | tx total (GiB) | rx total (GiB) | total (GiB) | avg tx GiB/s | avg rx GiB/s | peak tx GiB/s | peak rx GiB/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for phase in ["seed", "reuse"]:
        st = phase_stats.get(phase) or {}
        if not st:
            continue
        lines.append(
            f"| {phase} | {st['duration_s']:.3f} | {st['tx_total_GiB']:.3f} | {st['rx_total_GiB']:.3f} | {st['total_transfer_GiB']:.3f} | {st['avg_tx_GiB_s']:.3f} | {st['avg_rx_GiB_s']:.3f} | {st['peak_tx_GiB_s']:.3f} | {st['peak_rx_GiB_s']:.3f} |"
        )
    lines.append("")
    lines.append("## 5. 请求级汇总")
    lines.append("")
    lines.append(f"- request summary csv: `{escape_xml(rel_req)}`")
    lines.append("")
    lines.append("重点建议：")
    lines.append("")
    lines.append("- 看 `reuse` 请求的 `lmcache_remote_read_GiB` 和 `peak_rx_GiB_s`，这是最直接的 external prefix-cache prefill load 信号。")
    lines.append("- 看 `seed` 请求的 `lmcache_remote_write_GiB` 和 `peak_tx_GiB_s`，它反映首轮长前缀是怎么被写入外部 cache 的。")
    lines.append("- 如果 `lmcache_hit_ratio` 很高但 `peak_rx_GiB_s` 不高，通常意味着外部读回被更平滑地摊开了，而不是没有命中。")
    lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Analyze PCIe metrics for the external prefix-cache imitation workload.")
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--request_csv", required=True)
    ap.add_argument("--run_label", default="external_prefix_run")
    ap.add_argument("--out_svg", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    metrics_rows = load_metrics(args.metrics_csv)
    requests = load_requests(args.request_csv)
    windows = load_windows(requests)
    spans = build_request_spans(requests)
    selected_rows = [
        r for r in metrics_rows
        if windows["window_start_unix_ms"] <= r["ts_unix_ms"] <= windows["window_end_unix_ms"]
    ]
    if not selected_rows:
        raise SystemExit("no metrics rows overlap with request window")

    full_stats = summarize_intervals(metrics_rows, [(windows["window_start_unix_ms"], windows["window_end_unix_ms"])])
    phase_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in spans:
        phase_intervals[span["phase"]].append((span["start_unix_ms"], span["end_unix_ms"]))
    phase_stats = {phase: summarize_intervals(metrics_rows, intervals) for phase, intervals in phase_intervals.items()}

    out_dir = os.path.dirname(args.out_svg) or "."
    tx_svg = os.path.join(out_dir, "pcie_tx_timeline.svg")
    rx_svg = os.path.join(out_dir, "pcie_rx_timeline.svg")
    request_summary_csv = os.path.join(out_dir, "request_pcie_summary.csv")

    ensure_dir(out_dir)
    build_main_svg(selected_rows, spans, windows, args.out_svg, f"PCIe Timeline: {args.run_label}")
    build_direction_svg(selected_rows, spans, windows, tx_svg, f"PCIe TX Timeline: {args.run_label}", "pcie_tx_GiB_s", "#4C78A8", "TX GiB/s")
    build_direction_svg(selected_rows, spans, windows, rx_svg, f"PCIe RX Timeline: {args.run_label}", "pcie_rx_GiB_s", "#F58518", "RX GiB/s")
    write_window_csv(metrics_rows, spans, windows, args.out_csv)
    write_request_csv(metrics_rows, requests, request_summary_csv)
    write_summary_md(args.out_md, args.run_label, windows, full_stats, phase_stats, spans, request_summary_csv, args.out_svg, tx_svg, rx_svg)

    if args.out_json:
        ensure_dir(os.path.dirname(args.out_json) or ".")
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_label": args.run_label,
                    "windows": windows,
                    "full": full_stats,
                    "phases": phase_stats,
                    "requests": spans,
                    "request_summary_csv": request_summary_csv,
                },
                f,
                indent=2,
            )

    print(f"[ok] wrote {args.out_svg}")
    print(f"[ok] wrote {tx_svg}")
    print(f"[ok] wrote {rx_svg}")
    print(f"[ok] wrote {args.out_md}")
    print(f"[ok] wrote {args.out_csv}")
    print(f"[ok] wrote {request_summary_csv}")
    if args.out_json:
        print(f"[ok] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
