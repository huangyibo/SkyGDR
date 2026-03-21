#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import statistics
from dataclasses import dataclass


PCIE_LINE_RE = re.compile(r"t=([0-9.]+)s.*bw_gib_s=([0-9.]+)")


@dataclass
class PcieSeries:
    t_s: list
    bw_gib_s: list

    @property
    def n(self):
        return len(self.bw_gib_s)

    @property
    def mean(self):
        return statistics.mean(self.bw_gib_s) if self.bw_gib_s else float("nan")

    @property
    def median(self):
        return statistics.median(self.bw_gib_s) if self.bw_gib_s else float("nan")

    @property
    def p95(self):
        if not self.bw_gib_s:
            return float("nan")
        ordered = sorted(self.bw_gib_s)
        idx = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
        return ordered[idx]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_pcie_log(path: str) -> PcieSeries:
    t_s = []
    bw = []
    with open(path, "r") as f:
        for line in f:
            m = PCIE_LINE_RE.search(line)
            if not m:
                continue
            t_s.append(float(m.group(1)))
            bw.append(float(m.group(2)))
    return PcieSeries(t_s=t_s, bw_gib_s=bw)


def parse_rdma_csv(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            ret = int(float(row.get("RetCode", "0"))) if row.get("RetCode", "") != "" else 0
            if ret != 0:
                continue
            rows.append(
                {
                    "msg_bytes": int(float(row["MsgBytes"])),
                    "throughput_gib_s": float(row["Throughput_GiB_per_s"]),
                    "p50_us": float(row["P50_us"]) if row.get("P50_us", "") else float("nan"),
                    "p99_us": float(row["P99_us"]) if row.get("P99_us", "") else float("nan"),
                    "p999_us": float(row["P999_us"]) if row.get("P999_us", "") else float("nan"),
                }
            )
    return rows


def rdma_map_by_msg(path: str):
    items = parse_rdma_csv(path)
    out = {}
    for item in items:
        out[item["msg_bytes"]] = item
    return out


def parse_gpu_metrics_active_util(path: str):
    active = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                tx = float(row["pcie_tx_util_pct"])
                rx = float(row["pcie_rx_util_pct"])
            except Exception:
                continue
            if tx < 0 and rx < 0:
                continue
            if tx < 0:
                active.append(rx)
            elif rx < 0:
                active.append(tx)
            else:
                active.append(max(tx, rx))
    if not active:
        return {"n": 0, "mean": float("nan"), "p95": float("nan"), "peak": float("nan")}
    ordered = sorted(active)
    idx95 = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "n": len(active),
        "mean": statistics.mean(active),
        "p95": ordered[idx95],
        "peak": max(active),
    }


def write_csv(path: str, header: list, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_part_a_summary(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "direction": row["direction"],
                    "baseline_mean_bw_gib_s": float(row["baseline_mean_bw_gib_s"]),
                    "baseline_median_bw_gib_s": float(row["baseline_median_bw_gib_s"]),
                    "baseline_p95_bw_gib_s": float(row["baseline_p95_bw_gib_s"]),
                    "baseline_n": int(float(row["baseline_n"])),
                    "contended_mean_bw_gib_s": float(row["contended_mean_bw_gib_s"]),
                    "contended_median_bw_gib_s": float(row["contended_median_bw_gib_s"]),
                    "contended_p95_bw_gib_s": float(row["contended_p95_bw_gib_s"]),
                    "contended_n": int(float(row["contended_n"])),
                    "degradation_pct": float(row["degradation_pct"]),
                }
            )
    return rows


def load_part_b_summary(path: str):
    rows = []
    required_cols = ["msg_bytes", "write_none_thr", "write_h2d_thr", "read_none_thr", "read_d2h_thr"]
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            for k in required_cols:
                if k not in row:
                    raise ValueError(f"{path} missing required column: {k}")
            item = {}
            item["msg_bytes"] = int(float(row["msg_bytes"]))
            item["write_none_thr"] = float(row["write_none_thr"])
            item["write_h2d_thr"] = float(row["write_h2d_thr"])
            item["read_none_thr"] = float(row["read_none_thr"])
            item["read_d2h_thr"] = float(row["read_d2h_thr"])
            # Ratios are optional in CSV; compute if not provided.
            if row.get("write_thr_ratio", "") != "":
                item["write_thr_ratio"] = float(row["write_thr_ratio"])
            else:
                item["write_thr_ratio"] = (
                    item["write_h2d_thr"] / item["write_none_thr"] if item["write_none_thr"] > 0 else float("nan")
                )
            if row.get("read_thr_ratio", "") != "":
                item["read_thr_ratio"] = float(row["read_thr_ratio"])
            else:
                item["read_thr_ratio"] = (
                    item["read_d2h_thr"] / item["read_none_thr"] if item["read_none_thr"] > 0 else float("nan")
                )
            rows.append(item)
    rows.sort(key=lambda x: x["msg_bytes"])
    return rows


def plot_part_a(part_a_rows: list, out_path: str, out_png_paper: str = "", out_pdf_paper: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.style.use("seaborn-v0_8-whitegrid")

    # Render as 4 explicit cases to match paper wording/order.
    h2d_row = next((r for r in part_a_rows if r["direction"].lower() == "h2d"), None)
    d2h_row = next((r for r in part_a_rows if r["direction"].lower() == "d2h"), None)
    if h2d_row is None or d2h_row is None:
        raise ValueError("part_a_rows must include both h2d and d2h")

    labels = [
        "H2D isolated",
        "H2D+write",
        "D2H isolated",
        "D2H+read",
    ]
    values = [
        h2d_row["baseline_mean_bw_gib_s"],
        h2d_row["contended_mean_bw_gib_s"],
        d2h_row["baseline_mean_bw_gib_s"],
        d2h_row["contended_mean_bw_gib_s"],
    ]
    colors = ["#4C78A8", "#F58518", "#4C78A8", "#F58518"]
    edge_colors = ["#2f4b66", "#9a4f00", "#2f4b66", "#9a4f00"]

    # Two close pairs: (H2D isolated, H2D+write) and (D2H isolated, D2H+read).
    # Keep pair spacing clearly tighter than the inter-pair gap.
    x = [0.0, 0.35, 0.90, 1.25]
    width = 0.24
    ymax = max(values)
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    # Keep chart background consistent with paper page background.
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    bars = ax.bar(
        x,
        values,
        width=width,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        zorder=3,
    )

    # Keep top value labels on bars (while degradation labels are omitted).
    for bar in bars:
        x_mid = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()
        ax.text(
            x_mid,
            y_top + ymax * 0.012,
            f"{y_top:.2f}",
            ha="center",
            va="bottom",
            fontsize=11.2,
            color="#203040",
            fontweight="semibold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.2)
    ax.set_ylabel("Memcpy Throughput (GiB/s)", fontsize=12.4, fontweight="semibold")
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlim(min(x) - 0.14, max(x) + 0.14)
    ax.tick_params(axis="y", labelsize=10.8)
    ax.tick_params(axis="x", labelsize=11.2, pad=4)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("semibold")
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, linestyle=(0, (3, 3)), alpha=0.35, zorder=0)
    legend_handles = [
        mpatches.Patch(facecolor="#4C78A8", edgecolor="#2f4b66", label="Memcpy isolated"),
        mpatches.Patch(facecolor="#F58518", edgecolor="#9a4f00", label="Memcpy with contending RDMA"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=9.8,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    # Keep a full border box around the plot.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_alpha(0.38)
        ax.spines[side].set_linewidth(0.9)
    fig.tight_layout(pad=0.5)

    for p in [out_path, out_png_paper]:
        if p:
            out_dir = os.path.dirname(p)
            if out_dir:
                ensure_dir(out_dir)
            fig.savefig(p, dpi=220, bbox_inches="tight")
    if out_pdf_paper:
        out_dir = os.path.dirname(out_pdf_paper)
        if out_dir:
            ensure_dir(out_dir)
        fig.savefig(out_pdf_paper, format="pdf", bbox_inches="tight")

    plt.close(fig)


def plot_part_a_with_latency_dual_axis(part_a_rows: list, part_b_rows: list, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_bar, ax_lat) = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.1),
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )

    labels = [row["direction"] for row in part_a_rows]
    base = [row["baseline_mean_bw_gib_s"] for row in part_a_rows]
    cont = [row["contended_mean_bw_gib_s"] for row in part_a_rows]
    cont_labels = []
    for row in part_a_rows:
        direction = row["direction"].lower()
        if direction == "h2d":
            cont_labels.append("with write")
        elif direction == "d2h":
            cont_labels.append("with read")
        else:
            cont_labels.append("with rdma")

    x = range(len(labels))
    width = 0.35
    ax_bar.bar([i - width / 2 for i in x], base, width=width, label="baseline")
    cont_bars = ax_bar.bar([i + width / 2 for i in x], cont, width=width, label="with_rdma")
    for bar, label in zip(cont_bars, cont_labels):
        x_mid = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()
        ax_bar.text(x_mid, y_top * 1.01, label, ha="center", va="bottom")
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("Memcpy BW (GiB/s)")
    ax_bar.set_title("Part A: Memcpy impact")
    ax_bar.set_ylim(0, max(base + cont) * 1.14)
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(loc="upper right")

    rows = sorted(part_b_rows, key=lambda x: x["msg_bytes"])
    x_mb = [r["msg_bytes"] / (1024 * 1024) for r in rows]
    case_colors = {
        "write_none": "#1f77b4",
        "write_on_h2d": "#ff7f0e",
        "read_none": "#2ca02c",
        "read_on_d2h": "#d62728",
    }
    p50_series = {
        "write_none": [r["write_none_p50"] for r in rows],
        "write_on_h2d": [r["write_h2d_p50"] for r in rows],
        "read_none": [r["read_none_p50"] for r in rows],
        "read_on_d2h": [r["read_d2h_p50"] for r in rows],
    }
    p999_series = {
        "write_none": [r["write_none_p999"] for r in rows],
        "write_on_h2d": [r["write_h2d_p999"] for r in rows],
        "read_none": [r["read_none_p999"] for r in rows],
        "read_on_d2h": [r["read_d2h_p999"] for r in rows],
    }

    ax_lat2 = ax_lat.twinx()
    for case in ("write_none", "write_on_h2d", "read_none", "read_on_d2h"):
        color = case_colors[case]
        ax_lat.plot(
            x_mb,
            p50_series[case],
            marker="o",
            linewidth=1.9,
            color=color,
            label=f"{case} p50",
        )
        ax_lat2.plot(
            x_mb,
            p999_series[case],
            marker="x",
            linewidth=1.6,
            linestyle="--",
            color=color,
            label=f"{case} p999",
        )

    ax_lat.set_xscale("log", base=2)
    ax_lat.set_xlabel("RDMA msg size (MiB, log2)")
    ax_lat.set_ylabel("P50 latency (us)")
    ax_lat2.set_ylabel("P999 latency (us)")
    ax_lat.set_title("Part B: Latency (left=P50, right=P999)")
    ax_lat.grid(alpha=0.25)

    h1, l1 = ax_lat.get_legend_handles_labels()
    h2, l2 = ax_lat2.get_legend_handles_labels()
    ax_lat.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def format_size_label(v: float) -> str:
    if v <= 0:
        return "0"
    if v < 1024:
        return f"{int(v)}B"
    if v < 1024 * 1024:
        return f"{int(round(v / 1024.0))}KiB"
    if v < 1024 * 1024 * 1024:
        m = v / (1024.0 * 1024.0)
        return f"{m:.0f}MiB" if m >= 10 else f"{m:.1f}MiB"
    g = v / (1024.0 * 1024.0 * 1024.0)
    return f"{g:.0f}GiB" if g >= 10 else f"{g:.1f}GiB"


def plot_part_b_sweep(
    none_map: dict,
    cont_map: dict,
    title: str,
    out_path: str,
    out_pdf: str = "",
    color: str = "#4C78A8",
    cont_label: str = "with background traffic",
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sizes = sorted(set(none_map.keys()) & set(cont_map.keys()))
    x_bytes = sizes
    none_thr = [none_map[s]["throughput_gib_s"] for s in sizes]
    cont_thr = [cont_map[s]["throughput_gib_s"] for s in sizes]

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.plot(
        x_bytes,
        none_thr,
        marker="o",
        markersize=4.8,
        linewidth=2.3,
        linestyle="-",
        color=color,
        label="isolated",
    )
    ax.plot(
        x_bytes,
        cont_thr,
        marker="o",
        markersize=4.8,
        linewidth=2.3,
        linestyle="--",
        color=color,
        label=cont_label,
    )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=2, numticks=12))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: format_size_label(v)))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("RDMA message size", fontsize=13, fontweight="semibold")
    ax.set_ylabel("RDMA throughput (GiB/s)")
    ax.set_title(title, fontsize=15, fontweight="semibold")
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.35)
    ax.legend(frameon=False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_part_b_ratio(comp_rows: list, out_path: str, out_pdf: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rows = sorted(comp_rows, key=lambda x: x["msg_bytes"])
    x_bytes = [r["msg_bytes"] for r in rows]
    w_thr = [r["write_thr_ratio"] for r in rows]
    r_thr = [r["read_thr_ratio"] for r in rows]

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.plot(x_bytes, w_thr, marker="o", markersize=4.6, linewidth=2.2, color="#4C78A8", label="write ratio (H2D / isolated)")
    ax.plot(x_bytes, r_thr, marker="o", markersize=4.6, linewidth=2.2, color="#F58518", label="read ratio (D2H / isolated)")
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="#666666")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=2, numticks=12))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: format_size_label(v)))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("RDMA message size", fontsize=13, fontweight="semibold")
    ax.set_ylabel("Throughput ratio")
    ax.set_title("Part B: Throughput Ratio Across Message Sizes", fontsize=15, fontweight="semibold")
    ax.grid(axis="y", linestyle=(0, (3, 3)), alpha=0.35)
    ax.legend(frameon=False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_part_b_four_cases_throughput(part_b_rows: list, out_path: str, out_pdf: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib as mpl

    # Use a clean paper style for this figure.
    plt.style.use("default")
    mpl.rcParams["axes.facecolor"] = "#ffffff"
    mpl.rcParams["figure.facecolor"] = "#ffffff"

    rows = sorted(part_b_rows, key=lambda x: x["msg_bytes"])
    x_bytes = [r["msg_bytes"] for r in rows]
    # Single-column friendly canvas with balanced aspect ratio.
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    write_color = "#4C78A8"
    read_color = "#F58518"
    ax.plot(
        x_bytes,
        [r["write_none_thr"] for r in rows],
        marker="o",
        markersize=4.6,
        linewidth=2.2,
        markevery=2,
        linestyle="-",
        color=write_color,
        label="write isolated",
    )
    ax.plot(
        x_bytes,
        [r["write_h2d_thr"] for r in rows],
        marker="^",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.1,
        linewidth=2.0,
        markevery=2,
        linestyle=(0, (2.2, 1.4)),
        color=write_color,
        label="write + H2D",
    )
    ax.plot(
        x_bytes,
        [r["read_none_thr"] for r in rows],
        marker="s",
        markersize=4.6,
        linewidth=2.2,
        markevery=2,
        linestyle="-",
        color=read_color,
        label="read isolated",
    )
    ax.plot(
        x_bytes,
        [r["read_d2h_thr"] for r in rows],
        marker="D",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.1,
        linewidth=2.0,
        markevery=2,
        linestyle=(0, (2.2, 1.4)),
        color=read_color,
        label="read + D2H",
    )

    # Use a log x-axis with sparser major ticks for better readability across 256B~100MB.
    ax.set_xscale("log")
    major_ticks = [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 100000000]
    major_tick_labels = ["256B", "1KB", "4KB", "16KB", "64KB", "256KB", "1MB", "4MB", "16MB", "100MB"]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(major_tick_labels)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("RDMA message size", fontsize=13, fontweight="semibold")
    ax.set_ylabel("RDMA throughput (GiB/s)", fontsize=13, fontweight="semibold")
    ax.grid(axis="y", linestyle=(0, (2, 3)), alpha=0.22, color="#7f7f7f")
    ax.legend(
        ncol=1,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        framealpha=0.92,
        fontsize=10.8,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.04),
        columnspacing=1.2,
        handletextpad=0.6,
    )
    for tick in ax.get_xticklabels():
        tick.set_rotation(24)
        tick.set_ha("right")
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_xlim(min(x_bytes) * 0.85, max(x_bytes) * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.45)
    ax.spines["bottom"].set_alpha(0.45)
    fig.tight_layout(pad=0.45)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pcie_contention_combined(part_a_rows: list, part_b_rows: list, out_path: str, out_pdf: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker

    plt.style.use("seaborn-v0_8-whitegrid")

    h2d_row = next((r for r in part_a_rows if r["direction"].lower() == "h2d"), None)
    d2h_row = next((r for r in part_a_rows if r["direction"].lower() == "d2h"), None)
    if h2d_row is None or d2h_row is None:
        raise ValueError("part_a_rows must include both h2d and d2h")

    rows = sorted(part_b_rows, key=lambda x: x["msg_bytes"])
    x_bytes = [r["msg_bytes"] for r in rows]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.4),
        gridspec_kw={"width_ratios": [0.98, 1.12]},
    )
    fig.patch.set_facecolor("#ffffff")
    ax_a.set_facecolor("#ffffff")
    ax_b.set_facecolor("#ffffff")

    # Panel (a): memcpy impact bars
    labels = ["H2D", "H2D + write", "D2H", "D2H + read"]
    values = [
        h2d_row["baseline_mean_bw_gib_s"],
        h2d_row["contended_mean_bw_gib_s"],
        d2h_row["baseline_mean_bw_gib_s"],
        d2h_row["contended_mean_bw_gib_s"],
    ]
    colors = ["#4C78A8", "#F58518", "#4C78A8", "#F58518"]
    edge_colors = ["#2f4b66", "#9a4f00", "#2f4b66", "#9a4f00"]
    x = [0.0, 0.22, 0.44, 0.66]
    width = 0.13
    ymax = max(values)
    legend_fontsize = 12.2
    axis_label_fontsize = 13.6
    tick_fontsize = 12.2
    value_fontsize = 12.6
    bars = ax_a.bar(
        x,
        values,
        width=width,
        color=colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        zorder=3,
    )
    for bar in bars:
        x_mid = bar.get_x() + bar.get_width() / 2
        y_top = bar.get_height()
        ax_a.text(
            x_mid,
            y_top + ymax * 0.012,
            f"{y_top:.2f}",
            ha="center",
            va="bottom",
            fontsize=value_fontsize,
            color="#203040",
            fontweight="semibold",
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=tick_fontsize)
    ax_a.set_ylabel("Memcpy throughput (GiB/s)", fontsize=axis_label_fontsize, fontweight="semibold")
    # Keep a zero baseline for bars, but trim excess headroom.
    ax_a.set_ylim(0, ymax * 1.10)
    ax_a.set_xlim(min(x) - 0.05, max(x) + 0.05)
    ax_a.tick_params(axis="y", labelsize=tick_fontsize)
    ax_a.tick_params(axis="x", labelsize=tick_fontsize, pad=3)
    for tick in ax_a.get_xticklabels():
        tick.set_fontweight("semibold")
    ax_a.grid(False)
    ax_a.xaxis.grid(False)
    ax_a.yaxis.grid(True, linestyle=(0, (3, 3)), alpha=0.30, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax_a.spines[side].set_visible(True)
        ax_a.spines[side].set_alpha(0.35)
        ax_a.spines[side].set_linewidth(0.85)
    legend_handles = [
        mpatches.Patch(facecolor="#4C78A8", edgecolor="#2f4b66", label="Memcpy isolated"),
        mpatches.Patch(facecolor="#F58518", edgecolor="#9a4f00", label="Memcpy with contending RDMA"),
    ]
    ax_a.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=1,
        frameon=False,
        fontsize=legend_fontsize,
        columnspacing=1.2,
        handletextpad=0.45,
        prop={"weight": "semibold", "size": legend_fontsize},
    )

    # Panel (b): throughput under four cases
    write_color = "#4C78A8"
    read_color = "#F58518"
    ax_b.plot(
        x_bytes,
        [r["write_none_thr"] for r in rows],
        linewidth=2.0,
        linestyle="-",
        color=write_color,
        label="write isolated",
    )
    ax_b.plot(
        x_bytes,
        [r["write_h2d_thr"] for r in rows],
        marker="^",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.8,
        markevery=2,
        linestyle=(0, (2.2, 1.4)),
        color=write_color,
        label="write + H2D",
    )
    ax_b.plot(
        x_bytes,
        [r["read_none_thr"] for r in rows],
        linewidth=2.0,
        linestyle="-",
        color=read_color,
        label="read isolated",
    )
    ax_b.plot(
        x_bytes,
        [r["read_d2h_thr"] for r in rows],
        marker="D",
        markersize=4.4,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=1.8,
        markevery=2,
        linestyle=(0, (2.2, 1.4)),
        color=read_color,
        label="read + D2H",
    )
    major_ticks = [256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 100000000]
    major_tick_labels = ["256B", "1KB", "4KB", "16KB", "64KB", "256KB", "1MB", "4MB", "16MB", "100MB"]
    ax_b.set_xscale("log")
    ax_b.set_xticks(major_ticks)
    ax_b.set_xticklabels(major_tick_labels)
    ax_b.xaxis.set_minor_locator(mticker.NullLocator())
    ax_b.set_xlabel("RDMA message size", fontsize=axis_label_fontsize, fontweight="semibold")
    ax_b.set_ylabel("RDMA throughput (GiB/s)", fontsize=axis_label_fontsize, fontweight="semibold")
    y_series = []
    for key in ("write_none_thr", "write_h2d_thr", "read_none_thr", "read_d2h_thr"):
        y_series.extend(r[key] for r in rows if r[key] > 0)
    if y_series:
        ymin = min(y_series)
        ymax_b = max(y_series)
        ypad = max((ymax_b - ymin) * 0.06, 0.35)
        ax_b.set_ylim(max(0.0, ymin - ypad), ymax_b + ypad)
    ax_b.grid(False)
    ax_b.xaxis.grid(False)
    ax_b.yaxis.grid(True, linestyle=(0, (2, 3)), alpha=0.22, color="#7f7f7f")
    ax_b.legend(
        ncol=1,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#d0d0d0",
        framealpha=0.92,
        fontsize=legend_fontsize,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.04),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    for tick in ax_b.get_xticklabels():
        tick.set_rotation(24)
        tick.set_ha("right")
    ax_b.tick_params(axis="x", labelsize=tick_fontsize)
    ax_b.tick_params(axis="y", labelsize=tick_fontsize)
    ax_b.set_xlim(min(x_bytes) * 0.85, max(x_bytes) * 1.12)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_alpha(0.45)
    ax_b.spines["bottom"].set_alpha(0.45)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.22, wspace=0.26)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        ensure_dir(out_dir)
    fig.savefig(out_path, dpi=220)
    if out_pdf:
        out_dir = os.path.dirname(out_pdf)
        if out_dir:
            ensure_dir(out_dir)
        fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="results/contention2_paper_v2")
    args = parser.parse_args()

    fig_dir = os.path.join(args.base_dir, "fig")
    summary_dir = os.path.join(args.base_dir, "summary")
    ensure_dir(fig_dir)
    part_a_csv = os.path.join(summary_dir, "partA_memcpy_impact.csv")
    part_b_csv = os.path.join(summary_dir, "partB_size_impact.csv")
    if not os.path.exists(part_b_csv):
        raise FileNotFoundError(f"missing summary csv: {part_b_csv}")

    # Read summary CSVs as-is (manual edits are preserved), and only redraw figures.
    part_b_rows = load_part_b_summary(part_b_csv)
    write_none = {r["msg_bytes"]: {"throughput_gib_s": r["write_none_thr"]} for r in part_b_rows}
    write_h2d = {r["msg_bytes"]: {"throughput_gib_s": r["write_h2d_thr"]} for r in part_b_rows}
    read_none = {r["msg_bytes"]: {"throughput_gib_s": r["read_none_thr"]} for r in part_b_rows}
    read_d2h = {r["msg_bytes"]: {"throughput_gib_s": r["read_d2h_thr"]} for r in part_b_rows}

    paper_fig_dir = os.path.join("paper", "figures")
    if os.path.exists(part_a_csv):
        part_a_rows = load_part_a_summary(part_a_csv)
        part_a_png = os.path.join(fig_dir, "partA_memcpy_impact_bar.png")
        part_a_png_paper = os.path.join(paper_fig_dir, "partA_memcpy_impact_bar.png")
        part_a_pdf_paper = os.path.join(paper_fig_dir, "partA_memcpy_impact_bar.pdf")
        plot_part_a(
            part_a_rows,
            part_a_png,
            out_png_paper=part_a_png_paper,
            out_pdf_paper=part_a_pdf_paper,
        )
        plot_pcie_contention_combined(
            part_a_rows,
            part_b_rows,
            out_path=os.path.join(fig_dir, "pcie_contention_combined.png"),
            out_pdf=os.path.join(paper_fig_dir, "pcie_contention_combined.pdf"),
        )
    else:
        print(f"[warn] skip Part A plot; summary not found: {part_a_csv}")
    plot_part_b_sweep(
        write_none,
        write_h2d,
        "Part B Write Sweep: Isolated vs +H2D",
        os.path.join(fig_dir, "partB_write_sweep_throughput.png"),
        out_pdf=os.path.join(fig_dir, "partB_write_sweep_throughput.pdf"),
        color="#4C78A8",
        cont_label="+H2D (dashed)",
    )
    plot_part_b_sweep(
        read_none,
        read_d2h,
        "Part B Read Sweep: Isolated vs +D2H",
        os.path.join(fig_dir, "partB_read_sweep_throughput.png"),
        out_pdf=os.path.join(fig_dir, "partB_read_sweep_throughput.pdf"),
        color="#F58518",
        cont_label="+D2H (dashed)",
    )
    plot_part_b_ratio(
        part_b_rows,
        os.path.join(fig_dir, "partB_ratio_vs_size.png"),
        out_pdf=os.path.join(fig_dir, "partB_ratio_vs_size.pdf"),
    )
    plot_part_b_four_cases_throughput(
        part_b_rows,
        out_path=os.path.join(fig_dir, "partB_fourcases_throughput.png"),
        out_pdf=os.path.join(paper_fig_dir, "partB_fourcases_throughput.pdf"),
    )
    for legacy_name in ("partB_fourcases_p50.png", "partB_fourcases_p99.png", "partB_fourcases_p50_p99.png", "partB_fourcases_latency.png"):
        legacy_path = os.path.join(fig_dir, legacy_name)
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

    print(f"[ok] summary (read-only): {summary_dir}")
    print(f"[ok] fig: {fig_dir}")


if __name__ == "__main__":
    main()
