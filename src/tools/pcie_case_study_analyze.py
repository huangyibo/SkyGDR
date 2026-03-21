#!/usr/bin/env python3

import argparse
import csv
import math
import os
from pathlib import Path

PROTECT_MODES = ("LOW1", "LOW2", "LOW3", "LOW4", "LOW", "CUSTOM")


def to_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def load_csv_rows(path: Path):
    with path.open("r", newline="") as fp:
        return list(csv.DictReader(fp))


def is_protect_mode(mode: str) -> bool:
    return mode in PROTECT_MODES


def mode_level(mode: str) -> int:
    if mode == "LOW1" or mode == "LOW":
        return 1
    if mode == "LOW2":
        return 2
    if mode == "LOW3":
        return 3
    if mode == "LOW4":
        return 4
    if mode == "CUSTOM":
        return 1
    return 0


def final_restore_stats(path_text: str):
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    rows = load_csv_rows(path)
    if not rows:
        return None
    last = rows[-1]
    return {
        "elapsed_ms": to_float(last.get("elapsed_ms")),
        "avg_bw_gib_s": to_float(last.get("avg_bw_gib_s")),
    }


def nearest_row(rows, ts_unix_ms, max_delta_ms=500):
    if not rows:
        return None
    best = None
    best_delta = None
    for row in rows:
        row_ts = to_int(row.get("ts_unix_ms"))
        delta = abs(row_ts - ts_unix_ms)
        if best is None or delta < best_delta:
            best = row
            best_delta = delta
    if best_delta is None or best_delta > max_delta_ms:
        return None
    return best


def extract_controller_transitions(rows):
    transitions = []
    prev_mode = None
    for row in rows:
        mode = row.get("mode", "")
        if (mode == "HIGH" or is_protect_mode(mode)) and mode != prev_mode:
            transitions.append((mode, to_int(row.get("ts_unix_ms"))))
            prev_mode = mode
    return transitions


def extract_bg_transitions(rows):
    transitions = []
    prev_mode = None
    for row in rows:
        mode = row.get("control_mode", "")
        if (mode == "HIGH" or is_protect_mode(mode)) and mode != prev_mode:
            transitions.append((mode, to_int(row.get("ts_unix_ms"))))
            prev_mode = mode
    return transitions


def estimate_bg_clock_offset_ms(controller_rows, bg_rows):
    ctrl_transitions = extract_controller_transitions(controller_rows)
    bg_transitions = extract_bg_transitions(bg_rows)
    if len(ctrl_transitions) < 2 or len(bg_transitions) < 2:
        return 0, "absolute_ts_only"

    # Ignore the initial steady-state mode and align subsequent control changes.
    ctrl_changes = ctrl_transitions[1:]
    bg_changes = bg_transitions[1:]
    if not ctrl_changes or not bg_changes:
        return 0, "absolute_ts_only"

    best_offsets = None
    best_score = None
    need = len(ctrl_changes)
    for start in range(0, len(bg_changes) - need + 1):
        cand = bg_changes[start:start + need]
        cand_modes = [mode for mode, _ in cand]
        ctrl_modes = [mode for mode, _ in ctrl_changes]
        if cand_modes != ctrl_modes:
            continue

        offsets = [bg_ts - ctrl_ts for (_, ctrl_ts), (_, bg_ts) in zip(ctrl_changes, cand)]
        offsets_sorted = sorted(offsets)
        median = offsets_sorted[len(offsets_sorted) // 2]
        spread = max(abs(x - median) for x in offsets)
        score = (spread, abs(median))
        if best_score is None or score < best_score:
            best_score = score
            best_offsets = offsets_sorted

    if not best_offsets:
        return 0, "absolute_ts_only"

    return best_offsets[len(best_offsets) // 2], "matched_control_transitions"


def parse_args():
    ap = argparse.ArgumentParser(description="Merge and plot the PCIe case-study timeline.")
    ap.add_argument("--controller_csv", required=True)
    ap.add_argument("--bg_ts_csv", default="", help="Optional cpu_client time-series CSV copied back from the CPU machine")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_png", default="")
    ap.add_argument("--paper_out_pdf", default="")
    ap.add_argument("--plot_start_s", type=float, default=0.0)
    ap.add_argument("--plot_end_s", type=float, default=-1.0)
    ap.add_argument("--rx_threshold_pct", type=float, default=85.0)
    ap.add_argument("--pcie_peak_gib_s", type=float, default=29.0)
    ap.add_argument("--always_high_restore_csv", default="")
    ap.add_argument("--always_low_restore_csv", default="")
    ap.add_argument("--controlled_restore_csv", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    controller_rows = load_csv_rows(Path(args.controller_csv))
    bg_rows = load_csv_rows(Path(args.bg_ts_csv)) if args.bg_ts_csv else []
    bg_clock_offset_ms, align_method = estimate_bg_clock_offset_ms(controller_rows, bg_rows) if bg_rows else (0, "no_bg")

    merged = []
    for row in controller_rows:
        ts_unix_ms = to_int(row.get("ts_unix_ms"))
        bg_row = nearest_row(bg_rows, ts_unix_ms + bg_clock_offset_ms) if bg_rows else None
        restore_done = to_int(row.get("restore_done"))
        restore_inst_bw = to_float(row.get("restore_inst_bw_gib_s"))
        restore_smooth_bw = to_float(row.get("restore_smooth_bw_gib_s"))
        if restore_done:
            restore_inst_bw = 0.0
            restore_smooth_bw = 0.0
        merged.append(
            {
                "ts_unix_ms": ts_unix_ms,
                "mode": row.get("mode", ""),
                "decision": row.get("decision", ""),
                "restore_done": restore_done,
                "restore_elapsed_ms": to_int(row.get("restore_elapsed_ms")),
                "restore_completed_bytes": to_int(row.get("restore_completed_bytes")),
                "restore_remaining_bytes": to_int(row.get("restore_remaining_bytes")),
                "restore_inst_bw_gib_s": restore_inst_bw,
                "restore_smooth_bw_gib_s": restore_smooth_bw,
                "restore_guard_bw_gib_s": to_float(row.get("restore_guard_bw_gib_s")),
                "restore_avg_bw_gib_s": to_float(row.get("restore_avg_bw_gib_s")),
                "pcie_tx_util_pct": to_float(row.get("pcie_tx_util_pct")),
                "pcie_rx_util_pct": to_float(row.get("pcie_rx_util_pct")),
                "baseline_restore_gib_s": to_float(row.get("baseline_restore_gib_s")),
                "enter_thresh_gib_s": to_float(row.get("enter_thresh_gib_s")),
                "exit_thresh_gib_s": to_float(row.get("exit_thresh_gib_s")),
                "bg_throughput_gib_s": to_float(bg_row.get("throughput_gib_s")) if bg_row else float("nan"),
                "bg_pace_sleep_us": to_float(bg_row.get("pace_sleep_us")) if bg_row else float("nan"),
                "bg_p99_us": to_float(bg_row.get("p99_us")) if bg_row else float("nan"),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "ts_unix_ms",
                "mode",
                "decision",
                "restore_done",
                "restore_elapsed_ms",
                "restore_completed_bytes",
                "restore_remaining_bytes",
                "restore_inst_bw_gib_s",
                "restore_smooth_bw_gib_s",
                "restore_guard_bw_gib_s",
                "restore_avg_bw_gib_s",
                "pcie_tx_util_pct",
                "pcie_rx_util_pct",
                "baseline_restore_gib_s",
                "enter_thresh_gib_s",
                "exit_thresh_gib_s",
                "bg_throughput_gib_s",
                "bg_pace_sleep_us",
                "bg_p99_us",
            ]
        )
        for row in merged:
            writer.writerow(
                [
                    row["ts_unix_ms"],
                    row["mode"],
                    row["decision"],
                    row["restore_done"],
                    row["restore_elapsed_ms"],
                    row["restore_completed_bytes"],
                    row["restore_remaining_bytes"],
                    "" if math.isnan(row["restore_inst_bw_gib_s"]) else f"{row['restore_inst_bw_gib_s']:.6f}",
                    "" if math.isnan(row["restore_smooth_bw_gib_s"]) else f"{row['restore_smooth_bw_gib_s']:.6f}",
                    "" if math.isnan(row["restore_guard_bw_gib_s"]) else f"{row['restore_guard_bw_gib_s']:.6f}",
                    "" if math.isnan(row["restore_avg_bw_gib_s"]) else f"{row['restore_avg_bw_gib_s']:.6f}",
                    "" if math.isnan(row["pcie_tx_util_pct"]) else f"{row['pcie_tx_util_pct']:.6f}",
                    "" if math.isnan(row["pcie_rx_util_pct"]) else f"{row['pcie_rx_util_pct']:.6f}",
                    "" if math.isnan(row["baseline_restore_gib_s"]) else f"{row['baseline_restore_gib_s']:.6f}",
                    "" if math.isnan(row["enter_thresh_gib_s"]) else f"{row['enter_thresh_gib_s']:.6f}",
                    "" if math.isnan(row["exit_thresh_gib_s"]) else f"{row['exit_thresh_gib_s']:.6f}",
                    "" if math.isnan(row["bg_throughput_gib_s"]) else f"{row['bg_throughput_gib_s']:.6f}",
                    "" if math.isnan(row["bg_pace_sleep_us"]) else f"{row['bg_pace_sleep_us']:.6f}",
                    "" if math.isnan(row["bg_p99_us"]) else f"{row['bg_p99_us']:.6f}",
                ]
            )

    if args.out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[plot] matplotlib unavailable: {exc}")
            return 0

        t0 = merged[0]["ts_unix_ms"] if merged else 0
        t_s = [(row["ts_unix_ms"] - t0) / 1000.0 for row in merged]
        restore_bw = [
            0.0 if row["restore_done"]
            else (
                row["restore_smooth_bw_gib_s"] if not math.isnan(row["restore_smooth_bw_gib_s"])
                else row["restore_inst_bw_gib_s"]
            )
            for row in merged
        ]
        baseline = [row["baseline_restore_gib_s"] for row in merged]
        enter_line = [row["enter_thresh_gib_s"] for row in merged]
        pcie_rx = [row["pcie_rx_util_pct"] for row in merged]
        bg_thr = [row["bg_throughput_gib_s"] for row in merged]
        low_mask = [1 if is_protect_mode(row["mode"]) else 0 for row in merged]

        has_bg = any(not math.isnan(x) for x in bg_thr)
        fig, axes = plt.subplots(3 if has_bg else 2, 1, sharex=True, figsize=(12, 7))
        if not isinstance(axes, (list, tuple)):
            axes = list(axes)

        ax0 = axes[0]
        ax0.plot(t_s, restore_bw, label="restore_smooth_bw_gib_s", color="#1f77b4")
        ax0.plot(t_s, baseline, "--", label="baseline_restore_gib_s", color="#2ca02c")
        ax0.plot(t_s, enter_line, ":", label="enter_threshold_gib_s", color="#d62728")
        ax0.set_ylabel("Restore GiB/s")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right")

        ax1 = axes[1]
        ax1.plot(t_s, pcie_rx, label="pcie_rx_util_pct", color="#ff7f0e")
        ax1.fill_between(t_s, 0, 100, where=low_mask, alpha=0.12, color="#9467bd", label="LOW mode")
        ax1.set_ylabel("PCIe RX %")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        if has_bg:
            ax2 = axes[2]
            ax2.plot(t_s, bg_thr, label="bg_throughput_gib_s", color="#8c564b")
            ax2.set_ylabel("BG GiB/s")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right")
            ax2.set_xlabel("time_s")
        else:
            axes[-1].set_xlabel("time_s")

        fig.tight_layout()
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    if args.paper_out_pdf:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            from matplotlib import transforms as mtransforms
        except Exception as exc:
            print(f"[plot] matplotlib unavailable: {exc}")
            return 0

        plt.style.use("seaborn-v0_8-whitegrid")

        t0 = merged[0]["ts_unix_ms"] if merged else 0
        full_t_s = [(row["ts_unix_ms"] - t0) / 1000.0 for row in merged]
        plot_end_s = args.plot_end_s if args.plot_end_s >= 0 else (full_t_s[-1] if full_t_s else 0.0)

        filtered = [
            (t_s, row)
            for t_s, row in zip(full_t_s, merged)
            if t_s >= args.plot_start_s and t_s <= plot_end_s
        ]
        if not filtered:
            print("[plot] no rows left after applying plot window")
            return 0

        t_s = [t for t, _ in filtered]
        rows = [row for _, row in filtered]
        restore_bw_raw = [
            0.0 if row["restore_done"]
            else (
                row["restore_smooth_bw_gib_s"] if not math.isnan(row["restore_smooth_bw_gib_s"])
                else row["restore_inst_bw_gib_s"]
            )
            for row in rows
        ]
        restore_bw = []
        restore_started = False
        restore_ended = False
        restore_begin_x = None
        restore_end_x = None
        for t, bw in zip(t_s, restore_bw_raw):
            if math.isnan(bw):
                restore_bw.append(float("nan"))
                continue
            if restore_ended:
                restore_bw.append(float("nan"))
                continue
            restore_bw.append(bw)
            if bw > 0.1 and not restore_started:
                restore_started = True
                restore_begin_x = t
            elif restore_started and bw <= 0.1:
                restore_ended = True
                restore_end_x = t
        enter_line = [row["enter_thresh_gib_s"] for row in rows]
        bg_thr = [
            row["bg_throughput_gib_s"] if not math.isnan(row["bg_throughput_gib_s"]) else float("nan")
            for row in rows
        ]
        pcie_bw = [
            (row["pcie_rx_util_pct"] / 100.0) * args.pcie_peak_gib_s
            if not math.isnan(row["pcie_rx_util_pct"]) else float("nan")
            for row in rows
        ]

        stats_high = final_restore_stats(args.always_high_restore_csv)
        stats_low = final_restore_stats(args.always_low_restore_csv)
        stats_ctrl = final_restore_stats(args.controlled_restore_csv)
        markevery = max(1, len(t_s) // 10)

        fig = plt.figure(figsize=(3.45, 2.45))
        fig.patch.set_facecolor("#ffffff")
        gs = fig.add_gridspec(1, 1)
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_facecolor("#ffffff")

        h2d_color = "#4C78A8"
        rdma_color = "#F58518"
        threshold_color = h2d_color

        ax0.plot(
            t_s,
            restore_bw,
            marker="o",
            markersize=3.4,
            markerfacecolor="white",
            markeredgewidth=0.85,
            markevery=markevery,
            linewidth=2.2,
            linestyle="-",
            color=h2d_color,
            label="H2D",
        )
        ax0.plot(
            t_s,
            bg_thr,
            marker="s",
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.8,
            markevery=markevery,
            linewidth=2.0,
            linestyle="-",
            color=rdma_color,
            label="RDMA write",
        )
        ax0.plot(
            t_s,
            enter_line,
            linewidth=1.7,
            linestyle=(0, (3, 2)),
            color=threshold_color,
            alpha=0.8,
            label="H2D target",
        )
        ax0.set_ylabel("Bandwidth (GiB/s)", fontsize=8.0, fontweight="semibold")
        ax0.set_xlim(args.plot_start_s, plot_end_s)
        max_y = max(
            max(restore_bw) if restore_bw else 0.0,
            max((x for x in bg_thr if not math.isnan(x)), default=0.0),
            max((x for x in pcie_bw if not math.isnan(x)), default=0.0),
            max(enter_line) if enter_line else 0.0,
        )
        ax0.set_ylim(0.0, max(30.0, max_y * 1.05))
        ax0.tick_params(axis="y", labelsize=9.2)
        ax0.tick_params(axis="x", labelsize=9.2)
        ax0.grid(False)
        ax0.xaxis.grid(False)
        ax0.yaxis.grid(True, linestyle=(0, (2, 3)), alpha=0.22, color="#7f7f7f")
        ax0.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax0.set_xlabel("Time (s)", fontsize=7.0, fontweight="semibold", labelpad=0.8)

        text_transform = mtransforms.blended_transform_factory(ax0.transData, ax0.transAxes)
        for x_pos, label in ((restore_begin_x, "restore begin"), (restore_end_x, "restore end")):
            if x_pos is None:
                continue
            ax0.plot(
                [x_pos, x_pos],
                [0.0, -0.12],
                transform=text_transform,
                clip_on=False,
                color=h2d_color,
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                alpha=0.65,
                zorder=0,
            )
            ax0.text(
                x_pos,
                -0.14,
                label,
                transform=text_transform,
                ha="center",
                va="top",
                fontsize=7.8,
                color=h2d_color,
            )
        ax0.plot(
            t_s,
            pcie_bw,
            linewidth=1.9,
            linestyle=(0, (3, 2)),
            color="#B279A2",
            label="PCIe RX",
        )

        handles0, labels0 = ax0.get_legend_handles_labels()
        fig.legend(
            handles0,
            labels0,
            frameon=False,
            ncol=4,
            fontsize=6.9,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.875),
            borderaxespad=0.0,
            handletextpad=0.5,
            columnspacing=0.9,
        )

        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.spines["left"].set_linewidth(0.85)
        ax0.spines["bottom"].set_linewidth(0.85)
        ax0.spines["left"].set_alpha(0.45)
        ax0.spines["bottom"].set_alpha(0.45)
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.29, top=0.80)

        paper_out = Path(args.paper_out_pdf)
        paper_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_out, bbox_inches="tight")
        plt.close(fig)

    if merged:
        restore_finish_ms = max(row["restore_elapsed_ms"] for row in merged)
        low_samples = sum(1 for row in merged if is_protect_mode(row["mode"]))
        print(f"restore_finish_ms={restore_finish_ms}")
        print(f"low_samples={low_samples}")
        print(f"bg_align_method={align_method}")
        print(f"bg_clock_offset_ms={bg_clock_offset_ms}")
        if bg_rows:
            vals = [row["bg_throughput_gib_s"] for row in merged if not math.isnan(row["bg_throughput_gib_s"])]
            if vals:
                print(f"bg_throughput_mean_gib_s={sum(vals) / len(vals):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
