#!/usr/bin/env python3
"""
gpu_metrics_logger.py

Periodically log GPU metrics to CSV for contention analysis.

Adds "PCIe fabric occupancy" style signals by deriving utilization (%)
from NVML PCIe throughput and the GPU's PCIe link speed (gen/width).

Notes:
- NVML nvmlDeviceGetPcieThroughput returns a moving-average throughput sample
  in KB/s for TX/RX (host<->device). In practice this is commonly treated as KiB/s.
- "Utilization %" here means: throughput / (PCIe link theoretical one-direction bandwidth).
  This is an approximation but is usually good enough to spot contention and correlate with tail latency.

Example:
  python3 gpu_metrics_logger.py --gpu 0 --interval_ms 200 --out results/contention2/gpu_metrics.csv
  python3 gpu_metrics_logger.py --gpu 0 --interval_ms 200 --out - --live
"""

import argparse
import csv
import math
import os
import sys
import time

import pynvml

try:
    import psutil
except Exception:
    psutil = None


def safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def pcie_per_lane_GB_s(gen: int) -> float:
    """Approx effective GB/s per lane per direction for PCIe generations.

    Uses commonly cited usable throughput approximations:
      Gen1 ~0.25 GB/s, Gen2 ~0.50 GB/s, Gen3 ~0.985 GB/s,
      Gen4 ~1.969 GB/s, Gen5 ~3.938 GB/s
    """
    table = {
        1: 0.250,
        2: 0.500,
        3: 0.985,
        4: 1.969,
        5: 3.938,
        # Gen6 is uncommon on GPUs today; keep placeholder.
        6: 7.877,
    }
    return table.get(int(gen or 0), 0.0)


def kib_s_to_GB_s(kib_s: float) -> float:
    # NVML reports KB/s; we treat it as KiB/s and convert to decimal GB/s.
    return (kib_s * 1024.0) / 1e9


def kib_s_to_GiB_s(kib_s: float) -> float:
    return kib_s / (1024.0 * 1024.0)


def GB_s_to_GiB_s(GB_s: float) -> float:
    return (GB_s * 1e9) / (1024.0 * 1024.0 * 1024.0)


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def plot_csv(csv_path: str, out_path: str = None, show: bool = False, title: str = None):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}", file=sys.stderr)
        return 1

    if not os.path.exists(csv_path):
        print(f"[plot] csv not found: {csv_path}", file=sys.stderr)
        return 2

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            print("[plot] csv is empty", file=sys.stderr)
            return 3
        cols = reader.fieldnames or []

    # Prefer relative time if present; fallback to unix timestamp; else use sample index.
    time_col = None
    if "t_ms" in cols:
        time_col = "t_ms"
    elif "ts_unix_ms" in cols:
        time_col = "ts_unix_ms"

    ts_ms = []
    data = {c: [] for c in cols}
    for r in rows:
        if not time_col:
            continue
        if time_col not in r:
            continue
        t = _to_float(r.get(time_col, "nan"))
        ts_ms.append(t)
        for c in cols:
            data[c].append(_to_float(r.get(c, "nan")))

    if not time_col:
        # Fallback: plot against row index.
        ts_ms = list(range(len(rows)))
        for c in cols:
            data[c] = [_to_float(r.get(c, "nan")) for r in rows]
        time_col = "sample_idx"

    if time_col == "t_ms":
        t_s = [x / 1000.0 for x in ts_ms]
    elif time_col == "ts_unix_ms":
        t0 = ts_ms[0]
        t_s = [(x - t0) / 1000.0 for x in ts_ms]
    else:
        t_s = [float(x) for x in ts_ms]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    ax_pcie, ax_cum, ax_util, ax_clk = axes

    # --- PCIe subplot ---
    pcie_plotted = False
    if "pcie_tx_GiB_s" in data:
        ax_pcie.plot(t_s, data["pcie_tx_GiB_s"], label="pcie_tx_GiB_s")
        pcie_plotted = True
    elif "pcie_tx_GB_s" in data:
        tx_GiB_s = [GB_s_to_GiB_s(x) if not _is_nan(x) else float("nan") for x in data["pcie_tx_GB_s"]]
        ax_pcie.plot(t_s, tx_GiB_s, label="pcie_tx_GiB_s")
        pcie_plotted = True
    elif "pcie_tx_kbps" in data:
        tx_GiB_s = [kib_s_to_GiB_s(x) if not _is_nan(x) else float("nan") for x in data["pcie_tx_kbps"]]
        ax_pcie.plot(t_s, tx_GiB_s, label="pcie_tx_GiB_s")
        pcie_plotted = True

    if "pcie_rx_GiB_s" in data:
        ax_pcie.plot(t_s, data["pcie_rx_GiB_s"], label="pcie_rx_GiB_s")
        pcie_plotted = True
    elif "pcie_rx_GB_s" in data:
        rx_GiB_s = [GB_s_to_GiB_s(x) if not _is_nan(x) else float("nan") for x in data["pcie_rx_GB_s"]]
        ax_pcie.plot(t_s, rx_GiB_s, label="pcie_rx_GiB_s")
        pcie_plotted = True
    elif "pcie_rx_kbps" in data:
        rx_GiB_s = [kib_s_to_GiB_s(x) if not _is_nan(x) else float("nan") for x in data["pcie_rx_kbps"]]
        ax_pcie.plot(t_s, rx_GiB_s, label="pcie_rx_GiB_s")
        pcie_plotted = True

    if "pcie_total_GiB_s" in data:
        ax_pcie.plot(t_s, data["pcie_total_GiB_s"], label="pcie_total_GiB_s", linewidth=2.0)
        pcie_plotted = True

    ax_pcie.set_ylabel("PCIe GiB/s")
    ax_pcie.grid(True, alpha=0.3)

    ax_pcie_r = None
    if "pcie_tx_util_pct" in data or "pcie_rx_util_pct" in data:
        ax_pcie_r = ax_pcie.twinx()
        if "pcie_tx_util_pct" in data:
            ax_pcie_r.plot(t_s, data["pcie_tx_util_pct"], "--", label="pcie_tx_util_pct", alpha=0.7)
            pcie_plotted = True
        if "pcie_rx_util_pct" in data:
            ax_pcie_r.plot(t_s, data["pcie_rx_util_pct"], "--", label="pcie_rx_util_pct", alpha=0.7)
            pcie_plotted = True
        ax_pcie_r.set_ylabel("PCIe util %")

    # Combined legend
    if pcie_plotted:
        handles, labels = ax_pcie.get_legend_handles_labels()
        if ax_pcie_r:
            h2, l2 = ax_pcie_r.get_legend_handles_labels()
            handles += h2
            labels += l2
        ax_pcie.legend(handles, labels, loc="upper right")
    else:
        ax_pcie.set_visible(False)

    # --- cumulative PCIe volume subplot ---
    cum_plotted = False
    if "pcie_tx_cum_GiB" in data:
        ax_cum.plot(t_s, data["pcie_tx_cum_GiB"], label="pcie_tx_cum_GiB")
        cum_plotted = True
    if "pcie_rx_cum_GiB" in data:
        ax_cum.plot(t_s, data["pcie_rx_cum_GiB"], label="pcie_rx_cum_GiB")
        cum_plotted = True
    if "pcie_total_cum_GiB" in data:
        ax_cum.plot(t_s, data["pcie_total_cum_GiB"], label="pcie_total_cum_GiB", linewidth=2.0)
        cum_plotted = True
    ax_cum.set_ylabel("PCIe cum GiB")
    ax_cum.grid(True, alpha=0.3)
    if cum_plotted:
        ax_cum.legend(loc="upper left")
    else:
        ax_cum.set_visible(False)

    # --- GPU utilization subplot ---
    util_plotted = False
    if "util_gpu_pct" in data:
        ax_util.plot(t_s, data["util_gpu_pct"], label="util_gpu_pct")
        util_plotted = True
    if "util_mem_pct" in data:
        ax_util.plot(t_s, data["util_mem_pct"], label="util_mem_pct")
        util_plotted = True
    if "gpu_mem_used_GiB" in data:
        ax_util_r = ax_util.twinx()
        ax_util_r.plot(t_s, data["gpu_mem_used_GiB"], "--", label="gpu_mem_used_GiB", alpha=0.7)
        ax_util_r.set_ylabel("GPU mem GiB")
        util_plotted = True
    else:
        ax_util_r = None
    if "cpu_util_pct" in data:
        ax_util.plot(t_s, data["cpu_util_pct"], label="cpu_util_pct", alpha=0.7)
        util_plotted = True
    ax_util.set_ylabel("Util %")
    ax_util.grid(True, alpha=0.3)
    if util_plotted:
        handles, labels = ax_util.get_legend_handles_labels()
        if ax_util_r:
            h2, l2 = ax_util_r.get_legend_handles_labels()
            handles += h2
            labels += l2
        ax_util.legend(handles, labels, loc="upper right")
    else:
        ax_util.set_visible(False)

    # --- Clocks / Power subplot ---
    clk_plotted = False
    if "sm_clock_mhz" in data:
        ax_clk.plot(t_s, data["sm_clock_mhz"], label="sm_clock_mhz")
        clk_plotted = True
    if "mem_clock_mhz" in data:
        ax_clk.plot(t_s, data["mem_clock_mhz"], label="mem_clock_mhz")
        clk_plotted = True
    ax_clk.set_ylabel("Clock MHz")
    ax_clk.grid(True, alpha=0.3)

    ax_clk_r = None
    if "power_w" in data:
        ax_clk_r = ax_clk.twinx()
        ax_clk_r.plot(t_s, data["power_w"], "--", label="power_w", alpha=0.7)
        ax_clk_r.set_ylabel("Power W")
        clk_plotted = True

    if clk_plotted:
        handles, labels = ax_clk.get_legend_handles_labels()
        if ax_clk_r:
            h2, l2 = ax_clk_r.get_legend_handles_labels()
            handles += h2
            labels += l2
        ax_clk.legend(handles, labels, loc="upper right")
    else:
        ax_clk.set_visible(False)

    axes[-1].set_xlabel("time_s")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"[plot] saved {out_path}", file=sys.stderr)
    if show:
        plt.show()
    plt.close(fig)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="GPU index")
    ap.add_argument("--interval_ms", type=int, default=200, help="sampling interval in ms")
    ap.add_argument("--out", default="-", help="output CSV path or '-' for stdout")
    ap.add_argument("--append", action="store_true", help="append to existing file (skip header if non-empty)")
    ap.add_argument("--live", action="store_true", help="print a human-readable live line to stderr")
    ap.add_argument("--plot", action="store_true", help="plot CSV and exit (requires matplotlib)")
    ap.add_argument("--plot_in", default=None, help="CSV path to plot (default: --out)")
    ap.add_argument("--plot_out", default=None, help="output image path (png)")
    ap.add_argument("--plot_show", action="store_true", help="show interactive window")
    ap.add_argument("--plot_title", default=None, help="plot title")
    ap.add_argument("--plot_on_exit", action="store_true", help="after logging, plot --out to image")
    args = ap.parse_args()

    if args.plot:
        plot_in = args.plot_in or (args.out if args.out not in ("", "-") else None)
        if not plot_in:
            print("[plot] need --plot_in or a file path via --out", file=sys.stderr)
            return 2
        return plot_csv(plot_in, out_path=args.plot_out, show=args.plot_show, title=args.plot_title)

    if args.interval_ms <= 0:
        args.interval_ms = 200

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")

    # NVML function availability varies slightly by driver/pynvml version.
    get_curr_gen = getattr(pynvml, "nvmlDeviceGetCurrPcieLinkGeneration", None)
    get_curr_wid = getattr(pynvml, "nvmlDeviceGetCurrPcieLinkWidth", None)
    get_max_gen = getattr(pynvml, "nvmlDeviceGetMaxPcieLinkGeneration", None)
    get_max_wid = getattr(pynvml, "nvmlDeviceGetMaxPcieLinkWidth", None)
    get_replay = getattr(pynvml, "nvmlDeviceGetPcieReplayCounter", None)

    out_fp = None
    if args.out == "-" or args.out == "":
        out_fp = sys.stdout
    else:
        mode = "a" if args.append else "w"
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_fp = open(args.out, mode, buffering=1)

    # Keep the CSV minimal by default: only TX/RX-related metrics with explicit units.
    # Record both unix time and relative time so cross-process correlation stays robust.
    header = (
        "ts_unix_ms,t_ms,"
        "pcie_link_gen,pcie_link_width,pcie_link_ref_GiB_s,"
        "pcie_tx_GiB_s,pcie_rx_GiB_s,pcie_total_GiB_s,"
        "pcie_tx_util_pct,pcie_rx_util_pct,pcie_total_util_pct,"
        "pcie_tx_peak_util_pct,pcie_rx_peak_util_pct,pcie_total_peak_util_pct,"
        "pcie_tx_cum_GiB,pcie_rx_cum_GiB,pcie_total_cum_GiB,"
        "util_gpu_pct,util_mem_pct,"
        "gpu_mem_used_GiB,gpu_mem_total_GiB,gpu_mem_used_pct,"
        "sm_clock_mhz,mem_clock_mhz,power_w,"
        "cpu_util_pct,cpu_mem_used_GiB,cpu_mem_avail_GiB,cpu_mem_used_pct"
    )

    if out_fp is not sys.stdout:
        if args.append and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
            pass
        else:
            out_fp.write(header + "\n")
    else:
        out_fp.write(header + "\n")

    # Print one-time link capability info (TX/RX upper bound).
    curr_gen = safe_call(lambda: get_curr_gen(handle), -1) if get_curr_gen else -1
    curr_wid = safe_call(lambda: get_curr_wid(handle), -1) if get_curr_wid else -1
    max_gen = safe_call(lambda: get_max_gen(handle), -1) if get_max_gen else -1
    max_wid = safe_call(lambda: get_max_wid(handle), -1) if get_max_wid else -1

    link_cur_GiB_s = GB_s_to_GiB_s(pcie_per_lane_GB_s(curr_gen) * (curr_wid if curr_wid and curr_wid > 0 else 0))
    link_max_GiB_s = GB_s_to_GiB_s(pcie_per_lane_GB_s(max_gen) * (max_wid if max_wid and max_wid > 0 else 0))
    link_ref_GiB_s = link_max_GiB_s if link_max_GiB_s > 0 else link_cur_GiB_s

    print(f"[logger] GPU {args.gpu}: {name}", file=sys.stderr)
    print(f"[logger] interval_ms={args.interval_ms} out={args.out}", file=sys.stderr)
    if link_ref_GiB_s > 0:
        print(
            f"[pcie] TX/RX upper bound (per direction): "
            f"max gen{max_gen}x{max_wid}≈{link_max_GiB_s:.2f}GiB/s, "
            f"current gen{curr_gen}x{curr_wid}≈{link_cur_GiB_s:.2f}GiB/s; "
            f"NVML unit≈KiB/s (converted to GiB/s)",
            file=sys.stderr,
        )
    else:
        print("[pcie] link info unavailable; logging raw NVML KiB/s only", file=sys.stderr)

    interval_s = args.interval_ms / 1000.0
    next_t = time.time()
    t0 = time.monotonic()

    tx_peak_pct = 0.0
    rx_peak_pct = 0.0
    total_peak_pct = 0.0
    tx_cum_gib = 0.0
    rx_cum_gib = 0.0
    last_sample_t = None

    try:
        while True:
            now = time.time()
            if now < next_t:
                time.sleep(min(0.05, next_t - now))
                continue

            ts_unix_ms = int(time.time() * 1000.0)
            t_ms = int((time.monotonic() - t0) * 1000.0)

            pcie_tx_kib_s = safe_call(
                lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES), -1
            )
            pcie_rx_kib_s = safe_call(
                lambda: pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES), -1
            )

            tx_GiB_s = kib_s_to_GiB_s(pcie_tx_kib_s) if pcie_tx_kib_s is not None and pcie_tx_kib_s >= 0 else -1.0
            rx_GiB_s = kib_s_to_GiB_s(pcie_rx_kib_s) if pcie_rx_kib_s is not None and pcie_rx_kib_s >= 0 else -1.0
            total_GiB_s = (tx_GiB_s + rx_GiB_s) if (tx_GiB_s >= 0 and rx_GiB_s >= 0) else -1.0

            tx_pct = (100.0 * tx_GiB_s / link_ref_GiB_s) if (link_ref_GiB_s > 0 and tx_GiB_s >= 0) else -1.0
            rx_pct = (100.0 * rx_GiB_s / link_ref_GiB_s) if (link_ref_GiB_s > 0 and rx_GiB_s >= 0) else -1.0
            total_pct = (100.0 * total_GiB_s / link_ref_GiB_s) if (link_ref_GiB_s > 0 and total_GiB_s >= 0) else -1.0
            if tx_pct >= 0:
                tx_pct = min(tx_pct, 100.0)
            if rx_pct >= 0:
                rx_pct = min(rx_pct, 100.0)
            if total_pct >= 0:
                total_pct = min(total_pct, 200.0)

            if tx_pct >= 0:
                tx_peak_pct = max(tx_peak_pct, tx_pct)
            if rx_pct >= 0:
                rx_peak_pct = max(rx_peak_pct, rx_pct)
            if total_pct >= 0:
                total_peak_pct = max(total_peak_pct, total_pct)

            now_mono = time.monotonic()
            dt_s = 0.0 if last_sample_t is None else max(0.0, now_mono - last_sample_t)
            last_sample_t = now_mono
            if dt_s > 0.0:
                if tx_GiB_s >= 0:
                    tx_cum_gib += tx_GiB_s * dt_s
                if rx_GiB_s >= 0:
                    rx_cum_gib += rx_GiB_s * dt_s
            total_cum_gib = tx_cum_gib + rx_cum_gib

            util = safe_call(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle), None)
            mem = safe_call(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle), None)
            sm_clock = safe_call(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM), -1)
            mem_clock = safe_call(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM), -1)
            power_mw = safe_call(lambda: pynvml.nvmlDeviceGetPowerUsage(handle), -1)

            util_gpu_pct = float(getattr(util, "gpu", -1))
            util_mem_pct = float(getattr(util, "memory", -1))
            gpu_mem_used_gib = (float(mem.used) / (1024.0 ** 3)) if mem else -1.0
            gpu_mem_total_gib = (float(mem.total) / (1024.0 ** 3)) if mem else -1.0
            gpu_mem_used_pct = (100.0 * gpu_mem_used_gib / gpu_mem_total_gib) if gpu_mem_total_gib > 0 else -1.0
            power_w = (power_mw / 1000.0) if power_mw is not None and power_mw >= 0 else -1.0

            if psutil is not None:
                cpu_util_pct = float(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                cpu_mem_used_gib = float(vm.used) / (1024.0 ** 3)
                cpu_mem_avail_gib = float(vm.available) / (1024.0 ** 3)
                cpu_mem_used_pct = float(vm.percent)
            else:
                cpu_util_pct = -1.0
                cpu_mem_used_gib = -1.0
                cpu_mem_avail_gib = -1.0
                cpu_mem_used_pct = -1.0

            out_fp.write(
                f"{ts_unix_ms},{t_ms},"
                f"{curr_gen},{curr_wid},{link_ref_GiB_s:.6f},"
                f"{tx_GiB_s:.6f},{rx_GiB_s:.6f},{total_GiB_s:.6f},"
                f"{tx_pct:.3f},{rx_pct:.3f},{total_pct:.3f},"
                f"{tx_peak_pct:.3f},{rx_peak_pct:.3f},{total_peak_pct:.3f},"
                f"{tx_cum_gib:.6f},{rx_cum_gib:.6f},{total_cum_gib:.6f},"
                f"{util_gpu_pct:.3f},{util_mem_pct:.3f},"
                f"{gpu_mem_used_gib:.6f},{gpu_mem_total_gib:.6f},{gpu_mem_used_pct:.3f},"
                f"{float(sm_clock):.3f},{float(mem_clock):.3f},{power_w:.3f},"
                f"{cpu_util_pct:.3f},{cpu_mem_used_gib:.6f},{cpu_mem_avail_gib:.6f},{cpu_mem_used_pct:.3f}\n"
            )
            try:
                out_fp.flush()
            except BrokenPipeError:
                # Common when piping to tools like `head`; exit cleanly.
                break

            if args.live:
                if link_ref_GiB_s > 0:
                    print(
                        f"[pcie] t={t_ms / 1000.0:.3f}s "
                        f"tx={tx_GiB_s:.3f}GiB/s ({tx_pct:.2f}% of max, peak {tx_peak_pct:.2f}%) "
                        f"rx={rx_GiB_s:.3f}GiB/s ({rx_pct:.2f}% of max, peak {rx_peak_pct:.2f}%) "
                        f"total={total_GiB_s:.3f}GiB/s (peak {total_peak_pct:.2f}% of one-dir max) "
                        f"cum={total_cum_gib:.3f}GiB",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[pcie] t={t_ms / 1000.0:.3f}s tx={tx_GiB_s:.6f}GiB/s rx={rx_GiB_s:.6f}GiB/s (link unknown)",
                        file=sys.stderr,
                    )

            next_t += interval_s

    except KeyboardInterrupt:
        print("\n[logger] stopped.", file=sys.stderr)
    finally:
        try:
            if out_fp is not None and out_fp is not sys.stdout:
                out_fp.close()
        except Exception:
            pass
        pynvml.nvmlShutdown()
        if args.plot_on_exit and args.out not in ("", "-"):
            out_img = args.plot_out or (args.out + ".png")
            plot_csv(args.out, out_path=out_img, show=False, title=args.plot_title)


if __name__ == "__main__":
    main()
