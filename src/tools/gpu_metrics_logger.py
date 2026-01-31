#!/usr/bin/env python3
"""
gpu_metrics_logger.py

Periodically log GPU metrics to CSV for contention analysis.

Adds "PCIe fabric occupancy" style signals by deriving utilization (%)
from NVML PCIe throughput and the GPU's PCIe link speed (gen/width).

Notes:
- NVML nvmlDeviceGetPcieThroughput returns a moving-average throughput sample
  in KB/s for TX/RX (host<->device). See NVML docs/manpages. 
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


def safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def pcie_per_lane_gbps(gen: int) -> float:
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


def kbps_to_gbps(kbps: float) -> float:
    # NVML reports KB/s; in practice this is typically KiB/s. We keep 1024 for consistency.
    return (kbps * 1024.0) / 1e9


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

    ts_ms = []
    data = {c: [] for c in cols}
    for r in rows:
        if "ts_unix_ms" not in r:
            continue
        t = _to_float(r.get("ts_unix_ms", "nan"))
        ts_ms.append(t)
        for c in cols:
            data[c].append(_to_float(r.get(c, "nan")))

    if not ts_ms:
        print("[plot] no ts_unix_ms in csv", file=sys.stderr)
        return 4

    t0 = ts_ms[0]
    t_s = [(x - t0) / 1000.0 for x in ts_ms]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    ax_pcie, ax_util, ax_clk = axes

    # --- PCIe subplot ---
    pcie_plotted = False
    if "pcie_tx_kbps" in data:
        tx_gbps = [kbps_to_gbps(x) if not _is_nan(x) else float("nan") for x in data["pcie_tx_kbps"]]
        ax_pcie.plot(t_s, tx_gbps, label="pcie_tx_gbps")
        pcie_plotted = True
    if "pcie_rx_kbps" in data:
        rx_gbps = [kbps_to_gbps(x) if not _is_nan(x) else float("nan") for x in data["pcie_rx_kbps"]]
        ax_pcie.plot(t_s, rx_gbps, label="pcie_rx_gbps")
        pcie_plotted = True

    ax_pcie.set_ylabel("PCIe GB/s")
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

    # --- GPU utilization subplot ---
    util_plotted = False
    if "util_gpu_pct" in data:
        ax_util.plot(t_s, data["util_gpu_pct"], label="util_gpu_pct")
        util_plotted = True
    if "util_mem_pct" in data:
        ax_util.plot(t_s, data["util_mem_pct"], label="util_mem_pct")
        util_plotted = True
    ax_util.set_ylabel("Util %")
    ax_util.grid(True, alpha=0.3)
    if util_plotted:
        ax_util.legend(loc="upper right")
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

    header = (
        "ts_unix_ms,"
        "pcie_tx_kbps,pcie_rx_kbps,"
        "pcie_tx_util_pct,pcie_rx_util_pct,"
        "pcie_tx_peak_util_pct,pcie_rx_peak_util_pct,"
        "pcie_gen,pcie_width,pcie_max_gen,pcie_max_width,"
        "pcie_replay_cnt,"
        "util_gpu_pct,util_mem_pct,sm_clock_mhz,mem_clock_mhz,power_w"
    )

    if out_fp is not sys.stdout:
        if args.append and os.path.exists(args.out) and os.path.getsize(args.out) > 0:
            pass
        else:
            out_fp.write(header + "\n")
    else:
        out_fp.write(header + "\n")

    print(f"[logger] GPU {args.gpu}: {name}", file=sys.stderr)
    print(f"[logger] interval_ms={args.interval_ms} out={args.out}", file=sys.stderr)

    interval_s = args.interval_ms / 1000.0
    next_t = time.time()

    tx_peak_pct = 0.0
    rx_peak_pct = 0.0

    try:
        while True:
            now = time.time()
            if now < next_t:
                time.sleep(min(0.05, next_t - now))
                continue

            ts_ms = int(time.time() * 1000)

            # PCIe link info (current + max)
            curr_gen = safe_call(lambda: get_curr_gen(handle), -1) if get_curr_gen else -1
            curr_wid = safe_call(lambda: get_curr_wid(handle), -1) if get_curr_wid else -1
            max_gen = safe_call(lambda: get_max_gen(handle), -1) if get_max_gen else -1
            max_wid = safe_call(lambda: get_max_wid(handle), -1) if get_max_wid else -1

            link_gbps = pcie_per_lane_gbps(curr_gen) * (curr_wid if curr_wid > 0 else 0)
            # Avoid divide by zero; if we can't read link state, leave util as -1
            link_gbps = link_gbps if link_gbps > 0 else 0.0

            util = safe_call(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
            util_gpu = util.gpu if util else -1
            util_mem = util.memory if util else -1

            sm_clock = safe_call(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM), -1)
            mem_clock = safe_call(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM), -1)
            power_mw = safe_call(lambda: pynvml.nvmlDeviceGetPowerUsage(handle), -1)
            power_w = power_mw / 1000.0 if power_mw is not None and power_mw >= 0 else -1

            pcie_tx = safe_call(lambda: pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_TX_BYTES), -1)
            pcie_rx = safe_call(lambda: pynvml.nvmlDeviceGetPcieThroughput(
                handle, pynvml.NVML_PCIE_UTIL_RX_BYTES), -1)

            tx_gbps = kbps_to_gbps(pcie_tx) if pcie_tx is not None and pcie_tx >= 0 else -1.0
            rx_gbps = kbps_to_gbps(pcie_rx) if pcie_rx is not None and pcie_rx >= 0 else -1.0

            tx_pct = (100.0 * tx_gbps / link_gbps) if (link_gbps > 0 and tx_gbps >= 0) else -1.0
            rx_pct = (100.0 * rx_gbps / link_gbps) if (link_gbps > 0 and rx_gbps >= 0) else -1.0

            if tx_pct >= 0:
                tx_peak_pct = max(tx_peak_pct, tx_pct)
            if rx_pct >= 0:
                rx_peak_pct = max(rx_peak_pct, rx_pct)

            replay_cnt = safe_call(lambda: get_replay(handle), -1) if get_replay else -1

            out_fp.write(
                f"{ts_ms},"
                f"{pcie_tx},{pcie_rx},"
                f"{tx_pct:.3f},{rx_pct:.3f},"
                f"{tx_peak_pct:.3f},{rx_peak_pct:.3f},"
                f"{curr_gen},{curr_wid},{max_gen},{max_wid},"
                f"{replay_cnt},"
                f"{util_gpu},{util_mem},{sm_clock},{mem_clock},{power_w}\n"
            )
            out_fp.flush()

            if args.live:
                # A compact, grep-friendly status line
                # Example: [pcie] tx=1.23GB/s(3.8%,peak 12.1%) rx=...
                if link_gbps > 0:
                    print(
                        f"[pcie] gen{curr_gen}x{curr_wid} link≈{link_gbps:.2f}GB/s "
                        f"tx={tx_gbps:.3f}GB/s({tx_pct:.2f}%,peak {tx_peak_pct:.2f}%) "
                        f"rx={rx_gbps:.3f}GB/s({rx_pct:.2f}%,peak {rx_peak_pct:.2f}%) "
                        f"replay={replay_cnt} util_gpu={util_gpu}% util_mem={util_mem}% power={power_w:.1f}W",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"[pcie] tx={pcie_tx}KB/s rx={pcie_rx}KB/s (link unknown) replay={replay_cnt}",
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
