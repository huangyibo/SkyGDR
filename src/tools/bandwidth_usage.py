#!/usr/bin/env python3

import argparse
import csv
import sys
import time

import pynvml

# Peak memory bandwidth lookup table (GB/s)
PEAK_MEMORY_BANDWIDTH = {
    "A100-SXM4-40GB": 1555,
    "A100-SXM4-80GB": 2039,
    "A100-PCIE-40GB": 1555,
    "A100-PCIE-80GB": 2039,
    "H100-SXM": 3350,
    "H100-PCIE": 3000,
    "V100-SXM2-32GB": 900,
    "V100-PCIE-32GB": 785,
    "RTX 4090": 1008,
    "RTX 4090 D": 936,
    "RTX 4080": 717,
    "RTX 3090": 936,
    "RTX 3080": 760,
}


def get_peak_bw(model_name: str):
    for key in PEAK_MEMORY_BANDWIDTH:
        if key.lower() in model_name.lower():
            return PEAK_MEMORY_BANDWIDTH[key]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor GPU memory-controller utilization.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--interval_ms", type=int, default=500, help="sampling interval in milliseconds")
    parser.add_argument("--duration_s", type=float, default=0.0, help="monitor duration in seconds, 0 means infinite")
    parser.add_argument("--csv", action="store_true", help="output CSV instead of human-readable lines")
    parser.add_argument("--out", default="-", help="CSV output path when --csv is set, '-' means stdout")
    parser.add_argument("--quiet", action="store_true", help="suppress startup messages")
    return parser.parse_args()


def open_csv_writer(path: str):
    if path == "-":
        return csv.writer(sys.stdout), None
    fp = open(path, "w", newline="")
    return csv.writer(fp), fp


def main():
    args = parse_args()
    interval_s = max(args.interval_ms, 50) / 1000.0

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

    raw_name = pynvml.nvmlDeviceGetName(handle)
    name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
    peak_bw = get_peak_bw(name)

    if args.csv:
        writer, csv_fp = open_csv_writer(args.out)
        writer.writerow(
            [
                "ts_unix_ms",
                "elapsed_ms",
                "memory_util_pct",
                "estimated_mem_bw_GB_s",
                "peak_mem_bw_GB_s",
                "gpu_index",
                "gpu_name",
            ]
        )
    else:
        writer = None
        csv_fp = None

    if not args.quiet:
        print(f"[bw] Monitoring GPU {args.gpu}: {name}", file=sys.stderr)
        if peak_bw is None:
            print("[bw] Peak bandwidth lookup missing; estimated_mem_bw_GB_s will be empty", file=sys.stderr)
        else:
            print(f"[bw] Peak memory bandwidth reference: {peak_bw} GB/s", file=sys.stderr)
        if args.duration_s > 0:
            print(f"[bw] duration_s={args.duration_s}", file=sys.stderr)

    start = time.time()

    try:
        while True:
            now = time.time()
            elapsed_s = now - start
            if args.duration_s > 0 and elapsed_s >= args.duration_s:
                break

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_util = int(util.memory)
            est_bw = (mem_util / 100.0) * peak_bw if peak_bw is not None else None
            ts_ms = int(now * 1000)
            elapsed_ms = int(elapsed_s * 1000)

            if args.csv:
                writer.writerow(
                    [
                        ts_ms,
                        elapsed_ms,
                        mem_util,
                        f"{est_bw:.3f}" if est_bw is not None else "",
                        peak_bw if peak_bw is not None else "",
                        args.gpu,
                        name,
                    ]
                )
                if csv_fp is not None:
                    csv_fp.flush()
                else:
                    sys.stdout.flush()
            else:
                if est_bw is None:
                    print(f"Memory Util: {mem_util:3d}%")
                else:
                    print(f"Memory Util: {mem_util:3d}%   Estimated Bandwidth: {est_bw:7.1f} GB/s")

            time.sleep(interval_s)

    except KeyboardInterrupt:
        if not args.quiet:
            print("\n[bw] Stopped by keyboard interrupt.", file=sys.stderr)
    finally:
        if csv_fp is not None:
            csv_fp.close()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
