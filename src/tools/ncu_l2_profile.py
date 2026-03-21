#!/usr/bin/env python3
"""
ncu_l2_profile.py

Profile GPU L2 cache hit/miss using Nsight Compute (ncu) and write CSV.

Notes:
  - Requires ncu installed on the machine.
  - L2 metrics names can vary by GPU/driver. This script tries common names.
  - Use --list-metrics to query available L2 metrics on your system.
  - This script follows the "ncu -> .ncu-rep -> ncu -i --page raw --csv" workflow,
    because it is the most compatible across ncu versions.

Example:
  python3 ncu_l2_profile.py --list-metrics

  python3 ncu_l2_profile.py \
    --cmd "./bin/gpu_memhog --gb=16 --seconds=20 --op=copy --blocks=4096 --threads=256" \
    --kernel_id ":::1" \
    --launch_count 1 \
    --out ~/danyang/SkyGDR/results/contention1/ncu_l2.csv
"""

import argparse
import csv
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional


#
# Default settings (so you can run this script without a long command line)
# -----------------------------------------------------------------------
# You can still override any of these via CLI flags.
#
#
# NOTE: these defaults assume you run the script from the repo root (SkyGDR/),
# so we intentionally use relative paths (no absolute paths).
#
DEFAULT_CMD = "./src/bin/gpu_memhog --gb=16 --seconds=5 --op=copy --blocks=4096 --threads=256"
DEFAULT_KERNEL_ID = ":::1"
DEFAULT_OUT = "./results/contention1/l2_onekernel.csv"

COMMON_L2_METRICS = [
    # A100/ga100 full metric names (need --query-metrics-mode all)
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
]

# Heuristic patterns to find L2-related metrics if exact names differ.
L2_HINT_PATTERNS = [
    re.compile(r"^lts__.*hit.*rate", re.IGNORECASE),
    re.compile(r"^lts__.*miss.*rate", re.IGNORECASE),
    re.compile(r"^lts__.*hit.*sum", re.IGNORECASE),
    re.compile(r"^lts__.*miss.*sum", re.IGNORECASE),
]


def run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    capture: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess:
    if capture:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        # Let ncu print progress (it usually writes to stderr).
        p = subprocess.run(cmd, cwd=cwd)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr}")
    return p


def ncu_exists() -> bool:
    return subprocess.call(["which", "ncu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def list_metrics() -> None:
    if not ncu_exists():
        print("ncu not found in PATH")
        sys.exit(1)
    p = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"], capture=True)
    if p.returncode != 0:
        print(p.stderr)
        sys.exit(p.returncode)
    print(p.stdout)


def detect_metrics() -> List[str]:
    if not ncu_exists():
        return []
    p = run(["ncu", "--query-metrics", "--query-metrics-mode", "all"], capture=True)
    if p.returncode != 0:
        return []
    metrics = set()
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # ncu outputs "metric_name : description" in many versions
        name = line.split()[0]
        metrics.add(name)
    # First try exact known metric names.
    exact = [m for m in COMMON_L2_METRICS if m in metrics]
    if exact:
        return exact
    # Fallback: use regex to pick likely L2 hit/miss metrics.
    picked = []
    for name in metrics:
        for pat in L2_HINT_PATTERNS:
            if pat.match(name):
                picked.append(name)
                break
    # Prefer a small, stable set (up to 4) to keep output readable.
    return picked[:4]


def parse_ncu_csv_text(text: str, metrics: List[str]) -> Dict[str, float]:
    """
    Parse "ncu -i <rep> --page raw --csv" output and extract metric values.
    """
    results: Dict[str, float] = {}
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        # In raw CSV, one of the columns is the metric name; another is the value.
        # We use a heuristic:
        #  - Find a cell that matches one of the metric names.
        #  - Then take the first numeric cell after that (or anywhere) as value.
        metric_name: Optional[str] = None
        for cell in row:
            c = cell.strip()
            if c in metrics:
                metric_name = c
                break
        if not metric_name:
            continue
        for cell in row:
            c = cell.strip()
            if re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", c):
                results[metric_name] = float(c)
                break
    return results


def main():
    ap = argparse.ArgumentParser()
    # If you don't pass --cmd/--out, the script uses defaults above.
    ap.add_argument("--cmd", default=DEFAULT_CMD, help="command to profile (quoted)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output CSV file")
    ap.add_argument("--metrics", help="comma-separated metrics (optional)")
    ap.add_argument("--kernel_id", default=DEFAULT_KERNEL_ID, help='ncu --kernel-id value, e.g. ":::1" for the first kernel')
    ap.add_argument("--report_base", default="", help="output base name for .ncu-rep (default: derived from --out)")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="overwrite existing .ncu-rep/.raw.csv/.csv outputs (default: on)",
    )
    ap.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="do not overwrite existing outputs",
    )
    ap.add_argument(
        "--workdir",
        default="",
        help="working directory to run ncu/target in (default: repo root inferred from this script)",
    )
    ap.add_argument("--launch_count", type=int, default=0, help="profile only first N kernel launches (0 = ncu default)")
    ap.add_argument("--launch_skip", type=int, default=0, help="skip first N kernel launches before profiling")
    ap.add_argument("--profile_from_start", choices=["on", "off"], default="on", help="ncu profile-from-start flag")
    ap.add_argument("--live", action="store_true", help="loop and sample repeatedly (quasi-real-time)")
    ap.add_argument("--interval_ms", type=int, default=1000, help="interval between samples in live mode")
    ap.add_argument("--samples", type=int, default=0, help="number of samples in live mode (0 = infinite)")
    ap.add_argument("--append", action="store_true", help="append to CSV in live mode")
    ap.add_argument("--no_nvml", action="store_true", help="disable NVML sampling (memory/gpu util)")
    ap.add_argument("--list-metrics", action="store_true")
    ap.add_argument("--ncu_bin", default="ncu")
    args = ap.parse_args()

    if args.list_metrics:
        list_metrics()
        return

    if not ncu_exists():
        raise SystemExit("ncu not found in PATH")

    # Default workdir: repo root (SkyGDR/) inferred from this script location.
    script_dir = Path(__file__).resolve()
    repo_root = script_dir.parents[2]
    workdir = args.workdir.strip() or str(repo_root)

    # Normalize paths (handle ~ and relative paths).
    # IMPORTANT: keep relative paths relative to workdir so users don't need absolute paths.
    out_path = Path(os.path.expanduser(args.out))
    if not out_path.is_absolute():
        out_path = Path(workdir) / out_path
    out_csv = str(out_path)

    metrics = []
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = detect_metrics()

    if not metrics:
        raise SystemExit("No L2 metrics found. Run with --list-metrics to inspect available metrics.")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    report_base = args.report_base.strip()
    if not report_base:
        if out_csv.lower().endswith(".csv"):
            report_base = out_csv[:-4]
        else:
            report_base = out_csv
    rep_path = report_base + ".ncu-rep"
    raw_csv_path = report_base + ".raw.csv"

    # Optional NVML sampling (memory/gpu util) for "live" output
    nvml_ok = False
    nvml_handle = None
    if args.live and not args.no_nvml:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            nvml_ok = True
        except Exception:
            nvml_ok = False

    def run_one_sample(sample_idx: int) -> Dict[str, float]:
        # 1) Profile and produce a .ncu-rep report (most compatible way).
        ncu_profile_cmd = [
            args.ncu_bin,
            "-f" if args.overwrite else "",
            "--kernel-id",
            args.kernel_id,
            f"--metrics={','.join(metrics)}",
            "--target-processes",
            "all",
            "--profile-from-start",
            args.profile_from_start,
            "-o",
            report_base,
        ] + shlex.split(args.cmd)
        ncu_profile_cmd = [x for x in ncu_profile_cmd if x]
        if args.launch_skip > 0:
            ncu_profile_cmd.insert(1, f"--launch-skip={args.launch_skip}")
        if args.launch_count > 0:
            ncu_profile_cmd.insert(1, f"--launch-count={args.launch_count}")

        print(f"Running (profile)[{sample_idx}]:", " ".join(ncu_profile_cmd))
        p = run(ncu_profile_cmd, cwd=workdir, capture=False)
        if p.returncode != 0:
            if not args.overwrite and os.path.exists(rep_path):
                print(f"ncu profile failed: {rep_path} already exists (rerun with --overwrite)")
            else:
                print(f"ncu profile failed with rc={p.returncode}")
            sys.exit(p.returncode)

        # 2) Convert .ncu-rep to raw CSV so we can parse values.
        ncu_export_cmd = [args.ncu_bin, "-i", rep_path, "--page", "raw", "--csv"]
        print(f"Running (export)[{sample_idx}]:", " ".join(ncu_export_cmd))
        p2 = run(ncu_export_cmd, cwd=workdir, capture=True)
        if p2.returncode != 0:
            print(p2.stderr)
            sys.exit(p2.returncode)

        with open(raw_csv_path, "w", newline="") as f:
            f.write(p2.stdout)

        values = parse_ncu_csv_text(p2.stdout, metrics)
        return values

    def nvml_sample():
        if not nvml_ok:
            return -1, -1
        try:
            import pynvml  # type: ignore

            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            return util.memory, util.gpu
        except Exception:
            return -1, -1

    def write_header(writer):
        header = ["timestamp"] + metrics + ["l2_hit_rate"]
        if args.live:
            header += ["mem_util_pct", "gpu_util_pct"]
        writer.writerow(header)

    def append_row(writer, ts, values, mem_util, gpu_util):
        hit = values.get("lts__t_sectors_lookup_hit.sum", 0.0)
        miss = values.get("lts__t_sectors_lookup_miss.sum", 0.0)
        denom = hit + miss
        l2_hit_rate = (hit / denom) if denom > 0 else 0.0
        row = [ts] + [values.get(m) for m in metrics] + [l2_hit_rate]
        if args.live:
            row += [mem_util, gpu_util]
        writer.writerow(row)
        return l2_hit_rate

    if args.live:
        if args.interval_ms <= 0:
            args.interval_ms = 1000
        if args.launch_count == 0:
            args.launch_count = 1
        mode = "a" if args.append else "w"
        with open(out_csv, mode, newline="") as f:
            w = csv.writer(f)
            if not args.append or f.tell() == 0:
                write_header(w)
            i = 0
            while True:
                t0 = time.time()
                values = run_one_sample(i)
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                mem_util, gpu_util = nvml_sample()
                l2_hit_rate = append_row(w, ts, values, mem_util, gpu_util)
                f.flush()
                print(f"[live] ts={ts} l2_hit_rate={l2_hit_rate:.4f} mem_util={mem_util} gpu_util={gpu_util}")
                i += 1
                if args.samples > 0 and i >= args.samples:
                    break
                dt = time.time() - t0
                sleep_s = max(0.0, args.interval_ms / 1000.0 - dt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        if nvml_ok:
            try:
                import pynvml  # type: ignore

                pynvml.nvmlShutdown()
            except Exception:
                pass
        print("Saved:", out_csv)
        print("Saved (raw):", raw_csv_path)
        print("Saved (rep):", rep_path)
        return

    # One-shot mode (original behavior)
    values = run_one_sample(0)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        write_header(w)
        append_row(w, ts, values, -1, -1)

    if nvml_ok:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlShutdown()
        except Exception:
            pass

    print("Saved:", out_csv)
    print("Saved (raw):", raw_csv_path)
    print("Saved (rep):", rep_path)


if __name__ == "__main__":
    main()
