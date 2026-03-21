#!/usr/bin/env python3
"""
Run gpu_pcie_memcpy and bandwidth_usage together, then export merged CSV.

Example:
  python3 tools/run_pcie_memcpy_with_bandwidth.py \
    --seconds 60 --dir d2h \
    --out results/contention2/pcie_mem_bw_merged.csv
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


PCIE_SAMPLE_RE = re.compile(
    r"\[pcie\]\s+t=([0-9.]+)s\s+dir=(h2d|d2h)\s+bw_gib_s=([0-9.]+)\s+"
    r"bytes=([0-9]+)\s+streams=([0-9]+)\s+batch=([0-9]+)\s+inflight=([0-9]+)\s+"
    r"chunk_bytes=([0-9]+)\s+pinned=([0-9]+)"
)


def parse_args():
    here = Path(__file__).resolve()
    src_dir = here.parents[1]
    default_bin = src_dir / "bin" / "gpu_pcie_memcpy"
    default_bw_script = here.with_name("bandwidth_usage.py")

    parser = argparse.ArgumentParser(description="Launch gpu_pcie_memcpy + bandwidth logger and merge to CSV.")
    parser.add_argument("--pcie_bin", default=str(default_bin), help="path to gpu_pcie_memcpy binary")
    parser.add_argument("--bandwidth_script", default=str(default_bw_script), help="path to bandwidth_usage.py")
    parser.add_argument("--python_bin", default=sys.executable, help="python executable for bandwidth script")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dir", choices=["h2d", "d2h"], default="d2h")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--chunk_mb", type=float, default=128.0)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--inflight", type=int, default=8)
    parser.add_argument("--pinned", type=int, choices=[0, 1], default=1)
    parser.add_argument("--report_ms", type=int, default=1000)

    parser.add_argument("--bw_interval_ms", type=int, default=200, help="bandwidth_usage sample interval")

    parser.add_argument("--out", required=True, help="merged output CSV path")
    parser.add_argument("--raw_pcie_csv", default="", help="optional raw parsed pcie sample CSV")
    parser.add_argument("--raw_bw_csv", default="", help="optional raw bandwidth_usage sample CSV")
    parser.add_argument("--quiet", action="store_true", help="suppress child log forwarding")
    return parser.parse_args()


def ensure_parent(path_str: str):
    path = Path(path_str).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def terminate_process(proc: subprocess.Popen, name: str):
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            print(f"[warn] failed to stop {name} cleanly", file=sys.stderr)


def reader_pcie(proc, samples, raw_lines, quiet):
    for line in proc.stdout:
        ts_ms = int(time.time() * 1000)
        text = line.rstrip("\n")
        raw_lines.append((ts_ms, text))

        if not quiet:
            print(f"[pcie-log] {text}", file=sys.stderr)

        m = PCIE_SAMPLE_RE.search(text)
        if not m:
            continue

        samples.append(
            {
                "ts_unix_ms": ts_ms,
                "elapsed_s": float(m.group(1)),
                "dir": m.group(2),
                "bw_gib_s": float(m.group(3)),
                "bytes": int(m.group(4)),
                "streams": int(m.group(5)),
                "batch": int(m.group(6)),
                "inflight": int(m.group(7)),
                "chunk_bytes": int(m.group(8)),
                "pinned": int(m.group(9)),
            }
        )


def reader_bw(proc, samples, quiet):
    header = None
    for line in proc.stdout:
        text = line.strip()
        if not text:
            continue

        if header is None:
            if text.startswith("ts_unix_ms,"):
                header = text.split(",")
            elif not quiet:
                print(f"[bw-log] {text}", file=sys.stderr)
            continue

        row = next(csv.reader([text]))
        if len(row) != len(header):
            if not quiet:
                print(f"[bw-log] skip malformed row: {text}", file=sys.stderr)
            continue

        obj = dict(zip(header, row))
        try:
            samples.append(
                {
                    "ts_unix_ms": int(obj["ts_unix_ms"]),
                    "elapsed_ms": int(obj["elapsed_ms"]),
                    "memory_util_pct": int(obj["memory_util_pct"]),
                    "estimated_mem_bw_GB_s": float(obj["estimated_mem_bw_GB_s"]) if obj["estimated_mem_bw_GB_s"] else None,
                    "peak_mem_bw_GB_s": float(obj["peak_mem_bw_GB_s"]) if obj["peak_mem_bw_GB_s"] else None,
                    "gpu_index": int(obj["gpu_index"]),
                    "gpu_name": obj["gpu_name"],
                }
            )
        except (ValueError, KeyError):
            if not quiet:
                print(f"[bw-log] skip parse-failed row: {text}", file=sys.stderr)


def write_raw_pcie_csv(path: Path, raw_lines, parsed_samples):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "log_ts_unix_ms",
                "raw_line",
                "sample_ts_unix_ms",
                "elapsed_s",
                "dir",
                "bw_gib_s",
                "bytes",
                "streams",
                "batch",
                "inflight",
                "chunk_bytes",
                "pinned",
            ]
        )
        sample_idx = 0
        parsed_samples_sorted = sorted(parsed_samples, key=lambda x: x["ts_unix_ms"])
        for log_ts, line in raw_lines:
            sample = None
            while sample_idx < len(parsed_samples_sorted):
                candidate = parsed_samples_sorted[sample_idx]
                if candidate["ts_unix_ms"] >= log_ts - 100 and candidate["ts_unix_ms"] <= log_ts + 100:
                    sample = candidate
                    sample_idx += 1
                    break
                if candidate["ts_unix_ms"] < log_ts - 100:
                    sample_idx += 1
                    continue
                break
            if sample is None:
                writer.writerow([log_ts, line, "", "", "", "", "", "", "", "", "", ""])
            else:
                writer.writerow(
                    [
                        log_ts,
                        line,
                        sample["ts_unix_ms"],
                        sample["elapsed_s"],
                        sample["dir"],
                        sample["bw_gib_s"],
                        sample["bytes"],
                        sample["streams"],
                        sample["batch"],
                        sample["inflight"],
                        sample["chunk_bytes"],
                        sample["pinned"],
                    ]
                )


def write_raw_bw_csv(path: Path, bw_samples):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
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
        for s in sorted(bw_samples, key=lambda x: x["ts_unix_ms"]):
            writer.writerow(
                [
                    s["ts_unix_ms"],
                    s["elapsed_ms"],
                    s["memory_util_pct"],
                    "" if s["estimated_mem_bw_GB_s"] is None else f"{s['estimated_mem_bw_GB_s']:.3f}",
                    "" if s["peak_mem_bw_GB_s"] is None else f"{s['peak_mem_bw_GB_s']:.3f}",
                    s["gpu_index"],
                    s["gpu_name"],
                ]
            )


def write_merged_csv(path: Path, bw_samples, pcie_samples, args):
    bw_sorted = sorted(bw_samples, key=lambda x: x["ts_unix_ms"])
    pcie_sorted = sorted(pcie_samples, key=lambda x: x["ts_unix_ms"])

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ts_unix_ms",
                "elapsed_ms",
                "memory_util_pct",
                "estimated_mem_bw_GB_s",
                "peak_mem_bw_GB_s",
                "gpu_index",
                "gpu_name",
                "pcie_sample_ts_unix_ms",
                "pcie_sample_age_ms",
                "pcie_elapsed_s",
                "pcie_dir",
                "pcie_bw_gib_s",
                "pcie_bytes",
                "pcie_streams",
                "pcie_batch",
                "pcie_inflight",
                "pcie_chunk_bytes",
                "pcie_pinned",
                "run_device",
                "run_dir",
                "run_seconds",
                "run_chunk_mb",
                "run_streams",
                "run_batch",
                "run_inflight",
                "run_pinned",
                "run_report_ms",
                "run_bw_interval_ms",
            ]
        )

        pcie_idx = -1
        for bw in bw_sorted:
            bw_ts = bw["ts_unix_ms"]
            while pcie_idx + 1 < len(pcie_sorted) and pcie_sorted[pcie_idx + 1]["ts_unix_ms"] <= bw_ts:
                pcie_idx += 1
            pcie = pcie_sorted[pcie_idx] if pcie_idx >= 0 else None

            writer.writerow(
                [
                    bw["ts_unix_ms"],
                    bw["elapsed_ms"],
                    bw["memory_util_pct"],
                    "" if bw["estimated_mem_bw_GB_s"] is None else f"{bw['estimated_mem_bw_GB_s']:.3f}",
                    "" if bw["peak_mem_bw_GB_s"] is None else f"{bw['peak_mem_bw_GB_s']:.3f}",
                    bw["gpu_index"],
                    bw["gpu_name"],
                    "" if pcie is None else pcie["ts_unix_ms"],
                    "" if pcie is None else (bw_ts - pcie["ts_unix_ms"]),
                    "" if pcie is None else pcie["elapsed_s"],
                    "" if pcie is None else pcie["dir"],
                    "" if pcie is None else pcie["bw_gib_s"],
                    "" if pcie is None else pcie["bytes"],
                    "" if pcie is None else pcie["streams"],
                    "" if pcie is None else pcie["batch"],
                    "" if pcie is None else pcie["inflight"],
                    "" if pcie is None else pcie["chunk_bytes"],
                    "" if pcie is None else pcie["pinned"],
                    args.device,
                    args.dir,
                    args.seconds,
                    args.chunk_mb,
                    args.streams,
                    args.batch,
                    args.inflight,
                    args.pinned,
                    args.report_ms,
                    args.bw_interval_ms,
                ]
            )


def main():
    args = parse_args()
    pcie_bin = Path(os.path.expanduser(args.pcie_bin))
    bw_script = Path(os.path.expanduser(args.bandwidth_script))

    if not pcie_bin.exists():
        raise SystemExit(f"pcie binary not found: {pcie_bin}")
    if not bw_script.exists():
        raise SystemExit(f"bandwidth script not found: {bw_script}")

    out_path = ensure_parent(args.out)
    raw_pcie_path = ensure_parent(args.raw_pcie_csv) if args.raw_pcie_csv else None
    raw_bw_path = ensure_parent(args.raw_bw_csv) if args.raw_bw_csv else None

    pcie_cmd = [
        str(pcie_bin),
        f"--device={args.device}",
        f"--dir={args.dir}",
        f"--seconds={args.seconds}",
        f"--chunk_mb={args.chunk_mb}",
        f"--streams={args.streams}",
        f"--batch={args.batch}",
        f"--inflight={args.inflight}",
        f"--pinned={args.pinned}",
        f"--report_ms={args.report_ms}",
    ]
    bw_cmd = [
        args.python_bin,
        str(bw_script),
        "--gpu",
        str(args.device),
        "--interval_ms",
        str(args.bw_interval_ms),
        "--duration_s",
        str(args.seconds + 5),
        "--csv",
        "--out",
        "-",
        "--quiet",
    ]

    print("[run] launching gpu_pcie_memcpy:", " ".join(pcie_cmd), file=sys.stderr)
    print("[run] launching bandwidth_usage:", " ".join(bw_cmd), file=sys.stderr)

    pcie_samples = []
    pcie_raw_lines = []
    bw_samples = []

    pcie_proc = subprocess.Popen(
        pcie_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    bw_proc = subprocess.Popen(
        bw_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    stop_event = threading.Event()

    def handle_signal(sig_num, _frame):
        if stop_event.is_set():
            return
        stop_event.set()
        print(f"[run] received signal {sig_num}, stopping children...", file=sys.stderr)
        terminate_process(pcie_proc, "gpu_pcie_memcpy")
        terminate_process(bw_proc, "bandwidth_usage")

    old_sigint = signal.signal(signal.SIGINT, handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    t1 = threading.Thread(target=reader_pcie, args=(pcie_proc, pcie_samples, pcie_raw_lines, args.quiet), daemon=True)
    t2 = threading.Thread(target=reader_bw, args=(bw_proc, bw_samples, args.quiet), daemon=True)
    t1.start()
    t2.start()

    try:
        pcie_rc = pcie_proc.wait()
        terminate_process(bw_proc, "bandwidth_usage")
        bw_rc = bw_proc.wait()
        t1.join(timeout=2)
        t2.join(timeout=2)
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    if pcie_rc != 0:
        raise SystemExit(f"gpu_pcie_memcpy exited with code {pcie_rc}")
    if bw_rc not in (0, -15):
        raise SystemExit(f"bandwidth_usage exited with code {bw_rc}")

    if not bw_samples:
        raise SystemExit("no bandwidth samples collected")
    if not pcie_samples:
        raise SystemExit("no parsed pcie samples collected")

    write_merged_csv(out_path, bw_samples, pcie_samples, args)
    if raw_pcie_path is not None:
        write_raw_pcie_csv(raw_pcie_path, pcie_raw_lines, pcie_samples)
    if raw_bw_path is not None:
        write_raw_bw_csv(raw_bw_path, bw_samples)

    print(f"[done] merged_csv={out_path} rows={len(bw_samples)}", file=sys.stderr)
    print(f"[done] pcie_samples={len(pcie_samples)} bw_samples={len(bw_samples)}", file=sys.stderr)
    if raw_pcie_path is not None:
        print(f"[done] raw_pcie_csv={raw_pcie_path}", file=sys.stderr)
    if raw_bw_path is not None:
        print(f"[done] raw_bw_csv={raw_bw_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
