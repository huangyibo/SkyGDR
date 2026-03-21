#!/usr/bin/env python3

import argparse
import csv
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PROTECT_MODES = ("LOW1", "LOW2", "LOW3", "LOW4")


def parse_size(text: str) -> int:
    s = text.strip()
    if not s:
        return 0
    mult = 1
    suffix = s[-1].lower()
    if suffix in ("k", "m", "g"):
        s = s[:-1]
        if suffix == "k":
            mult = 1024
        elif suffix == "m":
            mult = 1024 ** 2
        else:
            mult = 1024 ** 3
    return int(float(s) * mult)


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


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def remove_if_exists(path: Path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def read_last_csv_row(path: Path):
    if not path.exists() or path.stat().st_size <= 0:
        return None
    with path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        last = None
        for row in reader:
            last = row
        return last


def fresh_row(row, min_ts_unix_ms: int):
    if not row:
        return None
    if "ts_unix_ms" not in row or row.get("ts_unix_ms", "") == "":
        return row
    row_ts = to_int(row.get("ts_unix_ms"), default=-1)
    if row_ts < min_ts_unix_ms:
        return None
    return row


def send_control(host: str, port: int, command: str, timeout_s: float = 2.0) -> str:
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.sendall((command.strip() + "\n").encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        data = sock.recv(4096)
    return data.decode("utf-8", errors="replace").strip()


def wait_for_control(host: str, port: int, timeout_s: float) -> str:
    deadline = time.time() + timeout_s
    last_err = "timeout"
    while time.time() < deadline:
        try:
            return send_control(host, port, "STATUS", timeout_s=1.0)
        except OSError as exc:
            last_err = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"control port {host}:{port} not reachable: {last_err}")


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


def restore_guard_bw(row):
    if not row:
        return float("nan")
    smooth = to_float(row.get("smooth_bw_gib_s"))
    if not math.isnan(smooth):
        return smooth
    return to_float(row.get("inst_bw_gib_s"))


def next_protect_mode(mode: str) -> str:
    if mode not in PROTECT_MODES:
        return PROTECT_MODES[0]
    idx = PROTECT_MODES.index(mode)
    if idx + 1 >= len(PROTECT_MODES):
        return mode
    return PROTECT_MODES[idx + 1]


def parse_args():
    here = Path(__file__).resolve()
    src_dir = here.parents[1]
    default_pcie_bin = src_dir / "bin" / "gpu_pcie_memcpy"
    default_gpu_metrics = here.with_name("gpu_metrics_logger.py")

    ap = argparse.ArgumentParser(description="Run the local GPU-side restore/controller for the PCIe case study.")
    ap.add_argument("--cpu_control_host", required=True, help="IP/host of the remote cpu_client control port")
    ap.add_argument("--cpu_control_port", type=int, required=True)
    ap.add_argument("--control_timeout_s", type=float, default=15.0)
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--pcie_bin", default=str(default_pcie_bin))
    ap.add_argument("--gpu_metrics_script", default=str(default_gpu_metrics))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu_metrics_interval_ms", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--dir", choices=["h2d", "d2h"], default="h2d")
    ap.add_argument("--chunk_mb", type=float, default=128.0)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--pinned", type=int, choices=[0, 1], default=1)
    ap.add_argument("--baseline_restore_gib_s", type=float, required=True)
    ap.add_argument("--restore_total_bytes", default="64G")
    ap.add_argument("--restore_max_outstanding_bytes", default="4G")
    ap.add_argument("--progress_ms", type=int, default=100)
    ap.add_argument("--progress_smooth_windows", type=int, default=5)
    ap.add_argument("--poll_ms", type=int, default=100)
    ap.add_argument("--warmup_ms", type=int, default=1000)
    ap.add_argument("--post_restore_ms", type=int, default=2000)
    ap.add_argument("--enter_ratio", type=float, default=0.65)
    ap.add_argument("--exit_ratio", type=float, default=0.80)
    ap.add_argument("--rx_threshold_pct", type=float, default=85.0)
    ap.add_argument("--enter_windows", type=int, default=1)
    ap.add_argument("--exit_windows", type=int, default=2)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--tag", default="case_study")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = Path(os.path.expanduser(args.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)

    restore_progress_csv = ensure_parent(log_dir / f"{args.tag}_restore_progress.csv")
    gpu_metrics_csv = ensure_parent(log_dir / f"{args.tag}_gpu_metrics.csv")
    controller_csv = ensure_parent(log_dir / f"{args.tag}_controller.csv")
    restore_log = ensure_parent(log_dir / f"{args.tag}_restore.log")
    gpu_metrics_log = ensure_parent(log_dir / f"{args.tag}_gpu_metrics.log")
    run_start_unix_ms = int(time.time() * 1000)

    # Reusing a tag should not let stale CSV rows leak into a new run.
    for path in (restore_progress_csv, gpu_metrics_csv, controller_csv, restore_log, gpu_metrics_log):
        remove_if_exists(path)

    baseline = args.baseline_restore_gib_s
    enter_thresh = baseline * args.enter_ratio
    exit_thresh = baseline * args.exit_ratio
    restore_total_bytes = parse_size(args.restore_total_bytes)
    restore_max_outstanding_bytes = parse_size(args.restore_max_outstanding_bytes)

    try:
        status = wait_for_control(args.cpu_control_host, args.cpu_control_port, args.control_timeout_s)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"[controller] initial remote status: {status}", file=sys.stderr)
    print(send_control(args.cpu_control_host, args.cpu_control_port, "HIGH"), file=sys.stderr)

    restore_cmd = [
        args.pcie_bin,
        f"--device={args.device}",
        f"--dir={args.dir}",
        "--seconds=0",
        f"--chunk_mb={args.chunk_mb}",
        f"--streams={args.streams}",
        f"--batch={args.batch}",
        f"--inflight={args.inflight}",
        f"--pinned={args.pinned}",
        f"--report_ms={args.progress_ms}",
        f"--progress_ms={args.progress_ms}",
        f"--progress_smooth_windows={max(args.progress_smooth_windows, 1)}",
        f"--total_bytes={restore_total_bytes}",
        f"--max_outstanding_bytes={restore_max_outstanding_bytes}",
        f"--progress_out={restore_progress_csv}",
    ]
    gpu_metrics_cmd = [
        args.python_bin,
        args.gpu_metrics_script,
        "--gpu",
        str(args.gpu),
        "--interval_ms",
        str(args.gpu_metrics_interval_ms),
        "--out",
        str(gpu_metrics_csv),
    ]

    restore_fp = restore_log.open("w")
    metrics_fp = gpu_metrics_log.open("w")
    restore_proc = None
    metrics_proc = None
    current_mode = "HIGH"
    protect_streak = 0
    restore_done = False
    stop_sent = False
    post_restore_deadline_ms = None

    try:
        metrics_proc = subprocess.Popen(gpu_metrics_cmd, stdout=metrics_fp, stderr=subprocess.STDOUT, text=True)
        time.sleep(max(args.warmup_ms, 0) / 1000.0)
        restore_proc = subprocess.Popen(restore_cmd, stdout=restore_fp, stderr=subprocess.STDOUT, text=True)

        with controller_csv.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "ts_unix_ms",
                    "mode",
                    "decision",
                    "restore_done",
                    "restore_elapsed_ms",
                    "restore_issued_bytes",
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
                    "control_response",
                ]
            )

            while True:
                now_ms = int(time.time() * 1000)
                restore_row = fresh_row(read_last_csv_row(restore_progress_csv), run_start_unix_ms)
                metrics_row = fresh_row(read_last_csv_row(gpu_metrics_csv), run_start_unix_ms)

                restore_elapsed_ms = to_int(restore_row.get("elapsed_ms")) if restore_row else 0
                restore_issued_bytes = to_int(restore_row.get("issued_bytes")) if restore_row else 0
                restore_completed_bytes = to_int(restore_row.get("completed_bytes")) if restore_row else 0
                restore_remaining_bytes = to_int(restore_row.get("remaining_bytes")) if restore_row else restore_total_bytes
                restore_inst_bw = to_float(restore_row.get("inst_bw_gib_s")) if restore_row else float("nan")
                restore_smooth_bw = to_float(restore_row.get("smooth_bw_gib_s")) if restore_row else float("nan")
                restore_guard = restore_guard_bw(restore_row) if restore_row else float("nan")
                restore_avg_bw = to_float(restore_row.get("avg_bw_gib_s")) if restore_row else float("nan")
                row_done = to_int(restore_row.get("done")) if restore_row else 0
                pcie_tx = to_float(metrics_row.get("pcie_tx_util_pct")) if metrics_row else float("nan")
                pcie_rx = to_float(metrics_row.get("pcie_rx_util_pct")) if metrics_row else float("nan")

                decision = "HOLD"
                response = ""
                if restore_row:
                    restore_done = bool(row_done) or restore_remaining_bytes <= 0
                    if restore_done:
                        restore_inst_bw = 0.0
                        restore_smooth_bw = 0.0
                        restore_guard = 0.0
                    if not restore_done:
                        bad = (not math.isnan(restore_guard) and restore_guard < enter_thresh and
                                   not math.isnan(pcie_rx) and pcie_rx >= args.rx_threshold_pct)
                        protect_streak = protect_streak + 1 if bad else 0
                        if protect_streak >= args.enter_windows:
                            target_mode = next_protect_mode(current_mode)
                            if target_mode != current_mode:
                                response = send_control(args.cpu_control_host, args.cpu_control_port, target_mode)
                                decision = f"SWITCH_{target_mode}" if current_mode == "HIGH" else f"ESCALATE_{target_mode}"
                                current_mode = target_mode
                            protect_streak = 0
                    else:
                        if post_restore_deadline_ms is None:
                            response = send_control(args.cpu_control_host, args.cpu_control_port, "HIGH")
                            current_mode = "HIGH"
                            decision = "RESTORE_DONE_HIGH"
                            post_restore_deadline_ms = now_ms + max(args.post_restore_ms, 0)
                            protect_streak = 0
                        elif not stop_sent and now_ms >= post_restore_deadline_ms:
                            response = send_control(args.cpu_control_host, args.cpu_control_port, "STOP")
                            current_mode = "STOP"
                            decision = "STOP"
                            stop_sent = True

                writer.writerow(
                    [
                        now_ms,
                        current_mode,
                        decision,
                        1 if restore_done else 0,
                        restore_elapsed_ms,
                        restore_issued_bytes,
                        restore_completed_bytes,
                        restore_remaining_bytes,
                        "" if math.isnan(restore_inst_bw) else f"{restore_inst_bw:.6f}",
                        "" if math.isnan(restore_smooth_bw) else f"{restore_smooth_bw:.6f}",
                        "" if math.isnan(restore_guard) else f"{restore_guard:.6f}",
                        "" if math.isnan(restore_avg_bw) else f"{restore_avg_bw:.6f}",
                        "" if math.isnan(pcie_tx) else f"{pcie_tx:.6f}",
                        "" if math.isnan(pcie_rx) else f"{pcie_rx:.6f}",
                        f"{baseline:.6f}",
                        f"{enter_thresh:.6f}",
                        f"{exit_thresh:.6f}",
                        response,
                    ]
                )
                fp.flush()

                restore_exited = restore_proc is not None and restore_proc.poll() is not None
                if stop_sent and restore_exited:
                    break
                if restore_exited and post_restore_deadline_ms is None:
                    break

                time.sleep(max(args.poll_ms, 50) / 1000.0)

        return 0
    except KeyboardInterrupt:
        print("[controller] interrupted, sending STOP to remote cpu_client", file=sys.stderr)
        try:
            print(send_control(args.cpu_control_host, args.cpu_control_port, "STOP"), file=sys.stderr)
        except OSError:
            pass
        return 130
    finally:
        if restore_proc is not None:
            terminate_process(restore_proc, "gpu_pcie_memcpy")
        if metrics_proc is not None:
            terminate_process(metrics_proc, "gpu_metrics_logger")
        restore_fp.close()
        metrics_fp.close()


if __name__ == "__main__":
    raise SystemExit(main())
