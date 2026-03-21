#!/usr/bin/env python3
"""
rdma_msgsize_sweep.py

Run cpu_client across multiple message sizes and write results to CSV.

Example (matches your known-good settings, write op, QD=1, span=1G):
  python3 rdma_msgsize_sweep.py \
    --cpu_client ./cpu_client \
    --server_ip 10.10.10.11 --tcp_port 33333 --ib_dev mlx5_0 \
    --iters 100000 --op write --port 1 --gid_idx 3 \
    --qd 1 --span 1G --pattern random --align 256 --mtu 1024 \
    --msg_sizes 8,128,1024,4096,8192,65536 \
    --warmup_iters 20000 --warmup_runs 1 \
    --out ~/danyang/SkyGDR/results/contention1/rdma_msgsize_sweep.csv
"""

import argparse
import csv
import os
import re
import subprocess
import time


def parse_msg_sizes(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def append_optional_client_tail(cmd, args):
    # cpu_client optional tail order:
    # [sample] [max_samples] [ts_ms] [ts_out] [write_ack] [read_min_qd] [rd_atomic] [qps]
    # When write_ack is present, keep ts placeholders so parsing stays aligned.
    need_ts_placeholders = (
        args.ts_ms != 0
        or args.ts_out != ""
        or args.write_ack is not None
        or args.read_min_qd is not None
        or args.rd_atomic is not None
        or args.qps != 1
    )
    if need_ts_placeholders:
        cmd.append(str(args.ts_ms))
        cmd.append(args.ts_out if args.ts_out != "" else "-")
    if args.write_ack is not None:
        cmd.append(str(args.write_ack))
    elif args.read_min_qd is not None or args.rd_atomic is not None or args.qps != 1:
        # Keep positional alignment for downstream optional args.
        cmd.append("0" if args.op == "read" else "1")
    if args.read_min_qd is not None or args.rd_atomic is not None or args.qps != 1:
        cmd.append(str(args.read_min_qd) if args.read_min_qd is not None else "0")
    if args.rd_atomic is not None or args.qps != 1:
        cmd.append(str(args.rd_atomic) if args.rd_atomic is not None else "0")
    if args.qps != 1:
        cmd.append(str(args.qps))
    return cmd


def parse_output(text: str):
    res = {
        "elapsed_s": None,
        "ops_per_s": None,
        "throughput_gib_s": None,
        "p50_us": None,
        "p90_us": None,
        "p99_us": None,
        "p999_us": None,
        "min_us": None,
        "max_us": None,
    }
    m = re.search(r"elapsed=([0-9.]+)\s*s", text)
    if m:
        res["elapsed_s"] = float(m.group(1))
    m = re.search(r"ops=([0-9.]+)\s*ops/s", text)
    if m:
        res["ops_per_s"] = float(m.group(1))
    m = re.search(r"throughput=([0-9.]+)\s*GiB/s", text)
    if m:
        res["throughput_gib_s"] = float(m.group(1))
    m = re.search(
        r"latency_us\s+samples=\d+\s+p50=([0-9.]+)\s+p90=([0-9.]+)\s+p99=([0-9.]+)\s+p999=([0-9.]+)\s+min=([0-9.]+)\s+max=([0-9.]+)",
        text,
    )
    if m:
        res["p50_us"] = float(m.group(1))
        res["p90_us"] = float(m.group(2))
        res["p99_us"] = float(m.group(3))
        res["p999_us"] = float(m.group(4))
        res["min_us"] = float(m.group(5))
        res["max_us"] = float(m.group(6))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu_client", default="./cpu_client")
    ap.add_argument("--server_ip", required=True)
    ap.add_argument("--tcp_port", type=int, required=True)
    ap.add_argument("--ib_dev", required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--op", choices=["read", "write"], required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--gid_idx", type=int, required=True)
    ap.add_argument("--qd", type=int, required=True)
    ap.add_argument("--span", required=True)
    ap.add_argument("--pattern", choices=["random", "seq"], required=True)
    ap.add_argument("--align", type=int, required=True)
    ap.add_argument("--mtu", type=int, required=True)
    ap.add_argument("--msg_sizes", required=True, help="comma-separated sizes, e.g., 8,128,1024,4096,8192,65536")
    ap.add_argument("--warmup_iters", type=int, default=0, help="iters for warmup run (0 disables warmup)")
    ap.add_argument("--warmup_runs", type=int, default=0, help="how many warmup runs per msg size")
    ap.add_argument("--warmup_sleep_ms", type=int, default=0, help="sleep between warmup and measurement")
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--ts_ms", type=int, default=0, help="optional cpu_client ts_ms arg")
    ap.add_argument("--ts_out", default="", help="optional cpu_client ts_out arg")
    ap.add_argument("--write_ack", type=int, choices=[0, 1], default=None, help="optional cpu_client write_ack arg")
    ap.add_argument("--write_ack_batch", type=int, default=None, help="deprecated; ignored (client now enforces per-write ACK)")
    ap.add_argument("--read_min_qd", type=int, default=None, help="optional cpu_client read_min_qd arg")
    ap.add_argument("--rd_atomic", type=int, default=None, help="optional cpu_client rd_atomic arg (0/omit=auto)")
    ap.add_argument("--qps", type=int, default=1, help="optional cpu_client qps arg (default 1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.write_ack_batch is not None:
        print("[warn] --write_ack_batch is deprecated and ignored; per-write ACK is always used")

    msg_sizes = parse_msg_sizes(args.msg_sizes)
    if not msg_sizes:
        raise SystemExit("msg_sizes is empty")

    out_path = os.path.expanduser(args.out)
    header = [
        "Timestamp",
        "RunId",
        "MsgBytes",
        "Elapsed_s",
        "Ops_per_s",
        "Throughput_GiB_per_s",
        "P50_us",
        "P90_us",
        "P99_us",
        "P999_us",
        "Min_us",
        "Max_us",
        "RetCode",
        # Parameters (appended at end so one CSV can hold many runs)
        "ServerIP",
        "TcpPort",
        "IbDev",
        "Op",
        "Iters",
        "Port",
        "GidIdx",
        "Qd",
        "Qps",
        "Span",
        "Pattern",
        "Align",
        "Mtu",
        "Sample",
        "MaxSamples",
        "WarmupIters",
        "WarmupRuns",
        "WarmupSleepMs",
        "CpuClient",
    ]

    need_header = True
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        # If the existing CSV already uses our header, keep appending without rewriting it.
        # Otherwise, append a new header block so the appended section is self-describing.
        with open(out_path, "r", newline="") as rf:
            first = rf.readline().strip()
        need_header = first != ",".join(header)

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            if os.path.getsize(out_path) > 0:
                w.writerow([])
            w.writerow(header)

        run_id = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        common_tail = [
            args.server_ip,
            str(args.tcp_port),
            args.ib_dev,
            args.op,
            str(args.iters),
            str(args.port),
            str(args.gid_idx),
            str(args.qd),
            str(args.qps),
            str(args.span),
            args.pattern,
            str(args.align),
            str(args.mtu),
            str(args.sample),
            str(args.max_samples),
            str(args.warmup_iters),
            str(args.warmup_runs),
            str(args.warmup_sleep_ms),
            args.cpu_client,
        ]

        for msg in msg_sizes:
            if args.warmup_iters > 0 and args.warmup_runs > 0:
                for wi in range(args.warmup_runs):
                    warm_cmd = [
                        args.cpu_client,
                        args.server_ip,
                        str(args.tcp_port),
                        args.ib_dev,
                        str(args.warmup_iters),
                        str(msg),
                        args.op,
                        str(args.port),
                        str(args.gid_idx),
                        str(args.qd),
                        str(args.span),
                        args.pattern,
                        str(args.align),
                        str(args.mtu),
                        str(args.sample),
                        str(args.max_samples),
                    ]
                    append_optional_client_tail(warm_cmd, args)
                    print(f"Warmup ({wi + 1}/{args.warmup_runs}) msg={msg}:", " ".join(warm_cmd))
                    wret, wout = run_cmd(warm_cmd)
                    if wret != 0:
                        print(f"[warmup-error] msg={msg} ret={wret}")
                        if wout:
                            print(wout, end="" if wout.endswith("\n") else "\n")
                if args.warmup_sleep_ms > 0:
                    time.sleep(args.warmup_sleep_ms / 1000.0)
            cmd = [
                args.cpu_client,
                args.server_ip,
                str(args.tcp_port),
                args.ib_dev,
                str(args.iters),
                str(msg),
                args.op,
                str(args.port),
                str(args.gid_idx),
                str(args.qd),
                str(args.span),
                args.pattern,
                str(args.align),
                str(args.mtu),
                str(args.sample),
                str(args.max_samples),
            ]
            append_optional_client_tail(cmd, args)
            print("Running:", " ".join(cmd))
            ret, out = run_cmd(cmd)
            if ret != 0:
                print(f"[run-error] msg={msg} ret={ret}")
                if out:
                    print(out, end="" if out.endswith("\n") else "\n")
            parsed = parse_output(out)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            w.writerow([
                ts,
                run_id,
                msg,
                parsed["elapsed_s"],
                parsed["ops_per_s"],
                parsed["throughput_gib_s"],
                parsed["p50_us"],
                parsed["p90_us"],
                parsed["p99_us"],
                parsed["p999_us"],
                parsed["min_us"],
                parsed["max_us"],
                ret,
                *common_tail,
            ])
            f.flush()


if __name__ == "__main__":
    main()
