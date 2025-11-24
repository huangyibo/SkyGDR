#!/usr/bin/env python3
"""
rdma_latency_sweep.py

Run RDMA client tests while sweeping gpu_memhog block counts on remote GPU server.

Assumptions:
- Run this script on the CPU server (where cpu_client runs).
- Passwordless SSH key from CPU -> GPU server is set up (or SSH will prompt).
- gpu_server, gpu_memhog exist on GPU server in REMOTE_DIR and are executable.
- cpu_client exists locally and is executable.

Output:
- results.csv with columns: block, ops_per_s, throughput_gib_s, elapsed_s, raw_output

Adjust CONFIG below as needed.
"""

import subprocess
import shlex
import time
import csv
import re
import os
import sys

### ========== CONFIG ========== ###
GPU_USER = "x_peic"              # e.g. "x_peic" or "ubuntu"
GPU_HOST = "10.200.0.27"                # GPU server IP or hostname
# directory on GPU server containing gpu_server, gpu_memhog
REMOTE_DIR = "/home/x_peic/SkyGDR/src/bin"
SSH_BIN = "ssh"
SCP_BIN = "scp"

# Command to launch gpu_server on GPU host (run once, persistent)
GPU_SERVER_CMD = "./gpu_server mlx5_1 1M 18515 1 3 1024"

# Command template to launch gpu_memhog on GPU host
# {blocks} will be substituted
GPU_MEMHOG_CMD_TEMPLATE = "./gpu_memhog -op=rw --blocks={blocks}"

# Command to run on CPU server (local). Use the full command line you gave.
CPU_CLIENT_CMD = "/home/x_peic/SkyGDR/src/bin/cpu_client 10.200.0.27 18515 mlx5_1 100000 65536 read 1 3 64 1M random 256 1024"

# Sweep parameters
BLOCK_START = 0
BLOCK_STEP = 16
BLOCK_MAX = 1024   # inclusive upper bound; adjust as needed
# memhog runs up to 60s, but we will wait only for client to finish then kill
MEMHOG_RUNTIME_SEC = 60
WAIT_AFTER_MEMHOG_START = 1.0  # wait before starting client

# Timeouts
CLIENT_TIMEOUT = 120  # seconds to wait for the cpu_client to finish before killing
SSH_TIMEOUT = 10

# Result file
RESULT_CSV = "../../results/results.csv"

# Whether to kill gpu_server at end (True/False)
KILL_GPU_SERVER_AT_END = True
### ========== END CONFIG ========== ###


def ssh_cmd(cmd, user=GPU_USER, host=GPU_HOST, timeout=None, get_output=True, shell=False):
    """
    Run ssh user@host <cmd>. Returns (returncode, stdout, stderr).
    If get_output=False, stdout/stderr are None and the subprocess is not captured.
    """
    target = f"{user}@{host}"
    # Use -o BatchMode=yes to fail quickly if no key and no password prompt desired.
    ssh_base = [SSH_BIN, "-o", "BatchMode=yes", target, "--", cmd]
    try:
        if get_output:
            p = subprocess.run(ssh_base, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout, text=True, shell=shell)
            return p.returncode, p.stdout, p.stderr
        else:
            p = subprocess.Popen(ssh_base)
            return None, None, None
    except subprocess.TimeoutExpired:
        return 124, "", "ssh timeout"


def ssh_run_nohup_background(cmd,
                             user=GPU_USER,
                             host=GPU_HOST,
                             remote_dir=REMOTE_DIR,
                             logname=None):
    """
    Launch remote command in background with nohup inside conda env rdma_env.
    """
    if logname is None:
        logname = "nohup_remote.log"

    # 激活 conda 的代码（适用于大多数 conda 安装）
    conda_init = (
        "source ~/anaconda3/etc/profile.d/conda.sh || "
        "source ~/miniconda3/etc/profile.d/conda.sh || "
        "source ~/miniforge3/etc/profile.d/conda.sh"
    )

    # 构造远端执行的完整命令
    full = (
        f"{conda_init} && "
        f"conda activate rdma_env && "
        f"cd {shlex.quote(remote_dir)} && "
        f"nohup {cmd} > {shlex.quote(logname)} 2>&1 & echo $!"
    )

    rc, out, err = ssh_cmd(full, user=user, host=host, timeout=SSH_TIMEOUT)
    return rc, out.strip() if out else out, err



def ssh_kill_by_cmdpattern(pattern, user=GPU_USER, host=GPU_HOST, remote_dir=REMOTE_DIR):
    """
    Kill remote processes that match the given pattern in their command line.
    Uses pkill -f 'pattern'.
    """
    cmd = f"pkill -f {shlex.quote(pattern)} || true"
    return ssh_cmd(cmd, user=user, host=host, timeout=SSH_TIMEOUT)


# parse sample client output, return dict with keys ops_per_s, throughput_gib_s, elapsed_s
def parse_client_output(text):
    # Example lines:
    # [client] done: iters=100000 msg=65536 qd=64 span=1073741824 pattern=random
    # [client] elapsed=1.302 s  ops=76813 ops/s  throughput=4.69 GiB/s
    res = {"ops_per_s": None, "throughput_gib_s": None,
           "elapsed_s": None, "raw": text}
    m = re.search(r"elapsed=([0-9.]+)\s*s", text)
    if m:
        try:
            res["elapsed_s"] = float(m.group(1))
        except:
            pass
    m = re.search(r"ops=([0-9.]+)\s*ops/s", text)
    if m:
        try:
            res["ops_per_s"] = float(m.group(1))
        except:
            pass
    m = re.search(r"throughput=([0-9.]+)\s*GiB/s", text)
    if m:
        try:
            res["throughput_gib_s"] = float(m.group(1))
        except:
            pass
    # fallback pattern variants
    if res["ops_per_s"] is None:
        m = re.search(r"ops/s[:=]?\s*([0-9.]+)", text)
        if m:
            res["ops_per_s"] = float(m.group(1))
    if res["throughput_gib_s"] is None:
        m = re.search(r"([\d.]+)\s*GiB/s", text)
        if m:
            res["throughput_gib_s"] = float(m.group(1))
    return res


def run_local_client_and_capture(cmd, timeout=CLIENT_TIMEOUT):
    """
    Run the cpu_client locally and capture stdout/stderr as text.
    Returns (returncode, combined_stdout_stderr)
    """
    try:
        p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout, text=True)
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s\n{str(e)}"


def ensure_gpu_server_running():
    # kill existing gpu_server processes that may conflict (be careful!)
    print("Killing any existing gpu_server (matching 'gpu_server') on remote host...")
    ssh_kill_by_cmdpattern("gpu_server")
    time.sleep(0.5)

    print("Starting gpu_server on GPU host (nohup) ...")
    rc, out, err = ssh_run_nohup_background(
        GPU_SERVER_CMD, logname="gpu_server.log")
    if rc != 0:
        print("Warning: starting gpu_server returned non-zero rc", rc, err)
    else:
        print("gpu_server start output:", out)
    # small sleep to let it bind ports
    time.sleep(1.0)


def stop_gpu_server():
    print("Stopping gpu_server on remote host...")
    ssh_kill_by_cmdpattern("gpu_server")
    time.sleep(0.3)


def main():
    # prepare CSV
    write_header = True
    if os.path.exists(RESULT_CSV):
        # backup existing file
        bak = RESULT_CSV + ".bak"
        print(f"{RESULT_CSV} exists — backing up to {bak}")
        os.replace(RESULT_CSV, bak)

    with open(RESULT_CSV, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["timestamp", "blocks", "ops_per_s", "throughput_gib_s",
                        "elapsed_s", "client_retcode", "notes", "raw_output"])

        # warm up
        for _ in range(5):
            print("=== Warming Up ===\n")
            
            # ensure gpu_server running once
            ensure_gpu_server_running()

            # ensure no leftover memhog
            ssh_kill_by_cmdpattern("gpu_memhog")

            # start memhog with blocks=b
            memhog_cmd = GPU_MEMHOG_CMD_TEMPLATE.format(blocks=0)
            remote_log = f"memhog_blocks_{0}.log"
            print(f"Starting remote memhog: {memhog_cmd}")
            rc, out, err = ssh_run_nohup_background(
                memhog_cmd, logname=remote_log)
            if rc != 0:
                notes = f"memhog start failed rc={rc} stderr={err}"
                print(notes)
            else:
                print("memhog started, out:", out)

            # wait for memhog to ramp
            print(
                f"Waiting {WAIT_AFTER_MEMHOG_START:.1f}s for memhog to ramp...")
            time.sleep(WAIT_AFTER_MEMHOG_START)

            # run local cpu_client
            print("Running local cpu_client command:")
            print("  ", CPU_CLIENT_CMD)
            retcode, output = run_local_client_and_capture(CPU_CLIENT_CMD)
            print("Client returned", retcode)
            print("Client output (truncated 300 chars):")
            print(output[:300])

            # stop memhog now
            print("Stopping remote memhog...")
            ssh_kill_by_cmdpattern("gpu_memhog")
            time.sleep(0.5)


        # sweep blocks
        b = BLOCK_START
        while b <= BLOCK_MAX:
            print(f"\n=== Testing blocks = {b} ===")
            # ensure gpu_server running once
            ensure_gpu_server_running()

            # ensure no leftover memhog
            ssh_kill_by_cmdpattern("gpu_memhog")

            # start memhog with blocks=b
            memhog_cmd = GPU_MEMHOG_CMD_TEMPLATE.format(blocks=b)
            remote_log = f"memhog_blocks_{b}.log"
            print(f"Starting remote memhog: {memhog_cmd}")
            rc, out, err = ssh_run_nohup_background(
                memhog_cmd, logname=remote_log)
            if rc != 0:
                notes = f"memhog start failed rc={rc} stderr={err}"
                print(notes)
            else:
                print("memhog started, out:", out)

            # wait for memhog to ramp
            print(
                f"Waiting {WAIT_AFTER_MEMHOG_START:.1f}s for memhog to ramp...")
            time.sleep(WAIT_AFTER_MEMHOG_START)

            # run local cpu_client
            print("Running local cpu_client command:")
            print("  ", CPU_CLIENT_CMD)
            retcode, output = run_local_client_and_capture(CPU_CLIENT_CMD)
            print("Client returned", retcode)
            print("Client output (truncated 300 chars):")
            print(output[:300])

            # parse
            parsed = parse_client_output(output)
            ops = parsed.get("ops_per_s")
            thr = parsed.get("throughput_gib_s")
            elapsed = parsed.get("elapsed_s")

            # stop memhog now
            print("Stopping remote memhog...")
            ssh_kill_by_cmdpattern("gpu_memhog")
            time.sleep(0.5)

            # write CSV row
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            notes = ""
            if retcode != 0:
                notes = f"client_ret_nonzero={retcode}"
            writer.writerow([ts, b, ops, thr, elapsed, retcode,
                            notes, parsed.get("raw", "")])

            csvf.flush()
            os.fsync(csvf.fileno())

            # increment
            b += BLOCK_STEP

        # finished sweep
        if KILL_GPU_SERVER_AT_END:
            stop_gpu_server()

    print("\nAll done. Results written to", RESULT_CSV)


if __name__ == "__main__":
    main()
