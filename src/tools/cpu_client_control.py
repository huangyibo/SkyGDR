#!/usr/bin/env python3

import argparse
import socket
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Send a control command to cpu_client's TCP control port.")
    ap.add_argument("--host", required=True, help="cpu_client host/IP")
    ap.add_argument("--port", type=int, required=True, help="cpu_client control port")
    ap.add_argument("command", help="HIGH / LOW / STOP / STATUS / 'SLEEP <us>'")
    ap.add_argument("--timeout_s", type=float, default=2.0)
    args = ap.parse_args()

    try:
        with socket.create_connection((args.host, args.port), timeout=args.timeout_s) as sock:
            sock.sendall((args.command.strip() + "\n").encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            resp = sock.recv(4096).decode("utf-8", errors="replace").strip()
    except OSError as exc:
        print(f"control connection failed: {exc}", file=sys.stderr)
        return 1

    print(resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
