#!/usr/bin/env python3
"""Send simple JSON-line controls to scene_runtime_external_control.py."""

from __future__ import annotations

import argparse
import json
import socket
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29001)
    parser.add_argument("--throttle", type=float, default=0.25)
    parser.add_argument("--steer", type=float, default=0.0)
    parser.add_argument("--brake", type=float, default=0.0)
    parser.add_argument("--interactive", action="store_true", help="Read throttle steer brake triples from stdin.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with socket.create_connection((args.host, args.port), timeout=30.0) as sock, sock.makefile("rwb") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            observation = json.loads(line.decode("utf-8"))
            if args.interactive:
                prompt = (
                    f"step={observation['step']} frame={observation['source_frame']} "
                    "enter throttle steer brake: "
                )
                text = input(prompt).strip()
                if text.lower() in {"q", "quit", "exit"}:
                    break
                parts = text.split()
                if len(parts) != 3:
                    print("expected three numbers, e.g. 0.3 0.0 0.0", file=sys.stderr)
                    command = {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
                else:
                    command = {"throttle": float(parts[0]), "steer": float(parts[1]), "brake": float(parts[2])}
            else:
                command = {"throttle": args.throttle, "steer": args.steer, "brake": args.brake}
            fp.write((json.dumps(command) + "\n").encode("utf-8"))
            fp.flush()
            print(
                f"step={observation['step']} frame={observation['source_frame']} "
                f"sent throttle={command['throttle']:.3f} steer={command['steer']:.3f} brake={command['brake']:.3f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
