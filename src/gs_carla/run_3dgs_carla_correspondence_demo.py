#!/usr/bin/env python3
"""Run CARLA/3DGS correspondence demos with one command.

Modes:

  frozen-ego-moving-3dgs
      CARLA hero stays braked. Rendering uses the frozen actor pose, so the
      camera/view stays fixed, while source_frame can advance for live instances.

  path-follow
      Starts the runtime plus pose_control_client. The hero follows the CARLA
      path, and 3DGS is rendered from the live actor pose.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("frozen-ego-fixed-3dgs", "frozen-ego-moving-3dgs", "path-follow"),
        default="frozen-ego-moving-3dgs",
        help="frozen-ego-fixed-3dgs freezes both CARLA hero and 3DGS camera; frozen-ego-moving-3dgs freezes only CARLA hero.",
    )
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=29001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--core-start-frame", type=int, default=315)
    parser.add_argument("--core-end-frame", type=int, default=350)
    parser.add_argument("--cameras", nargs="+", type=int, help="Camera ids to render into one pygame/video mosaic, e.g. --cameras 0 1 2 3 4 5.")
    parser.add_argument("--dynamic-rigid", type=Path, help="Exported RigidNodes pth. Defaults to rigid_nodes_gaussians.pth next to the scene package.")
    parser.add_argument("--follow-speed-kmh", type=float, default=3.0)
    parser.add_argument("--follow-lookahead-distance", type=float, default=10.0)
    parser.add_argument("--follow-start-z-offset", type=float, default=0.35)
    parser.add_argument("--follow-settle-ticks", type=int, default=30)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--source-frame-fps", type=float, default=10.0)
    parser.add_argument("--steer-gain", type=float, default=0.35)
    parser.add_argument("--max-steer", type=float, default=0.30)
    parser.add_argument("--max-render-frames", type=int, default=0)
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--pygame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--display-fps", type=float, default=0.0)
    parser.add_argument("--spectator-follow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/correspondence_demo"))
    parser.add_argument("--max-gaussians", type=int, default=5_000_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.02)
    parser.add_argument("--crop-radius", type=float, default=160.0)
    parser.add_argument("--far", type=float, default=220.0)
    parser.add_argument("--downscale", type=float, default=3.0)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument(
        "--live-instances",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Spawn/replay package instances. Defaults to false in frozen mode and true in path-follow mode.",
    )
    parser.add_argument("--max-instance-actors", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--client-connect-timeout", type=float, default=120.0)
    parser.add_argument(
        "--attach-existing-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In frozen demo modes, connect to an already-running scene_runtime_external_control.py "
            "on --control-host/--control-port and send path controls instead of starting another "
            "CARLA/3DGS renderer. Use --no-attach-existing-runtime to run the frozen runtime itself."
        ),
    )
    return parser.parse_args()


def append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def runtime_command(args: argparse.Namespace, tools_dir: Path, out_dir: Path) -> list[str]:
    frozen = args.mode in ("frozen-ego-fixed-3dgs", "frozen-ego-moving-3dgs")
    frozen_fixed = args.mode == "frozen-ego-fixed-3dgs"
    live_instances = (not frozen) if args.live_instances is None else bool(args.live_instances)
    max_render_frames = args.max_render_frames
    if frozen_fixed and max_render_frames <= 0:
        max_render_frames = 1
    render_pose_source = "actor"
    source_frame_mode = "timeline" if args.mode in ("frozen-ego-moving-3dgs", "path-follow") else "nearest"
    cmd = [
        sys.executable,
        "-m", "gs_carla.scene_runtime_external_control",
        "--scene-package",
        str(args.scene_package),
        "--control-host",
        args.control_host,
        "--control-port",
        str(args.control_port),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--core-start-frame",
        str(args.core_start_frame),
        "--core-end-frame",
        str(args.core_end_frame),
        *(["--cameras", *(str(camera) for camera in args.cameras)] if args.cameras is not None else []),
        *(["--dynamic-rigid", str(args.dynamic_rigid)] if args.dynamic_rigid is not None else []),
        "--render-pose-source",
        render_pose_source,
        "--source-frame-mode",
        source_frame_mode,
        "--source-frame-fps",
        str(args.source_frame_fps),
        "--vehicle-zrp-source",
        "mapping-frame",
        "--follow-speed-kmh",
        str(args.follow_speed_kmh),
        "--follow-start-z-offset",
        str(args.follow_start_z_offset),
        "--follow-settle-ticks",
        str(args.follow_settle_ticks),
        "--control-dt",
        str(args.control_dt),
        "--display-fps",
        str(args.display_fps),
        "--render-every-n",
        str(args.render_every_n),
        "--pose-output",
        str(out_dir / "runtime_pose.csv"),
        "--path-json-output",
        str(out_dir / "runtime_path.json"),
        "--instance-output",
        str(out_dir / "runtime_instances.csv"),
        "--traffic-bbox-output",
        str(out_dir / "runtime_traffic_bboxes.json"),
        "--video-output",
        str(out_dir / "runtime_render.mp4"),
        "--video-fps",
        str(args.video_fps),
        "--max-gaussians",
        str(args.max_gaussians),
        "--opacity-threshold",
        str(args.opacity_threshold),
        "--crop-radius",
        str(args.crop_radius),
        "--far",
        str(args.far),
        "--downscale",
        str(args.downscale),
    ]
    if max_render_frames > 0:
        cmd.extend(["--max-render-frames", str(max_render_frames)])
    if frozen:
        cmd.extend(["--render-fixed-source-frame", str(args.core_start_frame)])
    if args.max_instance_actors > 0:
        cmd.extend(["--max-instance-actors", str(args.max_instance_actors)])
    append_flag(cmd, "--pygame", args.pygame)
    append_flag(cmd, "--spectator-follow", args.spectator_follow)
    append_flag(cmd, "--no-save-frames", args.no_save_frames)
    append_flag(cmd, "--freeze-hero", frozen)
    append_flag(cmd, "--disable-control-server", frozen)
    append_flag(cmd, "--exit-after-max-render-frames", frozen_fixed)
    append_flag(cmd, "--no-live-instances", not live_instances)
    return cmd


def client_command(args: argparse.Namespace, tools_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m", "gs_carla.pose_control_client",
        "--scene-package",
        str(args.scene_package),
        "--control-host",
        args.control_host,
        "--control-port",
        str(args.control_port),
        "--connect-timeout",
        str(args.client_connect_timeout),
        "--core-start-frame",
        str(args.core_start_frame),
        "--core-end-frame",
        str(args.core_end_frame),
        "--control-mode",
        "path",
        "--follow-speed-kmh",
        str(args.follow_speed_kmh),
        "--follow-lookahead-distance",
        str(args.follow_lookahead_distance),
        "--control-dt",
        str(args.control_dt),
        "--steer-gain",
        str(args.steer_gain),
        "--max-steer",
        str(args.max_steer),
    ]


def connect_with_retry(args: argparse.Namespace) -> socket.socket:
    deadline = time.monotonic() + max(args.client_connect_timeout, 0.0)
    attempt = 0
    while True:
        attempt += 1
        try:
            sock = socket.create_connection((args.control_host, args.control_port), timeout=1.0)
            print(f"[demo] connected to runtime control server at {args.control_host}:{args.control_port}", flush=True)
            return sock
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"could not connect to {args.control_host}:{args.control_port} after {attempt} attempts"
                ) from exc
            print(f"[demo] waiting for {args.control_host}:{args.control_port} ({exc})", flush=True)
            time.sleep(1.0)


def run_brake_client(args: argparse.Namespace) -> int:
    sock = connect_with_retry(args)
    start_frame, end_frame = sorted((int(args.core_start_frame), int(args.core_end_frame)))
    duration_s = max(0.0, (end_frame - start_frame) / max(args.source_frame_fps, 1e-6))
    max_steps = max(1, int(duration_s / max(args.control_dt, 1e-6)))
    try:
        with sock, sock.makefile("rwb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                observation = json.loads(line.decode("utf-8"))
                step = int(observation["step"])
                command = {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
                f.write((json.dumps(command) + "\n").encode("utf-8"))
                f.flush()
                print(
                    f"[demo-brake] step={step} runtime_frame={observation.get('source_frame')} "
                    f"brake=1.000",
                    flush=True,
                )
                if step >= max_steps:
                    print("[demo-brake] frozen duration reached; disconnecting", flush=True)
                    break
    except KeyboardInterrupt:
        print("[demo-brake] interrupted; disconnecting", flush=True)
    return 0


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    mode_name = args.mode
    out_dir = args.output_root / mode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_control_runtime = not port_is_available(args.control_host, args.control_port)
    if mode_name in ("frozen-ego-fixed-3dgs", "frozen-ego-moving-3dgs") and existing_control_runtime:
        print(
            f"[demo] found existing runtime at {args.control_host}:{args.control_port}; "
            "running brake control only, with no extra CARLA/3DGS runtime and no extra pygame",
            flush=True,
        )
        if args.pygame:
            print("[demo] ignoring --pygame because rendering stays in the existing runtime window", flush=True)
        return run_brake_client(args)

    if mode_name == "path-follow" and existing_control_runtime:
        cmd = client_command(args, tools_dir)
        print(
            f"[demo] found existing runtime at {args.control_host}:{args.control_port}; "
            "running path control client only and leaving 3DGS/pygame rendering in that runtime",
            flush=True,
        )
        if args.pygame:
            print("[demo] ignoring --pygame because rendering stays in the existing runtime window", flush=True)
        print("+ " + " ".join(cmd), flush=True)
        return subprocess.call(cmd)

    runtime_cmd = runtime_command(args, tools_dir, out_dir)
    print("+ " + " ".join(runtime_cmd), flush=True)
    runtime = subprocess.Popen(runtime_cmd)
    client = None

    def handle_signal(signum, _frame):
        terminate(client)
        terminate(runtime)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.mode == "path-follow":
            time.sleep(3.0)
            cmd = client_command(args, tools_dir)
            print("+ " + " ".join(cmd), flush=True)
            client = subprocess.Popen(cmd)
            client_rc = client.wait()
            terminate(runtime)
            if client_rc != 0:
                return client_rc
            return 0
        return runtime.wait()
    finally:
        terminate(client)
        terminate(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
