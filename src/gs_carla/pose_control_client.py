#!/usr/bin/env python3
"""External pose-path controller client for scene_runtime_external_control.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import time
from pathlib import Path
from types import SimpleNamespace

from . import carla_xodr_live_3dgs_bridge_with_instances as base
from .load_xodr_town_and_dump_poses import (
    clamp,
    core_yaw_to_carla_yaw_deg,
    enu_to_sequence_start_local,
    lookahead_path_index,
    normalize_angle,
    read_mapping_origin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=29001)
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    parser.add_argument("--connect-retry-interval", type=float, default=1.0)
    parser.add_argument("--core-start-frame", type=int)
    parser.add_argument("--core-end-frame", type=int)
    parser.add_argument("--control-mode", choices=("path", "steer-bias", "constant-steer"), default="path")
    parser.add_argument("--steer-bias", type=float, default=0.0)
    parser.add_argument("--follow-speed-kmh", type=float, default=3.0)
    parser.add_argument("--follow-lookahead-distance", type=float, default=10.0)
    parser.add_argument("--stop-at-path-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-speed-threshold", type=float, default=0.05)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--speed-kp", type=float, default=0.35)
    parser.add_argument("--speed-ki", type=float, default=0.02)
    parser.add_argument("--speed-kd", type=float, default=0.02)
    parser.add_argument("--steer-gain", type=float, default=0.35)
    parser.add_argument("--max-steer", type=float, default=0.30)
    return parser.parse_args()


def package_path(package_file: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else package_file.parent / path


def load_config(cli: argparse.Namespace) -> argparse.Namespace:
    package = json.loads(cli.scene_package.expanduser().read_text(encoding="utf-8"))
    assets = package.get("assets", {})
    frames = package.get("frames", {})
    values = package.get("values", {})
    data = vars(cli).copy()
    data.update(
        {
            "mapping_pose": package_path(cli.scene_package, assets.get("mapping_pose")),
            "core_pose_csv": package_path(cli.scene_package, assets.get("core_pose_csv")),
            "sequence_origin_pose_file": package_path(cli.scene_package, assets.get("sequence_origin_pose_file")),
            "sequence_origin_frame": frames.get("sequence_origin_frame", 0),
            "core_start_frame": cli.core_start_frame if cli.core_start_frame is not None else frames.get("core_start_frame"),
            "core_end_frame": cli.core_end_frame if cli.core_end_frame is not None else frames.get("core_end_frame"),
            "core_pose_coordinate_frame": values.get("core_pose_coordinate_frame", "enu"),
            "no_core_y_flip": False,
            "core_pose_z": 0.5,
        }
    )
    return SimpleNamespace(**data)


def read_core_pose_sequence(args: argparse.Namespace) -> list[dict]:
    origin = None
    if args.core_pose_coordinate_frame == "enu":
        origin = read_mapping_origin(args.sequence_origin_pose_file or args.mapping_pose, args.sequence_origin_frame)
    start, end = sorted((args.core_start_frame, args.core_end_frame))
    poses: list[dict] = []
    with args.core_pose_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            if not start <= frame <= end:
                continue
            if args.core_pose_coordinate_frame == "enu":
                x_key = "enu_x" if "enu_x" in row else "x"
                y_key = "enu_y" if "enu_y" in row else "y"
                yaw_key = "enu_yaw" if "enu_yaw" in row else "yaw"
                local_x, local_y = enu_to_sequence_start_local(float(row[x_key]), float(row[y_key]), origin)
                local_yaw = normalize_angle(float(row[yaw_key]) - float(origin["yaw"]))
                carla_x, carla_y = (local_x, -local_y)
                carla_yaw = core_yaw_to_carla_yaw_deg(args, local_yaw)
            else:
                x_key = "x" if "x" in row else "corrected_x"
                y_key = "y" if "y" in row else "corrected_y"
                yaw_value = float(row.get("yaw", row.get("corrected_yaw", 0.0)))
                carla_x, carla_y = float(row[x_key]), float(row[y_key])
                carla_yaw = math.degrees(yaw_value) if abs(yaw_value) <= 2.0 * math.pi else yaw_value
            poses.append({"frame": frame, "x": carla_x, "y": carla_y, "z": args.core_pose_z, "yaw": carla_yaw})
    poses.sort(key=lambda item: item["frame"])
    return poses


def local_xy_from_pose(current: dict, target: dict) -> tuple[float, float]:
    dx = float(target["x"]) - float(current["x"])
    dy = float(target["y"]) - float(current["y"])
    yaw = math.radians(float(current["yaw"]))
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


class Controller:
    def __init__(self, args: argparse.Namespace, core_poses: list[dict]):
        self.args = args
        self.core_poses = core_poses
        self.pid_state: dict[str, float] = {}

    @staticmethod
    def stop_control() -> dict[str, float]:
        return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}

    def reached_path_end(self, observation: dict) -> bool:
        if not self.args.stop_at_path_end:
            return False
        if bool(observation.get("path_finished", False)):
            return True
        nearest_index = int(observation["nearest_index"])
        source_frame = int(observation["source_frame"])
        last_frame = int(self.core_poses[-1]["frame"])
        return nearest_index >= len(self.core_poses) - 1 or source_frame >= last_frame

    def control(self, observation: dict) -> dict[str, float]:
        if self.reached_path_end(observation):
            return self.stop_control()
        nearest_index = int(observation["nearest_index"])
        target_index = lookahead_path_index(self.core_poses, nearest_index, self.args.follow_lookahead_distance)
        target = self.core_poses[target_index]
        pose = observation["pose"]
        local_x, local_y = local_xy_from_pose(pose, target)
        lookahead = max(math.hypot(local_x, local_y), 1e-3)
        curvature = 2.0 * local_y / max(lookahead * lookahead, 1e-3)
        speed = float(observation["speed_mps"])
        steer_limit = self.args.max_steer if speed >= 1.0 else min(self.args.max_steer, 0.25)
        base_steer = clamp(self.args.steer_gain * curvature * 2.8, -steer_limit, steer_limit)
        steer_bias = self.args.steer_bias if self.args.control_mode in ("steer-bias", "constant-steer") else 0.0
        steer = clamp(steer_bias if self.args.control_mode == "constant-steer" else base_steer + steer_bias, -steer_limit, steer_limit)
        target_speed = max(self.args.follow_speed_kmh / 3.6, 0.0)
        dt = float(observation["dt"])
        speed_error = target_speed - speed
        self.pid_state["integral"] = clamp(self.pid_state.get("integral", 0.0) + speed_error * dt, -5.0, 5.0)
        derivative = (speed_error - self.pid_state.get("previous_error", speed_error)) / max(dt, 1e-6)
        self.pid_state["previous_error"] = speed_error
        accel_cmd = self.args.speed_kp * speed_error + self.args.speed_ki * self.pid_state["integral"] + self.args.speed_kd * derivative
        return {"throttle": clamp(accel_cmd, 0.0, 0.75), "steer": steer, "brake": clamp(-accel_cmd, 0.0, 0.8)}


def connect_with_retry(args: argparse.Namespace) -> socket.socket:
    deadline = time.monotonic() + max(args.connect_timeout, 0.0)
    attempt = 0
    while True:
        attempt += 1
        try:
            sock = socket.create_connection((args.control_host, args.control_port), timeout=args.connect_retry_interval)
            print(f"[pose-control] connected to {args.control_host}:{args.control_port}")
            return sock
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"could not connect to {args.control_host}:{args.control_port} after {attempt} attempts; "
                    "start scene_runtime_external_control.py first and wait for the control server to listen"
                ) from exc
            print(f"[pose-control] waiting for {args.control_host}:{args.control_port} ({exc})")
            time.sleep(max(args.connect_retry_interval, 0.1))


def main() -> int:
    args = load_config(parse_args())
    controller = Controller(args, read_core_pose_sequence(args))
    sock = connect_with_retry(args)
    try:
        with sock, sock.makefile("rwb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                observation = json.loads(line.decode("utf-8"))
                command = controller.control(observation)
                f.write((json.dumps(command) + "\n").encode("utf-8"))
                f.flush()
                reached_end = controller.reached_path_end(observation)
                status = " stopping-at-end" if reached_end else ""
                print(
                    f"[pose-control] step={observation['step']} frame={observation['source_frame']} "
                    f"steer={command['steer']:.3f} throttle={command['throttle']:.3f} "
                    f"brake={command['brake']:.3f}{status}"
                )
                if reached_end:
                    print("[pose-control] path end reached; sent stop control and disconnecting")
                    break
    except KeyboardInterrupt:
        print("[pose-control] interrupted; disconnecting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
