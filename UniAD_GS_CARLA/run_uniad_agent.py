#!/usr/bin/env python3
"""Run Bench2DriveZoo UniAD as the CARLA/3DGS JSON-line controller."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_CAMERA_MAP = {
    "CAM_FRONT": 0,
    "CAM_FRONT_LEFT": 1,
    "CAM_FRONT_RIGHT": 2,
    "CAM_BACK": 5,
    "CAM_BACK_LEFT": 3,
    "CAM_BACK_RIGHT": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="UniAD Python config file")
    parser.add_argument("--checkpoint", required=True, type=Path, help="UniAD .pth checkpoint")
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=29001)
    parser.add_argument("--camera-map", type=Path, help="Optional JSON mapping UniAD camera names to 3DGS camera ids")
    parser.add_argument("--carla-python-api", type=Path, help="CARLA root/PythonAPI directory when it is not already on PYTHONPATH")
    parser.add_argument("--connect-timeout", type=float, default=300.0)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    parser.add_argument("--log", type=Path, help="Optional JSONL log containing observations and controls")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--episode-result", type=Path, help="Runtime episode_result.json to score after socket closure")
    parser.add_argument("--score-output", type=Path, help="Output score JSON; defaults beside --episode-result")
    parser.add_argument("--score-wait-timeout", type=float, default=30.0)
    parser.add_argument("--collision-penalty", type=float, default=0.0)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_import_paths(carla_python_api: Path | None) -> None:
    root = repo_root()
    zoo = Path(os.environ.get("BENCH2DRIVE_ZOO_ROOT", root / "Bench2DriveZoo")).expanduser().resolve()
    bench = Path(os.environ.get("BENCH2DRIVE_ROOT", root / "Bench2Drive")).expanduser().resolve()
    paths = [
        root,
        zoo.parent,
        zoo,
        bench,
        bench / "leaderboard",
        bench / "scenario_runner",
    ]
    if carla_python_api is not None:
        api = carla_python_api.expanduser().resolve()
        if api.name != "PythonAPI":
            api = api / "PythonAPI"
        paths.extend([api, api / "carla"])
        dist = api / "carla" / "dist"
        if dist.exists():
            paths.extend(sorted(dist.glob("carla-*.egg")))
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_asset(package_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else package_path.parent / path


def load_route_xy(scene_package: Path) -> list[tuple[float, float]]:
    package = json.loads(scene_package.read_text(encoding="utf-8"))
    coordinate_frame = package.get("values", {}).get("core_pose_coordinate_frame", "enu")
    if coordinate_frame != "carla":
        raise ValueError(
            "This adapter currently requires values.core_pose_coordinate_frame='carla'; "
            f"the package declares {coordinate_frame!r}. Export the runtime CARLA route first."
        )
    route_path = resolve_asset(scene_package, package["assets"]["core_pose_csv"])
    start_value = package.get("frames", {}).get("core_start_frame")
    end_value = package.get("frames", {}).get("core_end_frame")
    start = int(start_value) if start_value is not None else -2**31
    end = int(end_value) if end_value is not None else 2**31 - 1
    points: list[tuple[float, float]] = []
    with route_path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            frame = int(row["frame"])
            if start <= frame <= end:
                points.append((float(row["x"]), float(row["y"])))
    if len(points) < 2:
        raise ValueError(f"route {route_path} contains fewer than two selected points")
    return points


def load_camera_map(path: Path | None) -> dict[str, int]:
    mapping = DEFAULT_CAMERA_MAP if path is None else json.loads(path.read_text(encoding="utf-8"))
    if set(mapping) != set(DEFAULT_CAMERA_MAP):
        raise ValueError(f"camera map must contain exactly: {sorted(DEFAULT_CAMERA_MAP)}")
    return {name: int(camera_id) for name, camera_id in mapping.items()}


def connect_with_retry(host: str, port: int, timeout: float, retry_interval: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection((host, port), timeout=30.0)
        except OSError as exc:
            last_error = exc
            time.sleep(max(0.05, retry_interval))
    raise TimeoutError(f"could not connect to {host}:{port} within {timeout}s: {last_error}")


def safe_control(control: Any) -> dict[str, float]:
    raw = {
        "throttle": float(control.throttle),
        "steer": float(control.steer),
        "brake": float(control.brake),
    }
    if not all(math.isfinite(value) for value in raw.values()):
        return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
    values = {
        "throttle": max(0.0, min(1.0, raw["throttle"])),
        "steer": max(-1.0, min(1.0, raw["steer"])),
        "brake": max(0.0, min(1.0, raw["brake"])),
    }
    if values["brake"] > 0.05:
        values["throttle"] = 0.0
    return values


def main() -> int:
    args = parse_args()
    initial_result_mtime = args.episode_result.stat().st_mtime_ns if args.episode_result and args.episode_result.exists() else None
    for label, path in (
        ("config", args.config),
        ("checkpoint", args.checkpoint),
        ("scene package", args.scene_package),
    ):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    configure_import_paths(args.carla_python_api)
    os.environ["IS_BENCH2DRIVE"] = "1"
    os.environ.pop("SAVE_PATH", None)

    import cv2

    from UniAD_GS_CARLA.team_code.uniad_gs_agent import GSUniadAgent, observation_to_input_data

    scene_package = args.scene_package.expanduser().resolve()
    camera_map = load_camera_map(args.camera_map)
    route_xy = load_route_xy(scene_package)
    config_and_checkpoint = f"{args.config.expanduser().resolve()}+{args.checkpoint.expanduser().resolve()}+gs_closed_loop"
    print(f"[uniad] loading model and checkpoint: {args.checkpoint}", flush=True)
    agent = GSUniadAgent(config_and_checkpoint, route_xy)
    log_fp = None
    processed = 0
    try:
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log_fp = args.log.open("w", encoding="utf-8")
        sock = connect_with_retry(args.host, args.control_port, args.connect_timeout, args.retry_interval)
        print(f"[uniad] connected to {args.host}:{args.control_port}", flush=True)
        with sock, sock.makefile("rwb") as fp:
            while True:
                line = fp.readline()
                if not line:
                    break
                observation = json.loads(line.decode("utf-8"))
                if observation.get("sensor_packet") is None:
                    command = {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
                    state = "warmup"
                else:
                    input_data = observation_to_input_data(observation, camera_map, cv2)
                    agent.set_external_observation(observation)
                    control = agent.run_step(input_data, float(observation["step"]) * float(observation["dt"]))
                    command = safe_control(control)
                    state = "inference"
                    processed += 1
                fp.write((json.dumps(command) + "\n").encode("utf-8"))
                fp.flush()
                if log_fp is not None:
                    log_fp.write(json.dumps({"observation": observation, "control": command, "state": state}) + "\n")
                    log_fp.flush()
                print(
                    f"[uniad] step={observation['step']} state={state} "
                    f"throttle={command['throttle']:.3f} steer={command['steer']:.3f} brake={command['brake']:.3f}",
                    flush=True,
                )
                if args.max_steps > 0 and processed >= args.max_steps:
                    break
    except KeyboardInterrupt:
        return 130
    finally:
        if log_fp is not None:
            log_fp.close()
        agent.destroy()
    if args.episode_result:
        deadline = time.monotonic() + max(0.0, args.score_wait_timeout)
        while time.monotonic() <= deadline:
            if args.episode_result.is_file():
                current_mtime = args.episode_result.stat().st_mtime_ns
                if initial_result_mtime is None or current_mtime != initial_result_mtime:
                    break
            time.sleep(0.1)
        else:
            raise TimeoutError(f"runtime did not write a fresh episode result: {args.episode_result}")
        from UniAD_GS_CARLA.score_episode import score_episode

        facts = json.loads(args.episode_result.read_text(encoding="utf-8"))
        score = score_episode(facts, args.collision_penalty)
        score_output = args.score_output or args.episode_result.with_name("uniad_score.json")
        score_output.parent.mkdir(parents=True, exist_ok=True)
        score_output.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
        print(
            f"[uniad] score success={score['success']} completion={score['route_completion_percent']:.2f} "
            f"driving_score={score['driving_score']:.2f}; wrote {score_output}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
