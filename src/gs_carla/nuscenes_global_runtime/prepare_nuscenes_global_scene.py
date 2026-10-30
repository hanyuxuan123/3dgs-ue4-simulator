#!/usr/bin/env python3
"""Prepare a global nuScenes-style scene package for the CARLA/3DGS runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=48)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--processed-origin-frame", type=int, default=0)
    parser.add_argument("--sequence-origin-frame", type=int, default=0)
    parser.add_argument("--export-background", action="store_true")
    parser.add_argument("--force-export-background", action="store_true")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for checkpoint export.")
    parser.add_argument("--ply", action="store_true", help="Also export a small background preview PLY.")
    parser.add_argument("--ply-max-points", type=int, default=500_000)
    parser.add_argument(
        "--core-yaw-source",
        choices=("trajectory", "xodr-waypoint", "lidar-pose"),
        default="trajectory",
        help="Yaw written to core_pose_global.csv for the CARLA hero body.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def matrix_to_rpy(mat: np.ndarray) -> tuple[float, float, float]:
    r = mat[:3, :3]
    sy = -float(r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
    return roll, pitch, yaw


def frame_pose_path(processed_scene: Path, frame: int) -> Path:
    return processed_scene / "lidar_pose" / f"{frame:03d}.txt"


def available_frames(processed_scene: Path, start: int, end: int) -> list[int]:
    frames = []
    for frame in range(start, end + 1):
        if frame_pose_path(processed_scene, frame).is_file():
            frames.append(frame)
    if len(frames) < 2:
        raise RuntimeError(f"Need at least two lidar_pose frames in [{start}, {end}] from {processed_scene}")
    return frames


def write_mapping_pose(path: Path, processed_scene: Path, frames: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        fp.write("lidar_frame x y z roll pitch yaw\n")
        for frame in frames:
            mat = load_matrix(frame_pose_path(processed_scene, frame))
            roll, pitch, yaw = matrix_to_rpy(mat)
            x, y, z = mat[:3, 3]
            fp.write(
                f"{frame} {x:.16e} {y:.16e} {z:.16e} "
                f"{roll:.16e} {pitch:.16e} {yaw:.16e}\n"
            )


def load_xodr_map(path: Path):
    import carla

    return carla, carla.Map(path.stem, path.read_text(encoding="utf-8"))


def xodr_waypoint_yaw_rad(carla_module, carla_map, x: float, y: float, z: float, fallback_yaw: float) -> float:
    waypoint = carla_map.get_waypoint(
        carla_module.Location(x=x, y=y, z=z),
        project_to_road=True,
        lane_type=carla_module.LaneType.Driving,
    )
    if waypoint is None:
        return fallback_yaw
    return math.radians(float(waypoint.transform.rotation.yaw))


def trajectory_yaws(processed_scene: Path, frames: list[int]) -> dict[int, float]:
    positions = {}
    for frame in frames:
        mat = load_matrix(frame_pose_path(processed_scene, frame))
        x, y, _z = mat[:3, 3]
        positions[frame] = (float(x), -float(y))

    yaws: dict[int, float] = {}
    for index, frame in enumerate(frames):
        if index == 0:
            left, right = frames[index], frames[index + 1]
        elif index == len(frames) - 1:
            left, right = frames[index - 1], frames[index]
        else:
            left, right = frames[index - 1], frames[index + 1]
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        yaws[frame] = math.atan2(y1 - y0, x1 - x0)
    return yaws


def write_core_pose(path: Path, processed_scene: Path, frames: list[int], args: argparse.Namespace) -> None:
    carla_module = None
    carla_map = None
    if args.core_yaw_source == "xodr-waypoint":
        carla_module, carla_map = load_xodr_map(args.xodr)
    path_yaws = trajectory_yaws(processed_scene, frames) if args.core_yaw_source == "trajectory" else {}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["frame", "x", "y", "z", "yaw"])
        writer.writeheader()
        for frame in frames:
            mat = load_matrix(frame_pose_path(processed_scene, frame))
            _roll, _pitch, yaw = matrix_to_rpy(mat)
            x, y, z = mat[:3, 3]
            # CARLA flips OpenDRIVE/map Y during import. Store CARLA-world poses
            # here, while mapping_pose.txt remains in the original global frame.
            carla_x = float(x)
            carla_y = -float(y)
            carla_yaw = -float(yaw)
            if args.core_yaw_source == "trajectory":
                carla_yaw = path_yaws[frame]
            if carla_map is not None:
                carla_yaw = xodr_waypoint_yaw_rad(carla_module, carla_map, carla_x, carla_y, float(z), carla_yaw)
            writer.writerow(
                {
                    "frame": frame,
                    "x": f"{carla_x:.16e}",
                    "y": f"{carla_y:.16e}",
                    "z": f"{z:.16e}",
                    "yaw": f"{carla_yaw:.16e}",
                }
            )


def export_background(args: argparse.Namespace, background_dir: Path) -> None:
    background_pth = background_dir / "background_gaussians.pth"
    sky_pth = background_dir / "sky_envlight.pth"
    if background_pth.exists() and sky_pth.exists() and not args.force_export_background:
        print(f"[prepare] background already exists: {background_dir}")
        return
    cmd = [
        args.python,
        "-m", "gs_carla.export_drivestudio_background",
        "--checkpoint",
        str(args.checkpoint),
        "--processed-scene",
        str(args.processed_scene),
        "--output-dir",
        str(background_dir),
    ]
    if args.config:
        cmd += ["--config", str(args.config)]
    if args.ply:
        cmd += ["--ply", str(background_dir / "background_preview.ply"), "--ply-max-points", str(args.ply_max_points)]
    print("[prepare] exporting DriveStudio background")
    subprocess.run(cmd, check=True)


def write_scene_package(
    path: Path,
    args: argparse.Namespace,
    mapping_pose: Path,
    core_pose: Path,
    background_dir: Path,
    frames: list[int],
) -> None:
    package = {
        "metadata": {
            "name": "nuscenes_global_3dgs_carla",
            "coordinate_frame": "nuscenes_global",
            "note": (
                "XODR, lidar_pose, core pose, and DriveStudio checkpoint are expected "
                "to use the same global XY frame. core_pose_coordinate_frame is carla "
                "so the runtime does not apply sequence-local origin conversion."
            ),
        },
        "assets": {
            "xodr": str(args.xodr.resolve()),
            "background": str((background_dir / "background_gaussians.pth").resolve()),
            "sky": str((background_dir / "sky_envlight.pth").resolve()),
            "processed_scene": str(args.processed_scene.resolve()),
            "mapping_pose": str(mapping_pose.resolve()),
            "core_pose_csv": str(core_pose.resolve()),
            "sequence_origin_pose_file": str(mapping_pose.resolve()),
        },
        "frames": {
            "core_start_frame": int(frames[0]),
            "core_end_frame": int(frames[-1]),
            "processed_origin_frame": int(args.processed_origin_frame),
            "processed_origin_mapping_frame": int(args.processed_origin_frame),
            "sequence_origin_frame": int(args.sequence_origin_frame),
            "instance_origin_sequence_frame": int(frames[0]),
        },
        "values": {
            "camera": int(args.camera),
            "core_pose_coordinate_frame": "carla",
            "carla_pose_frame": "opendrive-global",
            "gaussian_coordinate_frame": "mapping-absolute",
            "ego_frame_note": "lidar_pose +Y is vehicle-forward; CARLA actor +X is vehicle-forward.",
            "actor_camera_mount_frame": "trajectory-body",
            "instance_transform_mode": "mapping-absolute",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.xodr.is_file():
        raise FileNotFoundError(args.xodr)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.processed_scene.is_dir():
        raise FileNotFoundError(args.processed_scene)

    frames = available_frames(args.processed_scene, args.start_frame, args.end_frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_pose = args.output_dir / "mapping_pose.txt"
    core_pose = args.output_dir / "core_pose_global.csv"
    background_dir = args.output_dir / "background"
    scene_package = args.output_dir / "scene_package.json"

    write_mapping_pose(mapping_pose, args.processed_scene, frames)
    write_core_pose(core_pose, args.processed_scene, frames, args)
    if args.export_background:
        export_background(args, background_dir)
    write_scene_package(scene_package, args, mapping_pose, core_pose, background_dir, frames)

    print(f"[prepare] frames={frames[0]}..{frames[-1]} count={len(frames)}")
    print(f"[prepare] wrote {mapping_pose}")
    print(f"[prepare] wrote {core_pose}")
    print(f"[prepare] wrote {scene_package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
