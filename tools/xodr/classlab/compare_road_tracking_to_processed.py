#!/usr/bin/env python3
"""Compare ClassLab road tracking poses with mapping and processed poses."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-pose", required=True, type=Path, help="Raw result/mapping/mapping_pose.txt.")
    parser.add_argument("--road-corrected-enu", required=True, type=Path, help="Road tracking corrected_pose_enu.csv.")
    parser.add_argument("--processed-scene", type=Path, help="DriveStudio processed scene directory.")
    parser.add_argument(
        "--source-start-frame",
        type=int,
        help="Raw lidar_frame corresponding to processed frame 000. Required when --processed-scene is used.",
    )
    parser.add_argument(
        "--processed-coordinate-frame",
        choices=("as_stored", "drivestudio_local"),
        default="drivestudio_local",
        help=(
            "How to compare processed lidar_pose. as_stored uses the matrix translation/yaw directly; "
            "drivestudio_local converts DriveStudio stored XY/yaw to sequence_start_local as "
            "x=-stored_x, y=-stored_y, yaw=stored_yaw-pi/2."
        ),
    )
    parser.add_argument("--origin-frame", type=int, default=0, help="Frame used as sequence_start_local origin.")
    parser.add_argument(
        "--frames",
        default="250,300,350,400",
        help="Comma-separated raw lidar_frame ids to compare.",
    )
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_mapping_poses(path: Path) -> dict[int, dict[str, float]]:
    poses: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fp:
        header = fp.readline().strip().split()
        index = {name: idx for idx, name in enumerate(header)}
        required = ["lidar_frame", "x", "y", "z", "roll", "pitch", "yaw"]
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        for line in fp:
            if not line.strip():
                continue
            parts = line.split()
            frame = int(parts[index["lidar_frame"]])
            poses[frame] = {
                "x": float(parts[index["x"]]),
                "y": float(parts[index["y"]]),
                "z": float(parts[index["z"]]),
                "roll": float(parts[index["roll"]]),
                "pitch": float(parts[index["pitch"]]),
                "yaw": float(parts[index["yaw"]]),
            }
    return poses


def read_road_corrected_enu(path: Path) -> dict[int, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = {}
        for row in csv.DictReader(fp):
            frame = int(row["frame"])
            rows[frame] = {
                "x": float(row["enu_x"]),
                "y": float(row["enu_y"]),
                "yaw": float(row["enu_yaw"]),
            }
    return rows


def xyz_rpy_to_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rz @ ry @ rx
    matrix[:3, 3] = [x, y, z]
    return matrix


def matrix_yaw(matrix: np.ndarray) -> float:
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def to_sequence_start_local_xy_yaw(x: float, y: float, yaw: float, origin: dict[str, float]) -> tuple[float, float, float]:
    dx = x - origin["x"]
    dy = y - origin["y"]
    c = math.cos(-origin["yaw"])
    s = math.sin(-origin["yaw"])
    return (
        c * dx - s * dy,
        s * dx + c * dy,
        normalize_angle(yaw - origin["yaw"]),
    )


def mapping_local_pose(poses: dict[int, dict[str, float]], frame: int, origin: dict[str, float]) -> tuple[float, float, float]:
    pose = poses[frame]
    return to_sequence_start_local_xy_yaw(pose["x"], pose["y"], pose["yaw"], origin)


def road_local_pose(rows: dict[int, dict[str, float]], frame: int, origin: dict[str, float]) -> tuple[float, float, float]:
    pose = rows[frame]
    return to_sequence_start_local_xy_yaw(pose["x"], pose["y"], pose["yaw"], origin)


def load_processed_lidar_pose(processed_scene: Path, processed_frame: int) -> np.ndarray | None:
    path = processed_scene / "lidar_pose" / f"{processed_frame:03d}.txt"
    if not path.exists():
        return None
    return np.loadtxt(path, dtype=np.float64)


def processed_pose_for_compare(matrix: np.ndarray, coordinate_frame: str) -> tuple[float, float, float]:
    x = float(matrix[0, 3])
    y = float(matrix[1, 3])
    yaw = matrix_yaw(matrix)
    if coordinate_frame == "drivestudio_local":
        return (-x, -y, normalize_angle(yaw - math.pi * 0.5))
    return (x, y, yaw)


def pose_delta(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dx, dy, math.hypot(dx, dy), normalize_angle(b[2] - a[2])


def parse_frames(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    frames = parse_frames(args.frames)
    mapping = read_mapping_poses(args.mapping_pose)
    corrected = read_road_corrected_enu(args.road_corrected_enu)
    if args.origin_frame not in mapping:
        raise ValueError(f"origin frame {args.origin_frame} is not present in {args.mapping_pose}")
    if args.processed_scene and args.source_start_frame is None:
        raise ValueError("--source-start-frame is required with --processed-scene")
    origin = mapping[args.origin_frame]

    print(
        "frame,processed_frame,"
        "mapping_x,mapping_y,mapping_yaw,"
        "road_x,road_y,road_yaw,"
        "road_minus_mapping_dx,road_minus_mapping_dy,road_minus_mapping_dxy,road_minus_mapping_dyaw,"
        "processed_x,processed_y,processed_yaw,"
        "processed_minus_mapping_dx,processed_minus_mapping_dy,processed_minus_mapping_dxy,processed_minus_mapping_dyaw"
    )
    for frame in frames:
        if frame not in mapping:
            print(f"{frame},missing_mapping")
            continue
        mapping_pose = mapping_local_pose(mapping, frame, origin)
        road_pose = road_local_pose(corrected, frame, origin) if frame in corrected else None
        processed_frame = None
        processed_pose = None
        if args.processed_scene is not None:
            processed_frame = frame - int(args.source_start_frame)
            matrix = load_processed_lidar_pose(args.processed_scene, processed_frame)
            if matrix is not None:
                processed_pose = processed_pose_for_compare(matrix, args.processed_coordinate_frame)

        if road_pose is None:
            road_pose = (float("nan"), float("nan"), float("nan"))
            road_delta = (float("nan"), float("nan"), float("nan"), float("nan"))
        else:
            road_delta = pose_delta(mapping_pose, road_pose)

        if processed_pose is None:
            processed_pose = (float("nan"), float("nan"), float("nan"))
            processed_delta = (float("nan"), float("nan"), float("nan"), float("nan"))
        else:
            processed_delta = pose_delta(mapping_pose, processed_pose)

        values: Sequence[object] = (
            frame,
            "" if processed_frame is None else processed_frame,
            *mapping_pose,
            *road_pose,
            *road_delta,
            *processed_pose,
            *processed_delta,
        )
        print(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
