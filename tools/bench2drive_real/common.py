#!/usr/bin/env python3
"""Shared helpers for Bench2Drive-Real lightweight scenario generation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_SCENARIO_TYPES = (
    "ParkedObstacle",
    "ConstructionObstacle",
    "Accident",
    "HardBreakRoute",
    "HighwayCutIn",
    "ParkingCutIn",
    "PedestrianCrossing",
    "DynamicObjectCrossing",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def transform_from_xyz_yaw(x: float, y: float, z: float, yaw_rad: float) -> list[list[float]]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return [
        [c, -s, 0.0, float(x)],
        [s, c, 0.0, float(y)],
        [0.0, 0.0, 1.0, float(z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def yaw_from_points(x0: float, y0: float, x1: float, y1: float, fallback: float = 0.0) -> float:
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) + abs(dy) < 1e-6:
        return fallback
    return math.atan2(dy, dx)


def normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def carla_to_global_xyyaw(x: float, y: float, yaw_rad: float) -> tuple[float, float, float]:
    return float(x), -float(y), -float(yaw_rad)


def global_to_carla_xyyaw(x: float, y: float, yaw_rad: float) -> tuple[float, float, float]:
    return float(x), -float(y), -float(yaw_rad)


def offset_pose_global(
    x: float,
    y: float,
    z: float,
    yaw_rad: float,
    longitudinal: float,
    lateral: float,
    yaw_offset_rad: float = 0.0,
) -> dict[str, float]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return {
        "x": float(x + longitudinal * c - lateral * s),
        "y": float(y + longitudinal * s + lateral * c),
        "z": float(z),
        "yaw": float(normalize_angle_rad(yaw_rad + yaw_offset_rad)),
    }


def read_core_pose_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            frame = int(float(row["frame"]))
            x = float(row["x"])
            y = float(row["y"])
            z = float(row.get("z", 0.0) or 0.0)
            yaw = float(row["yaw"])
            gx, gy, gyaw = carla_to_global_xyyaw(x, y, yaw)
            rows.append(
                {
                    "frame": frame,
                    "carla_x": x,
                    "carla_y": y,
                    "carla_z": z,
                    "carla_yaw": yaw,
                    "global_x": gx,
                    "global_y": gy,
                    "global_z": z,
                    "global_yaw": gyaw,
                }
            )
    rows.sort(key=lambda item: item["frame"])
    return rows


def cumulative_route_s(rows: list[dict[str, float]], frame_key: str = "frame") -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    total = 0.0
    last = None
    for row in rows:
        item = dict(row)
        if last is not None:
            total += math.hypot(item["global_x"] - last["global_x"], item["global_y"] - last["global_y"])
        item["route_s"] = total
        out.append(item)
        last = item
    return out


def nearest_route_row_by_s(rows: list[dict[str, float]], target_s: float) -> dict[str, float]:
    return min(rows, key=lambda row: abs(float(row.get("route_s", 0.0)) - target_s))


def nearest_route_row_by_frame(rows: list[dict[str, float]], frame: int) -> dict[str, float]:
    return min(rows, key=lambda row: abs(int(row["frame"]) - int(frame)))


def build_instance_item(
    instance_id: str,
    class_name: str,
    frames: list[int],
    poses_global: list[dict[str, float]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mats = [transform_from_xyz_yaw(p["x"], p["y"], p["z"], p["yaw"]) for p in poses_global]
    payload: dict[str, Any] = {
        "id": instance_id,
        "class_name": class_name,
        "frame_annotations": {
            "frame_idx": [int(frame) for frame in frames],
            "obj_to_world": mats,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def interpolate_pose_on_route(
    route_rows: list[dict[str, float]],
    anchor_s: float,
    longitudinal: float,
    lateral: float,
    z_offset: float = 0.2,
    yaw_offset_rad: float = 0.0,
) -> dict[str, float]:
    row = nearest_route_row_by_s(route_rows, anchor_s + longitudinal)
    return offset_pose_global(
        row["global_x"],
        row["global_y"],
        row["global_z"] + z_offset,
        row["global_yaw"],
        0.0,
        lateral,
        yaw_offset_rad,
    )


def copy_scene_package_with_instances(
    scene_package: Path,
    output_scene_package: Path,
    instances_info: Path,
    instance_transform_mode: str = "mapping-absolute",
) -> None:
    package = read_json(scene_package)
    package.setdefault("assets", {})["instances_info"] = str(instances_info.resolve())
    package.setdefault("values", {})["instance_transform_mode"] = instance_transform_mode
    package.setdefault("metadata", {})["bench2drive_real"] = {
        "note": "Generated Bench2Drive-Real actor trajectories; obj_to_world is mapping/global absolute.",
        "instances_info": str(instances_info.resolve()),
    }
    write_json(output_scene_package, package)
