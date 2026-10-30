#!/usr/bin/env python3
"""Generate Bench2Drive-Real actor trajectories for CARLA and 3DGS actors."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import (
    SUPPORTED_SCENARIO_TYPES,
    build_instance_item,
    copy_scene_package_with_instances,
    cumulative_route_s,
    global_to_carla_xyyaw,
    interpolate_pose_on_route,
    nearest_route_row_by_frame,
    read_core_pose_csv,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", type=Path, help="Optional source scene package to patch with generated instances_info.")
    parser.add_argument("--core-pose-csv", type=Path, help="Required when --scene-package is omitted.")
    parser.add_argument("--templates", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-scene-package", type=Path)
    parser.add_argument("--types", nargs="+", default=list(SUPPORTED_SCENARIO_TYPES))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frames-before-trigger", type=int, default=10)
    parser.add_argument("--frames-after-trigger", type=int, default=38)
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 means one per available requested type.")
    parser.add_argument("--lane-width", type=float, default=3.5)
    return parser.parse_args()


def resolve_core_pose_csv(args: argparse.Namespace) -> Path:
    if args.core_pose_csv:
        return args.core_pose_csv
    if args.scene_package is None:
        raise ValueError("--core-pose-csv is required without --scene-package")
    package = read_json(args.scene_package)
    core = Path(package["assets"]["core_pose_csv"])
    if not core.is_absolute():
        core = args.scene_package.parent / core
    return core


def frames_for_anchor(anchor_frame: int, route_rows: list[dict[str, float]], before: int, after: int) -> list[int]:
    first = int(route_rows[0]["frame"])
    last = int(route_rows[-1]["frame"])
    start = max(first, anchor_frame - before)
    end = min(last, anchor_frame + after)
    return list(range(start, end + 1))


def pose_track_along_route(
    route_rows: list[dict[str, float]],
    frames: list[int],
    anchor_s: float,
    start_longitudinal: float,
    speed_mps: float,
    lateral: float,
    z_offset: float = 0.3,
    yaw_offset_rad: float = 0.0,
    dt: float = 0.1,
) -> list[dict[str, float]]:
    poses = []
    for idx, _frame in enumerate(frames):
        longitudinal = start_longitudinal + speed_mps * idx * dt
        poses.append(interpolate_pose_on_route(route_rows, anchor_s, longitudinal, lateral, z_offset, yaw_offset_rad))
    return poses


def static_pose_track(
    route_rows: list[dict[str, float]],
    frames: list[int],
    anchor_s: float,
    longitudinal: float,
    lateral: float,
    yaw_offset_rad: float = 0.0,
    z_offset: float = 0.3,
) -> list[dict[str, float]]:
    pose = interpolate_pose_on_route(route_rows, anchor_s, longitudinal, lateral, z_offset, yaw_offset_rad)
    return [dict(pose) for _ in frames]


def crossing_track(
    route_rows: list[dict[str, float]],
    frames: list[int],
    anchor_s: float,
    longitudinal: float,
    start_lateral: float,
    end_lateral: float,
    z_offset: float = 0.1,
) -> list[dict[str, float]]:
    poses = []
    denom = max(len(frames) - 1, 1)
    for idx, _frame in enumerate(frames):
        alpha = idx / denom
        lateral = (1.0 - alpha) * start_lateral + alpha * end_lateral
        row_pose = interpolate_pose_on_route(route_rows, anchor_s, longitudinal, lateral, z_offset, math.pi / 2.0)
        poses.append(row_pose)
    return poses


def scenario_tracks(
    scenario_type: str,
    anchor: dict[str, Any],
    route_rows: list[dict[str, float]],
    frames: list[int],
    lane_width: float,
    dt: float,
) -> list[dict[str, Any]]:
    anchor_s = float(anchor["route_s"])
    lane = float(anchor.get("map_features", {}).get("lane_width", lane_width) or lane_width)
    tracks: list[dict[str, Any]] = []

    if scenario_type == "ParkedObstacle":
        tracks.append(
            {
                "suffix": "parked_vehicle",
                "class_name": "vehicle.car",
                "behavior": "static_parked_obstacle",
                "poses": static_pose_track(route_rows, frames, anchor_s, 26.0, -0.55 * lane, 0.0),
            }
        )
    elif scenario_type == "ConstructionObstacle":
        tracks.append(
            {
                "suffix": "construction_vehicle",
                "class_name": "vehicle.truck",
                "behavior": "static_construction_obstacle",
                "poses": static_pose_track(route_rows, frames, anchor_s, 24.0, -0.35 * lane, 0.0),
            }
        )
    elif scenario_type == "Accident":
        tracks.append(
            {
                "suffix": "accident_vehicle_a",
                "class_name": "vehicle.car",
                "behavior": "static_accident",
                "poses": static_pose_track(route_rows, frames, anchor_s, 24.0, -0.25 * lane, math.radians(8.0)),
            }
        )
        tracks.append(
            {
                "suffix": "accident_vehicle_b",
                "class_name": "vehicle.car",
                "behavior": "static_accident",
                "poses": static_pose_track(route_rows, frames, anchor_s, 28.0, 0.25 * lane, math.radians(-12.0)),
            }
        )
    elif scenario_type == "HardBreakRoute":
        poses = []
        for idx, _frame in enumerate(frames):
            speed = 4.0 if idx < len(frames) * 0.45 else max(0.0, 4.0 * (1.0 - (idx - len(frames) * 0.45) / max(len(frames) * 0.25, 1.0)))
            poses.append(interpolate_pose_on_route(route_rows, anchor_s, 14.0 + speed * idx * dt, 0.0, 0.3))
        tracks.append(
            {
                "suffix": "hard_brake_lead",
                "class_name": "vehicle.car",
                "behavior": "lead_vehicle_hard_brake",
                "poses": poses,
            }
        )
    elif scenario_type == "HighwayCutIn":
        poses = []
        denom = max(len(frames) - 1, 1)
        for idx, _frame in enumerate(frames):
            alpha = min(1.0, max(0.0, (idx / denom - 0.25) / 0.45))
            lateral = (1.0 - alpha) * lane + alpha * 0.10 * lane
            poses.append(interpolate_pose_on_route(route_rows, anchor_s, 22.0 + 3.2 * idx * dt, lateral, 0.3))
        tracks.append(
            {
                "suffix": "highway_cutin_vehicle",
                "class_name": "vehicle.car",
                "behavior": "adjacent_lane_cut_in",
                "poses": poses,
            }
        )
    elif scenario_type == "ParkingCutIn":
        poses = []
        denom = max(len(frames) - 1, 1)
        for idx, _frame in enumerate(frames):
            alpha = min(1.0, max(0.0, (idx / denom - 0.15) / 0.55))
            lateral = (1.0 - alpha) * (-1.25 * lane) + alpha * (-0.05 * lane)
            poses.append(interpolate_pose_on_route(route_rows, anchor_s, 12.0 + 2.0 * idx * dt, lateral, 0.3))
        tracks.append(
            {
                "suffix": "parking_cutin_vehicle",
                "class_name": "vehicle.car",
                "behavior": "parking_cut_in",
                "poses": poses,
            }
        )
    elif scenario_type == "PedestrianCrossing":
        tracks.append(
            {
                "suffix": "pedestrian_crossing",
                "class_name": "human.pedestrian.adult",
                "behavior": "pedestrian_crossing",
                "poses": crossing_track(route_rows, frames, anchor_s, 18.0, -1.8 * lane, 1.8 * lane),
            }
        )
    elif scenario_type == "DynamicObjectCrossing":
        tracks.append(
            {
                "suffix": "bicycle_crossing",
                "class_name": "vehicle.bicycle",
                "behavior": "dynamic_object_crossing",
                "poses": crossing_track(route_rows, frames, anchor_s, 20.0, 1.8 * lane, -1.8 * lane, 0.2),
            }
        )
    return tracks


def carla_track_from_global(frames: list[int], poses: list[dict[str, float]]) -> list[dict[str, float]]:
    rows = []
    for frame, pose in zip(frames, poses):
        x, y, yaw = global_to_carla_xyyaw(pose["x"], pose["y"], pose["yaw"])
        rows.append(
            {
                "frame": int(frame),
                "x": x,
                "y": y,
                "z": float(pose["z"]),
                "yaw": math.degrees(yaw),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    core_pose_csv = resolve_core_pose_csv(args)
    route_rows = cumulative_route_s(read_core_pose_csv(core_pose_csv))
    templates = read_json(args.templates)
    anchors = read_json(args.anchors).get("anchors", [])
    template_by_type = {}
    for item in templates.get("templates", []):
        template_by_type.setdefault(item["scenario_type"], item)

    instances: dict[str, Any] = {}
    route_start_frame = int(route_rows[0]["frame"])
    frame_instances: dict[str, list[int]] = {str(int(row["frame"]) - route_start_frame): [] for row in route_rows}
    actor_trajectories: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    dt = 1.0 / max(args.fps, 1e-6)
    instance_index = 0
    selected = [a for a in anchors if a["scenario_type"] in set(args.types)]
    if args.max_scenarios > 0:
        selected = selected[: args.max_scenarios]

    for anchor in selected:
        scenario_type = anchor["scenario_type"]
        if scenario_type not in template_by_type:
            continue
        anchor_frame = int(anchor["frame"])
        frames = frames_for_anchor(anchor_frame, route_rows, args.frames_before_trigger, args.frames_after_trigger)
        tracks = scenario_tracks(scenario_type, anchor, route_rows, frames, args.lane_width, dt)
        actor_ids = []
        for track in tracks:
            instance_id = f"b2dr_{scenario_type}_{instance_index:03d}_{track['suffix']}"
            frame_idx = [frame - route_start_frame for frame in frames]
            instances[str(instance_index)] = build_instance_item(
                instance_id,
                track["class_name"],
                frame_idx,
                track["poses"],
                {
                    "bench2drive_real": {
                        "scenario_type": scenario_type,
                        "behavior": track["behavior"],
                        "anchor_id": anchor["anchor_id"],
                        "anchor_frame": anchor_frame,
                        "coordinate_frame": "mapping-absolute",
                    }
                },
            )
            for frame in frames:
                frame_instances.setdefault(str(frame - route_start_frame), []).append(instance_index)
            actor_trajectories.append(
                {
                    "track_id": instance_id,
                    "class_name": track["class_name"],
                    "scenario_type": scenario_type,
                    "behavior": track["behavior"],
                    "coordinate_frame": "carla",
                    "trajectory": carla_track_from_global(frames, track["poses"]),
                }
            )
            actor_ids.append(instance_id)
            instance_index += 1
        anchor_row = nearest_route_row_by_frame(route_rows, anchor_frame)
        scenarios.append(
            {
                "scenario_id": f"b2dr_{scenario_type}_{anchor_frame:04d}",
                "scenario_type": scenario_type,
                "template_id": template_by_type[scenario_type]["template_id"],
                "anchor": anchor,
                "trigger_frame": anchor_frame,
                "trigger_global_pose": {
                    "x": anchor_row["global_x"],
                    "y": anchor_row["global_y"],
                    "z": anchor_row["global_z"],
                    "yaw": anchor_row["global_yaw"],
                },
                "actor_track_ids": actor_ids,
            }
        )

    instances_dir = args.output_dir / "instances"
    write_json(instances_dir / "instances_info.json", instances)
    write_json(instances_dir / "frame_instances.json", frame_instances)
    write_json(args.output_dir / "actor_trajectories_carla.json", actor_trajectories)
    write_json(
        args.output_dir / "bench2drive_real_scenarios.json",
        {
            "name": "bench2drive_real_generated_scenarios",
            "core_pose_csv": str(core_pose_csv),
            "coordinate_contract": {
                "instances_info": "mapping-absolute obj_to_world; use instance_transform_mode=mapping-absolute",
                "actor_trajectories_carla": "CARLA world after OpenDRIVE y flip",
            },
            "fps": args.fps,
            "route_start_frame": route_start_frame,
            "scenarios": scenarios,
        },
    )
    if args.scene_package and args.output_scene_package:
        copy_scene_package_with_instances(
            args.scene_package,
            args.output_scene_package,
            instances_dir / "instances_info.json",
            "mapping-absolute",
        )
    print(f"Wrote {instances_dir / 'instances_info.json'}")
    print(f"Wrote {args.output_dir / 'actor_trajectories_carla.json'}")
    print(f"scenarios={len(scenarios)} actors={len(instances)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
