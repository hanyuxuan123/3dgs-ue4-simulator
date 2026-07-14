#!/usr/bin/env python3
"""Build traffic bbox records and 2D projections from a 3DGS CARLA scene package."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import carla_xodr_live_3dgs_bridge_with_instances as base
from load_xodr_town_with_instances import (
    carla_pose_from_instance_matrix,
    load_instance_tracks,
    make_instance_transform,
    make_local_se2_correction,
    spawn_instance_actors,
    update_instance_actors,
    write_instance_csv,
)
from render_aligned_mapping_or_carla_path_gsplat import mapping_pose_matrix, read_mapping_poses, transform_from_xyz_rpy
from render_background_gsplat import load_intrinsics_from_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--camera-path-json", type=Path, help="Use runtime camera path JSON for render-index/step aligned projections.")
    parser.add_argument("--live-carla", action="store_true", help="Replay traffic actors in a running CARLA world instead of projecting 2D boxes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--tick-world", action="store_true", help="Tick CARLA while replaying. Do not use when another synchronous runtime is ticking.")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--instance-class-prefix", default="vehicle.")
    parser.add_argument("--max-instance-actors", type=int, default=0)
    parser.add_argument("--instance-output", type=Path)
    parser.add_argument("--instance-z-offset", type=float, default=0.35)
    parser.add_argument("--no-instance-map-z", action="store_true")
    parser.add_argument("--instance-local-offset-x", type=float, default=0.0)
    parser.add_argument("--instance-local-offset-y", type=float, default=0.0)
    parser.add_argument("--instance-local-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--hidden-actor-z", type=float, default=-500.0)
    parser.add_argument("--draw-traffic-bboxes", action="store_true")
    parser.add_argument("--traffic-bbox-life-time", type=float, default=0.08)
    parser.add_argument("--bbox-length", type=float, default=4.6)
    parser.add_argument("--bbox-width", type=float, default=1.9)
    parser.add_argument("--bbox-height", type=float, default=1.6)
    parser.add_argument("--downscale", type=float, default=3.0)
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
            "instances_info": package_path(cli.scene_package, assets.get("instances_info")),
            "processed_scene": package_path(cli.scene_package, assets.get("processed_scene")),
            "mapping_pose": package_path(cli.scene_package, assets.get("mapping_pose")),
            "instance_origin_sequence_frame": frames.get("instance_origin_sequence_frame", 300),
            "processed_origin_frame": frames.get("processed_origin_frame", 0),
            "processed_origin_mapping_frame": frames.get("processed_origin_mapping_frame"),
            "sequence_origin_frame": frames.get("sequence_origin_frame", 0),
            "camera": values.get("camera", 0),
            "instance_transform_mode": values.get("instance_transform_mode", "mapping-absolute"),
        }
    )
    return SimpleNamespace(**data)


def bbox_corners(length: float, width: float, height: float) -> np.ndarray:
    xs = [-length / 2.0, length / 2.0]
    ys = [-width / 2.0, width / 2.0]
    zs = [0.0, height]
    corners = [[x, y, z, 1.0] for x in xs for y in ys for z in zs]
    return np.asarray(corners, dtype=np.float32).T


def project_points(k: np.ndarray, world_to_cam: np.ndarray, points_world: np.ndarray, width: int, height: int) -> dict[str, Any]:
    cam = world_to_cam @ points_world
    z = cam[2]
    valid = z > 1e-4
    if not np.any(valid):
        return {"visible": False, "points": [], "bbox_2d": None}
    uvw = k @ cam[:3, valid]
    uv = uvw[:2] / uvw[2:3]
    points = [{"u": float(u), "v": float(v)} for u, v in uv.T]
    inside = (uv[0] >= 0) & (uv[0] < width) & (uv[1] >= 0) & (uv[1] < height)
    bbox = {
        "xmin": float(np.min(uv[0])),
        "ymin": float(np.min(uv[1])),
        "xmax": float(np.max(uv[0])),
        "ymax": float(np.max(uv[1])),
    }
    return {"visible": bool(np.any(inside)), "points": points, "bbox_2d": bbox}


def load_camera_samples(path: Path | None, start: int, end: int, control_dt: float) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    samples = []
    for item in payload.get("frames", []):
        sequence_frame = int(item["frame"])
        if not start <= sequence_frame <= end:
            continue
        step = int(item.get("step", item.get("render_index", 0)))
        sample = {
            "render_index": int(item.get("render_index", len(samples))),
            "step": step,
            "elapsed_time": float(item.get("elapsed_time", step * control_dt)),
            "sequence_frame": sequence_frame,
            "camera_to_world": np.asarray(item["camera_to_3dgs_world"], dtype=np.float32),
            "camera_world": "3dgs",
        }
        samples.append(sample)
    return samples


def traffic_bbox_records(instance_actors: list[dict[str, Any]], sequence_frame: int, step: int, hidden_actor_z: float) -> list[dict[str, Any]]:
    objects = []
    for entry in instance_actors:
        actor = entry["actor"]
        transform = actor.get_transform()
        if transform.location.z <= hidden_actor_z + 10.0:
            continue
        bbox = actor.bounding_box
        center = transform.transform(bbox.location)
        extent = bbox.extent
        objects.append(
            {
                "step": step,
                "sequence_frame": sequence_frame,
                "track_id": entry["track"]["id"],
                "class_name": entry["track"]["class_name"],
                "actor_id": int(actor.id),
                "blueprint_id": entry.get("blueprint_id", ""),
                "center": {"x": float(center.x), "y": float(center.y), "z": float(center.z)},
                "extent": {"x": float(extent.x), "y": float(extent.y), "z": float(extent.z)},
                "yaw_deg": float(transform.rotation.yaw),
            }
        )
    return objects


def draw_traffic_bboxes(carla, world, objects: list[dict[str, Any]], life_time: float) -> None:
    color = carla.Color(255, 64, 64, 255)
    for item in objects:
        center = item["center"]
        extent = item["extent"]
        box = carla.BoundingBox(
            carla.Location(center["x"], center["y"], center["z"]),
            carla.Vector3D(extent["x"], extent["y"], extent["z"]),
        )
        world.debug.draw_box(box, carla.Rotation(yaw=item["yaw_deg"]), thickness=0.06, color=color, life_time=life_time)


def write_traffic_bbox_json(path: Path, frames: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "coordinate_frame": "carla",
                "bbox_convention": "center + half extent; actor bounding box transformed by CARLA actor transform",
                "frames": frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_instance_transform(args: argparse.Namespace, mapping_poses: dict[int, np.ndarray]) -> np.ndarray:
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)
    t_instance_anchor_abs = mapping_pose_matrix(mapping_poses, args.instance_origin_sequence_frame)
    t_abs_cam_origin = np.loadtxt(args.processed_scene / "extrinsics" / f"{args.processed_origin_frame:03d}_{args.camera}.txt").reshape(4, 4).astype(np.float32)
    t_seq_from_instance = make_instance_transform(
        args.instance_transform_mode,
        t_abs_from_sequence,
        t_instance_anchor_abs,
        t_abs_cam_origin,
    )
    return make_local_se2_correction(
        args.instance_local_offset_x,
        args.instance_local_offset_y,
        args.instance_local_yaw_offset_deg,
    ) @ t_seq_from_instance


def live_carla_replay(args: argparse.Namespace, tracks: list[dict[str, Any]], all_sequence_frames: list[int]) -> int:
    start = args.frame_start if args.frame_start is not None else all_sequence_frames[0]
    end = args.frame_end if args.frame_end is not None else all_sequence_frames[-1]
    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_seq_from_instance = build_instance_transform(args, mapping_poses)
    carla = base.import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    carla_map = world.get_map()
    instance_actors = spawn_instance_actors(carla, world, carla_map, tracks, t_seq_from_instance, args)
    instance_rows: list[dict[str, Any]] = []
    bbox_frames: list[dict[str, Any]] = []
    print(f"[traffic-live] spawned actors={len(instance_actors)} frames={start}..{end}")
    try:
        for step, sequence_frame in enumerate(range(start, end + 1)):
            rows = update_instance_actors(carla, carla_map, instance_actors, sequence_frame, t_seq_from_instance, args)
            instance_rows.extend(rows)
            objects = traffic_bbox_records(instance_actors, sequence_frame, step, args.hidden_actor_z)
            if objects:
                bbox_frames.append({"step": step, "elapsed_time": step * args.control_dt, "sequence_frame": sequence_frame, "objects": objects})
                if args.draw_traffic_bboxes:
                    draw_traffic_bboxes(carla, world, objects, args.traffic_bbox_life_time)
            if args.tick_world:
                world.tick()
            if args.real_time and args.control_dt > 0:
                time.sleep(args.control_dt)
    except KeyboardInterrupt:
        print("[traffic-live] interrupted; writing outputs")
    if args.instance_output:
        write_instance_csv(args.instance_output, instance_rows)
        print(f"Wrote {args.instance_output}")
    write_traffic_bbox_json(args.output, bbox_frames)
    print(f"Wrote {args.output}")
    print(f"frames={len(bbox_frames)} instance_rows={len(instance_rows)} bbox_objects={sum(len(frame['objects']) for frame in bbox_frames)}")
    return 0


def pose_to_matrix_sequence(pose: dict[str, float]) -> np.ndarray:
    seq_x = float(pose["x"])
    seq_y = -float(pose["y"])
    seq_z = float(pose.get("z", 0.0))
    seq_yaw = math.radians(-float(pose["yaw"]))
    return transform_from_xyz_rpy(seq_x, seq_y, seq_z, 0.0, 0.0, seq_yaw)


def main() -> int:
    args = load_config(parse_args())
    tracks = load_instance_tracks(args.instances_info, args.instance_class_prefix, args.max_instance_actors)
    all_sequence_frames = sorted(
        {
            int(instance_frame) + int(args.instance_origin_sequence_frame)
            for track in tracks
            for instance_frame in track["frames"].keys()
        }
    )
    if args.live_carla:
        return live_carla_replay(args, tracks, all_sequence_frames)

    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)
    t_instance_anchor_abs = mapping_pose_matrix(mapping_poses, args.instance_origin_sequence_frame)
    t_abs_cam_origin = np.loadtxt(args.processed_scene / "extrinsics" / f"{args.processed_origin_frame:03d}_{args.camera}.txt").reshape(4, 4).astype(np.float32)
    t_seq_from_instance = make_instance_transform(args.instance_transform_mode, t_abs_from_sequence, t_instance_anchor_abs, t_abs_cam_origin)
    t_abs_vehicle_origin = mapping_pose_matrix(mapping_poses, args.processed_origin_mapping_frame)
    t_camera_from_vehicle = np.linalg.inv(t_abs_vehicle_origin) @ t_abs_cam_origin
    t_3dgs_from_abs = np.linalg.inv(t_abs_cam_origin)
    k, width, height = load_intrinsics_from_processed(args.processed_scene, args.camera, args.downscale)

    start = args.frame_start if args.frame_start is not None else all_sequence_frames[0]
    end = args.frame_end if args.frame_end is not None else all_sequence_frames[-1]
    camera_samples = load_camera_samples(args.camera_path_json, start, end, args.control_dt)
    if camera_samples is None:
        camera_samples = []
        for sequence_frame in range(start, end + 1):
            t_abs_vehicle = mapping_pose_matrix(mapping_poses, sequence_frame)
            camera_samples.append(
                {
                    "render_index": len(camera_samples),
                    "step": len(camera_samples),
                    "elapsed_time": len(camera_samples) * args.control_dt,
                    "sequence_frame": sequence_frame,
                    "camera_to_world": t_abs_vehicle @ t_camera_from_vehicle,
                    "camera_world": "mapping_absolute",
                }
            )
    local_corners = bbox_corners(args.bbox_length, args.bbox_width, args.bbox_height)
    frames_out = []
    for sample in camera_samples:
        sequence_frame = int(sample["sequence_frame"])
        world_to_cam = np.linalg.inv(sample["camera_to_world"])
        instance_frame = sequence_frame - args.instance_origin_sequence_frame
        objects = []
        for track in tracks:
            mat = track["frames"].get(instance_frame)
            if mat is None:
                continue
            pose = carla_pose_from_instance_matrix(t_seq_from_instance, mat)
            t_seq_obj = pose_to_matrix_sequence(pose)
            t_abs_obj = t_abs_from_sequence @ t_seq_obj
            if sample["camera_world"] == "3dgs":
                points_world = t_3dgs_from_abs @ t_abs_obj @ local_corners
            else:
                points_world = t_abs_obj @ local_corners
            projection = project_points(k, world_to_cam, points_world, width, height)
            center = t_abs_obj[:3, 3]
            objects.append(
                {
                    "track_id": track["id"],
                    "class_name": track["class_name"],
                    "sequence_frame": sequence_frame,
                    "instance_frame": instance_frame,
                    "carla_pose": pose,
                    "abs_center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
                    "extent": {"x": args.bbox_length / 2.0, "y": args.bbox_width / 2.0, "z": args.bbox_height / 2.0},
                    "projection": projection,
                }
            )
        frames_out.append(
            {
                "render_index": int(sample["render_index"]),
                "step": int(sample["step"]),
                "elapsed_time": float(sample["elapsed_time"]),
                "sequence_frame": sequence_frame,
                "objects": objects,
            }
        )
    output = {
        "coordinate_frame": "carla_pose + runtime_camera_projection" if args.camera_path_json else "carla_pose + mapping_absolute_projection",
        "image_size": {"width": width, "height": height},
        "camera": args.camera,
        "frame_start": start,
        "frame_end": end,
        "camera_path_json": str(args.camera_path_json) if args.camera_path_json else None,
        "time_base": {"unit": "seconds", "elapsed_time": "step * control_dt", "control_dt": args.control_dt},
        "bbox_size_source": "constant fallback dimensions; replace with per-instance box_size if needed",
        "frames": frames_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    object_count = sum(len(frame["objects"]) for frame in frames_out)
    visible_count = sum(1 for frame in frames_out for obj in frame["objects"] if obj["projection"]["visible"])
    bbox_count = sum(1 for frame in frames_out for obj in frame["objects"] if obj["projection"]["bbox_2d"] is not None)
    print(f"Wrote {args.output}")
    print(f"frames={len(frames_out)} objects={object_count} visible={visible_count} bbox_2d={bbox_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
