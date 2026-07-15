#!/usr/bin/env python3
"""Load a ClassLab XODR town, drive a hero, and replay 3DGS instance actors.

This is a test harness for checking whether objects from
instances/instances_info.json can be mapped into the same CARLA local town used
by the hero closed-loop path follower.

Default coordinate chain for instances:

  T_instance_object
    -> T_seq_object = T_instance_object
    -> CARLA: (x, y, yaw) = (seq_x, -seq_y, -seq_yaw)

The instance frame index is treated as an offset from
--instance-origin-sequence-frame.  For example, frame_idx=0 maps to sequence
frame 300 when --instance-origin-sequence-frame=300.

If the actor cloud has the right relative layout but is globally shifted or
rotated from the XODR road, use the local SE(2) correction arguments to solve
that remaining map-frame alignment without touching per-object poses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .load_xodr_town_and_dump_poses import (
    clamp,
    import_carla,
    load_xodr_world,
    lookahead_path_index,
    make_path_follow_control,
    move_spectator,
    nearest_path_index,
    normalize_angle,
    path_length,
    pose_dict_to_map_aligned_transform,
    pose_dict_to_transform,
    read_core_pose_sequence,
    spawn_hero,
    transform_to_dict,
    write_pose_csv,
    write_pose_json,
)


Point = tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--vertex-distance", type=float, default=2.0)
    parser.add_argument("--max-road-length", type=float, default=500.0)
    parser.add_argument("--wall-height", type=float, default=0.0)
    parser.add_argument("--additional-width", type=float, default=0.6)
    parser.add_argument("--no-smooth-junctions", action="store_true")
    parser.add_argument("--no-mesh-visibility", action="store_true")

    parser.add_argument("--instances-info", required=True, type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--mapping-pose", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--processed-origin-frame", type=int, default=0)
    parser.add_argument("--processed-origin-mapping-frame", type=int, required=True)
    parser.add_argument("--sequence-origin-frame", type=int, default=0)
    parser.add_argument(
        "--instance-origin-sequence-frame",
        type=int,
        default=300,
        help="Sequence frame corresponding to instance frame_idx=0.",
    )
    parser.add_argument("--instance-class-prefix", default="vehicle.", help="Only replay classes with this prefix.")
    parser.add_argument("--max-instance-actors", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--instance-z-offset", type=float, default=0.35)
    parser.add_argument("--no-instance-map-z", action="store_true")
    parser.add_argument("--hidden-actor-z", type=float, default=-1000.0)
    parser.add_argument(
        "--instance-transform-mode",
        choices=("raw-local", "mapping-absolute", "sequence-anchor", "processed-camera-origin"),
        default="raw-local",
        help=(
            "raw-local treats instances_info obj_to_world as already in sequence/processed local world. "
            "mapping-absolute treats it as the same absolute mapping frame used by mapping_pose.txt. "
            "sequence-anchor treats it as local to --instance-origin-sequence-frame. "
            "processed-camera-origin uses the processed camera extrinsic origin conversion."
        ),
    )
    parser.add_argument(
        "--instance-local-offset-x",
        type=float,
        default=0.0,
        help="Extra x translation in sequence-local meters applied after --instance-transform-mode.",
    )
    parser.add_argument(
        "--instance-local-offset-y",
        type=float,
        default=0.0,
        help="Extra y translation in sequence-local meters applied after --instance-transform-mode before CARLA y flip.",
    )
    parser.add_argument(
        "--instance-local-yaw-offset-deg",
        type=float,
        default=0.0,
        help="Extra yaw rotation in sequence-local degrees applied after --instance-transform-mode.",
    )

    parser.add_argument("--core-pose-csv", required=True, type=Path)
    parser.add_argument("--core-start-frame", type=int, required=True)
    parser.add_argument("--core-end-frame", type=int, required=True)
    parser.add_argument("--core-pose-coordinate-frame", choices=("corrected_local", "enu", "carla"), default="enu")
    parser.add_argument("--sequence-origin-pose-file", type=Path)
    parser.add_argument("--no-core-y-flip", action="store_true")
    parser.add_argument("--core-pose-z", type=float, default=0.5)

    parser.add_argument("--hero-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--hero-role-name", default="hero")
    parser.add_argument("--follow-speed-kmh", type=float, default=3.0)
    parser.add_argument("--follow-lookahead-distance", type=float, default=10.0)
    parser.add_argument("--follow-finish-distance", type=float, default=3.0)
    parser.add_argument("--follow-max-seconds", type=float, default=0.0)
    parser.add_argument("--follow-start-z-offset", type=float, default=0.35)
    parser.add_argument("--no-follow-map-z", action="store_true")
    parser.add_argument("--follow-settle-ticks", type=int, default=30)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--speed-kp", type=float, default=0.35)
    parser.add_argument("--speed-ki", type=float, default=0.02)
    parser.add_argument("--speed-kd", type=float, default=0.02)
    parser.add_argument("--steer-gain", type=float, default=0.35)
    parser.add_argument("--max-steer", type=float, default=0.30)
    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument("--spectator-height", type=float, default=55.0)
    parser.add_argument("--spectator-distance", type=float, default=35.0)

    parser.add_argument("--pose-output", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--instance-output", type=Path)
    parser.add_argument("--debug-instance-conversion", type=int, default=5)
    return parser.parse_args()


def transform_from_xyz_rpy(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rz @ ry @ rx
    out[:3, 3] = [x, y, z]
    return out


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


def mapping_pose_matrix(poses: dict[int, dict[str, float]], frame: int) -> np.ndarray:
    if frame not in poses:
        raise ValueError(f"mapping frame {frame} is missing")
    p = poses[frame]
    return transform_from_xyz_rpy(p["x"], p["y"], p["z"], p["roll"], p["pitch"], p["yaw"])


def load_processed_extrinsic(scene: Path, processed_frame: int, camera: int) -> np.ndarray:
    return np.loadtxt(scene / "extrinsics" / f"{processed_frame:03d}_{camera}.txt").reshape(4, 4).astype(np.float32)


def yaw_from_matrix(mat: np.ndarray) -> float:
    return math.atan2(float(mat[1, 0]), float(mat[0, 0]))


def carla_pose_from_instance_matrix(t_seq_from_instance: np.ndarray, obj_to_world: list[list[float]]) -> dict[str, float]:
    t_instance_obj = np.asarray(obj_to_world, dtype=np.float32).reshape(4, 4)
    t_seq_obj = t_seq_from_instance @ t_instance_obj
    seq_yaw = yaw_from_matrix(t_seq_obj)
    return {
        "x": float(t_seq_obj[0, 3]),
        "y": float(-t_seq_obj[1, 3]),
        "z": float(t_seq_obj[2, 3]),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": math.degrees(-seq_yaw),
    }


def load_instance_tracks(path: Path, class_prefix: str, max_actors: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tracks: list[dict[str, Any]] = []
    for _, item in sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
        class_name = str(item.get("class_name", ""))
        if class_prefix and not class_name.startswith(class_prefix):
            continue
        ann = item.get("frame_annotations", {})
        frames = ann.get("frame_idx", [])
        mats = ann.get("obj_to_world", [])
        if not frames or not mats:
            continue
        frame_to_mat = {int(frame): mat for frame, mat in zip(frames, mats)}
        tracks.append(
            {
                "id": str(item.get("id", f"instance_{len(tracks)}")),
                "class_name": class_name,
                "frames": frame_to_mat,
            }
        )
        if max_actors > 0 and len(tracks) >= max_actors:
            break
    return tracks


def stable_index(key: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % modulo


def find_blueprint_candidates(blueprint_library, names: list[str]):
    candidates = []
    for name in names:
        try:
            candidates.append(blueprint_library.find(name))
        except Exception:
            continue
    return candidates


def blueprint_for_class(carla, blueprint_library, class_name: str, track_id: str, hero_blueprint: str):
    if "truck" in class_name:
        names = [
            "vehicle.carlamotors.carlacola",
            "vehicle.tesla.cybertruck",
            "vehicle.carlamotors.firetruck",
        ]
    elif "bus" in class_name:
        names = ["vehicle.mitsubishi.fusorosa"]
    elif "bicycle" in class_name:
        names = ["vehicle.diamondback.century"]
    elif "motorcycle" in class_name or "motorbike" in class_name:
        names = ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.harley-davidson.low_rider"]
    else:
        names = [
            "vehicle.audi.a2",
            "vehicle.audi.etron",
            "vehicle.audi.tt",
            "vehicle.bmw.grandtourer",
            "vehicle.chevrolet.impala",
            "vehicle.dodge.charger_2020",
            "vehicle.ford.mustang",
            "vehicle.jeep.wrangler_rubicon",
            "vehicle.lincoln.mkz_2017",
            "vehicle.mercedes.coupe",
            "vehicle.mini.cooper_s",
            "vehicle.nissan.patrol",
            "vehicle.seat.leon",
            "vehicle.toyota.prius",
            "vehicle.volkswagen.t2",
            "vehicle.tesla.cybertruck",
            "vehicle.carlamotors.carlacola",
        ]

    candidates = find_blueprint_candidates(blueprint_library, names)
    non_hero_candidates = [bp for bp in candidates if getattr(bp, "id", "") != hero_blueprint]
    if non_hero_candidates:
        candidates = non_hero_candidates
    if not candidates:
        candidates = [bp for bp in blueprint_library.filter("vehicle.*") if getattr(bp, "id", "") != hero_blueprint]
    if not candidates:
        candidates = list(blueprint_library.filter("vehicle.*"))
    if not candidates:
        raise RuntimeError("No CARLA vehicle blueprint is available for instance actors")
    return candidates[stable_index(f"{track_id}:{class_name}", len(candidates))]


def align_instance_transform(carla, carla_map, pose: dict[str, float], args: argparse.Namespace):
    transform = carla.Transform(
        carla.Location(x=pose["x"], y=pose["y"], z=pose["z"]),
        carla.Rotation(roll=0.0, pitch=0.0, yaw=pose["yaw"]),
    )
    if args.no_instance_map_z:
        transform.location.z += args.instance_z_offset
        return transform
    waypoint = carla_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        transform.location.z += args.instance_z_offset
        return transform
    transform.location.z = waypoint.transform.location.z + args.instance_z_offset
    transform.rotation.roll = waypoint.transform.rotation.roll
    transform.rotation.pitch = waypoint.transform.rotation.pitch
    return transform


def spawn_instance_actors(carla, world, carla_map, tracks: list[dict[str, Any]], t_seq_from_instance: np.ndarray, args: argparse.Namespace):
    blueprint_library = world.get_blueprint_library()
    actors = []
    for track in tracks:
        first_frame = min(track["frames"])
        pose = carla_pose_from_instance_matrix(t_seq_from_instance, track["frames"][first_frame])
        transform = align_instance_transform(carla, carla_map, pose, args)
        blueprint = blueprint_for_class(carla, blueprint_library, track["class_name"], track["id"], args.hero_blueprint)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", f"instance_{track['id']}")
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", colors[stable_index(track["id"], len(colors))])
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            hidden = carla.Transform(carla.Location(x=transform.location.x, y=transform.location.y, z=args.hidden_actor_z), transform.rotation)
            actor = world.try_spawn_actor(blueprint, hidden)
        if actor is None:
            print(f"[instances] skip spawn id={track['id']} class={track['class_name']}")
            continue
        actor.set_simulate_physics(False)
        actors.append({"track": track, "actor": actor, "blueprint_id": getattr(blueprint, "id", "")})
        print(
            f"[instances] spawned id={track['id']} class={track['class_name']} "
            f"blueprint={getattr(blueprint, 'id', '')} actor_id={actor.id}"
        )
    return actors


def hide_actor(carla, actor, args: argparse.Namespace) -> None:
    transform = actor.get_transform()
    transform.location.z = args.hidden_actor_z
    actor.set_transform(transform)


def update_instance_actors(
    carla,
    carla_map,
    instance_actors: list[dict[str, Any]],
    sequence_frame: int,
    t_seq_from_instance: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    instance_frame = sequence_frame - args.instance_origin_sequence_frame
    rows: list[dict[str, Any]] = []
    for entry in instance_actors:
        track = entry["track"]
        actor = entry["actor"]
        blueprint_id = entry.get("blueprint_id", "")
        mat = track["frames"].get(instance_frame)
        if mat is None:
            hide_actor(carla, actor, args)
            continue
        pose = carla_pose_from_instance_matrix(t_seq_from_instance, mat)
        transform = align_instance_transform(carla, carla_map, pose, args)
        actor.set_transform(transform)
        rows.append(
            {
                "sequence_frame": sequence_frame,
                "instance_frame": instance_frame,
                "track_id": track["id"],
                "class_name": track["class_name"],
                "blueprint_id": blueprint_id,
                "actor_id": int(actor.id),
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
                "yaw": float(transform.rotation.yaw),
            }
        )
    return rows


def write_instance_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sequence_frame",
        "instance_frame",
        "track_id",
        "class_name",
        "blueprint_id",
        "actor_id",
        "x",
        "y",
        "z",
        "yaw",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_instance_transform(
    mode: str,
    t_abs_from_sequence: np.ndarray,
    t_instance_anchor_abs: np.ndarray,
    t_abs_cam_origin: np.ndarray,
) -> np.ndarray:
    if mode == "raw-local":
        return np.eye(4, dtype=np.float32)
    if mode == "mapping-absolute":
        return np.linalg.inv(t_abs_from_sequence)
    if mode == "sequence-anchor":
        return np.linalg.inv(t_abs_from_sequence) @ t_instance_anchor_abs
    if mode == "processed-camera-origin":
        return np.linalg.inv(t_abs_from_sequence) @ t_abs_cam_origin
    raise ValueError(f"unsupported instance transform mode: {mode}")


def make_local_se2_correction(offset_x: float, offset_y: float, yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    correction = np.eye(4, dtype=np.float32)
    correction[0, 0] = math.cos(yaw)
    correction[0, 1] = -math.sin(yaw)
    correction[1, 0] = math.sin(yaw)
    correction[1, 1] = math.cos(yaw)
    correction[0, 3] = offset_x
    correction[1, 3] = offset_y
    return correction


def print_instance_conversion_debug(
    tracks: list[dict[str, Any]],
    t_abs_from_sequence: np.ndarray,
    t_instance_anchor_abs: np.ndarray,
    t_abs_cam_origin: np.ndarray,
    t_local_correction: np.ndarray,
    count: int,
) -> None:
    if count <= 0:
        return
    modes = ("raw-local", "mapping-absolute", "sequence-anchor", "processed-camera-origin")
    transforms = {
        mode: make_instance_transform(mode, t_abs_from_sequence, t_instance_anchor_abs, t_abs_cam_origin)
        for mode in modes
    }
    print("[instances-debug] first instance coordinate conversions:")
    for track in tracks[:count]:
        first_frame = min(track["frames"])
        mat = track["frames"][first_frame]
        parts = [f"id={track['id']} class={track['class_name']} instance_frame={first_frame}"]
        for mode in modes:
            pose = carla_pose_from_instance_matrix(transforms[mode], mat)
            parts.append(f"{mode}:x={pose['x']:.2f},y={pose['y']:.2f},yaw={pose['yaw']:.1f}")
        corrected_pose = carla_pose_from_instance_matrix(t_local_correction @ transforms["mapping-absolute"], mat)
        parts.append(
            f"mapping-absolute+local:x={corrected_pose['x']:.2f},"
            f"y={corrected_pose['y']:.2f},yaw={corrected_pose['yaw']:.1f}"
        )
        print("[instances-debug] " + " | ".join(parts))


def main() -> int:
    args = parse_args()
    if args.core_pose_coordinate_frame == "enu" and args.sequence_origin_pose_file is None:
        args.sequence_origin_pose_file = args.mapping_pose

    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)
    t_instance_anchor_abs = mapping_pose_matrix(mapping_poses, args.instance_origin_sequence_frame)
    t_abs_cam_origin = load_processed_extrinsic(args.processed_scene, args.processed_origin_frame, args.camera)
    t_seq_from_instance = make_instance_transform(
        args.instance_transform_mode,
        t_abs_from_sequence,
        t_instance_anchor_abs,
        t_abs_cam_origin,
    )
    t_local_correction = make_local_se2_correction(
        args.instance_local_offset_x,
        args.instance_local_offset_y,
        args.instance_local_yaw_offset_deg,
    )
    t_seq_from_instance = t_local_correction @ t_seq_from_instance

    carla = import_carla()
    client, world = load_xodr_world(carla, args)
    carla_map = world.get_map()
    core_poses = read_core_pose_sequence(args)
    if len(core_poses) < 2:
        raise RuntimeError("Need at least two core poses")

    first_transform = pose_dict_to_transform(carla, core_poses[0])
    hero = spawn_hero(carla, world, first_transform, args.hero_blueprint, args.hero_role_name)
    print(f"spawned hero actor id={hero.id}")

    tracks = load_instance_tracks(args.instances_info, args.instance_class_prefix, args.max_instance_actors)
    print(f"[instances] loaded tracks={len(tracks)} from {args.instances_info}")
    print_instance_conversion_debug(
        tracks,
        t_abs_from_sequence,
        t_instance_anchor_abs,
        t_abs_cam_origin,
        t_local_correction,
        args.debug_instance_conversion,
    )

    original_settings = world.get_settings()
    sync_settings = world.get_settings()
    sync_settings.synchronous_mode = True
    sync_settings.fixed_delta_seconds = args.control_dt
    world.apply_settings(sync_settings)

    instance_actors: list[dict[str, Any]] = []
    outputs: list[dict] = []
    instance_rows: list[dict[str, Any]] = []
    try:
        start_pose = dict(core_poses[0])
        if args.no_follow_map_z:
            start_pose["z"] = float(start_pose["z"]) + float(args.follow_start_z_offset)
            start_transform = pose_dict_to_transform(carla, start_pose)
        else:
            start_transform, _ = pose_dict_to_map_aligned_transform(
                carla, carla_map, start_pose, args.follow_start_z_offset
            )
        hero.set_simulate_physics(False)
        hero.set_transform(start_transform)
        world.tick()
        hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        hero.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        hero.set_simulate_physics(True)

        instance_actors = spawn_instance_actors(carla, world, carla_map, tracks, t_seq_from_instance, args)
        for _ in range(max(0, args.follow_settle_ticks)):
            hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            world.tick()

        target_speed = max(args.follow_speed_kmh / 3.6, 0.1)
        max_seconds = args.follow_max_seconds
        if max_seconds <= 0.0:
            max_seconds = path_length(core_poses) / target_speed + 20.0
        max_steps = max(1, int(math.ceil(max_seconds / max(args.control_dt, 1e-3))))

        nearest_index = 0
        elapsed_time = 0.0
        pid_state: dict = {}
        for step in range(max_steps):
            transform = hero.get_transform()
            nearest_index, nearest_dist = nearest_path_index(core_poses, transform.location.x, transform.location.y, nearest_index)
            source_frame = int(core_poses[nearest_index]["frame"])
            instance_rows.extend(update_instance_actors(carla, carla_map, instance_actors, source_frame, t_seq_from_instance, args))

            target_index = lookahead_path_index(core_poses, nearest_index, args.follow_lookahead_distance)
            target = core_poses[target_index]
            control, speed, _, lateral_error = make_path_follow_control(carla, args, hero, target, args.control_dt, pid_state)
            hero.apply_control(control)
            world.tick()

            actual = transform_to_dict(hero.get_transform(), source="hero_controller_with_instances")
            heading_error = normalize_angle(math.radians(float(target["yaw"])) - math.radians(float(actual["yaw"])))
            actual.update(
                {
                    "frame": source_frame,
                    "loop": 0,
                    "elapsed_time": elapsed_time,
                    "speed_mps": speed,
                    "speed_kmh": speed * 3.6,
                    "target_index": target_index,
                    "target_frame": int(target["frame"]),
                    "target_x": float(target["x"]),
                    "target_y": float(target["y"]),
                    "target_yaw": float(target["yaw"]),
                    "distance_error": nearest_dist,
                    "lateral_error": lateral_error,
                    "heading_error_deg": math.degrees(heading_error),
                    "throttle": float(control.throttle),
                    "steer": float(control.steer),
                    "brake": float(control.brake),
                    "visible_instance_count": sum(1 for row in instance_rows if row["sequence_frame"] == source_frame),
                }
            )
            outputs.append(actual)
            if args.spectator_follow:
                move_spectator(carla, world, hero.get_transform(), args.spectator_height, args.spectator_distance)
            elapsed_time += args.control_dt
            if nearest_index >= len(core_poses) - 2 and nearest_dist <= args.follow_finish_distance:
                break
        hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        world.tick()
    finally:
        world.apply_settings(original_settings)

    write_pose_csv(args.pose_output, outputs)
    if args.instance_output:
        write_instance_csv(args.instance_output, instance_rows)
    if args.json_output:
        write_pose_json(
            args.json_output,
            outputs,
            {
                "xodr": str(args.xodr),
                "instances_info": str(args.instances_info),
                "instance_origin_sequence_frame": args.instance_origin_sequence_frame,
                "instance_transform_mode": args.instance_transform_mode,
                "instance_local_offset_x": args.instance_local_offset_x,
                "instance_local_offset_y": args.instance_local_offset_y,
                "instance_local_yaw_offset_deg": args.instance_local_yaw_offset_deg,
                "instance_tracks_loaded": len(tracks),
                "instance_actors_spawned": len(instance_actors),
                "pose_count": len(outputs),
            },
        )
    print(f"loaded world: {carla_map.name}")
    print(f"hero poses: {len(outputs)} wrote {args.pose_output}")
    print(f"instance actors: loaded={len(tracks)} spawned={len(instance_actors)} rows={len(instance_rows)}")
    if args.instance_output:
        print(f"wrote {args.instance_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
