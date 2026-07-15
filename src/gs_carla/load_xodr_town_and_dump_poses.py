#!/usr/bin/env python3
"""Load a local OpenDRIVE town in CARLA and export waypoint/hero poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", required=True, type=Path, help="OpenDRIVE .xodr file to load.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--pose-output", required=True, type=Path, help="CSV path for exported CARLA poses.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON path with transform matrices.")
    parser.add_argument("--waypoint-spacing", type=float, default=2.0, help="Meters between exported map waypoints.")
    parser.add_argument("--max-poses", type=int, default=0, help="Limit exported poses. 0 means all sampled waypoints.")
    parser.add_argument("--road-id", type=int, help="Optional road_id filter.")
    parser.add_argument("--lane-id", type=int, help="Optional lane_id filter.")
    parser.add_argument(
        "--core-pose-csv",
        type=Path,
        help="Optional corrected pose CSV used to crop replay/export waypoints to a frame interval.",
    )
    parser.add_argument("--core-start-frame", type=int, help="Start frame for waypoint crop, e.g. 250.")
    parser.add_argument("--core-end-frame", type=int, help="End frame for waypoint crop, e.g. 450.")
    parser.add_argument(
        "--core-pose-coordinate-frame",
        choices=("corrected_local", "enu", "carla"),
        default="corrected_local",
        help=(
            "Coordinate frame of --core-pose-csv. Use corrected_local when XODR was written in the same "
            "tracking-window local frame; use enu with --sequence-origin-pose-file when XODR was written "
            "as sequence_start_local; use carla when the CSV is already in CARLA/XODR coordinates."
        ),
    )
    parser.add_argument(
        "--sequence-origin-pose-file",
        type=Path,
        help="mapping_pose.txt defining frame-0 origin when converting core ENU poses to sequence_start_local.",
    )
    parser.add_argument("--sequence-origin-frame", type=int, default=0)
    parser.add_argument(
        "--no-core-y-flip",
        action="store_true",
        help=(
            "Do not flip Y when matching core poses to CARLA waypoints. By default, core poses from XODR/sequence "
            "coordinates are converted to CARLA world coordinates as (x, -y), matching CARLA's OpenDRIVE import."
        ),
    )
    parser.add_argument("--spawn-hero", action="store_true", help="Spawn a role_name=hero vehicle at the first pose.")
    parser.add_argument("--hero-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--hero-role-name", default="hero")
    parser.add_argument(
        "--replay-sampled-poses",
        action="store_true",
        help="Kinematically move the hero through exported waypoint poses and record the actual actor poses.",
    )
    parser.add_argument(
        "--replay-core-poses",
        action="store_true",
        help=(
            "Replay poses read directly from --core-pose-csv instead of following CARLA map waypoints. "
            "Use this when validating mapping_pose/3DGS alignment; the XODR is only used as the visual town."
        ),
    )
    parser.add_argument(
        "--follow-core-poses",
        action="store_true",
        help=(
            "Physically drive the hero with VehicleControl to follow poses read from --core-pose-csv. "
            "This uses pure-pursuit steering plus speed control and records tracking error."
        ),
    )
    parser.add_argument(
        "--core-pose-z",
        type=float,
        default=0.5,
        help="CARLA z used for direct --replay-core-poses transforms.",
    )
    parser.add_argument("--replay-loops", type=int, default=1, help="Number of times to replay sampled poses.")
    parser.add_argument("--replay-delay", type=float, default=0.03, help="Sleep seconds between replayed poses.")
    parser.add_argument(
        "--replay-speed-kmh",
        type=float,
        help="Replay speed in km/h. When set, this overrides --replay-delay using waypoint distance.",
    )
    parser.add_argument("--spectator-follow", action="store_true", help="Move spectator with the hero during replay.")
    parser.add_argument("--spectator-height", type=float, default=55.0)
    parser.add_argument("--spectator-distance", type=float, default=35.0)
    parser.add_argument("--follow-speed-kmh", type=float, default=10.0)
    parser.add_argument("--follow-lookahead-distance", type=float, default=6.0)
    parser.add_argument("--follow-finish-distance", type=float, default=3.0)
    parser.add_argument("--follow-max-seconds", type=float, default=0.0, help="0 means infer from path length and speed.")
    parser.add_argument(
        "--follow-start-z-offset",
        type=float,
        default=0.35,
        help="Extra z offset applied only when initializing physical --follow-core-poses.",
    )
    parser.add_argument(
        "--no-follow-map-z",
        action="store_true",
        help="Do not snap physical --follow-core-poses initialization z/pitch/roll to the nearest CARLA map waypoint.",
    )
    parser.add_argument(
        "--follow-settle-ticks",
        type=int,
        default=10,
        help="Physics ticks with brake applied before starting --follow-core-poses control.",
    )
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--speed-kp", type=float, default=0.35)
    parser.add_argument("--speed-ki", type=float, default=0.02)
    parser.add_argument("--speed-kd", type=float, default=0.02)
    parser.add_argument("--steer-gain", type=float, default=0.85)
    parser.add_argument("--max-steer", type=float, default=0.65)
    parser.add_argument("--vertex-distance", type=float, default=2.0)
    parser.add_argument("--max-road-length", type=float, default=500.0)
    parser.add_argument("--wall-height", type=float, default=0.0)
    parser.add_argument("--additional-width", type=float, default=0.6)
    parser.add_argument("--no-smooth-junctions", action="store_true")
    parser.add_argument("--no-mesh-visibility", action="store_true")
    return parser.parse_args()


def import_carla():
    try:
        import carla  # type: ignore

        return carla
    except ImportError:
        dist = Path(__file__).resolve().parents[2] / "PythonAPI" / "carla" / "dist"
        candidates = sorted(dist.glob(f"carla-*-cp{sys.version_info.major}{sys.version_info.minor}-*.whl"))
        if candidates:
            sys.path.insert(0, str(candidates[-1]))
            import carla  # type: ignore

            return carla
        raise


def load_xodr_world(carla, args: argparse.Namespace):
    if not args.xodr.is_file():
        raise FileNotFoundError(args.xodr)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    params = carla.OpendriveGenerationParameters(
        vertex_distance=args.vertex_distance,
        max_road_length=args.max_road_length,
        wall_height=args.wall_height,
        additional_width=args.additional_width,
        smooth_junctions=not args.no_smooth_junctions,
        enable_mesh_visibility=not args.no_mesh_visibility,
    )
    world = client.generate_opendrive_world(args.xodr.read_text(encoding="utf-8"), params)
    return client, world


def transform_to_dict(transform, road_id=None, lane_id=None, s=None, source: str = "waypoint") -> dict:
    loc = transform.location
    rot = transform.rotation
    return {
        "source": source,
        "frame": None,
        "road_id": road_id,
        "lane_id": lane_id,
        "s": s,
        "x": float(loc.x),
        "y": float(loc.y),
        "z": float(loc.z),
        "roll": float(rot.roll),
        "pitch": float(rot.pitch),
        "yaw": float(rot.yaw),
        "matrix": transform.get_matrix(),
    }


def location_distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def xy_distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def path_length(poses: list[dict]) -> float:
    return sum(xy_distance(a, b) for a, b in zip(poses, poses[1:]))


def xy_distance_sq(point_xy: tuple[float, float], waypoint) -> float:
    loc = waypoint.transform.location
    return (loc.x - point_xy[0]) ** 2 + (loc.y - point_xy[1]) ** 2


def read_mapping_origin(path: Path, origin_frame: int) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as fp:
        header = fp.readline().strip().split()
        index = {name: idx for idx, name in enumerate(header)}
        required = ["lidar_frame", "x", "y", "yaw"]
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        for line in fp:
            if not line.strip():
                continue
            parts = line.split()
            if int(parts[index["lidar_frame"]]) == origin_frame:
                return {
                    "x": float(parts[index["x"]]),
                    "y": float(parts[index["y"]]),
                    "yaw": float(parts[index["yaw"]]),
                }
    raise ValueError(f"origin frame {origin_frame} is not present in {path}")


def enu_to_sequence_start_local(x: float, y: float, origin: dict[str, float]) -> tuple[float, float]:
    dx = x - origin["x"]
    dy = y - origin["y"]
    c = math.cos(-origin["yaw"])
    s = math.sin(-origin["yaw"])
    return (c * dx - s * dy, s * dx + c * dy)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def core_xy_to_carla_xy(args: argparse.Namespace, xy: tuple[float, float]) -> tuple[float, float]:
    if args.core_pose_coordinate_frame == "carla" or args.no_core_y_flip:
        return xy
    return (xy[0], -xy[1])


def core_yaw_to_carla_yaw_deg(args: argparse.Namespace, yaw_rad: float) -> float:
    if args.core_pose_coordinate_frame == "carla":
        return math.degrees(yaw_rad)
    if args.no_core_y_flip:
        return math.degrees(yaw_rad)
    return math.degrees(-yaw_rad)


def read_core_frame_xy(args: argparse.Namespace, frame: int) -> tuple[float, float]:
    if args.core_pose_csv is None:
        raise ValueError("--core-pose-csv is required for core frame filtering")
    with args.core_pose_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next((item for item in reader if int(item["frame"]) == frame), None)
    if row is None:
        raise ValueError(f"frame {frame} is not present in {args.core_pose_csv}")

    if args.core_pose_coordinate_frame == "corrected_local":
        return core_xy_to_carla_xy(args, (float(row["corrected_x"]), float(row["corrected_y"])))
    if args.core_pose_coordinate_frame == "carla":
        if "x" in row and "y" in row:
            return (float(row["x"]), float(row["y"]))
        if "corrected_x" in row and "corrected_y" in row:
            return (float(row["corrected_x"]), float(row["corrected_y"]))
        raise ValueError(f"{args.core_pose_csv} has no x/y or corrected_x/corrected_y columns")

    if not args.sequence_origin_pose_file:
        raise ValueError("--core-pose-coordinate-frame=enu needs --sequence-origin-pose-file")
    x_key = "enu_x" if "enu_x" in row else "x"
    y_key = "enu_y" if "enu_y" in row else "y"
    origin = read_mapping_origin(args.sequence_origin_pose_file, args.sequence_origin_frame)
    return core_xy_to_carla_xy(args, enu_to_sequence_start_local(float(row[x_key]), float(row[y_key]), origin))


def core_row_to_carla_pose(args: argparse.Namespace, row: dict, origin: dict[str, float] | None) -> dict:
    frame = int(row["frame"])
    if args.core_pose_coordinate_frame == "corrected_local":
        x = float(row.get("corrected_x", row.get("x")))
        y = float(row.get("corrected_y", row.get("y")))
        yaw = float(row.get("corrected_yaw", row.get("yaw", 0.0)))
        carla_x, carla_y = core_xy_to_carla_xy(args, (x, y))
        carla_yaw = core_yaw_to_carla_yaw_deg(args, yaw)
    elif args.core_pose_coordinate_frame == "enu":
        if origin is None:
            raise ValueError("--core-pose-coordinate-frame=enu needs --sequence-origin-pose-file")
        x_key = "enu_x" if "enu_x" in row else "x"
        y_key = "enu_y" if "enu_y" in row else "y"
        yaw_key = "enu_yaw" if "enu_yaw" in row else "yaw"
        local_x, local_y = enu_to_sequence_start_local(float(row[x_key]), float(row[y_key]), origin)
        local_yaw = normalize_angle(float(row[yaw_key]) - float(origin["yaw"]))
        carla_x, carla_y = core_xy_to_carla_xy(args, (local_x, local_y))
        carla_yaw = core_yaw_to_carla_yaw_deg(args, local_yaw)
    else:
        x_key = "x" if "x" in row else "corrected_x"
        y_key = "y" if "y" in row else "corrected_y"
        yaw_value = float(row.get("yaw", row.get("corrected_yaw", 0.0)))
        # CARLA CSV yaw is conventionally degrees. If the value looks like radians, convert it.
        carla_yaw = math.degrees(yaw_value) if abs(yaw_value) <= 2.0 * math.pi else yaw_value
        carla_x, carla_y = float(row[x_key]), float(row[y_key])
    return {
        "frame": frame,
        "x": carla_x,
        "y": carla_y,
        "z": args.core_pose_z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": carla_yaw,
    }


def read_core_pose_sequence(args: argparse.Namespace) -> list[dict]:
    if args.core_pose_csv is None:
        raise ValueError("--core-pose-csv is required with --replay-core-poses")
    if args.core_start_frame is None or args.core_end_frame is None:
        raise ValueError("--core-start-frame and --core-end-frame are required with --replay-core-poses")
    origin = None
    if args.core_pose_coordinate_frame == "enu":
        if not args.sequence_origin_pose_file:
            raise ValueError("--core-pose-coordinate-frame=enu needs --sequence-origin-pose-file")
        origin = read_mapping_origin(args.sequence_origin_pose_file, args.sequence_origin_frame)
    start, end = sorted((args.core_start_frame, args.core_end_frame))
    poses: list[dict] = []
    with args.core_pose_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            if start <= frame <= end:
                poses.append(core_row_to_carla_pose(args, row, origin))
    if not poses:
        raise RuntimeError(f"No core poses found in [{start}, {end}] from {args.core_pose_csv}")
    poses.sort(key=lambda item: item["frame"])
    return poses


def pose_dict_to_transform(carla, pose: dict):
    return carla.Transform(
        carla.Location(x=float(pose["x"]), y=float(pose["y"]), z=float(pose["z"])),
        carla.Rotation(
            roll=float(pose.get("roll", 0.0)),
            pitch=float(pose.get("pitch", 0.0)),
            yaw=float(pose["yaw"]),
        ),
    )


def pose_dict_to_map_aligned_transform(carla, carla_map, pose: dict, z_offset: float):
    transform = pose_dict_to_transform(carla, pose)
    waypoint = carla_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        transform.location.z += z_offset
        return transform, None
    aligned = carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=waypoint.transform.location.z + z_offset,
        ),
        carla.Rotation(
            roll=waypoint.transform.rotation.roll,
            pitch=waypoint.transform.rotation.pitch,
            yaw=transform.rotation.yaw,
        ),
    )
    return aligned, waypoint


def sorted_waypoints(waypoints: Iterable, road_id: int | None, lane_id: int | None) -> list:
    filtered = []
    for waypoint in waypoints:
        if road_id is not None and int(waypoint.road_id) != road_id:
            continue
        if lane_id is not None and int(waypoint.lane_id) != lane_id:
            continue
        filtered.append(waypoint)
    return sorted(filtered, key=lambda wp: (int(wp.road_id), int(wp.lane_id), float(wp.s)))


def crop_waypoints_to_core_frames(waypoints: list, args: argparse.Namespace) -> tuple[list, dict]:
    if args.core_pose_csv is None:
        return waypoints, {}
    if args.core_start_frame is None or args.core_end_frame is None:
        raise ValueError("--core-start-frame and --core-end-frame are required with --core-pose-csv")

    start_xy = read_core_frame_xy(args, args.core_start_frame)
    end_xy = read_core_frame_xy(args, args.core_end_frame)
    start_wp = min(waypoints, key=lambda wp: xy_distance_sq(start_xy, wp))
    end_wp = min(waypoints, key=lambda wp: xy_distance_sq(end_xy, wp))
    route_road_id = int(start_wp.road_id)
    route_lane_id = int(start_wp.lane_id)
    s0, s1 = sorted((float(start_wp.s), float(end_wp.s)))
    cropped = [
        wp
        for wp in waypoints
        if int(wp.road_id) == route_road_id
        and int(wp.lane_id) == route_lane_id
        and s0 <= float(wp.s) <= s1
    ]
    if not cropped:
        raise RuntimeError(
            f"Core frame crop produced no waypoints: road={route_road_id} lane={route_lane_id} s=[{s0}, {s1}]"
        )
    metadata = {
        "core_pose_csv": str(args.core_pose_csv),
        "core_pose_coordinate_frame": args.core_pose_coordinate_frame,
        "core_start_frame": args.core_start_frame,
        "core_end_frame": args.core_end_frame,
        "core_start_xy": start_xy,
        "core_end_xy": end_xy,
        "core_y_flipped_for_carla": args.core_pose_coordinate_frame != "carla" and not args.no_core_y_flip,
        "core_road_id": route_road_id,
        "core_lane_id": route_lane_id,
        "core_s_min": s0,
        "core_s_max": s1,
        "core_start_nearest_s": float(start_wp.s),
        "core_end_nearest_s": float(end_wp.s),
        "core_start_nearest_distance_m": math.sqrt(xy_distance_sq(start_xy, start_wp)),
        "core_end_nearest_distance_m": math.sqrt(xy_distance_sq(end_xy, end_wp)),
    }
    return cropped, metadata


def write_pose_csv(path: Path, poses: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "source",
        "frame",
        "loop",
        "elapsed_time",
        "speed_mps",
        "speed_kmh",
        "road_id",
        "lane_id",
        "s",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "target_index",
        "target_frame",
        "target_x",
        "target_y",
        "target_yaw",
        "distance_error",
        "lateral_error",
        "heading_error_deg",
        "throttle",
        "steer",
        "brake",
        "start_nearest_waypoint_distance",
        "start_nearest_road_id",
        "start_nearest_lane_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, pose in enumerate(poses):
            row = {field: pose.get(field) for field in fields}
            row["index"] = idx
            writer.writerow(row)


def write_pose_json(path: Path, poses: list[dict], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "poses": poses}, indent=2) + "\n", encoding="utf-8")


def move_spectator(carla, world, target_transform, height: float, distance: float) -> None:
    yaw = math.radians(target_transform.rotation.yaw)
    loc = target_transform.location
    spectator_loc = carla.Location(
        x=loc.x - math.cos(yaw) * distance,
        y=loc.y - math.sin(yaw) * distance,
        z=loc.z + height,
    )
    spectator_rot = carla.Rotation(pitch=-70.0, yaw=target_transform.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(spectator_loc, spectator_rot))


def spawn_hero(carla, world, transform, blueprint_id: str, role_name: str):
    blueprint_library = world.get_blueprint_library()
    blueprint = blueprint_library.find(blueprint_id)
    blueprint.set_attribute("role_name", role_name)
    spawn_transform = carla.Transform(transform.location, transform.rotation)
    spawn_transform.location.z += 0.5
    actor = world.try_spawn_actor(blueprint, spawn_transform)
    if actor is None:
        raise RuntimeError(f"Failed to spawn {blueprint_id} at first exported pose")
    return actor


def replay_hero(
    carla,
    world,
    hero,
    waypoints: list,
    loops: int,
    delay: float,
    speed_kmh: float | None,
    spectator_follow: bool,
    spectator_height: float,
    spectator_distance: float,
) -> list[dict]:
    poses = []
    loops = max(1, loops)
    speed_mps = speed_kmh / 3.6 if speed_kmh is not None else None
    elapsed_time = 0.0
    for loop_idx in range(loops):
        previous_transform = None
        for waypoint in waypoints:
            transform = waypoint.transform
            transform.location.z += 0.5
            hero.set_transform(transform)
            if spectator_follow:
                move_spectator(carla, world, transform, spectator_height, spectator_distance)
            world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick()
            pose = transform_to_dict(hero.get_transform(), waypoint.road_id, waypoint.lane_id, waypoint.s, "hero")
            pose["loop"] = loop_idx
            pose["elapsed_time"] = elapsed_time
            pose["speed_mps"] = float(speed_mps) if speed_mps is not None else None
            pose["speed_kmh"] = float(speed_kmh) if speed_kmh is not None else None
            poses.append(pose)
            step_delay = delay
            if speed_mps is not None and previous_transform is not None:
                distance = location_distance(previous_transform.location, transform.location)
                step_delay = distance / max(speed_mps, 1e-6)
            previous_transform = transform
            elapsed_time += step_delay
            if step_delay > 0.0:
                time.sleep(step_delay)
    return poses


def replay_core_pose_transforms(
    carla,
    world,
    hero,
    core_poses: list[dict],
    loops: int,
    delay: float,
    speed_kmh: float | None,
    spectator_follow: bool,
    spectator_height: float,
    spectator_distance: float,
) -> list[dict]:
    poses = []
    loops = max(1, loops)
    speed_mps = speed_kmh / 3.6 if speed_kmh is not None else None
    elapsed_time = 0.0
    for loop_idx in range(loops):
        previous_transform = None
        for core_pose in core_poses:
            transform = pose_dict_to_transform(carla, core_pose)
            hero.set_transform(transform)
            if spectator_follow:
                move_spectator(carla, world, transform, spectator_height, spectator_distance)
            world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick()
            pose = transform_to_dict(hero.get_transform(), source="hero_core_pose")
            pose["frame"] = int(core_pose["frame"])
            pose["loop"] = loop_idx
            pose["elapsed_time"] = elapsed_time
            pose["speed_mps"] = float(speed_mps) if speed_mps is not None else None
            pose["speed_kmh"] = float(speed_kmh) if speed_kmh is not None else None
            poses.append(pose)
            step_delay = delay
            if speed_mps is not None and previous_transform is not None:
                distance = location_distance(previous_transform.location, transform.location)
                step_delay = distance / max(speed_mps, 1e-6)
            previous_transform = transform
            elapsed_time += step_delay
            if step_delay > 0.0:
                time.sleep(step_delay)
    return poses


def vehicle_speed_mps(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)


def nearest_path_index(path: list[dict], x: float, y: float, start_index: int) -> tuple[int, float]:
    if not path:
        raise ValueError("empty path")
    begin = max(0, min(start_index, len(path) - 1))
    end = min(len(path), begin + 80)
    best_index = begin
    best_dist = float("inf")
    for index in range(begin, end):
        dist = math.hypot(float(path[index]["x"]) - x, float(path[index]["y"]) - y)
        if dist < best_dist:
            best_index = index
            best_dist = dist
    return best_index, best_dist


def lookahead_path_index(path: list[dict], start_index: int, lookahead_distance: float) -> int:
    if start_index >= len(path) - 1:
        return len(path) - 1
    distance = 0.0
    previous = path[start_index]
    for index in range(start_index + 1, len(path)):
        distance += xy_distance(previous, path[index])
        if distance >= lookahead_distance:
            return index
        previous = path[index]
    return len(path) - 1


def local_xy_from_transform(transform, target: dict) -> tuple[float, float]:
    loc = transform.location
    yaw = math.radians(transform.rotation.yaw)
    dx = float(target["x"]) - float(loc.x)
    dy = float(target["y"]) - float(loc.y)
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return local_x, local_y


def make_path_follow_control(carla, args: argparse.Namespace, hero, target: dict, dt: float, pid_state: dict) -> tuple:
    transform = hero.get_transform()
    local_x, local_y = local_xy_from_transform(transform, target)
    lookahead = max(math.hypot(local_x, local_y), 1e-3)
    curvature = 2.0 * local_y / max(lookahead * lookahead, 1e-3)
    target_speed = max(args.follow_speed_kmh / 3.6, 0.0)
    speed = vehicle_speed_mps(hero)
    steer_limit = args.max_steer
    if speed < 1.0:
        steer_limit = min(steer_limit, 0.25)
    steer = clamp(args.steer_gain * curvature * 2.8, -steer_limit, steer_limit)
    speed_error = target_speed - speed
    pid_state["integral"] = clamp(pid_state.get("integral", 0.0) + speed_error * dt, -5.0, 5.0)
    derivative = (speed_error - pid_state.get("previous_error", speed_error)) / max(dt, 1e-6)
    pid_state["previous_error"] = speed_error
    accel_cmd = args.speed_kp * speed_error + args.speed_ki * pid_state["integral"] + args.speed_kd * derivative
    throttle = clamp(accel_cmd, 0.0, 0.75)
    brake = clamp(-accel_cmd, 0.0, 0.8)
    if target_speed < 1e-3:
        throttle = 0.0
        brake = 1.0
    control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
    return control, speed, local_x, local_y


def follow_core_pose_path(
    carla,
    world,
    carla_map,
    hero,
    core_poses: list[dict],
    args: argparse.Namespace,
) -> list[dict]:
    if len(core_poses) < 2:
        raise ValueError("--follow-core-poses needs at least two core poses")

    original_settings = world.get_settings()
    sync_settings = world.get_settings()
    sync_settings.synchronous_mode = True
    sync_settings.fixed_delta_seconds = args.control_dt
    world.apply_settings(sync_settings)

    start_pose = dict(core_poses[0])
    if args.no_follow_map_z:
        start_pose["z"] = float(start_pose["z"]) + float(args.follow_start_z_offset)
        start_transform = pose_dict_to_transform(carla, start_pose)
        start_waypoint = None
    else:
        start_transform, start_waypoint = pose_dict_to_map_aligned_transform(
            carla,
            carla_map,
            start_pose,
            args.follow_start_z_offset,
        )
    hero.set_simulate_physics(False)
    hero.set_transform(start_transform)
    world.tick()
    hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    hero.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    hero.set_simulate_physics(True)
    for _ in range(max(0, args.follow_settle_ticks)):
        hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        world.tick()

    target_speed = max(args.follow_speed_kmh / 3.6, 0.1)
    max_seconds = args.follow_max_seconds
    if max_seconds <= 0.0:
        max_seconds = path_length(core_poses) / target_speed + 20.0
    max_steps = max(1, int(math.ceil(max_seconds / max(args.control_dt, 1e-3))))

    outputs: list[dict] = []
    nearest_index = 0
    elapsed_time = 0.0
    pid_state: dict = {}
    try:
        for step in range(max_steps):
            transform = hero.get_transform()
            nearest_index, nearest_dist = nearest_path_index(core_poses, transform.location.x, transform.location.y, nearest_index)
            target_index = lookahead_path_index(core_poses, nearest_index, args.follow_lookahead_distance)
            target = core_poses[target_index]
            control, speed, _, lateral_error = make_path_follow_control(carla, args, hero, target, args.control_dt, pid_state)
            hero.apply_control(control)
            world.tick()
            actual = transform_to_dict(hero.get_transform(), source="hero_controller")
            heading_error = normalize_angle(math.radians(float(target["yaw"])) - math.radians(float(actual["yaw"])))
            actual["frame"] = int(core_poses[nearest_index]["frame"])
            actual["loop"] = 0
            actual["elapsed_time"] = elapsed_time
            actual["speed_mps"] = speed
            actual["speed_kmh"] = speed * 3.6
            actual["target_index"] = target_index
            actual["target_frame"] = int(target["frame"])
            actual["target_x"] = float(target["x"])
            actual["target_y"] = float(target["y"])
            actual["target_yaw"] = float(target["yaw"])
            actual["distance_error"] = nearest_dist
            actual["lateral_error"] = lateral_error
            actual["heading_error_deg"] = math.degrees(heading_error)
            actual["throttle"] = float(control.throttle)
            actual["steer"] = float(control.steer)
            actual["brake"] = float(control.brake)
            if step == 0 and start_waypoint is not None:
                actual["start_nearest_waypoint_distance"] = location_distance(start_transform.location, start_waypoint.transform.location)
                actual["start_nearest_road_id"] = int(start_waypoint.road_id)
                actual["start_nearest_lane_id"] = int(start_waypoint.lane_id)
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
    return outputs


def main() -> int:
    args = parse_args()
    if args.replay_core_poses and args.replay_sampled_poses:
        raise ValueError("--replay-core-poses and --replay-sampled-poses are mutually exclusive")
    if args.follow_core_poses and (args.replay_core_poses or args.replay_sampled_poses):
        raise ValueError("--follow-core-poses is mutually exclusive with replay modes")
    carla = import_carla()
    client, world = load_xodr_world(carla, args)
    carla_map = world.get_map()
    core_metadata = {}
    hero = None
    if args.replay_core_poses or args.follow_core_poses:
        core_poses = read_core_pose_sequence(args)
        if args.max_poses > 0:
            core_poses = core_poses[: args.max_poses]
        first_transform = pose_dict_to_transform(carla, core_poses[0])
        move_spectator(carla, world, first_transform, args.spectator_height, args.spectator_distance)
        exported = []
        for core_pose in core_poses:
            pose = transform_to_dict(pose_dict_to_transform(carla, core_pose), source="core_pose")
            pose["frame"] = int(core_pose["frame"])
            exported.append(pose)
        if args.spawn_hero or args.replay_core_poses:
            hero = spawn_hero(carla, world, first_transform, args.hero_blueprint, args.hero_role_name)
            print(f"spawned hero actor id={hero.id}")
        if args.follow_core_poses:
            if hero is None:
                hero = spawn_hero(carla, world, first_transform, args.hero_blueprint, args.hero_role_name)
                print(f"spawned hero actor id={hero.id}")
            exported = follow_core_pose_path(carla, world, carla_map, hero, core_poses, args)
        else:
            exported = replay_core_pose_transforms(
                carla,
                world,
                hero,
                core_poses,
                args.replay_loops,
                args.replay_delay,
                args.replay_speed_kmh,
                args.spectator_follow,
                args.spectator_height,
                args.spectator_distance,
            )
        move_spectator(carla, world, hero.get_transform(), args.spectator_height, args.spectator_distance)
        core_metadata = {
            "core_pose_csv": str(args.core_pose_csv),
            "core_pose_coordinate_frame": args.core_pose_coordinate_frame,
            "core_start_frame": args.core_start_frame,
            "core_end_frame": args.core_end_frame,
            "core_y_flipped_for_carla": args.core_pose_coordinate_frame != "carla" and not args.no_core_y_flip,
            "core_replay_mode": "controller_follow_core_poses" if args.follow_core_poses else "direct_core_poses",
            "core_pose_z": args.core_pose_z,
            "follow_speed_kmh": args.follow_speed_kmh if args.follow_core_poses else None,
            "follow_lookahead_distance": args.follow_lookahead_distance if args.follow_core_poses else None,
            "control_dt": args.control_dt if args.follow_core_poses else None,
            "speed_kp": args.speed_kp if args.follow_core_poses else None,
            "speed_ki": args.speed_ki if args.follow_core_poses else None,
            "speed_kd": args.speed_kd if args.follow_core_poses else None,
            "steer_gain": args.steer_gain if args.follow_core_poses else None,
            "max_steer": args.max_steer if args.follow_core_poses else None,
            "follow_start_z_offset": args.follow_start_z_offset if args.follow_core_poses else None,
            "follow_uses_map_z": args.follow_core_poses and not args.no_follow_map_z,
        }
    else:
        waypoints = sorted_waypoints(
            carla_map.generate_waypoints(args.waypoint_spacing),
            args.road_id,
            args.lane_id,
        )
        waypoints, core_metadata = crop_waypoints_to_core_frames(waypoints, args)
        if args.max_poses > 0:
            waypoints = waypoints[: args.max_poses]
        if not waypoints:
            raise RuntimeError("No waypoints found. Check the XODR and optional road/lane filters.")

        move_spectator(carla, world, waypoints[0].transform, args.spectator_height, args.spectator_distance)
        exported = [
            transform_to_dict(wp.transform, wp.road_id, wp.lane_id, wp.s, "waypoint")
            for wp in waypoints
        ]

        if args.spawn_hero or args.replay_sampled_poses:
            hero = spawn_hero(carla, world, waypoints[0].transform, args.hero_blueprint, args.hero_role_name)
            print(f"spawned hero actor id={hero.id}")

        if args.replay_sampled_poses:
            exported = replay_hero(
                carla,
                world,
                hero,
                waypoints,
                args.replay_loops,
                args.replay_delay,
                args.replay_speed_kmh,
                args.spectator_follow,
                args.spectator_height,
                args.spectator_distance,
            )
            move_spectator(carla, world, hero.get_transform(), args.spectator_height, args.spectator_distance)

    write_pose_csv(args.pose_output, exported)
    metadata = {
        "xodr": str(args.xodr),
        "map_name": carla_map.name,
        "waypoint_spacing": args.waypoint_spacing,
        "road_id_filter": args.road_id,
        "lane_id_filter": args.lane_id,
        "pose_count": len(exported),
        "host": args.host,
        "port": args.port,
    }
    metadata.update(core_metadata)
    if args.json_output:
        write_pose_json(args.json_output, exported, metadata)

    print(f"loaded world: {carla_map.name}")
    print(f"topology edges: {len(carla_map.get_topology())}")
    print(f"exported poses: {len(exported)}")
    print(f"wrote: {args.pose_output}")
    if args.json_output:
        print(f"wrote: {args.json_output}")
    print("CARLA window should now show the generated local OpenDRIVE town.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
