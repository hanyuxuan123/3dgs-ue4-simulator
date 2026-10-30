#!/usr/bin/env python3
"""Generate CARLA Traffic Manager actor trajectories in DriveStudio instance format.

The script loads an OpenDRIVE town, samples an ego reference route, spawns NPC
vehicles with Traffic Manager, records their CARLA-frame transforms, filters
unsafe tracks, and writes:

  instances/instances_info.json
  instances/frame_instances.json
  ego_route_carla.json
  generation_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", required=True, type=Path, help="OpenDRIVE .xodr file to load.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-actors", type=int, default=24)
    parser.add_argument(
        "--target-instances",
        type=int,
        default=0,
        help="Keep only this many accepted tracks in the output. 0 keeps all accepted tracks.",
    )
    parser.add_argument("--max-spawn-attempts", type=int, default=300)
    parser.add_argument(
        "--spawn-source",
        choices=("auto", "spawn-points", "generated-waypoints", "both"),
        default="auto",
        help="Where to sample NPC spawn transforms. generated-waypoints is more robust for custom XODR maps.",
    )
    parser.add_argument("--spawn-waypoint-spacing", type=float, default=4.0)
    parser.add_argument("--vehicle-filter", default="vehicle.*")
    parser.add_argument("--ego-route-spacing", type=float, default=2.0)
    parser.add_argument("--ego-route-length", type=float, default=90.0)
    parser.add_argument("--ego-route-speed-mps", type=float, default=6.0)
    parser.add_argument("--ego-start-index", type=int, default=0, help="Index into shuffled spawn points.")
    parser.add_argument("--task-start-frame", type=int, default=20)
    parser.add_argument("--task-end-frame", type=int, default=140)
    parser.add_argument("--spawn-near-ego-min", type=float, default=12.0)
    parser.add_argument("--spawn-near-ego-max", type=float, default=85.0)
    parser.add_argument("--ego-clearance", type=float, default=7.0, help="Min actor center distance to ego route during task interval.")
    parser.add_argument("--actor-collision-margin", type=float, default=0.75)
    parser.add_argument("--min-track-frames", type=int, default=20)
    parser.add_argument("--ignore-lights-percentage", type=float, default=30.0)
    parser.add_argument("--global-distance-to-leading", type=float, default=4.0)
    parser.add_argument("--global-speed-difference", type=float, default=15.0, help="Positive means NPCs drive slower than limit.")
    parser.add_argument("--draw-debug", action="store_true")
    parser.add_argument("--destroy-actors", action="store_true", default=True)
    parser.add_argument("--keep-actors", dest="destroy_actors", action="store_false")
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


def transform_matrix(transform) -> list[list[float]]:
    return [[float(v) for v in row] for row in transform.get_matrix()]


def loc_xy(location) -> tuple[float, float]:
    return float(location.x), float(location.y)


def xy_distance(a, b) -> float:
    ax, ay = loc_xy(a)
    bx, by = loc_xy(b)
    return math.hypot(ax - bx, ay - by)


def distance_to_route_xy(location, route: list[dict[str, Any]], frame_index: int | None = None) -> float:
    x, y = loc_xy(location)
    if frame_index is not None and route:
        begin = max(0, frame_index - 8)
        end = min(len(route), frame_index + 9)
        candidates = route[begin:end] or route
    else:
        candidates = route
    return min(math.hypot(x - float(p["x"]), y - float(p["y"])) for p in candidates)


def route_length(route: list[dict[str, Any]]) -> float:
    return sum(
        math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
        for a, b in zip(route, route[1:])
    )


def sample_ego_route(carla, carla_map, spawn_points: list, args: argparse.Namespace) -> list[dict[str, Any]]:
    if not spawn_points:
        raise RuntimeError("CARLA map has no spawn points")
    start = spawn_points[args.ego_start_index % len(spawn_points)]
    waypoint = carla_map.get_waypoint(start.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if waypoint is None:
        raise RuntimeError("Failed to resolve ego start waypoint")

    sampled = [waypoint]
    while route_length([waypoint_to_pose(w) for w in sampled]) < args.ego_route_length:
        next_waypoints = sampled[-1].next(args.ego_route_spacing)
        if not next_waypoints:
            break
        sampled.append(next_waypoints[0])
        if len(sampled) > 2000:
            break
    poses = [waypoint_to_pose(w) for w in sampled]
    if len(poses) < 2:
        raise RuntimeError("Failed to sample a usable ego route")
    return resample_route_by_time(poses, args.frames, args.ego_route_speed_mps, 1.0 / args.fps)


def waypoint_to_pose(waypoint) -> dict[str, Any]:
    t = waypoint.transform
    return {
        "road_id": int(waypoint.road_id),
        "lane_id": int(waypoint.lane_id),
        "s": float(waypoint.s),
        "x": float(t.location.x),
        "y": float(t.location.y),
        "z": float(t.location.z),
        "roll": float(t.rotation.roll),
        "pitch": float(t.rotation.pitch),
        "yaw": float(t.rotation.yaw),
        "matrix": transform_matrix(t),
    }


def interpolate_pose(a: dict[str, Any], b: dict[str, Any], ratio: float) -> dict[str, Any]:
    def lerp(key: str) -> float:
        return float(a[key]) * (1.0 - ratio) + float(b[key]) * ratio

    yaw_a = math.radians(float(a["yaw"]))
    yaw_b = math.radians(float(b["yaw"]))
    dyaw = math.atan2(math.sin(yaw_b - yaw_a), math.cos(yaw_b - yaw_a))
    yaw = math.degrees(yaw_a + dyaw * ratio)
    return {
        "road_id": a.get("road_id"),
        "lane_id": a.get("lane_id"),
        "s": lerp("s"),
        "x": lerp("x"),
        "y": lerp("y"),
        "z": lerp("z"),
        "roll": lerp("roll"),
        "pitch": lerp("pitch"),
        "yaw": yaw,
    }


def resample_route_by_time(route: list[dict[str, Any]], frames: int, speed_mps: float, dt: float) -> list[dict[str, Any]]:
    arcs = [0.0]
    for a, b in zip(route, route[1:]):
        arcs.append(arcs[-1] + math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])))
    out = []
    for frame in range(frames):
        target_s = min(frame * dt * speed_mps, arcs[-1])
        j = 0
        while j < len(arcs) - 2 and arcs[j + 1] < target_s:
            j += 1
        span = max(arcs[j + 1] - arcs[j], 1e-6) if j < len(arcs) - 1 else 1.0
        pose = interpolate_pose(route[j], route[min(j + 1, len(route) - 1)], (target_s - arcs[j]) / span)
        pose["frame"] = frame
        pose["elapsed_time"] = frame * dt
        out.append(pose)
    return out


def actor_radius(actor) -> float:
    e = actor.bounding_box.extent
    return math.hypot(float(e.x), float(e.y))


def actor_class_name(actor) -> str:
    if actor.type_id.startswith("vehicle."):
        suffix = actor.type_id.split(".")[-1]
        if any(k in suffix for k in ("truck", "carlacola", "firetruck", "ambulance")):
            return "vehicle.truck"
        if any(k in suffix for k in ("bus", "coach")):
            return "vehicle.bus"
        if "motorcycle" in suffix or "bike" in suffix:
            return "vehicle.motorcycle"
        return "vehicle.car"
    return actor.type_id


def choose_spawn_points(carla, carla_map, spawn_points: list, ego_route: list[dict[str, Any]], args: argparse.Namespace) -> list:
    shuffled = list(spawn_points)
    random.shuffle(shuffled)
    chosen = []
    for transform in shuffled:
        dist = distance_to_route_xy(transform.location, ego_route)
        if not (args.spawn_near_ego_min <= dist <= args.spawn_near_ego_max):
            continue
        if any(xy_distance(transform.location, other.location) < 8.0 for other in chosen):
            continue
        waypoint = carla_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            continue
        chosen.append(transform)
        if len(chosen) >= args.num_actors:
            break
    return chosen


def generated_waypoint_transforms(carla_map, spacing: float) -> list:
    waypoints = carla_map.generate_waypoints(max(spacing, 0.5))
    transforms = []
    seen = set()
    for waypoint in waypoints:
        key = (
            int(waypoint.road_id),
            int(waypoint.lane_id),
            round(float(waypoint.s), 1),
        )
        if key in seen:
            continue
        seen.add(key)
        transforms.append(waypoint.transform)
    return transforms


def candidate_spawn_transforms(carla_map, args: argparse.Namespace) -> list:
    spawn_points = list(carla_map.get_spawn_points())
    generated = generated_waypoint_transforms(carla_map, args.spawn_waypoint_spacing)
    if args.spawn_source == "spawn-points":
        return spawn_points
    if args.spawn_source == "generated-waypoints":
        return generated
    if args.spawn_source == "both":
        return spawn_points + generated
    return spawn_points if len(spawn_points) >= args.num_actors * 3 else spawn_points + generated


def spawn_traffic(carla, world, traffic_manager, carla_map, ego_route, args: argparse.Namespace):
    blueprints = list(world.get_blueprint_library().filter(args.vehicle_filter))
    blueprints = [bp for bp in blueprints if int(bp.get_attribute("number_of_wheels")) >= 4] if blueprints else []
    if not blueprints:
        raise RuntimeError(f"No vehicle blueprints matched {args.vehicle_filter}")
    spawn_points = candidate_spawn_transforms(carla_map, args)
    random.shuffle(spawn_points)
    selected = choose_spawn_points(carla, carla_map, spawn_points, ego_route, args)
    print(
        "[spawn] "
        f"source={args.spawn_source} raw_candidates={len(spawn_points)} "
        f"selected_after_distance_filter={len(selected)} "
        f"distance_range=[{args.spawn_near_ego_min:.1f},{args.spawn_near_ego_max:.1f}]",
        flush=True,
    )
    actors = []
    attempts = 0
    failed_spawns = 0
    for transform in selected:
        if attempts >= args.max_spawn_attempts or len(actors) >= args.num_actors:
            break
        attempts += 1
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "tm_instance")
        if blueprint.has_attribute("color"):
            colors = blueprint.get_attribute("color").recommended_values
            if colors:
                blueprint.set_attribute("color", random.choice(colors))
        spawn_transform = carla.Transform(transform.location, transform.rotation)
        spawn_transform.location.z += 0.4
        actor = world.try_spawn_actor(blueprint, spawn_transform)
        if actor is None:
            failed_spawns += 1
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(actor, args.ignore_lights_percentage)
        actors.append(actor)
    print(f"[spawn] spawned={len(actors)} failed_spawns={failed_spawns} attempts={attempts}", flush=True)
    return actors


def record_actor(actor, frame: int, elapsed_time: float) -> dict[str, Any]:
    transform = actor.get_transform()
    extent = actor.bounding_box.extent
    return {
        "frame": frame,
        "elapsed_time": elapsed_time,
        "transform": transform,
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "yaw": float(transform.rotation.yaw),
        "box_size": [float(extent.x) * 2.0, float(extent.y) * 2.0, float(extent.z) * 2.0],
        "class_name": actor_class_name(actor),
        "type_id": actor.type_id,
    }


def collect_trajectories(carla, client, world, actors: list, traffic_manager, args: argparse.Namespace) -> dict[int, list[dict[str, Any]]]:
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)
    traffic_manager.set_global_distance_to_leading_vehicle(args.global_distance_to_leading)
    traffic_manager.global_percentage_speed_difference(args.global_speed_difference)

    records: dict[int, list[dict[str, Any]]] = {int(actor.id): [] for actor in actors}
    try:
        for _ in range(10):
            world.tick()
        for frame in range(args.frames):
            world.tick()
            elapsed = frame / args.fps
            for actor in list(actors):
                if actor.is_alive:
                    records[int(actor.id)].append(record_actor(actor, frame, elapsed))
            if args.draw_debug and frame % max(1, int(args.fps)) == 0:
                world.debug.draw_string(actors[0].get_location(), f"frame {frame}", life_time=1.0) if actors else None
    finally:
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)
    return records


def track_passes_ego_clearance(track: list[dict[str, Any]], ego_route: list[dict[str, Any]], args: argparse.Namespace) -> bool:
    begin = max(0, args.task_start_frame)
    end = min(args.frames - 1, args.task_end_frame)
    if begin > end:
        return True
    for row in track:
        frame = int(row["frame"])
        if begin <= frame <= end:
            loc = row["transform"].location
            if distance_to_route_xy(loc, ego_route, frame) < args.ego_clearance:
                return False
    return True


def tracks_collide(a: list[dict[str, Any]], b: list[dict[str, Any]], margin: float) -> bool:
    by_frame = {int(row["frame"]): row for row in b}
    for row in a:
        other = by_frame.get(int(row["frame"]))
        if other is None:
            continue
        ra = math.hypot(row["box_size"][0] * 0.5, row["box_size"][1] * 0.5)
        rb = math.hypot(other["box_size"][0] * 0.5, other["box_size"][1] * 0.5)
        dist = math.hypot(float(row["x"]) - float(other["x"]), float(row["y"]) - float(other["y"]))
        if dist < ra + rb + margin:
            return True
    return False


def filter_tracks(records: dict[int, list[dict[str, Any]]], ego_route: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, list[dict[str, Any]]]]:
    candidates = []
    for actor_id, track in records.items():
        if len(track) < args.min_track_frames:
            continue
        if not track_passes_ego_clearance(track, ego_route, args):
            continue
        candidates.append((actor_id, track))

    accepted: list[tuple[int, list[dict[str, Any]]]] = []
    for actor_id, track in candidates:
        if any(tracks_collide(track, other, args.actor_collision_margin) for _, other in accepted):
            continue
        accepted.append((actor_id, track))
    return accepted


def write_outputs(out_dir: Path, accepted: list[tuple[int, list[dict[str, Any]]]], ego_route: list[dict[str, Any]], args: argparse.Namespace) -> None:
    instances_dir = out_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    instances_info: dict[str, Any] = {}
    frame_instances: dict[str, list[int]] = {str(frame): [] for frame in range(args.frames)}
    raw_tracks: dict[str, Any] = {}

    for instance_index, (actor_id, track) in enumerate(accepted):
        frame_idx = [int(row["frame"]) for row in track]
        obj_to_world = [transform_matrix(row["transform"]) for row in track]
        box_size = [row["box_size"] for row in track]
        for frame in frame_idx:
            frame_instances[str(frame)].append(instance_index)
        class_name = track[0].get("class_name", "vehicle.car")
        instances_info[str(instance_index)] = {
            "id": f"carla_tm_actor_{actor_id}",
            "class_name": class_name,
            "frame_annotations": {
                "frame_idx": frame_idx,
                "obj_to_world": obj_to_world,
                "box_size": box_size,
            },
        }
        raw_tracks[str(instance_index)] = {
            "actor_id": actor_id,
            "frames": [
                {
                    "frame": int(row["frame"]),
                    "elapsed_time": float(row["elapsed_time"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "yaw": float(row["yaw"]),
                    "box_size": row["box_size"],
                    "class_name": row.get("class_name", class_name),
                    "type_id": row.get("type_id", ""),
                }
                for row in track
            ],
        }

    (instances_dir / "instances_info.json").write_text(json.dumps(instances_info, indent=2) + "\n", encoding="utf-8")
    (instances_dir / "frame_instances.json").write_text(json.dumps(frame_instances, indent=2) + "\n", encoding="utf-8")
    (out_dir / "ego_route_carla.json").write_text(
        json.dumps(
            {
                "coordinate_frame": "carla",
                "fps": args.fps,
                "task_interval": {"start_frame": args.task_start_frame, "end_frame": args.task_end_frame},
                "poses": ego_route,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "actor_trajectories_carla.json").write_text(json.dumps(raw_tracks, indent=2) + "\n", encoding="utf-8")
    (out_dir / "generation_summary.json").write_text(
        json.dumps(
            {
                "num_instances": len(accepted),
                "frames": args.frames,
                "fps": args.fps,
                "ego_clearance": args.ego_clearance,
                "actor_collision_margin": args.actor_collision_margin,
                "outputs": {
                    "instances_info": str(instances_dir / "instances_info.json"),
                    "frame_instances": str(instances_dir / "frame_instances.json"),
                    "ego_route": str(out_dir / "ego_route_carla.json"),
                    "raw_tracks": str(out_dir / "actor_trajectories_carla.json"),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    carla = import_carla()
    globals()["carla"] = carla
    client, world = load_xodr_world(carla, args)
    carla_map = world.get_map()
    spawn_points = candidate_spawn_transforms(carla_map, args)
    print(f"[map] spawn_points={len(carla_map.get_spawn_points())} generated_candidates={len(generated_waypoint_transforms(carla_map, args.spawn_waypoint_spacing))}", flush=True)
    random.shuffle(spawn_points)
    traffic_manager = client.get_trafficmanager(args.tm_port)

    actors = []
    try:
        ego_route = sample_ego_route(carla, carla_map, spawn_points, args)
        actors = spawn_traffic(carla, world, traffic_manager, carla_map, ego_route, args)
        if not actors:
            raise RuntimeError("No NPC actors spawned. Try increasing --spawn-near-ego-max or using another --ego-start-index.")
        print(f"spawned actors={len(actors)}; recording {args.frames} frames at {args.fps:.1f} FPS", flush=True)
        records = collect_trajectories(carla, client, world, actors, traffic_manager, args)
        accepted = filter_tracks(records, ego_route, args)
        if args.target_instances > 0:
            accepted = accepted[: args.target_instances]
        write_outputs(args.out_dir, accepted, ego_route, args)
        print(f"accepted tracks={len(accepted)} out_dir={args.out_dir}", flush=True)
    finally:
        if args.destroy_actors and actors:
            actor_ids = [int(actor.id) for actor in actors]
            try:
                client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in actor_ids])
            except Exception as exc:
                print(f"WARNING: failed to batch destroy actors {actor_ids}: {exc}", flush=True)
            time.sleep(0.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
