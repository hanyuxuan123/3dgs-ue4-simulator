#!/usr/bin/env python3
"""Start CARLA/XODR + 3DGS and accept vehicle controls from an external process."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

import carla_xodr_live_3dgs_bridge_with_instances as base
from load_xodr_town_and_dump_poses import (
    move_spectator,
    nearest_path_index,
    pose_dict_to_map_aligned_transform,
    pose_dict_to_transform,
    transform_to_dict,
    vehicle_speed_mps,
)
from load_xodr_town_with_instances import (
    load_instance_tracks,
    make_instance_transform,
    make_local_se2_correction,
    print_instance_conversion_debug,
    spawn_instance_actors,
    update_instance_actors,
    write_instance_csv,
)
from render_aligned_mapping_or_carla_path_gsplat import mapping_pose_matrix, read_mapping_poses
from render_background_gsplat import Profiler, load_background, load_intrinsics_from_processed, load_sky, render, resolve_device, write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=29001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--vertex-distance", type=float, default=2.0)
    parser.add_argument("--max-road-length", type=float, default=500.0)
    parser.add_argument("--wall-height", type=float, default=0.0)
    parser.add_argument("--additional-width", type=float, default=0.6)
    parser.add_argument("--no-smooth-junctions", action="store_true")
    parser.add_argument("--no-mesh-visibility", action="store_true")
    parser.add_argument("--core-start-frame", type=int)
    parser.add_argument("--core-end-frame", type=int)
    parser.add_argument("--instance-class-prefix", default="vehicle.")
    parser.add_argument("--no-live-instances", action="store_true", help="Do not spawn package instances in this runtime; useful when a separate traffic process owns them.")
    parser.add_argument("--max-instance-actors", type=int, default=0)
    parser.add_argument("--instance-z-offset", type=float, default=0.35)
    parser.add_argument("--no-instance-map-z", action="store_true")
    parser.add_argument("--instance-local-offset-x", type=float, default=0.0)
    parser.add_argument("--instance-local-offset-y", type=float, default=0.0)
    parser.add_argument("--instance-local-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--debug-instance-conversion", type=int, default=5)
    parser.add_argument("--hidden-actor-z", type=float, default=-500.0)
    parser.add_argument("--follow-start-z-offset", type=float, default=0.35)
    parser.add_argument("--follow-settle-ticks", type=int, default=30)
    parser.add_argument("--follow-speed-kmh", type=float, default=3.0)
    parser.add_argument("--follow-finish-distance", type=float, default=3.0)
    parser.add_argument("--follow-max-seconds", type=float, default=0.0)
    parser.add_argument("--exit-on-path-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--no-follow-map-z", action="store_true")
    parser.add_argument("--hero-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--hero-role-name", default="hero")
    parser.add_argument("--carla-z-mode", choices=("zero", "actor"), default="zero")
    parser.add_argument("--render-pose-source", choices=("actor", "reference"), default="actor")
    parser.add_argument("--vehicle-zrp-source", choices=("mapping-frame", "sequence-flat"), default="mapping-frame")
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--max-render-frames", type=int, default=0)
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--display-fps", type=float, default=10.0)
    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument("--spectator-height", type=float, default=55.0)
    parser.add_argument("--spectator-distance", type=float, default=35.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-pattern", default="live_{index:04d}_cam{camera}.png")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--pose-output", type=Path)
    parser.add_argument("--instance-output", type=Path)
    parser.add_argument("--collision-output", type=Path)
    parser.add_argument("--traffic-bbox-output", type=Path)
    parser.add_argument("--draw-traffic-bboxes", action="store_true")
    parser.add_argument("--traffic-bbox-life-time", type=float, default=0.08)
    parser.add_argument("--path-json-output", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-gaussians", type=int, default=2_000_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--crop-radius", type=float, default=180.0)
    parser.add_argument("--near", type=float, default=0.2)
    parser.add_argument("--far", type=float, default=250.0)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--downscale", type=float, default=2.0)
    parser.add_argument("--background-rgb", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--sh-degree", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def package_path(package_file: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else package_file.parent / path


def args_from_package(cli: argparse.Namespace) -> argparse.Namespace:
    package = json.loads(cli.scene_package.expanduser().read_text(encoding="utf-8"))
    assets = package.get("assets", {})
    frames = package.get("frames", {})
    values = package.get("values", {})
    data = vars(cli).copy()
    data.update(
        {
            "xodr": package_path(cli.scene_package, assets.get("xodr")),
            "background": package_path(cli.scene_package, assets.get("background")),
            "sky": package_path(cli.scene_package, assets.get("sky")),
            "processed_scene": package_path(cli.scene_package, assets.get("processed_scene")),
            "mapping_pose": package_path(cli.scene_package, assets.get("mapping_pose")),
            "instances_info": package_path(cli.scene_package, assets.get("instances_info")),
            "core_pose_csv": package_path(cli.scene_package, assets.get("core_pose_csv")),
            "sequence_origin_pose_file": package_path(cli.scene_package, assets.get("sequence_origin_pose_file")),
            "instance_origin_sequence_frame": frames.get("instance_origin_sequence_frame", 300),
            "processed_origin_frame": frames.get("processed_origin_frame", 0),
            "processed_origin_mapping_frame": frames.get("processed_origin_mapping_frame"),
            "sequence_origin_frame": frames.get("sequence_origin_frame", 0),
            "core_start_frame": cli.core_start_frame if cli.core_start_frame is not None else frames.get("core_start_frame"),
            "core_end_frame": cli.core_end_frame if cli.core_end_frame is not None else frames.get("core_end_frame"),
            "camera": values.get("camera", 0),
            "instance_transform_mode": values.get("instance_transform_mode", "mapping-absolute"),
            "core_pose_coordinate_frame": values.get("core_pose_coordinate_frame", "enu"),
            "no_core_y_flip": False,
            "core_pose_z": 0.5,
        }
    )
    return SimpleNamespace(**data)


class JsonLineControlServer:
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.sock.setblocking(False)
        self.conn: socket.socket | None = None
        self.file = None
        print(f"[control-server] listening on {host}:{port}; rendering starts without a client")

    def _try_accept(self) -> None:
        if self.conn is not None:
            return
        try:
            self.conn, addr = self.sock.accept()
        except BlockingIOError:
            return
        self.conn.setblocking(True)
        self.file = self.conn.makefile("rwb")
        print(f"[control-server] connected from {addr}")

    def _disconnect_client(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        print("[control-server] client disconnected; falling back to default brake control")

    @staticmethod
    def default_control() -> dict[str, float]:
        return {"throttle": 0.0, "steer": 0.0, "brake": 1.0}

    def request_control(self, observation: dict[str, Any]) -> dict[str, float]:
        self._try_accept()
        if self.file is None:
            return self.default_control()
        try:
            self.file.write((json.dumps(observation) + "\n").encode("utf-8"))
            self.file.flush()
            line = self.file.readline()
        except OSError:
            self._disconnect_client()
            return self.default_control()
        if not line:
            self._disconnect_client()
            return self.default_control()
        msg = json.loads(line.decode("utf-8"))
        return {
            "throttle": float(msg.get("throttle", 0.0)),
            "steer": float(msg.get("steer", 0.0)),
            "brake": float(msg.get("brake", 0.0)),
        }

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
        if self.conn is not None:
            self.conn.close()
        self.sock.close()


def write_pose_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_collision_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def traffic_bbox_records(instance_actors: list[dict], sequence_frame: int, step: int, hidden_actor_z: float) -> list[dict]:
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


def draw_traffic_bboxes(carla, world, objects: list[dict], life_time: float) -> None:
    color = carla.Color(255, 64, 64, 255)
    for item in objects:
        center = item["center"]
        extent = item["extent"]
        box = carla.BoundingBox(
            carla.Location(center["x"], center["y"], center["z"]),
            carla.Vector3D(extent["x"], extent["y"], extent["z"]),
        )
        world.debug.draw_box(box, carla.Rotation(yaw=item["yaw_deg"]), thickness=0.06, color=color, life_time=life_time)


def main() -> int:
    args = args_from_package(parse_args())
    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_abs_vehicle_origin = mapping_pose_matrix(mapping_poses, args.processed_origin_mapping_frame)
    t_abs_cam_origin = np.loadtxt(args.processed_scene / "extrinsics" / f"{args.processed_origin_frame:03d}_{args.camera}.txt").reshape(4, 4).astype(np.float32)
    t_camera_from_vehicle = np.linalg.inv(t_abs_vehicle_origin) @ t_abs_cam_origin
    t_3dgs_from_abs = np.linalg.inv(t_abs_cam_origin)
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)
    t_instance_anchor_abs = mapping_pose_matrix(mapping_poses, args.instance_origin_sequence_frame)
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

    carla = base.import_carla()
    client, world = base.load_xodr_world(carla, args)
    carla_map = world.get_map()
    core_poses = base.read_core_pose_sequence(args)
    hero = base.spawn_hero(carla, world, pose_dict_to_transform(carla, core_poses[0]), args.hero_blueprint, args.hero_role_name)

    original_settings = world.get_settings()
    sync_settings = world.get_settings()
    sync_settings.synchronous_mode = True
    sync_settings.fixed_delta_seconds = args.control_dt
    world.apply_settings(sync_settings)

    device = resolve_device(args.device)
    profiler = Profiler(args.profile, device)
    k, width, height = load_intrinsics_from_processed(args.processed_scene, args.camera, args.downscale)
    bg = load_background(args.background, device)
    sky = load_sky(args.sky, device)
    pygame, screen = base.init_pygame(args.pygame, width, height)
    display_delay = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    control_server = None
    pose_rows: list[dict] = []
    path_rows: list[dict] = []
    video_frames: list[np.ndarray] = []
    instance_rows: list[dict] = []
    collision_rows: list[dict] = []
    traffic_bbox_frames: list[dict] = []
    collision_events: list[dict] = []
    collision_sensor = None
    tracks: list[dict] = []
    instance_actors: list[dict] = []
    rendered = 0
    nearest_index = 0

    try:
        start_pose = dict(core_poses[0])
        if args.no_follow_map_z:
            start_pose["z"] = float(start_pose["z"]) + float(args.follow_start_z_offset)
            start_transform = pose_dict_to_transform(carla, start_pose)
        else:
            start_transform, _ = pose_dict_to_map_aligned_transform(carla, carla_map, start_pose, args.follow_start_z_offset)
        hero.set_simulate_physics(False)
        hero.set_transform(start_transform)
        world.tick()
        hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        hero.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        hero.set_simulate_physics(True)
        if args.instances_info is not None and not args.no_live_instances:
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
            instance_actors = spawn_instance_actors(carla, world, carla_map, tracks, t_seq_from_instance, args)
            print(f"[instances] spawned actors={len(instance_actors)}")
        collision_sensor, collision_events = base.attach_collision_sensor(carla, world, hero)
        for _ in range(max(0, args.follow_settle_ticks)):
            hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            world.tick()

        control_server = JsonLineControlServer(args.control_host, args.control_port)
        max_steps = None
        if args.follow_max_seconds > 0.0:
            max_steps = max(1, int(math.ceil(args.follow_max_seconds / max(args.control_dt, 1e-3))))

        step = 0
        while max_steps is None or step < max_steps:
            live_transform = hero.get_transform()
            nearest_index, nearest_dist = nearest_path_index(core_poses, live_transform.location.x, live_transform.location.y, nearest_index)
            source_frame = int(core_poses[nearest_index]["frame"])
            path_finished = nearest_index >= len(core_poses) - 1 or source_frame >= int(core_poses[-1]["frame"])
            current_instance_rows: list[dict] = []
            if instance_actors:
                current_instance_rows = update_instance_actors(
                    carla,
                    carla_map,
                    instance_actors,
                    source_frame,
                    t_seq_from_instance,
                    args,
                )
                instance_rows.extend(current_instance_rows)
                bbox_objects = traffic_bbox_records(instance_actors, source_frame, step, args.hidden_actor_z)
                if bbox_objects:
                    traffic_bbox_frames.append({"step": step, "sequence_frame": source_frame, "objects": bbox_objects})
                    if args.draw_traffic_bboxes:
                        draw_traffic_bboxes(carla, world, bbox_objects, args.traffic_bbox_life_time)
            actual = transform_to_dict(live_transform, source="runtime_observation")
            observation = {
                "step": step,
                "dt": args.control_dt,
                "source_frame": source_frame,
                "nearest_index": nearest_index,
                "nearest_dist": nearest_dist,
                "path_finished": path_finished,
                "pose": actual,
                "speed_mps": vehicle_speed_mps(hero),
            }
            cmd = control_server.request_control(observation)
            control = carla.VehicleControl(throttle=cmd["throttle"], steer=cmd["steer"], brake=cmd["brake"])
            hero.apply_control(control)
            collision_start = len(collision_events)
            world.tick()
            step_collision_events = collision_events[collision_start:]
            for event in step_collision_events:
                row = dict(event)
                row.update({"step": step, "elapsed_time": step * args.control_dt, "source_frame": source_frame})
                collision_rows.append(row)
            live_transform = hero.get_transform()
            actual = transform_to_dict(live_transform, source="external_control_runtime")
            actual.update(
                {
                    "step": step,
                    "elapsed_time": step * args.control_dt,
                    "frame": source_frame,
                    "speed_mps": observation["speed_mps"],
                    "speed_kmh": observation["speed_mps"] * 3.6,
                    "distance_error": nearest_dist,
                    "throttle": float(control.throttle),
                    "steer": float(control.steer),
                    "brake": float(control.brake),
                    "collision_count": len(collision_events),
                    "step_collision_count": len(step_collision_events),
                    "last_collision_actor_id": int(collision_events[-1]["other_actor_id"]) if collision_events else -1,
                    "last_collision_actor_type": collision_events[-1]["other_actor_type"] if collision_events else "",
                    "visible_instance_count": len(current_instance_rows),
                }
            )
            pose_rows.append(actual)

            should_render = step % args.render_every_n == 0 and not (args.max_render_frames > 0 and rendered >= args.max_render_frames)
            if should_render:
                if args.render_pose_source == "reference":
                    t_abs_vehicle = mapping_pose_matrix(mapping_poses, source_frame)
                    reference_frame = source_frame
                else:
                    t_seq_vehicle = base.carla_transform_to_sequence_matrix(live_transform, args.carla_z_mode)
                    t_abs_vehicle = t_abs_from_sequence @ t_seq_vehicle
                    if args.vehicle_zrp_source == "mapping-frame":
                        t_abs_vehicle = base.merge_xy_yaw_with_mapping_zrp(t_abs_vehicle, mapping_poses, source_frame)
                    reference_frame = None
                cam_to_world = t_3dgs_from_abs @ t_abs_vehicle @ t_camera_from_vehicle
                image, _ = render(bg, sky, cam_to_world, k, width, height, args, device, profiler)
                if args.video_output is not None:
                    video_frames.append(image)
                if args.output_dir is not None and not args.no_save_frames:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(args.output_dir / args.output_pattern.format(index=rendered, camera=args.camera))
                if pygame is not None and not base.show_pygame_frame(pygame, screen, image, display_delay):
                    break
                path_rows.append(
                    {
                        "render_index": rendered,
                        "step": step,
                        "elapsed_time": step * args.control_dt,
                        "frame": source_frame,
                        "reference_frame": reference_frame,
                        "visible_instance_count": len(current_instance_rows),
                        "collision_count": len(collision_rows),
                        "camera_to_3dgs_world": cam_to_world.tolist(),
                    }
                )
                rendered += 1
                print(f"[runtime] render={rendered} step={step} frame={source_frame} steer={control.steer:.3f}")

            if args.spectator_follow:
                move_spectator(carla, world, live_transform, args.spectator_height, args.spectator_distance)
            if args.exit_on_path_end and path_finished:
                break
            if args.exit_on_path_end and nearest_index >= len(core_poses) - 2 and nearest_dist <= args.follow_finish_distance:
                break
            step += 1
    except KeyboardInterrupt:
        print("[runtime] interrupted; writing outputs and shutting down")
    finally:
        hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        world.tick()
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()
        world.apply_settings(original_settings)
        if pygame is not None:
            pygame.quit()
        if control_server is not None:
            control_server.close()

    if args.pose_output:
        write_pose_csv(args.pose_output, pose_rows)
        print(f"Wrote {args.pose_output}")
    if args.instance_output:
        write_instance_csv(args.instance_output, instance_rows)
        print(f"Wrote {args.instance_output}")
    if args.collision_output:
        write_collision_csv(args.collision_output, collision_rows)
        print(f"Wrote {args.collision_output} collisions={len(collision_rows)}")
    if args.traffic_bbox_output:
        args.traffic_bbox_output.parent.mkdir(parents=True, exist_ok=True)
        args.traffic_bbox_output.write_text(
            json.dumps(
                {
                    "coordinate_frame": "carla",
                    "bbox_convention": "center + half extent; actor bounding box transformed by CARLA actor transform",
                    "frames": traffic_bbox_frames,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.traffic_bbox_output}")
    if args.path_json_output:
        args.path_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.path_json_output.write_text(
            json.dumps(
                {
                    "metadata": {
                        "instances_info": str(args.instances_info) if args.instances_info is not None else None,
                        "instance_origin_sequence_frame": args.instance_origin_sequence_frame,
                        "instance_transform_mode": args.instance_transform_mode,
                        "instance_local_offset_x": args.instance_local_offset_x,
                        "instance_local_offset_y": args.instance_local_offset_y,
                        "instance_local_yaw_offset_deg": args.instance_local_yaw_offset_deg,
                        "instance_tracks_loaded": len(tracks),
                        "instance_actors_spawned": len(instance_actors),
                        "collision_count": len(collision_rows),
                    },
                    "frames": path_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.path_json_output}")
    if args.video_output is not None:
        write_video(args.video_output, video_frames, args.video_fps)
        print(f"Wrote {args.video_output}")
    profiler.print()
    print(
        f"[runtime] steps={len(pose_rows)} rendered={rendered} "
        f"instance_actors={len(instance_actors)} instance_rows={len(instance_rows)} collisions={len(collision_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
