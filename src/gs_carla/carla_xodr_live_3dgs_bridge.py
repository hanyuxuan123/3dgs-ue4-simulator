#!/usr/bin/env python3
"""Run a CARLA OpenDRIVE closed loop and render the followed pose in 3DGS.

The bridge uses the same coordinate convention as the existing offline tools:

  CARLA OpenDRIVE world pose -> sequence-start local pose:
      (x, y, yaw)_seq = (x_carla, -y_carla, -yaw_carla)

  sequence-start local pose -> mapping absolute pose:
      T_abs_vehicle = T_abs_sequence_origin @ T_sequence_vehicle

  mapping absolute camera pose -> DriveStudio/3DGS local world:
      T_3dgs_camera = T_3dgs_from_abs @ T_abs_vehicle @ T_camera_from_vehicle

`T_camera_from_vehicle` and `T_3dgs_from_abs` are calibrated from a known
processed frame and its matching mapping_pose frame, exactly as in
render_aligned_mapping_or_carla_path_gsplat.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .load_xodr_town_and_dump_poses import (
    clamp,
    core_yaw_to_carla_yaw_deg,
    enu_to_sequence_start_local,
    import_carla,
    load_xodr_world,
    local_xy_from_transform,
    lookahead_path_index,
    move_spectator,
    nearest_path_index,
    normalize_angle,
    path_length,
    pose_dict_to_map_aligned_transform,
    pose_dict_to_transform,
    read_mapping_origin,
    spawn_hero,
    transform_to_dict,
    vehicle_speed_mps,
    xy_distance,
)
from .render_aligned_mapping_or_carla_path_gsplat import (
    mapping_pose_matrix,
    read_mapping_poses,
    transform_from_xyz_rpy,
)
from .render_background_gsplat import (
    Profiler,
    load_background,
    load_intrinsics_from_processed,
    load_sky,
    render,
    resolve_device,
    write_video,
)


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

    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--sky", type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--mapping-pose", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--processed-origin-frame", type=int, default=0)
    parser.add_argument(
        "--processed-origin-mapping-frame",
        type=int,
        required=True,
        help="mapping_pose lidar_frame matching --processed-origin-frame.",
    )
    parser.add_argument(
        "--sequence-origin-frame",
        type=int,
        default=0,
        help="mapping_pose frame defining the CARLA/XODR sequence-local origin.",
    )

    parser.add_argument("--core-pose-csv", required=True, type=Path)
    parser.add_argument("--core-start-frame", type=int, required=True)
    parser.add_argument("--core-end-frame", type=int, required=True)
    parser.add_argument(
        "--core-pose-coordinate-frame",
        choices=("corrected_local", "enu", "carla"),
        default="enu",
    )
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

    parser.add_argument(
        "--carla-z-mode",
        choices=("zero", "actor"),
        default="zero",
        help="Use zero sequence-local z for 3DGS, or use the live CARLA actor z.",
    )
    parser.add_argument(
        "--render-pose-source",
        choices=("actor", "reference"),
        default="actor",
        help=(
            "actor renders from the live CARLA hero transform. reference renders from the nearest "
            "core pose while CARLA still runs, useful for isolating controller/actor-origin errors."
        ),
    )
    parser.add_argument(
        "--vehicle-zrp-source",
        choices=("mapping-frame", "sequence-flat"),
        default="mapping-frame",
        help=(
            "For actor rendering, mapping-frame uses z/roll/pitch from the nearest mapping_pose frame "
            "and x/y/yaw from the live CARLA actor. sequence-flat keeps the older flat z=0, roll=pitch=0 path."
        ),
    )
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--max-render-frames", type=int, default=0)
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--display-fps", type=float, default=10.0)
    parser.add_argument("--loop-timing", action="store_true", help="Print coarse live-loop timing without render internals.")
    parser.add_argument("--loop-timing-every", type=int, default=10, help="Print loop timing every N rendered frames.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-pattern", default="live_{index:04d}_cam{camera}.png")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--pose-output", type=Path)
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


def read_core_pose_sequence(args: argparse.Namespace) -> list[dict]:
    origin = None
    if args.core_pose_coordinate_frame == "enu":
        origin_file = args.sequence_origin_pose_file or args.mapping_pose
        origin = read_mapping_origin(origin_file, args.sequence_origin_frame)
    start, end = sorted((args.core_start_frame, args.core_end_frame))
    poses: list[dict] = []
    with args.core_pose_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            if not start <= frame <= end:
                continue
            if args.core_pose_coordinate_frame == "enu":
                x_key = "enu_x" if "enu_x" in row else "x"
                y_key = "enu_y" if "enu_y" in row else "y"
                yaw_key = "enu_yaw" if "enu_yaw" in row else "yaw"
                local_x, local_y = enu_to_sequence_start_local(float(row[x_key]), float(row[y_key]), origin)
                local_yaw = normalize_angle(float(row[yaw_key]) - float(origin["yaw"]))
                carla_x, carla_y = (local_x, local_y) if args.no_core_y_flip else (local_x, -local_y)
                carla_yaw = core_yaw_to_carla_yaw_deg(args, local_yaw)
            elif args.core_pose_coordinate_frame == "corrected_local":
                x = float(row.get("corrected_x", row.get("x")))
                y = float(row.get("corrected_y", row.get("y")))
                yaw = float(row.get("corrected_yaw", row.get("yaw", 0.0)))
                carla_x, carla_y = (x, y) if args.no_core_y_flip else (x, -y)
                carla_yaw = core_yaw_to_carla_yaw_deg(args, yaw)
            else:
                x_key = "x" if "x" in row else "corrected_x"
                y_key = "y" if "y" in row else "corrected_y"
                yaw_value = float(row.get("yaw", row.get("corrected_yaw", 0.0)))
                carla_x, carla_y = float(row[x_key]), float(row[y_key])
                carla_yaw = math.degrees(yaw_value) if abs(yaw_value) <= 2.0 * math.pi else yaw_value
            poses.append(
                {
                    "frame": frame,
                    "x": carla_x,
                    "y": carla_y,
                    "z": args.core_pose_z,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": carla_yaw,
                }
            )
    poses.sort(key=lambda item: item["frame"])
    if len(poses) < 2:
        raise RuntimeError(f"Need at least two core poses in [{start}, {end}] from {args.core_pose_csv}")
    return poses


def make_path_follow_control(carla, args: argparse.Namespace, hero, target: dict, dt: float, pid_state: dict) -> tuple:
    transform = hero.get_transform()
    local_x, local_y = local_xy_from_transform(transform, target)
    lookahead = max(math.hypot(local_x, local_y), 1e-3)
    curvature = 2.0 * local_y / max(lookahead * lookahead, 1e-3)
    target_speed = max(args.follow_speed_kmh / 3.6, 0.0)
    speed = vehicle_speed_mps(hero)
    steer_limit = args.max_steer if speed >= 1.0 else min(args.max_steer, 0.25)
    steer = clamp(args.steer_gain * curvature * 2.8, -steer_limit, steer_limit)
    speed_error = target_speed - speed
    pid_state["integral"] = clamp(pid_state.get("integral", 0.0) + speed_error * dt, -5.0, 5.0)
    derivative = (speed_error - pid_state.get("previous_error", speed_error)) / max(dt, 1e-6)
    pid_state["previous_error"] = speed_error
    accel_cmd = args.speed_kp * speed_error + args.speed_ki * pid_state["integral"] + args.speed_kd * derivative
    throttle = clamp(accel_cmd, 0.0, 0.75)
    brake = clamp(-accel_cmd, 0.0, 0.8)
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake), speed, lateral_error(local_y)


def lateral_error(value: float) -> float:
    return float(value)


def carla_transform_to_sequence_matrix(transform, z_mode: str) -> np.ndarray:
    loc = transform.location
    rot = transform.rotation
    seq_x = float(loc.x)
    seq_y = -float(loc.y)
    seq_z = float(loc.z) if z_mode == "actor" else 0.0
    seq_yaw = math.radians(-float(rot.yaw))
    return transform_from_xyz_rpy(seq_x, seq_y, seq_z, 0.0, 0.0, seq_yaw)


def carla_pose_dict_to_sequence_matrix(pose: dict, z_mode: str) -> np.ndarray:
    seq_x = float(pose["x"])
    seq_y = -float(pose["y"])
    seq_z = float(pose.get("z", 0.0)) if z_mode == "actor" else 0.0
    seq_yaw = math.radians(-float(pose["yaw"]))
    return transform_from_xyz_rpy(seq_x, seq_y, seq_z, 0.0, 0.0, seq_yaw)


def yaw_from_matrix(mat: np.ndarray) -> float:
    return math.atan2(float(mat[1, 0]), float(mat[0, 0]))


def merge_xy_yaw_with_mapping_zrp(
    xy_yaw_abs: np.ndarray,
    mapping_poses: dict[int, dict[str, float]],
    frame: int,
) -> np.ndarray:
    if frame not in mapping_poses:
        return xy_yaw_abs
    ref = mapping_poses[frame]
    return transform_from_xyz_rpy(
        float(xy_yaw_abs[0, 3]),
        float(xy_yaw_abs[1, 3]),
        float(ref["z"]),
        float(ref["roll"]),
        float(ref["pitch"]),
        yaw_from_matrix(xy_yaw_abs),
    )


def init_pygame(enabled: bool, width: int, height: int):
    if not enabled:
        return None, None
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA XODR live 3DGS bridge")
    return pygame, screen


def show_pygame_frame(pygame, screen, image: np.ndarray, delay_s: float) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    if delay_s > 0:
        time.sleep(delay_s)
    return True


def write_pose_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.render_every_n <= 0:
        raise ValueError("--render-every-n must be positive")
    if args.no_save_frames and args.video_output is None and not args.pygame:
        raise ValueError("--no-save-frames needs --video-output or --pygame")

    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_abs_vehicle_origin = mapping_pose_matrix(mapping_poses, args.processed_origin_mapping_frame)
    t_abs_cam_origin = np.loadtxt(
        args.processed_scene / "extrinsics" / f"{args.processed_origin_frame:03d}_{args.camera}.txt"
    ).reshape(4, 4).astype(np.float32)
    t_camera_from_vehicle = np.linalg.inv(t_abs_vehicle_origin) @ t_abs_cam_origin
    t_3dgs_from_abs = np.linalg.inv(t_abs_cam_origin)
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)

    carla = import_carla()
    client, world = load_xodr_world(carla, args)
    carla_map = world.get_map()
    core_poses = read_core_pose_sequence(args)

    first_transform = pose_dict_to_transform(carla, core_poses[0])
    hero = spawn_hero(carla, world, first_transform, args.hero_blueprint, args.hero_role_name)
    print(f"spawned hero actor id={hero.id}")

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
    pygame, screen = init_pygame(args.pygame, width, height)
    display_delay = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    video_frames: list[np.ndarray] = []
    pose_rows: list[dict] = []
    path_rows: list[dict] = []

    start_pose = dict(core_poses[0])
    if args.no_follow_map_z:
        start_pose["z"] = float(start_pose["z"]) + float(args.follow_start_z_offset)
        start_transform = pose_dict_to_transform(carla, start_pose)
    else:
        start_transform, _ = pose_dict_to_map_aligned_transform(
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

    nearest_index = 0
    pid_state: dict = {}
    rendered = 0
    timing_rows: list[dict[str, float]] = []
    timing_every = max(1, args.loop_timing_every)
    t0 = time.perf_counter()
    try:
        for step in range(max_steps):
            loop_t0 = time.perf_counter()
            spectator_moved = False
            transform = hero.get_transform()
            nearest_index, nearest_dist = nearest_path_index(core_poses, transform.location.x, transform.location.y, nearest_index)
            target_index = lookahead_path_index(core_poses, nearest_index, args.follow_lookahead_distance)
            target = core_poses[target_index]
            control, speed, lat_error = make_path_follow_control(carla, args, hero, target, args.control_dt, pid_state)
            hero.apply_control(control)
            tick_t0 = time.perf_counter()
            world.tick()
            tick_time = time.perf_counter() - tick_t0

            live_transform = hero.get_transform()
            actual = transform_to_dict(live_transform, source="hero_controller")
            heading_error = normalize_angle(math.radians(float(target["yaw"])) - math.radians(float(actual["yaw"])))
            actual.update(
                {
                    "step": step,
                    "elapsed_time": step * args.control_dt,
                    "frame": int(core_poses[nearest_index]["frame"]),
                    "speed_mps": speed,
                    "speed_kmh": speed * 3.6,
                    "target_index": target_index,
                    "target_frame": int(target["frame"]),
                    "target_x": float(target["x"]),
                    "target_y": float(target["y"]),
                    "target_yaw": float(target["yaw"]),
                    "distance_error": nearest_dist,
                    "lateral_error": lat_error,
                    "heading_error_deg": math.degrees(heading_error),
                    "throttle": float(control.throttle),
                    "steer": float(control.steer),
                    "brake": float(control.brake),
                }
            )
            pose_rows.append(actual)

            should_render = step % args.render_every_n == 0
            if args.max_render_frames > 0 and rendered >= args.max_render_frames:
                should_render = False
            if should_render:
                pose_t0 = time.perf_counter()
                source_frame = int(core_poses[nearest_index]["frame"])
                if args.render_pose_source == "reference":
                    render_source_pose = core_poses[nearest_index]
                    t_abs_vehicle = mapping_pose_matrix(mapping_poses, source_frame)
                else:
                    render_source_pose = None
                    t_seq_vehicle = carla_transform_to_sequence_matrix(live_transform, args.carla_z_mode)
                    t_abs_vehicle = t_abs_from_sequence @ t_seq_vehicle
                    if args.vehicle_zrp_source == "mapping-frame":
                        t_abs_vehicle = merge_xy_yaw_with_mapping_zrp(t_abs_vehicle, mapping_poses, source_frame)
                cam_to_world = t_3dgs_from_abs @ t_abs_vehicle @ t_camera_from_vehicle
                pose_time = time.perf_counter() - pose_t0
                render_t0 = time.perf_counter()
                image, _ = render(bg, sky, cam_to_world, k, width, height, args, device, profiler)
                render_time = time.perf_counter() - render_t0
                output_t0 = time.perf_counter()
                if args.video_output is not None:
                    video_frames.append(image)
                if args.output_dir is not None and not args.no_save_frames:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    out = args.output_dir / args.output_pattern.format(index=rendered, camera=args.camera)
                    Image.fromarray(image).save(out)
                output_time = time.perf_counter() - output_t0
                display_time = 0.0
                if pygame is not None and not show_pygame_frame(pygame, screen, image, display_delay):
                    break
                if pygame is not None:
                    display_time = time.perf_counter() - output_t0 - output_time
                path_rows.append(
                    {
                        "render_index": rendered,
                        "step": step,
                        "frame": source_frame,
                        "render_pose_source": args.render_pose_source,
                        "vehicle_zrp_source": args.vehicle_zrp_source,
                        "reference_frame": int(render_source_pose["frame"]) if render_source_pose is not None else None,
                        "camera_to_3dgs_world": cam_to_world.tolist(),
                    }
                )
                rendered += 1
                elapsed = time.perf_counter() - t0
                spectator_time = 0.0
                if args.spectator_follow:
                    spectator_t0 = time.perf_counter()
                    move_spectator(carla, world, live_transform, args.spectator_height, args.spectator_distance)
                    spectator_time = time.perf_counter() - spectator_t0
                    spectator_moved = True
                loop_time = time.perf_counter() - loop_t0
                if args.loop_timing:
                    timing_rows.append(
                        {
                            "tick": tick_time,
                            "pose": pose_time,
                            "render": render_time,
                            "output": output_time,
                            "display": display_time,
                            "spectator": spectator_time,
                            "loop": loop_time,
                        }
                    )
                    if rendered % timing_every == 0:
                        recent = timing_rows[-timing_every:]
                        avg = {
                            key: sum(row[key] for row in recent) / len(recent)
                            for key in recent[0]
                        }
                        print(
                            "[live-3dgs-timing] "
                            f"last={len(recent)} "
                            f"tick={avg['tick']:.4f}s pose={avg['pose']:.4f}s "
                            f"render={avg['render']:.4f}s output={avg['output']:.4f}s "
                            f"display={avg['display']:.4f}s spectator={avg['spectator']:.4f}s "
                            f"loop={avg['loop']:.4f}s"
                        )
                print(
                    f"[live-3dgs] render {rendered} step={step} "
                    f"target_frame={target['frame']} speed_kmh={speed * 3.6:.2f} "
                    f"track_error_m={nearest_dist:.3f} avg_render_fps={rendered / max(elapsed, 1e-9):.2f}"
                )

            if args.spectator_follow and not spectator_moved:
                move_spectator(carla, world, live_transform, args.spectator_height, args.spectator_distance)
            if nearest_index >= len(core_poses) - 2 and nearest_dist <= args.follow_finish_distance:
                break
    finally:
        hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        world.tick()
        world.apply_settings(original_settings)
        if pygame is not None:
            pygame.quit()

    if args.pose_output:
        write_pose_csv(args.pose_output, pose_rows)
        print(f"Wrote {args.pose_output}")
    if args.path_json_output:
        args.path_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.path_json_output.write_text(
            json.dumps(
                {
                    "metadata": {
                        "xodr": str(args.xodr),
                        "processed_scene": str(args.processed_scene),
                        "mapping_pose": str(args.mapping_pose),
                        "camera": args.camera,
                        "processed_origin_frame": args.processed_origin_frame,
                        "processed_origin_mapping_frame": args.processed_origin_mapping_frame,
                        "sequence_origin_frame": args.sequence_origin_frame,
                        "carla_z_mode": args.carla_z_mode,
                        "render_pose_source": args.render_pose_source,
                        "vehicle_zrp_source": args.vehicle_zrp_source,
                    },
                    "frames": path_rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {args.path_json_output}")
    if args.video_output is not None:
        write_video(args.video_output, video_frames, args.video_fps)
        print(f"Wrote {args.video_output}")
    profiler.print()
    print(f"[live-3dgs] controller_steps={len(pose_rows)} rendered_frames={rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
