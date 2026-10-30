#!/usr/bin/env python3
"""Start CARLA/XODR + 3DGS and accept vehicle controls from an external process."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import select
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from . import carla_xodr_live_3dgs_bridge_with_instances as base
from .load_xodr_town_and_dump_poses import (
    move_spectator,
    nearest_path_index,
    pose_dict_to_map_aligned_transform,
    pose_dict_to_transform,
    transform_to_dict,
    vehicle_speed_mps,
)
from .load_xodr_town_with_instances import (
    load_instance_tracks,
    make_instance_transform,
    make_local_se2_correction,
    print_instance_conversion_debug,
    spawn_instance_actors,
    update_instance_actors,
    write_instance_csv,
)
from .render_aligned_mapping_or_carla_path_gsplat import mapping_pose_matrix, read_mapping_poses
from .render_background_gsplat import (
    Profiler,
    concat_gaussians,
    load_background,
    load_intrinsics_from_processed,
    load_rigid_nodes,
    load_sky,
    render,
    resolve_device,
    rigid_nodes_for_frame,
    write_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=29001)
    parser.add_argument(
        "--disable-control-server",
        action="store_true",
        help="Do not bind a TCP control socket; use default brake control every step.",
    )
    parser.add_argument(
        "--wait-for-control-client",
        action="store_true",
        help="Block at each control step until a JSON-line control client is connected.",
    )
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
    parser.add_argument("--cameras", nargs="+", type=int, help="Camera ids to render into one pygame/video mosaic, e.g. --cameras 0 1 2 3 4 5.")
    parser.add_argument("--dynamic-rigid", type=Path, help="Exported RigidNodes pth to splat with the background.")
    parser.add_argument(
        "--render-dynamic-gaussians",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Splat checkpoint RigidNodes dynamic Gaussians in the 3DGS view; does not affect live CARLA instances.",
    )
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
    parser.add_argument(
        "--carla-pose-frame",
        choices=("sequence-local", "mapping-absolute", "opendrive-global"),
        default="sequence-local",
        help=(
            "How to interpret live CARLA actor x/y/yaw before 3DGS rendering. "
            "sequence-local uses the ClassLab/OpenDRIVE y-flip convention. "
            "mapping-absolute uses CARLA x/y/yaw directly as mapping/global coordinates. "
            "opendrive-global converts CARLA's imported OpenDRIVE pose back with x, -y, -yaw."
        ),
    )
    parser.add_argument("--render-pose-source", choices=("actor", "reference"), default="actor")
    parser.add_argument(
        "--render-fixed-source-frame",
        type=int,
        help=(
            "Use this fixed mapping/core frame for reference rendering, or for actor render z/roll/pitch "
            "when --vehicle-zrp-source mapping-frame is used, while source_frame can still advance for "
            "time-varying instances or logging."
        ),
    )
    parser.add_argument(
        "--source-frame-mode",
        choices=("nearest", "timeline"),
        default="nearest",
        help=(
            "nearest uses the CARLA hero's nearest reference pose as source_frame. "
            "timeline advances source_frame by elapsed simulation time, useful for moving 3DGS/instances while the hero is frozen."
        ),
    )
    parser.add_argument("--source-frame-fps", type=float, default=10.0)
    parser.add_argument(
        "--freeze-hero",
        action="store_true",
        help="Keep the CARLA hero braked while source_frame/rendering can still advance in timeline mode.",
    )
    parser.add_argument(
        "--kinematic-core-replay",
        action="store_true",
        help="Move the hero by set_transform along core poses instead of using physics controls.",
    )
    parser.add_argument("--vehicle-zrp-source", choices=("mapping-frame", "sequence-flat"), default="mapping-frame")
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--max-render-frames", type=int, default=0)
    parser.add_argument(
        "--exit-after-max-render-frames",
        action="store_true",
        help="Exit the runtime as soon as --max-render-frames has been rendered.",
    )
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--display-fps", type=float, default=10.0)
    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument("--spectator-height", type=float, default=55.0)
    parser.add_argument("--spectator-distance", type=float, default=35.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-pattern", default="live_{index:04d}_cam{camera}.png")
    parser.add_argument(
        "--publish-rendered-observations",
        action="store_true",
        help=(
            "Publish the latest rendered multi-camera image paths and calibration in each control observation. "
            "Images are written to --output-dir even when --no-save-frames is set."
        ),
    )
    parser.add_argument(
        "--publish-expert-state",
        action="store_true",
        help="Publish a privileged route window and live traffic boxes for expert-data collection.",
    )
    parser.add_argument(
        "--expert-route-points",
        type=int,
        default=80,
        help="Number of reference poses ahead of the ego included in expert_state.route.",
    )
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--pose-output", type=Path)
    parser.add_argument("--instance-output", type=Path)
    parser.add_argument("--collision-output", type=Path)
    parser.add_argument("--episode-result-output", type=Path, help="Write machine-readable episode termination facts as JSON.")
    parser.add_argument("--terminate-on-collision", action="store_true")
    parser.add_argument(
        "--collision-impulse-threshold",
        type=float,
        default=0.0,
        help="Only collision events with impulse_norm at or above this value terminate the episode.",
    )
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
            "dynamic_rigid": cli.dynamic_rigid
            or package_path(cli.scene_package, assets.get("dynamic_rigid"))
            or package_path(cli.scene_package, "rigid_nodes_gaussians.pth"),
            "core_pose_csv": package_path(cli.scene_package, assets.get("core_pose_csv")),
            "sequence_origin_pose_file": package_path(cli.scene_package, assets.get("sequence_origin_pose_file")),
            "instance_origin_sequence_frame": frames.get("instance_origin_sequence_frame", 300),
            "processed_origin_frame": frames.get("processed_origin_frame", 0),
            "processed_origin_mapping_frame": frames.get("processed_origin_mapping_frame"),
            "sequence_origin_frame": frames.get("sequence_origin_frame", 0),
            "core_start_frame": cli.core_start_frame if cli.core_start_frame is not None else frames.get("core_start_frame"),
            "core_end_frame": cli.core_end_frame if cli.core_end_frame is not None else frames.get("core_end_frame"),
            "camera": values.get("camera", 0),
            "cameras": cli.cameras if cli.cameras is not None else [values.get("camera", 0)],
            "carla_pose_frame": values.get("carla_pose_frame", cli.carla_pose_frame),
            "gaussian_coordinate_frame": values.get("gaussian_coordinate_frame", "drive_studio_local"),
            "actor_camera_mount_frame": values.get("actor_camera_mount_frame", "mapping-pose"),
            "instance_transform_mode": values.get("instance_transform_mode", "mapping-absolute"),
            "core_pose_coordinate_frame": values.get("core_pose_coordinate_frame", "enu"),
            "no_core_y_flip": False,
            "core_pose_z": 0.5,
        }
    )
    return SimpleNamespace(**data)


class JsonLineControlServer:
    def __init__(self, host: str, port: int, wait_for_client: bool = False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.sock.setblocking(False)
        self.conn: socket.socket | None = None
        self.file = None
        self.wait_for_client = wait_for_client
        behavior = "waiting for a client" if wait_for_client else "rendering starts without a client"
        print(f"[control-server] listening on {host}:{port}; {behavior}")

    def _try_accept(self) -> None:
        if self.conn is not None:
            return
        if self.wait_for_client:
            select.select([self.sock], [], [])
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
        if self.wait_for_client:
            print("[control-server] client disconnected; waiting for a new client")
        else:
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


def make_mosaic(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]
    rows = []
    for start in range(0, len(images), 3):
        row = images[start : start + 3]
        if len(row) < 3:
            pad = np.zeros_like(images[0])
            row = row + [pad] * (3 - len(row))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def carla_transform_to_mapping_absolute_matrix(transform, z_mode: str, z_offset: float = 0.0) -> np.ndarray:
    loc = transform.location
    rot = transform.rotation
    z_value = float(loc.z) if z_mode == "actor" else 0.0
    return base.transform_from_xyz_rpy(
        float(loc.x),
        float(loc.y),
        z_value + float(z_offset),
        0.0,
        0.0,
        math.radians(float(rot.yaw)),
    )


def carla_transform_to_opendrive_global_matrix(transform, z_mode: str, z_offset: float = 0.0) -> np.ndarray:
    loc = transform.location
    rot = transform.rotation
    z_value = float(loc.z) if z_mode == "actor" else 0.0
    return base.transform_from_xyz_rpy(
        float(loc.x),
        -float(loc.y),
        z_value + float(z_offset),
        0.0,
        0.0,
        math.radians(-float(rot.yaw)),
    )


def actor_from_lidar_frame_correction() -> np.ndarray:
    correction = np.eye(4, dtype=np.float32)
    angle = -0.5 * math.pi
    c, s = math.cos(angle), math.sin(angle)
    correction[:3, :3] = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return correction


def merge_z_with_mapping_pose(
    xy_yaw_abs: np.ndarray,
    mapping_poses: dict[int, dict[str, float]],
    frame: int,
) -> np.ndarray:
    if frame not in mapping_poses:
        return xy_yaw_abs
    out = np.array(xy_yaw_abs, copy=True)
    out[2, 3] = float(mapping_poses[frame]["z"])
    return out


def main() -> int:
    args = args_from_package(parse_args())
    if args.publish_rendered_observations and args.output_dir is None:
        raise ValueError("--publish-rendered-observations requires --output-dir")
    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_abs_vehicle_origin = mapping_pose_matrix(mapping_poses, args.processed_origin_mapping_frame)
    t_abs_cam_origin = np.loadtxt(args.processed_scene / "extrinsics" / f"{args.processed_origin_frame:03d}_{args.camera}.txt").reshape(4, 4).astype(np.float32)
    if args.gaussian_coordinate_frame == "mapping-absolute":
        t_3dgs_from_abs = np.eye(4, dtype=np.float32)
    else:
        t_3dgs_from_abs = np.linalg.inv(t_abs_cam_origin)
    core_poses = base.read_core_pose_sequence(args)
    t_actor_from_lidar = actor_from_lidar_frame_correction()
    camera_models = []
    for camera in args.cameras:
        t_abs_cam = np.loadtxt(args.processed_scene / "extrinsics" / f"{args.processed_origin_frame:03d}_{camera}.txt").reshape(4, 4).astype(np.float32)
        k, width, height = load_intrinsics_from_processed(args.processed_scene, camera, args.downscale)
        camera_models.append(
            {
                "camera": int(camera),
                "t_camera_from_vehicle": np.linalg.inv(t_abs_vehicle_origin) @ t_abs_cam,
                "k": k,
                "width": width,
                "height": height,
            }
        )
    if not camera_models:
        raise ValueError("--cameras must contain at least one camera id")
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)
    t_instance_anchor_abs = mapping_pose_matrix(mapping_poses, args.instance_origin_sequence_frame)
    # The global nuScenes runtime replays the hero directly in CARLA/OpenDRIVE
    # global coordinates (see prepare_nuscenes_global_scene.write_core_pose).
    # Therefore a mapping-absolute instance matrix is already in the required
    # CARLA X/Y frame; carla_pose_from_instance_matrix below performs the only
    # needed convention conversion (the CARLA Y flip).  Applying
    # inv(t_abs_from_sequence) here would incorrectly move an actor near the
    # local origin, kilometres away from the global-coordinate hero.
    if args.instance_transform_mode == "mapping-absolute":
        t_seq_from_instance = np.eye(4, dtype=np.float32)
    else:
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
    hero = base.spawn_hero(carla, world, pose_dict_to_transform(carla, core_poses[0]), args.hero_blueprint, args.hero_role_name)

    original_settings = world.get_settings()
    sync_settings = world.get_settings()
    sync_settings.synchronous_mode = True
    sync_settings.fixed_delta_seconds = args.control_dt
    world.apply_settings(sync_settings)

    device = resolve_device(args.device)
    profiler = Profiler(args.profile, device)
    bg = load_background(args.background, device)
    rigid = load_rigid_nodes(args.dynamic_rigid, device)
    has_dynamic_gaussians = (
        args.render_dynamic_gaussians and rigid is not None and int(rigid["means"].shape[0]) > 0
    )
    if rigid is not None and not args.render_dynamic_gaussians:
        print("[runtime] checkpoint dynamic Gaussians disabled; rendering background/sky only.")
    sky = load_sky(args.sky, device)
    first_width, first_height = camera_models[0]["width"], camera_models[0]["height"]
    mosaic_cols = min(len(camera_models), 3)
    mosaic_rows = int(math.ceil(len(camera_models) / 3))
    pygame, screen = base.init_pygame(args.pygame, first_width * mosaic_cols, first_height * mosaic_rows)
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
    latest_sensor_packet: dict[str, Any] | None = None
    nearest_index = 0
    termination_reason = "unknown"
    if args.instances_info is not None and not args.no_live_instances and rigid is None:
        print("[runtime] note: --live-instances replays CARLA actors for traffic state/bboxes; no exported RigidNodes dynamic gaussians were found for splatting.")

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
        hero.set_simulate_physics(not args.kinematic_core_replay)
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

        actor_global_z_offset = 0.0
        if args.render_pose_source == "actor":
            origin_frame = int(core_poses[0]["frame"])
            t_origin_abs = mapping_pose_matrix(mapping_poses, origin_frame)
            origin_transform = hero.get_transform()
            if args.carla_pose_frame == "mapping-absolute":
                t_actor_origin_abs = carla_transform_to_mapping_absolute_matrix(origin_transform, "actor")
            elif args.carla_pose_frame == "opendrive-global":
                t_actor_origin_abs = carla_transform_to_opendrive_global_matrix(origin_transform, "actor")
            else:
                t_actor_origin_abs = t_abs_from_sequence @ base.carla_transform_to_sequence_matrix(origin_transform, "actor")
            actor_global_z_offset = float(t_origin_abs[2, 3] - t_actor_origin_abs[2, 3])
            print(
                "[runtime] actor pose drives 3DGS camera; "
                f"source_frame is content-only; actor_global_z_offset={actor_global_z_offset:.6f}"
            )

        if not args.disable_control_server:
            control_server = JsonLineControlServer(args.control_host, args.control_port, args.wait_for_control_client)
        else:
            print("[control-server] disabled; using default brake control")
        max_steps = None
        if args.follow_max_seconds > 0.0:
            max_steps = max(1, int(math.ceil(args.follow_max_seconds / max(args.control_dt, 1e-3))))

        step = 0
        while max_steps is None or step < max_steps:
            live_transform = hero.get_transform()
            nearest_index, nearest_dist = nearest_path_index(core_poses, live_transform.location.x, live_transform.location.y, nearest_index)
            if args.source_frame_mode == "timeline":
                timeline_index = min(
                    int(math.floor(step * args.control_dt * max(args.source_frame_fps, 1e-6))),
                    len(core_poses) - 1,
                )
                content_source_index = timeline_index
            else:
                content_source_index = nearest_index
            source_frame = int(core_poses[content_source_index]["frame"])
            if args.kinematic_core_replay:
                replay_pose = dict(core_poses[content_source_index])
                if args.no_follow_map_z:
                    replay_pose["z"] = float(replay_pose["z"]) + float(args.follow_start_z_offset)
                    replay_transform = pose_dict_to_transform(carla, replay_pose)
                else:
                    replay_transform, _ = pose_dict_to_map_aligned_transform(carla, carla_map, replay_pose, args.follow_start_z_offset)
                hero.set_transform(replay_transform)
                live_transform = hero.get_transform()
                nearest_index = content_source_index
                nearest_dist = 0.0
            end_distance = math.hypot(
                float(core_poses[-1]["x"]) - float(live_transform.location.x),
                float(core_poses[-1]["y"]) - float(live_transform.location.y),
            )
            progress_frame = int(core_poses[nearest_index]["frame"])
            path_finished = nearest_index >= len(core_poses) - 2 and end_distance <= args.follow_finish_distance
            current_instance_rows: list[dict] = []
            bbox_objects: list[dict] = []
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
            acceleration = hero.get_acceleration()
            angular_velocity = hero.get_angular_velocity()
            yaw_rad = math.radians(float(live_transform.rotation.yaw))
            yaw_cos, yaw_sin = math.cos(yaw_rad), math.sin(yaw_rad)
            acceleration_local = [
                yaw_cos * float(acceleration.x) + yaw_sin * float(acceleration.y),
                -yaw_sin * float(acceleration.x) + yaw_cos * float(acceleration.y),
                float(acceleration.z),
            ]
            angular_velocity_local = [
                yaw_cos * float(angular_velocity.x) + yaw_sin * float(angular_velocity.y),
                -yaw_sin * float(angular_velocity.x) + yaw_cos * float(angular_velocity.y),
                float(angular_velocity.z),
            ]
            observation = {
                "step": step,
                "dt": args.control_dt,
                "source_frame": source_frame,
                "actor_progress_frame": progress_frame,
                "nearest_index": nearest_index,
                "nearest_dist": nearest_dist,
                "path_finished": path_finished,
                "pose": actual,
                "speed_mps": vehicle_speed_mps(hero),
                "acceleration_mps2": acceleration_local,
                "angular_velocity_dps": angular_velocity_local,
                # Match CARLA's IMU compass convention: north=0, east=pi/2.
                "compass_rad": math.atan2(
                    math.sin(math.pi / 2.0 + yaw_rad),
                    math.cos(math.pi / 2.0 + yaw_rad),
                ),
                "sensor_packet": latest_sensor_packet,
            }
            if args.publish_expert_state:
                route_end = min(len(core_poses), nearest_index + max(1, args.expert_route_points))
                observation["expert_state"] = {
                    "route": [
                        {
                            "frame": int(pose["frame"]),
                            "x": float(pose["x"]),
                            "y": float(pose["y"]),
                            "z": float(pose.get("z", 0.0)),
                            "yaw": float(pose.get("yaw", 0.0)),
                        }
                        for pose in core_poses[nearest_index:route_end]
                    ],
                    "traffic_objects": bbox_objects,
                }
            cmd = control_server.request_control(observation) if control_server is not None else JsonLineControlServer.default_control()
            control = carla.VehicleControl(throttle=cmd["throttle"], steer=cmd["steer"], brake=cmd["brake"])
            if args.freeze_hero or args.kinematic_core_replay:
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
            hero.apply_control(control)
            collision_start = len(collision_events)
            world.tick()
            step_collision_events = collision_events[collision_start:]
            for event in step_collision_events:
                row = dict(event)
                row.update({"step": step, "elapsed_time": step * args.control_dt, "source_frame": source_frame})
                collision_rows.append(row)
            live_transform = hero.get_transform()
            nearest_index, nearest_dist = nearest_path_index(core_poses, live_transform.location.x, live_transform.location.y, nearest_index)
            end_distance = math.hypot(
                float(core_poses[-1]["x"]) - float(live_transform.location.x),
                float(core_poses[-1]["y"]) - float(live_transform.location.y),
            )
            progress_frame = int(core_poses[nearest_index]["frame"])
            path_finished = nearest_index >= len(core_poses) - 2 and end_distance <= args.follow_finish_distance
            actual = transform_to_dict(live_transform, source="external_control_runtime")
            actual.update(
                {
                    "step": step,
                    "elapsed_time": step * args.control_dt,
                    "frame": progress_frame,
                    "source_frame": source_frame,
                    "source_frame_mode": args.source_frame_mode,
                    "source_index": content_source_index,
                    "nearest_index": nearest_index,
                    "actor_progress_frame": progress_frame,
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
                dynamic_frame = int(source_frame) - int(args.processed_origin_mapping_frame)
                dynamic_gaussians = rigid_nodes_for_frame(rigid, dynamic_frame) if has_dynamic_gaussians else None
                render_gaussians = concat_gaussians(bg, dynamic_gaussians)
                if args.render_pose_source == "reference":
                    render_frame = args.render_fixed_source_frame if args.render_fixed_source_frame is not None else source_frame
                    t_abs_vehicle = mapping_pose_matrix(mapping_poses, render_frame)
                    reference_frame = render_frame
                else:
                    if args.carla_pose_frame == "mapping-absolute":
                        t_abs_vehicle = carla_transform_to_mapping_absolute_matrix(
                            live_transform,
                            "actor",
                            actor_global_z_offset,
                        )
                    elif args.carla_pose_frame == "opendrive-global":
                        t_abs_vehicle = carla_transform_to_opendrive_global_matrix(
                            live_transform,
                            "actor",
                            actor_global_z_offset,
                        )
                    else:
                        t_seq_vehicle = base.carla_transform_to_sequence_matrix(live_transform, "actor")
                        t_abs_vehicle = t_abs_from_sequence @ t_seq_vehicle
                        t_abs_vehicle[2, 3] += actor_global_z_offset
                    reference_frame = None
                images = []
                camera_rows = []
                image_paths: dict[str, str] = {}
                for model in camera_models:
                    camera_from_vehicle = (
                        model["t_camera_from_vehicle"]
                        if args.render_pose_source == "reference" or args.actor_camera_mount_frame == "mapping-pose"
                        else t_actor_from_lidar @ model["t_camera_from_vehicle"]
                    )
                    cam_to_world = t_3dgs_from_abs @ t_abs_vehicle @ camera_from_vehicle
                    image, _ = render(render_gaussians, sky, cam_to_world, model["k"], model["width"], model["height"], args, device, profiler)
                    images.append(image)
                    camera_rows.append(
                        {
                            "camera": model["camera"],
                            "camera_to_3dgs_world": cam_to_world.tolist(),
                            "intrinsic": model["k"].tolist(),
                            "width": int(model["width"]),
                            "height": int(model["height"]),
                        }
                    )
                    if args.output_dir is not None and (not args.no_save_frames or args.publish_rendered_observations):
                        args.output_dir.mkdir(parents=True, exist_ok=True)
                        if args.publish_rendered_observations and args.no_save_frames:
                            image_path = args.output_dir / f"agent_latest_cam{model['camera']}.png"
                        else:
                            image_path = args.output_dir / args.output_pattern.format(index=rendered, camera=model["camera"])
                        Image.fromarray(image).save(image_path)
                        image_paths[str(model["camera"])] = str(image_path.resolve())
                if args.publish_rendered_observations:
                    latest_sensor_packet = {
                        "step": step,
                        "render_index": rendered,
                        "source_frame": source_frame,
                        "images": image_paths,
                        "cameras": camera_rows,
                    }
                image = make_mosaic(images)
                if args.video_output is not None:
                    video_frames.append(image)
                if pygame is not None and not base.show_pygame_frame(pygame, screen, image, display_delay):
                    break
                path_rows.append(
                    {
                        "render_index": rendered,
                        "step": step,
                        "elapsed_time": step * args.control_dt,
                        "frame": progress_frame,
                        "source_frame": source_frame,
                        "actor_progress_frame": progress_frame,
                        "reference_frame": reference_frame,
                        "render_fixed_source_frame": args.render_fixed_source_frame,
                        "visible_instance_count": len(current_instance_rows),
                        "collision_count": len(collision_rows),
                        "cameras": camera_rows,
                        "camera_to_3dgs_world": camera_rows[0]["camera_to_3dgs_world"],
                    }
                )
                rendered += 1
                print(
                    f"[runtime] render={rendered} step={step} actor_frame={progress_frame} "
                    f"source_frame={source_frame} steer={control.steer:.3f}"
                )
                if args.exit_after_max_render_frames and args.max_render_frames > 0 and rendered >= args.max_render_frames:
                    break

            if args.spectator_follow:
                move_spectator(carla, world, live_transform, args.spectator_height, args.spectator_distance)
            if args.terminate_on_collision and any(
                float(event.get("impulse_norm", 0.0)) >= args.collision_impulse_threshold
                for event in step_collision_events
            ):
                termination_reason = "collision"
                break
            if args.exit_on_path_end and path_finished:
                termination_reason = "route_completed"
                break
            step += 1
        if termination_reason == "unknown":
            termination_reason = "timeout" if max_steps is not None and step >= max_steps else "runtime_stopped"
    except KeyboardInterrupt:
        termination_reason = "interrupted"
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
        destroy_ids = []
        if hero is not None:
            destroy_ids.append(int(hero.id))
        destroy_ids.extend(int(entry["actor"].id) for entry in instance_actors if entry.get("actor") is not None)
        if destroy_ids:
            try:
                client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in destroy_ids])
            except Exception as exc:
                print(f"WARNING: failed to destroy actors {destroy_ids}: {exc}")

    if args.pose_output:
        write_pose_csv(args.pose_output, pose_rows)
        print(f"Wrote {args.pose_output}")
    if args.instance_output:
        write_instance_csv(args.instance_output, instance_rows)
        print(f"Wrote {args.instance_output}")
    if args.collision_output:
        write_collision_csv(args.collision_output, collision_rows)
        print(f"Wrote {args.collision_output} collisions={len(collision_rows)}")
    if args.episode_result_output:
        route_distances = [0.0]
        for previous, current in zip(core_poses, core_poses[1:]):
            route_distances.append(
                route_distances[-1]
                + math.hypot(float(current["x"]) - float(previous["x"]), float(current["y"]) - float(previous["y"]))
            )
        total_route_m = route_distances[-1]
        completed_route_m = route_distances[min(nearest_index, len(route_distances) - 1)]
        completion = 100.0 if total_route_m <= 1e-6 else 100.0 * completed_route_m / total_route_m
        if termination_reason == "route_completed":
            completed_route_m = total_route_m
            completion = 100.0
        qualifying_collisions = [
            event
            for event in collision_rows
            if float(event.get("impulse_norm", 0.0)) >= args.collision_impulse_threshold
        ]
        result = {
            "schema": "carla_3dgs_episode_result_v1",
            "termination_reason": termination_reason,
            "route_completion_percent": max(0.0, min(100.0, completion)),
            "route_completed_m": completed_route_m,
            "route_total_m": total_route_m,
            "final_nearest_index": nearest_index,
            "route_pose_count": len(core_poses),
            "final_distance_to_route_m": nearest_dist,
            "final_distance_to_endpoint_m": end_distance,
            "elapsed_time_s": float(pose_rows[-1]["elapsed_time"]) if pose_rows else 0.0,
            "steps": len(pose_rows),
            "rendered_frames": rendered,
            "collision_count": len(qualifying_collisions),
            "all_collision_event_count": len(collision_rows),
            "collision_impulse_threshold": args.collision_impulse_threshold,
            "traffic_manager_enabled": False,
            "traffic_source": "deterministic_instances_info" if args.instances_info is not None else "none",
            "success": termination_reason == "route_completed" and not qualifying_collisions,
        }
        args.episode_result_output.parent.mkdir(parents=True, exist_ok=True)
        args.episode_result_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.episode_result_output}: reason={termination_reason} success={result['success']}")
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
                        "source_frame_mode": args.source_frame_mode,
                        "source_frame_fps": args.source_frame_fps,
                        "render_fixed_source_frame": args.render_fixed_source_frame,
                        "freeze_hero": args.freeze_hero,
                        "instance_local_offset_x": args.instance_local_offset_x,
                        "instance_local_offset_y": args.instance_local_offset_y,
                        "instance_local_yaw_offset_deg": args.instance_local_yaw_offset_deg,
                        "instance_tracks_loaded": len(tracks),
                        "instance_actors_spawned": len(instance_actors),
                        "collision_count": len(collision_rows),
                        "traffic_manager_enabled": False,
                        "traffic_source": "deterministic_instances_info" if args.instances_info is not None else "none",
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
