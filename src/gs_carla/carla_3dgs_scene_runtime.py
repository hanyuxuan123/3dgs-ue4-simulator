#!/usr/bin/env python3
"""Main runtime: load scene package, run CARLA/XODR, render 3DGS, and accept control signals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from . import carla_xodr_live_3dgs_bridge_with_instances as base
from .load_xodr_town_and_dump_poses import (
    move_spectator,
    path_length,
    pose_dict_to_map_aligned_transform,
    pose_dict_to_transform,
    transform_to_dict,
)
from .load_xodr_town_with_instances import make_instance_transform, make_local_se2_correction
from .pose_control_signal import PoseControlSignal
from .render_aligned_mapping_or_carla_path_gsplat import mapping_pose_matrix, read_mapping_poses
from .render_background_gsplat import (
    Profiler,
    load_background,
    load_intrinsics_from_processed,
    load_sky,
    render,
    resolve_device,
    write_video,
)
from .traffic_bbox_carla_loader import TrafficBBoxCarlaLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
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

    parser.add_argument("--render-pose-source", choices=("actor", "reference"), default="actor")
    parser.add_argument("--vehicle-zrp-source", choices=("mapping-frame", "sequence-flat"), default="mapping-frame")
    parser.add_argument("--carla-z-mode", choices=("zero", "actor"), default="zero")
    parser.add_argument("--control-mode", choices=("path", "steer-bias", "constant-steer"), default="path")
    parser.add_argument("--steer-bias", type=float, default=0.0)
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

    parser.add_argument("--hero-blueprint", default="vehicle.tesla.model3")
    parser.add_argument("--hero-role-name", default="hero")
    parser.add_argument("--instance-class-prefix", default="vehicle.")
    parser.add_argument("--max-instance-actors", type=int, default=0)
    parser.add_argument("--instance-z-offset", type=float, default=0.35)
    parser.add_argument("--no-instance-map-z", action="store_true")
    parser.add_argument("--hidden-actor-z", type=float, default=-1000.0)
    parser.add_argument("--instance-local-offset-x", type=float, default=0.0)
    parser.add_argument("--instance-local-offset-y", type=float, default=0.0)
    parser.add_argument("--instance-local-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--debug-instance-conversion", type=int, default=5)

    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument("--spectator-height", type=float, default=55.0)
    parser.add_argument("--spectator-distance", type=float, default=35.0)
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--max-render-frames", type=int, default=0)
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--display-fps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-pattern", default="live_{index:04d}_cam{camera}.png")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--pose-output", type=Path)
    parser.add_argument("--instance-output", type=Path)
    parser.add_argument("--collision-output", type=Path)
    parser.add_argument("--path-json-output", type=Path)
    parser.add_argument("--traffic-bbox-output", type=Path)
    parser.add_argument("--draw-traffic-bboxes", action="store_true")
    parser.add_argument("--traffic-bbox-life-time", type=float, default=0.08)

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
            "instances_info": package_path(cli.scene_package, assets.get("instances_info")),
            "background": package_path(cli.scene_package, assets.get("background")),
            "sky": package_path(cli.scene_package, assets.get("sky")),
            "processed_scene": package_path(cli.scene_package, assets.get("processed_scene")),
            "mapping_pose": package_path(cli.scene_package, assets.get("mapping_pose")),
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
    missing = [
        key
        for key in ("xodr", "background", "processed_scene", "mapping_pose", "core_pose_csv", "core_start_frame", "core_end_frame")
        if data.get(key) is None
    ]
    if missing:
        raise ValueError(f"scene package is missing required fields: {missing}")
    return SimpleNamespace(**data)


class Scene3DGSRuntime:
    """Main loop that receives controls from PoseControlSignal."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pose_rows: list[dict[str, Any]] = []
        self.path_rows: list[dict[str, Any]] = []
        self.collision_rows: list[dict[str, Any]] = []
        self.video_frames: list[np.ndarray] = []

    def run(self) -> int:
        args = self.args
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
        t_instance_anchor_abs = mapping_pose_matrix(mapping_poses, args.instance_origin_sequence_frame)
        t_seq_from_instance = make_instance_transform(
            args.instance_transform_mode,
            t_abs_from_sequence,
            t_instance_anchor_abs,
            t_abs_cam_origin,
        )
        t_seq_from_instance = make_local_se2_correction(
            args.instance_local_offset_x,
            args.instance_local_offset_y,
            args.instance_local_yaw_offset_deg,
        ) @ t_seq_from_instance

        carla = base.import_carla()
        client, world = base.load_xodr_world(carla, args)
        carla_map = world.get_map()
        core_poses = base.read_core_pose_sequence(args)
        hero = base.spawn_hero(carla, world, pose_dict_to_transform(carla, core_poses[0]), args.hero_blueprint, args.hero_role_name)
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
        pygame, screen = base.init_pygame(args.pygame, width, height)
        display_delay = 1.0 / args.display_fps if args.display_fps > 0 else 0.0

        traffic = None
        collision_sensor = None
        collision_events: list[dict[str, Any]] = []
        rendered = 0
        controller = PoseControlSignal(carla, args, core_poses)
        try:
            self._place_hero(carla, world, carla_map, hero, core_poses[0])
            traffic = TrafficBBoxCarlaLoader.from_instances_info(carla, world, carla_map, args, t_seq_from_instance)
            for _ in range(max(0, args.follow_settle_ticks)):
                hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
                world.tick()
            collision_sensor, collision_events = base.attach_collision_sensor(carla, world, hero)
            max_steps = self._max_steps(core_poses)
            for step in range(max_steps):
                signal = controller.step(hero)
                hero.apply_control(signal["control"])
                instance_rows = traffic.update(signal["source_frame"], step) if traffic is not None else []
                collision_start = len(collision_events)
                world.tick()
                step_events = collision_events[collision_start:]
                self._append_collision_rows(step, signal["source_frame"], step_events)
                live_transform = hero.get_transform()
                self._append_pose_row(step, live_transform, signal, collision_events, step_events, instance_rows)
                if self._should_render(step, rendered):
                    rendered += self._render_frame(
                        step,
                        rendered,
                        live_transform,
                        signal,
                        mapping_poses,
                        t_abs_from_sequence,
                        t_3dgs_from_abs,
                        t_camera_from_vehicle,
                        bg,
                        sky,
                        k,
                        width,
                        height,
                        device,
                        profiler,
                        pygame,
                        screen,
                        display_delay,
                        instance_rows,
                    )
                if args.spectator_follow:
                    move_spectator(carla, world, live_transform, args.spectator_height, args.spectator_distance)
                if controller.nearest_index >= len(core_poses) - 2 and signal["nearest_dist"] <= args.follow_finish_distance:
                    break
        finally:
            hero.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            world.tick()
            if collision_sensor is not None:
                collision_sensor.stop()
                collision_sensor.destroy()
            world.apply_settings(original_settings)
            if pygame is not None:
                pygame.quit()

        self._write_outputs(traffic, profiler, rendered)
        return 0

    def _place_hero(self, carla, world, carla_map, hero, start_pose: dict[str, Any]) -> None:
        pose = dict(start_pose)
        if self.args.no_follow_map_z:
            pose["z"] = float(pose["z"]) + float(self.args.follow_start_z_offset)
            transform = pose_dict_to_transform(carla, pose)
        else:
            transform, _ = pose_dict_to_map_aligned_transform(carla, carla_map, pose, self.args.follow_start_z_offset)
        hero.set_simulate_physics(False)
        hero.set_transform(transform)
        world.tick()
        hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        hero.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        hero.set_simulate_physics(True)

    def _max_steps(self, core_poses: list[dict[str, Any]]) -> int:
        target_speed = max(self.args.follow_speed_kmh / 3.6, 0.1)
        max_seconds = self.args.follow_max_seconds
        if max_seconds <= 0.0:
            max_seconds = path_length(core_poses) / target_speed + 20.0
        return max(1, int(math.ceil(max_seconds / max(self.args.control_dt, 1e-3))))

    def _should_render(self, step: int, rendered: int) -> bool:
        if step % self.args.render_every_n != 0:
            return False
        return not (self.args.max_render_frames > 0 and rendered >= self.args.max_render_frames)

    def _append_collision_rows(self, step: int, source_frame: int, events: list[dict[str, Any]]) -> None:
        for event in events:
            row = dict(event)
            row.update({"step": step, "elapsed_time": step * self.args.control_dt, "source_frame": source_frame})
            self.collision_rows.append(row)

    def _append_pose_row(self, step: int, transform, signal: dict[str, Any], collision_events: list[dict[str, Any]], step_events: list[dict[str, Any]], instance_rows: list[dict[str, Any]]) -> None:
        target = signal["target"]
        actual = transform_to_dict(transform, source="hero_controller")
        heading_error = base.normalize_angle(math.radians(float(target["yaw"])) - math.radians(float(actual["yaw"])))
        control = signal["control"]
        actual.update(
            {
                "step": step,
                "elapsed_time": step * self.args.control_dt,
                "frame": signal["source_frame"],
                "speed_mps": signal["speed"],
                "speed_kmh": signal["speed"] * 3.6,
                "target_index": signal["target_index"],
                "target_frame": int(target["frame"]),
                "target_x": float(target["x"]),
                "target_y": float(target["y"]),
                "target_yaw": float(target["yaw"]),
                "distance_error": signal["nearest_dist"],
                "lateral_error": signal["lateral_error"],
                "heading_error_deg": math.degrees(heading_error),
                "throttle": float(control.throttle),
                "base_steer": signal["base_steer"],
                "steer_bias": signal["steer_bias"],
                "steer": float(control.steer),
                "brake": float(control.brake),
                "control_mode": self.args.control_mode,
                "collision_count": len(collision_events),
                "step_collision_count": len(step_events),
                "last_collision_actor_id": int(collision_events[-1]["other_actor_id"]) if collision_events else -1,
                "last_collision_actor_type": collision_events[-1]["other_actor_type"] if collision_events else "",
                "visible_instance_count": len(instance_rows),
            }
        )
        self.pose_rows.append(actual)

    def _render_frame(
        self,
        step: int,
        rendered: int,
        live_transform,
        signal: dict[str, Any],
        mapping_poses: dict[int, dict[str, float]],
        t_abs_from_sequence: np.ndarray,
        t_3dgs_from_abs: np.ndarray,
        t_camera_from_vehicle: np.ndarray,
        bg,
        sky,
        k,
        width: int,
        height: int,
        device,
        profiler,
        pygame,
        screen,
        display_delay: float,
        instance_rows: list[dict[str, Any]],
    ) -> int:
        source_frame = signal["source_frame"]
        if self.args.render_pose_source == "reference":
            render_source_pose = signal["target"]
            t_abs_vehicle = mapping_pose_matrix(mapping_poses, source_frame)
        else:
            render_source_pose = None
            t_seq_vehicle = base.carla_transform_to_sequence_matrix(live_transform, self.args.carla_z_mode)
            t_abs_vehicle = t_abs_from_sequence @ t_seq_vehicle
            if self.args.vehicle_zrp_source == "mapping-frame":
                t_abs_vehicle = base.merge_xy_yaw_with_mapping_zrp(t_abs_vehicle, mapping_poses, source_frame)
        cam_to_world = t_3dgs_from_abs @ t_abs_vehicle @ t_camera_from_vehicle
        image, _ = render(bg, sky, cam_to_world, k, width, height, self.args, device, profiler)
        if self.args.video_output is not None:
            self.video_frames.append(image)
        if self.args.output_dir is not None and not self.args.no_save_frames:
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            out = self.args.output_dir / self.args.output_pattern.format(index=rendered, camera=self.args.camera)
            Image.fromarray(image).save(out)
        if pygame is not None and not base.show_pygame_frame(pygame, screen, image, display_delay):
            return 0
        self.path_rows.append(
            {
                "render_index": rendered,
                "step": step,
                "frame": source_frame,
                "render_pose_source": self.args.render_pose_source,
                "vehicle_zrp_source": self.args.vehicle_zrp_source,
                "visible_instance_count": len(instance_rows),
                "reference_frame": int(render_source_pose["frame"]) if render_source_pose is not None else None,
                "camera_to_3dgs_world": cam_to_world.tolist(),
            }
        )
        print(
            f"[scene-runtime] render {rendered + 1} step={step} "
            f"target_frame={signal['target']['frame']} speed_kmh={signal['speed'] * 3.6:.2f} "
            f"track_error_m={signal['nearest_dist']:.3f} steer={float(signal['control'].steer):.3f}"
        )
        return 1

    def _write_outputs(self, traffic: TrafficBBoxCarlaLoader | None, profiler, rendered: int) -> None:
        if self.args.pose_output:
            base.write_pose_csv(self.args.pose_output, self.pose_rows)
            print(f"Wrote {self.args.pose_output}")
        if traffic is not None:
            traffic.write_outputs()
        if self.args.collision_output:
            base.write_collision_csv(self.args.collision_output, self.collision_rows)
            print(f"Wrote {self.args.collision_output} collisions={len(self.collision_rows)}")
        if self.args.path_json_output:
            self.args.path_json_output.parent.mkdir(parents=True, exist_ok=True)
            self.args.path_json_output.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "scene_package": str(self.args.scene_package),
                            "control_mode": self.args.control_mode,
                            "steer_bias": self.args.steer_bias,
                            "collision_count": len(self.collision_rows),
                        },
                        "frames": self.path_rows,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {self.args.path_json_output}")
        if self.args.video_output is not None:
            write_video(self.args.video_output, self.video_frames, self.args.video_fps)
            print(f"Wrote {self.args.video_output}")
        profiler.print()
        print(f"[scene-runtime] controller_steps={len(self.pose_rows)} rendered_frames={rendered}")


def main() -> int:
    args = args_from_package(parse_args())
    return Scene3DGSRuntime(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
