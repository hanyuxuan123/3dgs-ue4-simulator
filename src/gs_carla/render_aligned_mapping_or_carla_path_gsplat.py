#!/usr/bin/env python3
"""Render 3DGS from mapping_pose or CARLA-followed poses in DriveStudio local world.

Coordinate idea:
  processed extrinsics are absolute mapping-world camera_to_world matrices.
  DriveStudio local world used by render_background_gsplat is:
      T_ds_from_abs = inv(processed extrinsics[processed_origin_frame, camera])

For mapping_pose:
      T_ds_camera = T_ds_from_abs @ T_abs_vehicle(mapping_frame) @ T_camera_from_vehicle

For CARLA followed poses exported by load_xodr_town_and_dump_poses.py:
      CARLA x/y/yaw are converted back to sequence_start_local by (x, -y, -yaw),
      then composed under mapping_pose[sequence_origin_frame].
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
from scipy.spatial.transform import Rotation, Slerp

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
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--sky", type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--mapping-pose", required=True, type=Path)
    parser.add_argument("--trajectory-source", choices=("mapping-pose", "carla-csv"), default="mapping-pose")
    parser.add_argument("--carla-csv", type=Path, help="CSV exported by load_xodr_town_and_dump_poses.py follow/replay mode.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--processed-origin-frame", type=int, default=0)
    parser.add_argument(
        "--processed-origin-mapping-frame",
        type=int,
        default=0,
        help="Raw mapping_pose lidar_frame corresponding to processed frame --processed-origin-frame.",
    )
    parser.add_argument("--sequence-origin-frame", type=int, default=0)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--samples", type=int, default=0, help="0 means keep input trajectory sampling.")
    parser.add_argument("--output-calibration", type=Path)
    parser.add_argument("--output-path-json", type=Path)
    parser.add_argument("--transform-only", action="store_true", help="Only export calibration/path JSON; do not render.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-pattern", default="aligned_{index:04d}_cam{camera}.png")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--display-fps", type=float, default=10.0)
    parser.add_argument("--no-save-frames", action="store_true")
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


def carla_csv_rows(path: Path, start: int, end: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("frame") not in (None, "")]
    selected = [row for row in rows if start <= int(float(row["frame"])) <= end]
    if not selected:
        raise RuntimeError(f"No rows in frame range [{start}, {end}] from {path}")
    selected.sort(key=lambda row: int(float(row["frame"])))
    return selected


def carla_row_to_sequence_local_matrix(row: dict[str, str]) -> np.ndarray:
    carla_x = float(row["x"])
    carla_y = float(row["y"])
    carla_z = float(row.get("z", 0.0) or 0.0)
    yaw_value = float(row.get("yaw", row.get("rotation_yaw", 0.0)) or 0.0)
    yaw_deg = math.degrees(yaw_value) if abs(yaw_value) <= 2.0 * math.pi else yaw_value
    # OpenDRIVE import convention used elsewhere in this repo: carla_y = -sequence_local_y.
    seq_x = carla_x
    seq_y = -carla_y
    seq_yaw = math.radians(-yaw_deg)
    return transform_from_xyz_rpy(seq_x, seq_y, carla_z, 0.0, 0.0, seq_yaw)


def mapping_trajectory_matrices(poses: dict[int, dict[str, float]], start: int, end: int) -> tuple[list[int], list[np.ndarray]]:
    frames = [frame for frame in sorted(poses) if start <= frame <= end]
    if not frames:
        raise RuntimeError(f"No mapping poses in frame range [{start}, {end}]")
    return frames, [mapping_pose_matrix(poses, frame) for frame in frames]


def carla_trajectory_matrices(
    poses: dict[int, dict[str, float]],
    carla_csv: Path,
    start: int,
    end: int,
    sequence_origin_frame: int,
) -> tuple[list[int], list[np.ndarray]]:
    t_abs_from_seq = mapping_pose_matrix(poses, sequence_origin_frame)
    rows = carla_csv_rows(carla_csv, start, end)
    frames = [int(float(row["frame"])) for row in rows]
    mats = [t_abs_from_seq @ carla_row_to_sequence_local_matrix(row) for row in rows]
    return frames, mats


def interpolate_matrices(frames: list[int], mats: list[np.ndarray], samples: int) -> tuple[list[float], list[np.ndarray]]:
    if samples <= 0 or samples == len(mats):
        return [float(frame) for frame in frames], mats
    if samples < 2:
        raise ValueError("--samples must be 0 or >= 2")
    if len(mats) < 2:
        raise ValueError("Need at least two trajectory matrices for interpolation")
    key_t = np.linspace(0.0, 1.0, len(mats), dtype=np.float64)
    out_t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    frame_values = np.interp(out_t, key_t, np.asarray(frames, dtype=np.float64)).tolist()
    translations = np.stack([mat[:3, 3] for mat in mats], axis=0)
    interp_trans = np.stack([np.interp(out_t, key_t, translations[:, axis]) for axis in range(3)], axis=1)
    rotations = Rotation.from_matrix(np.stack([mat[:3, :3] for mat in mats], axis=0))
    interp_rots = Slerp(key_t, rotations)(out_t).as_matrix()
    out = []
    for rot, trans in zip(interp_rots, interp_trans):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot.astype(np.float32)
        mat[:3, 3] = trans.astype(np.float32)
        out.append(mat)
    return frame_values, out


def maybe_init_pygame(enabled: bool, width: int, height: int):
    if not enabled:
        return None, None
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA/mapping_pose aligned 3DGS render")
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


def main() -> None:
    args = parse_args()
    if args.trajectory_source == "carla-csv" and args.carla_csv is None:
        raise ValueError("--trajectory-source=carla-csv requires --carla-csv")
    if args.no_save_frames and args.video_output is None and not args.pygame:
        raise ValueError("--no-save-frames needs --video-output or --pygame")

    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_abs_vehicle_origin = mapping_pose_matrix(mapping_poses, args.processed_origin_mapping_frame)
    t_abs_cam_origin = load_processed_extrinsic(args.processed_scene, args.processed_origin_frame, args.camera)
    t_camera_from_vehicle = np.linalg.inv(t_abs_vehicle_origin) @ t_abs_cam_origin
    t_ds_from_abs = np.linalg.inv(t_abs_cam_origin)
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)
    t_ds_from_sequence = t_ds_from_abs @ t_abs_from_sequence

    if args.trajectory_source == "mapping-pose":
        frames, vehicle_mats_abs = mapping_trajectory_matrices(mapping_poses, args.start_frame, args.end_frame)
    else:
        frames, vehicle_mats_abs = carla_trajectory_matrices(
            mapping_poses,
            args.carla_csv,
            args.start_frame,
            args.end_frame,
            args.sequence_origin_frame,
        )
    frame_values, vehicle_mats_abs = interpolate_matrices(frames, vehicle_mats_abs, args.samples)
    camera_mats_ds = [t_ds_from_abs @ vehicle_abs @ t_camera_from_vehicle for vehicle_abs in vehicle_mats_abs]

    if args.output_calibration:
        args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
        args.output_calibration.write_text(
            json.dumps(
                {
                    "processed_scene": str(args.processed_scene),
                    "mapping_pose": str(args.mapping_pose),
                    "camera": args.camera,
                    "processed_origin_frame": args.processed_origin_frame,
                    "processed_origin_mapping_frame": args.processed_origin_mapping_frame,
                    "sequence_origin_frame": args.sequence_origin_frame,
                    "T_camera_from_vehicle": t_camera_from_vehicle.tolist(),
                    "T_3dgs_from_mapping_abs": t_ds_from_abs.tolist(),
                    "T_3dgs_from_sequence_local": t_ds_from_sequence.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {args.output_calibration}")

    if args.output_path_json:
        args.output_path_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_path_json.write_text(
            json.dumps(
                {
                    "trajectory_source": args.trajectory_source,
                    "frames": [
                        {"frame": float(frame), "camera_to_3dgs_world": mat.tolist()}
                        for frame, mat in zip(frame_values, camera_mats_ds)
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {args.output_path_json}")

    if args.transform_only:
        print(
            f"[aligned] transform-only source={args.trajectory_source} "
            f"samples={len(camera_mats_ds)} frame_range={frame_values[0]:.3f}-{frame_values[-1]:.3f}"
        )
        return

    device = resolve_device(args.device)
    profiler = Profiler(args.profile, device)
    k, width, height = load_intrinsics_from_processed(args.processed_scene, args.camera, args.downscale)
    profiler.mark("load camera/intrinsics")
    bg = load_background(args.background, device)
    profiler.mark("load background")
    sky = load_sky(args.sky, device)
    profiler.mark("load sky")

    pygame, screen = maybe_init_pygame(args.pygame, width, height)
    display_delay = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    video_frames: list[np.ndarray] = []
    start_time = time.perf_counter()
    for index, cam_to_world in enumerate(camera_mats_ds):
        image, _ = render(bg, sky, cam_to_world, k, width, height, args, device, profiler)
        if args.video_output is not None:
            video_frames.append(image)
        if args.output_dir is not None and not args.no_save_frames:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out = args.output_dir / args.output_pattern.format(index=index, camera=args.camera)
            Image.fromarray(image).save(out)
        if pygame is not None:
            if not show_pygame_frame(pygame, screen, image, display_delay):
                break
        elapsed = time.perf_counter() - start_time
        print(
            f"[aligned] sample {index + 1}/{len(camera_mats_ds)} "
            f"source_frame={frame_values[index]:.3f} "
            f"avg_render_fps={(index + 1) / max(elapsed, 1e-9):.2f}"
        )

    elapsed = time.perf_counter() - start_time
    print(
        f"[aligned] source={args.trajectory_source} samples={len(camera_mats_ds)} "
        f"render_elapsed={elapsed:.4f}s avg_render_fps={len(camera_mats_ds) / max(elapsed, 1e-9):.2f}"
    )
    if args.video_output is not None:
        write_video(args.video_output, video_frames, args.video_fps)
        print(f"Wrote {args.video_output}")
    profiler.print()
    if pygame is not None:
        pygame.quit()


if __name__ == "__main__":
    main()
