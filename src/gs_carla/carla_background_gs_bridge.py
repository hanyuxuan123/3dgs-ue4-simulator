#!/usr/bin/env python3
"""Read a CARLA hero pose and render DriveStudio background gaussians."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .render_background_gsplat import load_background, load_sky, render, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--xodr", type=Path, help="Optional OpenDRIVE file to load into CARLA.")
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--sky", type=Path, help="Optional DriveStudio EnvLight cubemap exported as sky_envlight.pth.")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hero-role-name", default="hero")
    parser.add_argument("--camera-local", type=Path, help="JSON 4x4 CARLA sensor-local transform from hero to camera.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fx", type=float, default=900.0)
    parser.add_argument("--fy", type=float, default=900.0)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-gaussians", type=int, default=2_000_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--crop-radius", type=float, default=180.0)
    parser.add_argument("--near", type=float, default=0.2)
    parser.add_argument("--far", type=float, default=250.0)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--background-rgb", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=None,
        help="Spherical harmonics degree for exported gaussian colors. Defaults to auto-detect from features_rest.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-camera-pose", type=Path)
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


def rotation_matrix_from_carla(rot) -> np.ndarray:
    cy, sy = math.cos(math.radians(rot.yaw)), math.sin(math.radians(rot.yaw))
    cp, sp = math.cos(math.radians(rot.pitch)), math.sin(math.radians(rot.pitch))
    cr, sr = math.cos(math.radians(rot.roll)), math.sin(math.radians(rot.roll))
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def carla_transform_to_matrix(transform) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rotation_matrix_from_carla(transform.rotation)
    mat[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]
    return mat


def load_matrix_json(path: Path, key: str) -> np.ndarray:
    data = json.loads(path.read_text())
    matrix = data.get(key, data.get("matrix"))
    if matrix is None:
        raise ValueError(f"{path} must contain {key} or matrix")
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.size != 16:
        raise ValueError(f"{path}:{key} must contain 16 values")
    return arr.reshape(4, 4)


def carla_sensor_to_opencv_camera_matrix() -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return mat


def find_hero(world, role_name: str):
    actors = world.get_actors().filter("vehicle.*")
    for actor in actors:
        if actor.attributes.get("role_name") == role_name:
            return actor
    if len(actors) > 0:
        print(f"No vehicle with role_name={role_name!r}; using first vehicle id={actors[0].id}")
        return actors[0]
    raise RuntimeError("No CARLA vehicle actor found. Spawn a hero vehicle first.")


def maybe_load_xodr(client, xodr_path: Path | None):
    if not xodr_path:
        return client.get_world()
    xodr = xodr_path.read_text()
    import carla  # type: ignore

    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
    return client.generate_opendrive_world(xodr, params)


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image

    carla = import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = maybe_load_xodr(client, args.xodr)
    hero = find_hero(world, args.hero_role_name)

    t_ds_from_carla = load_matrix_json(args.calibration, "T_drivestudio_from_carla")
    t_hero_from_camera_sensor = load_matrix_json(args.camera_local, "T_hero_from_camera") if args.camera_local else np.eye(4, dtype=np.float32)
    t_carla_from_hero = carla_transform_to_matrix(hero.get_transform())
    t_carla_from_camera_sensor = t_carla_from_hero @ t_hero_from_camera_sensor
    t_ds_from_camera_opencv = t_ds_from_carla @ t_carla_from_camera_sensor @ carla_sensor_to_opencv_camera_matrix()

    if args.dump_camera_pose:
        args.dump_camera_pose.parent.mkdir(parents=True, exist_ok=True)
        args.dump_camera_pose.write_text(json.dumps({"camera_to_world": t_ds_from_camera_opencv.tolist()}, indent=2) + "\n")

    device = resolve_device(args.device)
    k = np.array(
        [
            [args.fx, 0.0, args.cx if args.cx is not None else args.width / 2.0],
            [0.0, args.fy, args.cy if args.cy is not None else args.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    bg = load_background(args.background, device)
    sky = load_sky(args.sky, device)
    image, _ = render(bg, sky, t_ds_from_camera_opencv, k, args.width, args.height, args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f"Rendered CARLA actor id={hero.id} transform={hero.get_transform()}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
