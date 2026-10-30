#!/usr/bin/env python3
"""Convert Traffic Manager CARLA-frame instances to bridge insert assets.

`generate_tm_instance_trajectories.py` writes `obj_to_world` matrices in CARLA's
world frame. The 3DGS bridge can replay raw sequence-local instances with:

    --instance-transform-mode raw-local

This utility converts the generated CARLA matrices into that raw-local frame so
the bridge maps them back to the same CARLA poses during replay:

    x_seq = x_carla
    y_seq = -y_carla
    yaw_seq = -yaw_carla

When mapping pose and processed camera metadata are provided, it also writes a
3DGS-local copy where the origin is a processed camera frame:

    T_3dgs_obj = inv(T_abs_cam0) @ T_abs_seq_origin @ T_seq_obj
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tm-out-dir",
        required=True,
        type=Path,
        help="Output directory produced by generate_tm_instance_trajectories.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Insert root directory. Outputs are written under this directory.",
    )
    parser.add_argument("--mapping-pose", type=Path, help="mapping_pose.txt used by the scene.")
    parser.add_argument("--processed-scene", type=Path, help="DriveStudio processed scene directory.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--processed-origin-frame", type=int, default=0)
    parser.add_argument(
        "--processed-origin-mapping-frame",
        type=int,
        help="Mapping frame corresponding to --processed-origin-frame. Required to derive a 3DGS origin from a sequence frame.",
    )
    parser.add_argument("--sequence-origin-frame", type=int, default=0)
    parser.add_argument(
        "--insert-origin-sequence-frame",
        type=int,
        default=300,
        help="Sequence frame aligned with insert instance frame_idx=0, for example 310.",
    )
    parser.add_argument(
        "--3dgs-origin-processed-frame",
        dest="g3d_origin_processed_frame",
        type=int,
        help="Processed frame used as 3DGS local origin. Defaults to --processed-origin-frame.",
    )
    parser.add_argument(
        "--3dgs-origin-sequence-frame",
        dest="g3d_origin_sequence_frame",
        type=int,
        help=(
            "Sequence frame used as 3DGS local origin. Converted to a processed frame using "
            "--processed-origin-mapping-frame and --processed-origin-frame."
        ),
    )
    parser.add_argument(
        "--id-prefix",
        default="insert_",
        help="Prefix added to generated instance ids.",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def transform_from_xyz_rpy(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
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


def carla_to_sequence_matrix(matrix: list[list[float]]) -> list[list[float]]:
    t_carla = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    flip_y = np.diag([1.0, -1.0, 1.0, 1.0])
    t_seq = flip_y @ t_carla @ flip_y
    return t_seq.tolist()


def transform_instance_matrices(instances: dict, id_prefix: str, transform_fn) -> dict:
    converted = {}
    for key, item in instances.items():
        out = dict(item)
        out["id"] = f"{id_prefix}{item.get('id', key)}"
        ann = dict(item.get("frame_annotations", {}))
        ann["obj_to_world"] = [transform_fn(mat) for mat in ann.get("obj_to_world", [])]
        out["frame_annotations"] = ann
        converted[str(key)] = out
    return converted


def offset_frame_instances(frame_instances: dict, offset: int) -> dict:
    return {str(int(frame) + offset): value for frame, value in frame_instances.items()}


def resolve_3dgs_origin_processed_frame(args: argparse.Namespace) -> int:
    if args.g3d_origin_processed_frame is not None and args.g3d_origin_sequence_frame is not None:
        raise ValueError("Use only one of --3dgs-origin-processed-frame or --3dgs-origin-sequence-frame")
    if args.g3d_origin_processed_frame is not None:
        return args.g3d_origin_processed_frame
    if args.g3d_origin_sequence_frame is not None:
        if args.processed_origin_mapping_frame is None:
            raise ValueError("--processed-origin-mapping-frame is required with --3dgs-origin-sequence-frame")
        return args.processed_origin_frame + (args.g3d_origin_sequence_frame - args.processed_origin_mapping_frame)
    return args.processed_origin_frame


def maybe_build_3dgs_transform(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int] | None:
    if args.mapping_pose is None and args.processed_scene is None:
        return None
    if args.mapping_pose is None or args.processed_scene is None:
        raise ValueError("--mapping-pose and --processed-scene must be provided together for 3dgs_origin output")
    mapping_poses = read_mapping_poses(args.mapping_pose)
    t_abs_from_sequence = mapping_pose_matrix(mapping_poses, args.sequence_origin_frame)
    origin_processed_frame = resolve_3dgs_origin_processed_frame(args)
    t_abs_cam0 = np.loadtxt(
        args.processed_scene / "extrinsics" / f"{origin_processed_frame:03d}_{args.camera}.txt"
    ).reshape(4, 4).astype(np.float64)
    t_3dgs_from_abs = np.linalg.inv(t_abs_cam0)
    return t_abs_from_sequence, t_3dgs_from_abs, origin_processed_frame


def main() -> int:
    args = parse_args()
    source_instances = args.tm_out_dir / "instances" / "instances_info.json"
    source_frame_instances = args.tm_out_dir / "instances" / "frame_instances.json"
    if not source_instances.is_file():
        raise FileNotFoundError(source_instances)
    if not source_frame_instances.is_file():
        raise FileNotFoundError(source_frame_instances)

    source = load_json(source_instances)
    frame_instances = load_json(source_frame_instances)

    carla_dir = args.output_dir / "carla_aligned"
    sequence_dir = args.output_dir / "sequence_local"
    g3d_dir = args.output_dir / "3dgs_origin"

    carla_instances = transform_instance_matrices(source, args.id_prefix, lambda mat: mat)
    sequence_instances = transform_instance_matrices(source, args.id_prefix, carla_to_sequence_matrix)

    write_json(carla_dir / "instances_info.json", carla_instances)
    write_json(carla_dir / "frame_instances.json", frame_instances)
    write_json(carla_dir / "frame_instances_sequence.json", offset_frame_instances(frame_instances, args.insert_origin_sequence_frame))

    write_json(sequence_dir / "instances_info.json", sequence_instances)
    write_json(sequence_dir / "frame_instances.json", frame_instances)
    write_json(sequence_dir / "frame_instances_sequence.json", offset_frame_instances(frame_instances, args.insert_origin_sequence_frame))

    transform_pair = maybe_build_3dgs_transform(args)
    wrote_3dgs = False
    origin_processed_frame = None
    if transform_pair is not None:
        t_abs_from_sequence, t_3dgs_from_abs, origin_processed_frame = transform_pair

        def carla_to_3dgs_matrix(matrix: list[list[float]]) -> list[list[float]]:
            t_seq_obj = np.asarray(carla_to_sequence_matrix(matrix), dtype=np.float64).reshape(4, 4)
            t_3dgs_obj = t_3dgs_from_abs @ t_abs_from_sequence @ t_seq_obj
            return t_3dgs_obj.tolist()

        g3d_instances = transform_instance_matrices(source, args.id_prefix, carla_to_3dgs_matrix)
        write_json(g3d_dir / "instances_info.json", g3d_instances)
        write_json(g3d_dir / "frame_instances.json", frame_instances)
        write_json(g3d_dir / "frame_instances_sequence.json", offset_frame_instances(frame_instances, args.insert_origin_sequence_frame))
        wrote_3dgs = True

    summary = {
        "source": str(args.tm_out_dir),
        "insert_origin_sequence_frame": args.insert_origin_sequence_frame,
        "num_instances": len(source),
        "num_frames": len(frame_instances),
        "state_transition": {
            "frame_mapping": "sequence_frame = insert_origin_sequence_frame + instance_frame",
            "example": f"if insert_origin_sequence_frame={args.insert_origin_sequence_frame}, instance_frame=0 maps to sequence_frame={args.insert_origin_sequence_frame}; instance_frame=10 maps to sequence_frame={args.insert_origin_sequence_frame + 10}",
            "carla_to_sequence": "T_seq_obj = diag(1,-1,1,1) @ T_carla_obj @ diag(1,-1,1,1)",
            "sequence_to_carla_replay": "x_carla=x_seq, y_carla=-y_seq, yaw_carla=-yaw_seq",
            "sequence_to_abs": "T_abs_obj = T_abs_from_sequence_origin @ T_seq_obj",
            "abs_to_3dgs_origin": "T_3dgs_obj = inv(T_abs_cam_processed_origin) @ T_abs_obj",
            "3dgs_origin_note": "3DGS local axes are camera-local axes; z is camera-forward depth, not CARLA vertical height.",
        },
        "3dgs_origin": {
            "processed_frame": origin_processed_frame,
            "sequence_frame": args.g3d_origin_sequence_frame,
            "camera": args.camera,
        },
        "outputs": {
            "carla_aligned_instances_info": str(carla_dir / "instances_info.json"),
            "sequence_local_instances_info": str(sequence_dir / "instances_info.json"),
            "3dgs_origin_instances_info": str(g3d_dir / "instances_info.json") if wrote_3dgs else None,
        },
        "bridge_usage": {
            "instances_info": str(sequence_dir / "instances_info.json"),
            "instance_transform_mode": "raw-local",
            "instance_origin_sequence_frame": args.insert_origin_sequence_frame,
        },
    }
    write_json(args.output_dir / "info.json", summary)
    print(f"Wrote {carla_dir / 'instances_info.json'}")
    print(f"Wrote {sequence_dir / 'instances_info.json'}")
    if wrote_3dgs:
        print(f"Wrote {g3d_dir / 'instances_info.json'}")
    print(f"Wrote {args.output_dir / 'info.json'}")
    print("For CARLA replay bridge, use insert/sequence_local/instances_info.json with --instance-transform-mode raw-local.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
