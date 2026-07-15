#!/usr/bin/env python3
"""Write a small JSON manifest for a CARLA OpenDRIVE + 3DGS + instances scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default="classlab_3dgs_carla_scene")
    parser.add_argument("--description", default="")

    parser.add_argument("--xodr", required=True, type=Path)
    parser.add_argument("--instances-info", type=Path)
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--sky", type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--mapping-pose", required=True, type=Path)
    parser.add_argument("--core-pose-csv", required=True, type=Path)
    parser.add_argument("--sequence-origin-pose-file", type=Path)

    parser.add_argument("--instance-origin-sequence-frame", type=int, default=300)
    parser.add_argument(
        "--instance-transform-mode",
        choices=("raw-local", "mapping-absolute", "sequence-anchor", "processed-camera-origin"),
        default="mapping-absolute",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--processed-origin-frame", type=int, default=0)
    parser.add_argument("--processed-origin-mapping-frame", type=int, required=True)
    parser.add_argument("--sequence-origin-frame", type=int, default=0)
    parser.add_argument("--core-start-frame", type=int)
    parser.add_argument("--core-end-frame", type=int)
    parser.add_argument(
        "--core-pose-coordinate-frame",
        choices=("corrected_local", "enu", "carla"),
        default="enu",
    )
    parser.add_argument(
        "--relative-to-output",
        action="store_true",
        help="Store relative paths when an input is under the package output directory.",
    )
    return parser.parse_args()


def encode_path(path: Path | None, base: Path, relative: bool) -> str | None:
    if path is None:
        return None
    resolved = path.expanduser()
    if relative:
        try:
            return str(resolved.resolve().relative_to(base.resolve()))
        except ValueError:
            pass
    return str(resolved.resolve())


def main() -> int:
    args = parse_args()
    base = args.output.parent
    package = {
        "schema": "carla_3dgs_scene_package_v1",
        "metadata": {
            "name": args.name,
            "description": args.description,
        },
        "assets": {
            "xodr": encode_path(args.xodr, base, args.relative_to_output),
            "instances_info": encode_path(args.instances_info, base, args.relative_to_output),
            "background": encode_path(args.background, base, args.relative_to_output),
            "sky": encode_path(args.sky, base, args.relative_to_output),
            "processed_scene": encode_path(args.processed_scene, base, args.relative_to_output),
            "mapping_pose": encode_path(args.mapping_pose, base, args.relative_to_output),
            "core_pose_csv": encode_path(args.core_pose_csv, base, args.relative_to_output),
            "sequence_origin_pose_file": encode_path(args.sequence_origin_pose_file, base, args.relative_to_output),
        },
        "frames": {
            "instance_origin_sequence_frame": args.instance_origin_sequence_frame,
            "processed_origin_frame": args.processed_origin_frame,
            "processed_origin_mapping_frame": args.processed_origin_mapping_frame,
            "sequence_origin_frame": args.sequence_origin_frame,
            "core_start_frame": args.core_start_frame,
            "core_end_frame": args.core_end_frame,
        },
        "values": {
            "camera": args.camera,
            "instance_transform_mode": args.instance_transform_mode,
            "core_pose_coordinate_frame": args.core_pose_coordinate_frame,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
