#!/usr/bin/env python3
"""Crop a ClassLab roadlane tracking output without rerunning correction."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


MAP_PRODUCTS = [
    "road_network_info_local.json",
    "road_network_info.json",
    "road_fill_info_local.json",
    "road_fill_info.json",
    "roadlane_info_local.json",
    "roadlane_info.json",
    "road_polygon_info.json",
    "road_instruction_info.json",
    "roadlane_bev.png",
    "roadlane_id_bev.png",
    "road_fill_bev.png",
    "road_direction_bev.png",
    "road_instruction_bev.png",
    "classlab_multi_road_seq0.xodr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path, help="Global/wide-window tracking output directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Cropped output directory.")
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame", required=True, type=int)
    parser.add_argument(
        "--copy-map-products",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy map/network/XODR products from the source directory. Defaults to true.",
    )
    return parser.parse_args()


def crop_csv(source: Path, target: Path, start_frame: int, end_frame: int) -> int:
    if not source.exists():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as src, target.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"{source} has no CSV header")
        if "frame" not in reader.fieldnames:
            raise ValueError(f"{source} has no frame column")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = 0
        for row in reader:
            frame = int(row["frame"])
            if start_frame <= frame <= end_frame:
                writer.writerow(row)
                count += 1
    return count


def copy_product(source_dir: Path, output_dir: Path, name: str, start_frame: int, end_frame: int) -> bool:
    source = source_dir / name
    if not source.exists():
        return False
    target = output_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        metadata = data.setdefault("metadata", {})
        metadata["cropped_from_tracking_dir"] = str(source_dir)
        metadata["crop_start_frame"] = int(start_frame)
        metadata["crop_end_frame"] = int(end_frame)
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        shutil.copy2(source, target)
    return True


def main() -> int:
    args = parse_args()
    if args.start_frame > args.end_frame:
        raise ValueError("--start-frame must be <= --end-frame")
    if not args.source_dir.is_dir():
        raise FileNotFoundError(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pose_count = crop_csv(
        args.source_dir / "corrected_pose.csv",
        args.output_dir / "corrected_pose.csv",
        args.start_frame,
        args.end_frame,
    )
    pose_enu_count = crop_csv(
        args.source_dir / "corrected_pose_enu.csv",
        args.output_dir / "corrected_pose_enu.csv",
        args.start_frame,
        args.end_frame,
    )

    copied = []
    if args.copy_map_products:
        for name in MAP_PRODUCTS:
            if copy_product(args.source_dir, args.output_dir, name, args.start_frame, args.end_frame):
                copied.append(name)

    print(f"cropped corrected_pose.csv rows: {pose_count}")
    print(f"cropped corrected_pose_enu.csv rows: {pose_enu_count}")
    print(f"copied map products: {len(copied)}")
    print(f"wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
