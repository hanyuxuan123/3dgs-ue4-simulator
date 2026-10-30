#!/usr/bin/env python3
"""Convert live CARLA replay actual poses back to 3DGS instances_info.json."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import build_instance_item, carla_to_global_xyyaw, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="CSV written by traffic_scenario_runtime.py.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--use", choices=("actual", "target"), default="actual")
    parser.add_argument("--route-start-frame", type=int, default=0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def row_pose_global(row: dict[str, Any], prefix: str) -> dict[str, float]:
    x = float(row[f"{prefix}_x"])
    y = float(row[f"{prefix}_y"])
    z = float(row[f"{prefix}_z"])
    yaw_deg = float(row[f"{prefix}_yaw"])
    gx, gy, gyaw = carla_to_global_xyyaw(x, y, math.radians(yaw_deg))
    return {"x": gx, "y": gy, "z": z, "yaw": gyaw}


def main() -> int:
    args = parse_args()
    prefix = args.use
    rows = read_rows(args.csv)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["track_id"]].append(row)
    instances = {}
    frame_instances: dict[str, list[int]] = {}
    for index, (track_id, track_rows) in enumerate(sorted(grouped.items())):
        track_rows.sort(key=lambda item: (int(float(item["frame"])), int(float(item["step"]))))
        frames = [int(float(row["frame"])) - int(args.route_start_frame) for row in track_rows]
        poses = [row_pose_global(row, prefix) for row in track_rows]
        class_name = track_rows[0].get("class_name", "vehicle.car")
        scenario_type = track_rows[0].get("scenario_type", "")
        instances[str(index)] = build_instance_item(
            track_id,
            class_name,
            frames,
            poses,
            {
                "bench2drive_real": {
                    "scenario_type": scenario_type,
                    "source": f"carla_{prefix}_pose_csv",
                    "coordinate_frame": "mapping-absolute",
                }
            },
        )
        for frame in frames:
            frame_instances.setdefault(str(frame), []).append(index)
    write_json(args.output_dir / "instances_info.json", instances)
    write_json(args.output_dir / "frame_instances.json", frame_instances)
    print(f"Wrote {args.output_dir / 'instances_info.json'}")
    print(f"Wrote {args.output_dir / 'frame_instances.json'}")
    print(f"tracks={len(instances)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
