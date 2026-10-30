#!/usr/bin/env python3
"""Find nuScenes/CARLA route anchors suitable for Bench2Drive-Real templates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import (
    SUPPORTED_SCENARIO_TYPES,
    cumulative_route_s,
    read_core_pose_csv,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", type=Path, help="Prepared nuScenes global scene_package.json.")
    parser.add_argument("--core-pose-csv", type=Path)
    parser.add_argument("--xodr", type=Path)
    parser.add_argument("--templates", type=Path, help="Optional templates JSON from bench2drive_xml_parser.py.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--types", nargs="+", default=list(SUPPORTED_SCENARIO_TYPES))
    parser.add_argument("--frame-margin", type=int, default=4)
    parser.add_argument("--min-anchor-spacing-m", type=float, default=18.0)
    parser.add_argument("--prefer-carla-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--target-mode",
        choices=("spread", "template"),
        default="spread",
        help="spread distributes requested scenario types over the current route; template uses source Bench2Drive trigger fractions.",
    )
    return parser.parse_args()


def resolve_from_scene_package(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.scene_package is None:
        if args.core_pose_csv is None:
            raise ValueError("--core-pose-csv is required without --scene-package")
        return args.core_pose_csv, args.xodr
    package = read_json(args.scene_package)
    assets = package.get("assets", {})
    base = args.scene_package.parent
    core = Path(assets["core_pose_csv"])
    if not core.is_absolute():
        core = base / core
    xodr = args.xodr or (Path(assets["xodr"]) if assets.get("xodr") else None)
    if xodr is not None and not xodr.is_absolute():
        xodr = base / xodr
    return core, xodr


def route_curvature(rows: list[dict[str, float]], index: int, window: int = 2) -> float:
    left = max(0, index - window)
    right = min(len(rows) - 1, index + window)
    return abs(rows[right]["global_yaw"] - rows[left]["global_yaw"]) / max(rows[right]["route_s"] - rows[left]["route_s"], 1e-3)


def load_carla_map(xodr: Path | None):
    if xodr is None:
        return None, None
    try:
        import carla  # type: ignore
    except Exception as exc:
        print(f"[anchors] CARLA import unavailable, using route-only anchors: {exc}")
        return None, None
    try:
        return carla, carla.Map(xodr.stem, xodr.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[anchors] failed to load XODR map, using route-only anchors: {exc}")
        return None, None


def waypoint_features(carla, carla_map, row: dict[str, float]) -> dict[str, Any]:
    if carla is None or carla_map is None:
        return {
            "has_map_waypoint": False,
            "is_junction": False,
            "lane_width": 3.5,
            "has_left_driving_lane": True,
            "has_right_driving_lane": True,
        }
    waypoint = carla_map.get_waypoint(
        carla.Location(x=row["carla_x"], y=row["carla_y"], z=row["carla_z"]),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        return {"has_map_waypoint": False, "is_junction": False, "lane_width": 3.5}
    left = waypoint.get_left_lane()
    right = waypoint.get_right_lane()
    return {
        "has_map_waypoint": True,
        "road_id": int(waypoint.road_id),
        "lane_id": int(waypoint.lane_id),
        "is_junction": bool(waypoint.is_junction),
        "lane_width": float(waypoint.lane_width),
        "has_left_driving_lane": bool(left and left.lane_type == carla.LaneType.Driving),
        "has_right_driving_lane": bool(right and right.lane_type == carla.LaneType.Driving),
    }


def type_score(scenario_type: str, row: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    curvature = float(row.get("curvature", 0.0))
    straight = curvature < 0.04
    is_junction = bool(row.get("is_junction", False))
    has_adjacent = bool(row.get("has_left_driving_lane", True) or row.get("has_right_driving_lane", True))
    score = 0.5
    if scenario_type in {"ParkedObstacle", "ConstructionObstacle", "Accident", "HardBreakRoute"}:
        score += 0.35 if straight else -0.15
        reasons.append("straight route" if straight else "curved route fallback")
        score += 0.10 if not is_junction else -0.20
    elif scenario_type in {"HighwayCutIn", "ParkingCutIn"}:
        score += 0.25 if straight else -0.10
        score += 0.25 if has_adjacent else -0.25
        reasons.append("adjacent lane available" if has_adjacent else "no adjacent lane from map")
    elif scenario_type in {"PedestrianCrossing", "DynamicObjectCrossing"}:
        score += 0.20 if not is_junction else 0.10
        score += 0.15 if straight else 0.0
        reasons.append("front camera crossing candidate")
    return score, reasons


def choose_anchors(
    rows: list[dict[str, Any]],
    scenario_types: list[str],
    min_spacing_m: float,
    target_fraction_by_type: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    used_s: list[float] = []
    route_total = max(float(rows[-1]["route_s"]), 1e-6)
    target_fraction_by_type = target_fraction_by_type or {}
    for scenario_type in scenario_types:
        candidates = []
        target_s = target_fraction_by_type.get(scenario_type, 0.5) * route_total
        for row in rows:
            score, reasons = type_score(scenario_type, row)
            score -= 0.35 * min(abs(float(row["route_s"]) - target_s) / route_total, 1.0)
            if any(abs(row["route_s"] - s) < min_spacing_m for s in used_s):
                score -= 0.35
            candidates.append((score, row, reasons))
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, row, reasons = candidates[0]
        used_s.append(row["route_s"])
        anchors.append(
            {
                "anchor_id": f"{scenario_type}_{int(row['frame']):04d}",
                "scenario_type": scenario_type,
                "score": float(score),
                "reasons": reasons + [f"target_route_fraction={target_fraction_by_type.get(scenario_type, 0.5):.3f}"],
                "frame": int(row["frame"]),
                "route_s": float(row["route_s"]),
                "global_pose": {
                    "x": float(row["global_x"]),
                    "y": float(row["global_y"]),
                    "z": float(row["global_z"]),
                    "yaw": float(row["global_yaw"]),
                },
                "carla_pose": {
                    "x": float(row["carla_x"]),
                    "y": float(row["carla_y"]),
                    "z": float(row["carla_z"]),
                    "yaw": float(row["carla_yaw"]),
                },
                "map_features": {
                    key: value
                    for key, value in row.items()
                    if key
                    in {
                        "has_map_waypoint",
                        "road_id",
                        "lane_id",
                        "is_junction",
                        "lane_width",
                        "has_left_driving_lane",
                        "has_right_driving_lane",
                    }
                },
            }
        )
    return anchors


def main() -> int:
    args = parse_args()
    core_pose_csv, xodr = resolve_from_scene_package(args)
    route_rows = cumulative_route_s(read_core_pose_csv(core_pose_csv))
    if len(route_rows) < 3:
        raise RuntimeError(f"Need at least three core poses from {core_pose_csv}")
    first = int(route_rows[0]["frame"])
    last = int(route_rows[-1]["frame"])
    route_rows = [row for row in route_rows if first + args.frame_margin <= int(row["frame"]) <= last - args.frame_margin]
    carla, carla_map = load_carla_map(xodr) if args.prefer_carla_map else (None, None)
    enriched = []
    for index, row in enumerate(route_rows):
        item = dict(row)
        item["curvature"] = route_curvature(route_rows, index)
        item.update(waypoint_features(carla, carla_map, item))
        enriched.append(item)
    wanted = args.types
    target_fraction_by_type: dict[str, float] = {}
    if args.templates:
        templates = read_json(args.templates)
        present = {item["scenario_type"] for item in templates.get("templates", [])}
        wanted = [name for name in wanted if name in present]
        for scenario_type in wanted:
            fractions = [
                float(item.get("trigger_route_fraction", 0.5))
                for item in templates.get("templates", [])
                if item.get("scenario_type") == scenario_type
            ]
            if fractions:
                fractions.sort()
                target_fraction_by_type[scenario_type] = fractions[len(fractions) // 2]
    if args.target_mode == "spread" and wanted:
        if len(wanted) == 1:
            target_fraction_by_type[wanted[0]] = 0.5
        else:
            for index, scenario_type in enumerate(wanted):
                target_fraction_by_type[scenario_type] = 0.12 + 0.76 * index / (len(wanted) - 1)
    anchors = choose_anchors(enriched, wanted, args.min_anchor_spacing_m, target_fraction_by_type)
    write_json(
        args.output,
        {
            "core_pose_csv": str(core_pose_csv),
            "xodr": str(xodr) if xodr else None,
            "coordinate_frame": "nuscenes_global_and_carla_opendrive_import",
            "route_frame_range": [first, last],
            "anchors": anchors,
        },
    )
    print(f"Wrote {args.output}")
    print(f"anchors={len(anchors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
