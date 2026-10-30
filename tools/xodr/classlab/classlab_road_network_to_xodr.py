#!/usr/bin/env python3
"""Convert a local ClassLab road-fill track into a simple single-road OpenDRIVE map."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree import ElementTree as ET


Point = tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a minimal CARLA-loadable .xodr from "
            "track_classlab_roadlanes_bev_semantic.py road_fills."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--road-network",
        type=Path,
        help="Existing road_network_info_local.json from the semantic BEV lane tracker.",
    )
    source.add_argument(
        "--sequence-dir",
        type=Path,
        help="Raw ClassLab sequence directory. The semantic tracker will be run first.",
    )
    parser.add_argument(
        "--tracking-output-dir",
        type=Path,
        help="Output directory for intermediate semantic tracking files when --sequence-dir is used.",
    )
    parser.add_argument("--lane-csv", type=Path, help="Optional lane.csv override for the semantic tracker.")
    parser.add_argument("--pose-file", type=Path, help="Optional mapping_pose.txt override for the semantic tracker.")
    parser.add_argument("--start-frame", type=int, help="First frame for semantic tracking.")
    parser.add_argument("--end-frame", type=int, help="Last frame for semantic tracking.")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path, help="Output .xodr path.")
    parser.add_argument(
        "--road-fill-id",
        type=int,
        help="Use a specific road_fill_id. Defaults to the longest fill closest to n=0.",
    )
    parser.add_argument(
        "--road-fill-ids",
        help="Comma-separated road_fill_id list to export, e.g. 2,0. Overrides automatic road fill selection.",
    )
    parser.add_argument(
        "--multi-lane",
        action="store_true",
        help="Generate one OpenDRIVE road with all selected road_fills as lanes.",
    )
    parser.add_argument(
        "--separate-roads",
        action="store_true",
        help="Write each selected road_fill as a separate one-lane OpenDRIVE road.",
    )
    parser.add_argument(
        "--reverse-road-fill-ids",
        help="Comma-separated selected road_fill_id list whose exported OpenDRIVE direction should be reversed.",
    )
    parser.add_argument(
        "--traffic-direction",
        choices=("auto", "with_pose_s", "against_pose_s", "all"),
        default="auto",
        help="Direction filter for --multi-lane. auto uses the road_fill closest to n=0 at --target-s.",
    )
    parser.add_argument(
        "--target-s",
        type=float,
        default=0.0,
        help="Prefer a road_fill whose s_range covers this local longitudinal position.",
    )
    parser.add_argument(
        "--target-s-tolerance",
        type=float,
        default=5.0,
        help="Tolerance in meters when checking whether a road_fill covers --target-s.",
    )
    parser.add_argument(
        "--centerline-key",
        default="traffic_manager_centerline_world",
        choices=("traffic_manager_centerline_world", "centerline_world"),
        help="Centerline field to use from road_fills.",
    )
    parser.add_argument(
        "--xodr-coordinate-frame",
        choices=("corrected_local", "sequence_start_local", "enu"),
        default="sequence_start_local",
        help=(
            "Coordinate frame used for the written OpenDRIVE geometry. "
            "corrected_local keeps the selected frame-window origin; "
            "sequence_start_local uses a fixed whole-sequence origin frame and is the default "
            "so XODR files generated from different frame windows share one CARLA world frame; "
            "enu writes mapping ENU coordinates."
        ),
    )
    parser.add_argument(
        "--sequence-origin-frame",
        type=int,
        default=0,
        help="Frame used as origin when --xodr-coordinate-frame=sequence_start_local. Defaults to frame 0.",
    )
    parser.add_argument(
        "--sequence-origin-pose-file",
        type=Path,
        help="mapping_pose.txt used to define the sequence origin. Defaults to --pose-file or <sequence-dir>/result/mapping/mapping_pose.txt.",
    )
    parser.add_argument("--lane-width", type=float, help="Override lane width in meters.")
    parser.add_argument("--min-segment-length", type=float, default=0.25)
    parser.add_argument("--max-segment-length", type=float, default=6.0)
    parser.add_argument(
        "--prepend-length",
        type=float,
        default=10.0,
        help="Extend the road backward from the first centerline point so the hero origin is on the road.",
    )
    parser.add_argument(
        "--append-length",
        type=float,
        default=0.0,
        help="Optionally extend the road forward from the last centerline point.",
    )
    parser.add_argument(
        "--lane-side",
        choices=("right", "left"),
        default="right",
        help="Place the single OpenDRIVE driving lane to this side of the reference line.",
    )
    parser.add_argument("--road-id", default="1")
    parser.add_argument("--road-name", default="classlab_local_road")
    return parser.parse_args()


def run_semantic_tracker(args: argparse.Namespace) -> Path:
    if args.tracking_output_dir is None:
        raise ValueError("--tracking-output-dir is required with --sequence-dir")

    import track_classlab_roadlanes_bev_semantic as semantic

    tracker_argv = [
        "track_classlab_roadlanes_bev_semantic.py",
        "--sequence_dir",
        str(args.sequence_dir),
        "--output_dir",
        str(args.tracking_output_dir),
        "--frame_step",
        str(args.frame_step),
    ]
    if args.lane_csv:
        tracker_argv.extend(["--lane_csv", str(args.lane_csv)])
    if args.pose_file:
        tracker_argv.extend(["--pose_file", str(args.pose_file)])
    if args.start_frame is not None:
        tracker_argv.extend(["--start_frame", str(args.start_frame)])
    if args.end_frame is not None:
        tracker_argv.extend(["--end_frame", str(args.end_frame)])

    old_argv = sys.argv
    try:
        sys.argv = tracker_argv
        tracker_args = semantic.base.parse_args()
    finally:
        sys.argv = old_argv
    semantic.process_sequence(tracker_args, args.sequence_dir, args.tracking_output_dir)
    return args.tracking_output_dir / "road_network_info_local.json"


def load_road_network(args: argparse.Namespace) -> dict:
    path = args.road_network if args.road_network else run_semantic_tracker(args)
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    data["_source_path"] = str(path)
    return data


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
    if not poses:
        raise ValueError(f"No mapping poses in {path}")
    return poses


def resolve_sequence_origin_pose_file(args: argparse.Namespace, source: dict) -> Path:
    if args.sequence_origin_pose_file:
        return args.sequence_origin_pose_file
    if args.pose_file:
        return args.pose_file
    if args.sequence_dir:
        return args.sequence_dir / "result" / "mapping" / "mapping_pose.txt"
    metadata_pose_file = source.get("metadata", {}).get("pose_file")
    if metadata_pose_file:
        return Path(metadata_pose_file)
    raise ValueError(
        "--xodr-coordinate-frame=sequence_start_local needs --sequence-origin-pose-file, "
        "--pose-file, --sequence-dir, or metadata.pose_file"
    )


def local_point_to_enu(point: Sequence[float], transform: dict) -> Point:
    c = float(transform["cos_yaw"])
    s = float(transform["sin_yaw"])
    x0 = float(transform["origin_enu_x"])
    y0 = float(transform["origin_enu_y"])
    z = float(point[2]) if len(point) > 2 else 0.0
    return (x0 + c * float(point[0]) - s * float(point[1]), y0 + s * float(point[0]) + c * float(point[1]), z)


def enu_point_to_origin_local(point: Sequence[float], origin_pose: dict[str, float]) -> Point:
    dx = float(point[0]) - float(origin_pose["x"])
    dy = float(point[1]) - float(origin_pose["y"])
    yaw0 = float(origin_pose["yaw"])
    c = math.cos(-yaw0)
    s = math.sin(-yaw0)
    z = float(point[2]) if len(point) > 2 else 0.0
    return (c * dx - s * dy, s * dx + c * dy, z)


def transform_point_for_xodr(point: Sequence[float], args: argparse.Namespace, source: dict, origin_pose: dict[str, float] | None) -> Point:
    if args.xodr_coordinate_frame == "corrected_local":
        return point_tuple(point)
    local_to_enu = source.get("metadata", {}).get("local_to_enu")
    if not local_to_enu:
        raise ValueError(f"{source.get('_source_path', 'road_network')} has no metadata.local_to_enu")
    enu = local_point_to_enu(point, local_to_enu)
    if args.xodr_coordinate_frame == "enu":
        return enu
    if origin_pose is None:
        raise ValueError("sequence_start_local needs an origin pose")
    return enu_point_to_origin_local(enu, origin_pose)


def prepare_road_fills_for_xodr(
    road_fills: Sequence[dict],
    args: argparse.Namespace,
    source: dict,
) -> list[dict]:
    if args.xodr_coordinate_frame == "corrected_local":
        return [copy.deepcopy(fill) for fill in road_fills]

    origin_pose = None
    origin_frame = None
    if args.xodr_coordinate_frame == "sequence_start_local":
        pose_file = resolve_sequence_origin_pose_file(args, source)
        poses = read_mapping_poses(pose_file)
        origin_frame = args.sequence_origin_frame
        if origin_frame not in poses:
            raise ValueError(f"origin frame {origin_frame} is not present in {pose_file}")
        origin_pose = poses[origin_frame]

    transformed: list[dict] = []
    for fill in road_fills:
        item = copy.deepcopy(fill)
        for key in ("polygon_world", "centerline_world", "traffic_manager_centerline_world"):
            if key in item:
                item[key] = [
                    list(transform_point_for_xodr(point, args, source, origin_pose))
                    for point in item.get(key, [])
                ]
        transformed.append(item)

    metadata = source.setdefault("metadata", {})
    metadata["xodr_coordinate_frame"] = args.xodr_coordinate_frame
    if args.xodr_coordinate_frame == "sequence_start_local":
        metadata["xodr_sequence_origin_frame"] = int(origin_frame)
        metadata["xodr_sequence_origin_pose"] = origin_pose
    return transformed


def fill_center_n(fill: dict) -> float:
    n0, n1 = fill.get("n_range", [0.0, 0.0])
    return (float(n0) + float(n1)) * 0.5


def fill_length(fill: dict) -> float:
    s0, s1 = fill.get("s_range", [0.0, 0.0])
    return float(s1) - float(s0)


def fill_covers_target(fill: dict, target_s: float, target_s_tolerance: float) -> bool:
    s0, s1 = fill.get("s_range", [0.0, 0.0])
    return float(s0) - target_s_tolerance <= target_s <= float(s1) + target_s_tolerance


def fill_center_s(fill: dict) -> float:
    s0, s1 = fill.get("s_range", [0.0, 0.0])
    return (float(s0) + float(s1)) * 0.5


def choose_road_fill(
    road_fills: Sequence[dict],
    road_fill_id: int | None,
    target_s: float,
    target_s_tolerance: float,
) -> dict:
    if not road_fills:
        raise ValueError("road_network contains no road_fills")
    if road_fill_id is not None:
        for fill in road_fills:
            if int(fill.get("road_fill_id", -1)) == road_fill_id:
                return fill
        raise ValueError(f"No road_fill_id={road_fill_id} found")

    candidates = [fill for fill in road_fills if fill_covers_target(fill, target_s, target_s_tolerance)]
    if not candidates:
        candidates = list(road_fills)

    def score(fill: dict) -> tuple[float, float]:
        return (-abs(fill_center_n(fill)), fill_length(fill))

    return max(candidates, key=score)


def parse_road_fill_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    if not ids:
        raise ValueError("--road-fill-ids did not contain any ids")
    return ids


def choose_road_fills_by_ids(road_fills: Sequence[dict], road_fill_ids: Sequence[int]) -> list[dict]:
    by_id = {int(fill.get("road_fill_id", -1)): fill for fill in road_fills}
    missing = [road_fill_id for road_fill_id in road_fill_ids if road_fill_id not in by_id]
    if missing:
        raise ValueError(f"No road_fill_id(s) found: {missing}")
    return [by_id[road_fill_id] for road_fill_id in road_fill_ids]


def choose_multi_lane_fills(
    road_fills: Sequence[dict],
    target_fill: dict,
    target_s: float,
    target_s_tolerance: float,
    traffic_direction: str,
) -> list[dict]:
    candidates = [fill for fill in road_fills if fill_covers_target(fill, target_s, target_s_tolerance)]
    if not candidates:
        effective_target_s = fill_center_s(target_fill)
        candidates = [
            fill for fill in road_fills if fill_covers_target(fill, effective_target_s, target_s_tolerance)
        ]
    if not candidates:
        candidates = [target_fill]

    if traffic_direction == "auto":
        traffic_direction = str(target_fill.get("traffic_direction", "all"))
    if traffic_direction != "all":
        candidates = [fill for fill in candidates if fill.get("traffic_direction") == traffic_direction]
    if not candidates:
        candidates = [target_fill]

    # n increases to the left in the corrected local road coordinates.
    return sorted(candidates, key=fill_center_n, reverse=True)


def point_tuple(value: Sequence[float]) -> Point:
    z = float(value[2]) if len(value) > 2 else 0.0
    return (float(value[0]), float(value[1]), z)


def distance_2d(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def resample_polyline(points: Sequence[Point], max_segment_length: float) -> list[Point]:
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        length = distance_2d(a, b)
        if length <= 1e-6:
            continue
        steps = max(1, int(math.ceil(length / max_segment_length)))
        for step in range(1, steps + 1):
            t = step / steps
            out.append(
                (
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                )
            )
    return out


def extend_polyline(points: Sequence[Point], prepend_length: float, append_length: float) -> list[Point]:
    if len(points) < 2:
        return list(points)
    out = list(points)
    if prepend_length > 0.0:
        first, second = out[0], out[1]
        length = distance_2d(first, second)
        if length > 1e-6:
            ux, uy = (second[0] - first[0]) / length, (second[1] - first[1]) / length
            out.insert(0, (first[0] - ux * prepend_length, first[1] - uy * prepend_length, first[2]))
    if append_length > 0.0:
        before, last = out[-2], out[-1]
        length = distance_2d(before, last)
        if length > 1e-6:
            ux, uy = (last[0] - before[0]) / length, (last[1] - before[1]) / length
            out.append((last[0] + ux * append_length, last[1] + uy * append_length, last[2]))
    return out


def filter_short_segments(points: Iterable[Point], min_segment_length: float) -> list[Point]:
    filtered: list[Point] = []
    for point in points:
        if not filtered or distance_2d(filtered[-1], point) >= min_segment_length:
            filtered.append(point)
    if len(filtered) == 1:
        return []
    return filtered


def offset_centerline_to_reference(points: Sequence[Point], lane_width: float, lane_side: str) -> list[Point]:
    if len(points) < 2:
        return list(points)

    half_width = lane_width * 0.5
    reference: list[Point] = []
    for idx, point in enumerate(points):
        if idx == 0:
            prev_point, next_point = point, points[idx + 1]
        elif idx == len(points) - 1:
            prev_point, next_point = points[idx - 1], point
        else:
            prev_point, next_point = points[idx - 1], points[idx + 1]

        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            reference.append(point)
            continue
        ux, uy = dx / norm, dy / norm
        left_x, left_y = -uy, ux
        sign = 1.0 if lane_side == "right" else -1.0
        reference.append((point[0] + left_x * half_width * sign, point[1] + left_y * half_width * sign, point[2]))
    return reference


def add_text(parent: ET.Element, tag: str, **attrib: object) -> ET.Element:
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attrib.items()})


def add_road_to_xodr(
    root: ET.Element,
    points: Sequence[Point],
    lane_widths: Sequence[float],
    road_id: str,
    road_name: str,
    lane_side: str,
) -> None:
    road_length = sum(distance_2d(a, b) for a, b in zip(points, points[1:]))
    road = add_text(root, "road", name=road_name, length=f"{road_length:.6f}", id=road_id, junction="-1")
    add_text(road, "type", s="0.0", type="town")

    plan_view = add_text(road, "planView")
    s = 0.0
    for start, end in zip(points, points[1:]):
        length = distance_2d(start, end)
        if length <= 1e-6:
            continue
        hdg = math.atan2(end[1] - start[1], end[0] - start[0])
        geom = add_text(
            plan_view,
            "geometry",
            s=f"{s:.6f}",
            x=f"{start[0]:.6f}",
            y=f"{start[1]:.6f}",
            hdg=f"{hdg:.12f}",
            length=f"{length:.6f}",
        )
        add_text(geom, "line")
        s += length

    lanes = add_text(road, "lanes")
    lane_section = add_text(lanes, "laneSection", s="0.0")
    center = add_text(lane_section, "center")
    center_lane = add_text(center, "lane", id="0", type="none", level="false")
    add_text(center_lane, "roadMark", sOffset="0.0", type="broken", weight="standard", color="white", width="0.15")

    lane_group_name = "right" if lane_side == "right" else "left"
    lane_group = add_text(lane_section, lane_group_name)
    for index, lane_width in enumerate(lane_widths, start=1):
        lane_id = -index if lane_side == "right" else index
        lane = add_text(lane_group, "lane", id=str(lane_id), type="driving", level="false")
        add_text(lane, "width", sOffset="0.0", a=f"{lane_width:.6f}", b="0.0", c="0.0", d="0.0")
        add_text(lane, "roadMark", sOffset="0.0", type="solid", weight="standard", color="white", width="0.15")


def build_xodr(
    points: Sequence[Point],
    lane_widths: Sequence[float],
    args: argparse.Namespace,
    source: dict,
    road_fills: Sequence[dict],
) -> ET.ElementTree:
    root = ET.Element("OpenDRIVE")
    add_text(
        root,
        "header",
        revMajor="1",
        revMinor="4",
        name=args.road_name,
        version="1.00",
        date=datetime.utcnow().isoformat(timespec="seconds"),
        north=max(p[1] for p in points),
        south=min(p[1] for p in points),
        east=max(p[0] for p in points),
        west=min(p[0] for p in points),
        vendor="classlab",
    )

    add_road_to_xodr(root, points, lane_widths, args.road_id, args.road_name, args.lane_side)
    road = root.find("road")
    user_data = add_text(road, "userData")
    add_text(
        user_data,
        "classlab",
        source=source.get("_source_path", ""),
        road_fill_ids=",".join(str(fill.get("road_fill_id", "")) for fill in road_fills),
        centerline_key=args.centerline_key,
        lane_side=args.lane_side,
        multi_lane=str(args.multi_lane).lower(),
        xodr_coordinate_frame=source.get("metadata", {}).get("xodr_coordinate_frame", args.xodr_coordinate_frame),
        xodr_sequence_origin_frame=source.get("metadata", {}).get("xodr_sequence_origin_frame", ""),
        xodr_sequence_origin_x=source.get("metadata", {}).get("xodr_sequence_origin_pose", {}).get("x", ""),
        xodr_sequence_origin_y=source.get("metadata", {}).get("xodr_sequence_origin_pose", {}).get("y", ""),
        xodr_sequence_origin_yaw=source.get("metadata", {}).get("xodr_sequence_origin_pose", {}).get("yaw", ""),
    )
    return ET.ElementTree(root)


def prepare_reference_for_fill(
    fill: dict,
    args: argparse.Namespace,
    reverse_road_fill_ids: set[int] | None = None,
) -> tuple[list[Point], float]:
    raw_points = fill.get(args.centerline_key) or fill.get("centerline_world")
    if not raw_points or len(raw_points) < 2:
        raise ValueError(f"road_fill_id={fill.get('road_fill_id')} has no usable centerline")
    fill_id = int(fill.get("road_fill_id", -1))
    if reverse_road_fill_ids and fill_id in reverse_road_fill_ids:
        raw_points = list(reversed(raw_points))
    lane_width = float(args.lane_width if args.lane_width is not None else fill.get("median_width", 3.6))
    centerline = [point_tuple(point) for point in raw_points]
    centerline = extend_polyline(centerline, args.prepend_length, args.append_length)
    centerline = resample_polyline(centerline, args.max_segment_length)
    centerline = filter_short_segments(centerline, args.min_segment_length)
    reference = offset_centerline_to_reference(centerline, lane_width, args.lane_side)
    reference = filter_short_segments(reference, args.min_segment_length)
    if len(reference) < 2:
        raise ValueError(f"road_fill_id={fill.get('road_fill_id')} has too few valid points for OpenDRIVE")
    return reference, lane_width


def build_separate_roads_xodr(
    references: Sequence[tuple[dict, Sequence[Point], float]],
    args: argparse.Namespace,
    source: dict,
) -> ET.ElementTree:
    all_points = [point for _, points, _ in references for point in points]
    root = ET.Element("OpenDRIVE")
    add_text(
        root,
        "header",
        revMajor="1",
        revMinor="4",
        name=args.road_name,
        version="1.00",
        date=datetime.utcnow().isoformat(timespec="seconds"),
        north=max(p[1] for p in all_points),
        south=min(p[1] for p in all_points),
        east=max(p[0] for p in all_points),
        west=min(p[0] for p in all_points),
        vendor="classlab",
    )

    for index, (fill, points, lane_width) in enumerate(references, start=1):
        road_id = str(index)
        road_name = f"{args.road_name}_{fill.get('road_fill_id')}"
        add_road_to_xodr(root, points, [lane_width], road_id, road_name, args.lane_side)
        road = root.findall("road")[-1]
        user_data = add_text(road, "userData")
        add_text(
            user_data,
            "classlab",
            source=source.get("_source_path", ""),
            road_fill_ids=str(fill.get("road_fill_id", "")),
            centerline_key=args.centerline_key,
            lane_side=args.lane_side,
            separate_roads="true",
            xodr_coordinate_frame=source.get("metadata", {}).get("xodr_coordinate_frame", args.xodr_coordinate_frame),
            xodr_sequence_origin_frame=source.get("metadata", {}).get("xodr_sequence_origin_frame", ""),
            xodr_sequence_origin_x=source.get("metadata", {}).get("xodr_sequence_origin_pose", {}).get("x", ""),
            xodr_sequence_origin_y=source.get("metadata", {}).get("xodr_sequence_origin_pose", {}).get("y", ""),
            xodr_sequence_origin_yaw=source.get("metadata", {}).get("xodr_sequence_origin_pose", {}).get("yaw", ""),
        )
    return ET.ElementTree(root)


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in element:
            indent_xml(child, level + 1)
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def main() -> int:
    args = parse_args()
    data = load_road_network(args)
    road_fill_ids = parse_road_fill_ids(args.road_fill_ids)
    reverse_road_fill_ids = set(parse_road_fill_ids(args.reverse_road_fill_ids) or [])
    if road_fill_ids is not None:
        road_fills = choose_road_fills_by_ids(data.get("road_fills", []), road_fill_ids)
    else:
        road_fill = choose_road_fill(data.get("road_fills", []), args.road_fill_id, args.target_s, args.target_s_tolerance)
        road_fills = (
            choose_multi_lane_fills(
                data.get("road_fills", []),
                road_fill,
                args.target_s,
                args.target_s_tolerance,
                args.traffic_direction,
            )
            if args.multi_lane
            else [road_fill]
        )
    road_fills = prepare_road_fills_for_xodr(road_fills, args, data)

    if args.separate_roads:
        references = []
        for fill in road_fills:
            reference, lane_width = prepare_reference_for_fill(fill, args, reverse_road_fill_ids)
            references.append((fill, reference, lane_width))
        tree = build_separate_roads_xodr(references, args, data)
        indent_xml(tree.getroot())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tree.write(args.output, encoding="UTF-8", xml_declaration=True)
        selected_ids = ",".join(str(fill.get("road_fill_id")) for fill in road_fills)
        widths = ",".join(f"{lane_width:.3f}" for _, _, lane_width in references)
        point_counts = ",".join(str(len(reference)) for _, reference, _ in references)
        print(f"selected road_fill_ids={selected_ids} lane_widths={widths}m points={point_counts} separate_roads=true")
        print(f"wrote {args.output}")
        return 0

    reference_fill = road_fills[0]
    lane_widths = [
        float(args.lane_width if args.lane_width is not None else fill.get("median_width", 3.6))
        for fill in road_fills
    ]
    reference, _ = prepare_reference_for_fill(reference_fill, args, reverse_road_fill_ids)

    tree = build_xodr(reference, lane_widths, args, data, road_fills)
    indent_xml(tree.getroot())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="UTF-8", xml_declaration=True)
    selected_ids = ",".join(str(fill.get("road_fill_id")) for fill in road_fills)
    widths = ",".join(f"{width:.3f}" for width in lane_widths)
    print(f"selected road_fill_ids={selected_ids} lane_widths={widths}m points={len(reference)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
