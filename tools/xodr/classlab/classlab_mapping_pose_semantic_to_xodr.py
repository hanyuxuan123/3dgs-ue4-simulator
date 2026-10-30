#!/usr/bin/env python3
"""Generate XODR from ClassLab road fills built with mapping_pose trajectory.

This is a mapping-pose-first pipeline:

1. Optionally run track_classlab_roadlanes_bev_semantic_mapping_pose.py logic to
   build semantic road fills without lane-matching pose correction.
2. Export selected road_fills as OpenDRIVE roads.

The exported road direction follows each fill's traffic_manager_centerline_world.
The default layout intentionally keeps the two driving directions as separate
roads and gives each selected road two lanes.  This matches CARLA/libOpenDRIVE
behavior for the ClassLab straight-road semantic fill better than putting both
directions into one OpenDRIVE road.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET

import track_classlab_roadlanes_bev_semantic_mapping_pose as mapping_semantic


Point = tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence-dir", "--sequence_dir", type=Path, help="Raw ClassLab sequence directory.")
    source.add_argument("--road-network", "--road_network", type=Path, help="Existing road_network_info_local.json.")
    parser.add_argument(
        "--tracking-output-dir",
        "--tracking_output_dir",
        type=Path,
        help="Output directory for mapping-pose semantic road fill products when --sequence-dir is used.",
    )
    parser.add_argument("--lane-csv", "--lane_csv", type=Path)
    parser.add_argument("--pose-file", "--pose_file", type=Path)
    parser.add_argument("--start-frame", "--start_frame", type=int)
    parser.add_argument("--end-frame", "--end_frame", type=int)
    parser.add_argument("--frame-step", "--frame_step", type=int, default=1)
    parser.add_argument("--sequence-origin-frame", "--sequence_origin_frame", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--debug-plot", "--debug_plot", type=Path)
    parser.add_argument(
        "--road-fill-ids",
        "--road_fill_ids",
        help=(
            "Comma-separated road_fill_id list. For the default ClassLab 4-lane export, "
            "use 3,0 or omit this argument to auto-select the main with/against fills."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("separate-roads", "bidirectional-road"),
        default="separate-roads",
        help="OpenDRIVE layout. bidirectional-road builds one road with left/right lanes from selected road_fills.",
    )
    parser.add_argument(
        "--reverse-road-fill-ids",
        "--reverse_road_fill_ids",
        help="Comma-separated selected road_fill_id list to reverse after traffic-manager direction is applied.",
    )
    parser.add_argument(
        "--centerline-key",
        "--centerline_key",
        choices=("traffic_manager_centerline_world", "centerline_world"),
        default="traffic_manager_centerline_world",
    )
    parser.add_argument("--lane-width", "--lane_width", type=float)
    parser.add_argument(
        "--lanes-per-road",
        "--lanes_per_road",
        type=int,
        default=2,
        help="Number of same-direction driving lanes generated for each selected road_fill centerline.",
    )
    parser.add_argument("--lane-side", "--lane_side", choices=("right", "left"), default="right")
    parser.add_argument("--prepend-length", "--prepend_length", type=float, default=10.0)
    parser.add_argument("--append-length", "--append_length", type=float, default=0.0)
    parser.add_argument("--max-segment-length", "--max_segment_length", type=float, default=6.0)
    parser.add_argument("--min-segment-length", "--min_segment_length", type=float, default=0.25)
    parser.add_argument("--road-name", "--road_name", default="classlab_mapping_pose_road")
    return parser.parse_args()


def parse_id_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def point_tuple(value: Sequence[float]) -> Point:
    z = float(value[2]) if len(value) > 2 else 0.0
    return (float(value[0]), float(value[1]), z)


def distance_2d(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def extend_polyline(points: Sequence[Point], prepend_length: float, append_length: float) -> list[Point]:
    output = list(points)
    if len(output) < 2:
        return output
    if prepend_length > 0.0:
        first, second = output[0], output[1]
        length = distance_2d(first, second)
        if length > 1e-6:
            ux = (second[0] - first[0]) / length
            uy = (second[1] - first[1]) / length
            output.insert(0, (first[0] - ux * prepend_length, first[1] - uy * prepend_length, first[2]))
    if append_length > 0.0:
        before, last = output[-2], output[-1]
        length = distance_2d(before, last)
        if length > 1e-6:
            ux = (last[0] - before[0]) / length
            uy = (last[1] - before[1]) / length
            output.append((last[0] + ux * append_length, last[1] + uy * append_length, last[2]))
    return output


def resample_polyline(points: Sequence[Point], max_segment_length: float) -> list[Point]:
    if len(points) < 2:
        return list(points)
    output = [points[0]]
    for a, b in zip(points, points[1:]):
        length = distance_2d(a, b)
        if length <= 1e-6:
            continue
        steps = max(1, int(math.ceil(length / max_segment_length)))
        for step in range(1, steps + 1):
            t = step / steps
            output.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))
    return output


def filter_short_segments(points: Sequence[Point], min_segment_length: float) -> list[Point]:
    output: list[Point] = []
    for point in points:
        if not output or distance_2d(output[-1], point) >= min_segment_length:
            output.append(point)
    return output if len(output) > 1 else []


def offset_centerline_to_reference(points: Sequence[Point], lane_width: float, lane_side: str) -> list[Point]:
    half_width = lane_width * 0.5
    reference: list[Point] = []
    for index, point in enumerate(points):
        prev_point = points[index - 1] if index > 0 else point
        next_point = points[index + 1] if index < len(points) - 1 else point
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            reference.append(point)
            continue
        left_x = -dy / norm
        left_y = dx / norm
        sign = 1.0 if lane_side == "right" else -1.0
        reference.append((point[0] + left_x * half_width * sign, point[1] + left_y * half_width * sign, point[2]))
    return reference


def add_text(parent: ET.Element, tag: str, **attrib: object) -> ET.Element:
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attrib.items()})


def add_road(
    root: ET.Element,
    road_id: int,
    road_name: str,
    points: Sequence[Point],
    lane_width: float,
    lanes_per_road: int,
) -> ET.Element:
    road_length = sum(distance_2d(a, b) for a, b in zip(points, points[1:]))
    road = add_text(root, "road", name=road_name, length=f"{road_length:.6f}", id=str(road_id), junction="-1")
    add_text(road, "type", s="0.0", type="town")
    plan_view = add_text(road, "planView")
    road_s = 0.0
    for start, end in zip(points, points[1:]):
        length = distance_2d(start, end)
        if length <= 1e-6:
            continue
        hdg = math.atan2(end[1] - start[1], end[0] - start[0])
        geometry = add_text(plan_view, "geometry", s=f"{road_s:.6f}", x=f"{start[0]:.6f}", y=f"{start[1]:.6f}", hdg=f"{hdg:.12f}", length=f"{length:.6f}")
        add_text(geometry, "line")
        road_s += length
    lanes = add_text(road, "lanes")
    lane_section = add_text(lanes, "laneSection", s="0.0")
    center = add_text(lane_section, "center")
    center_lane = add_text(center, "lane", id="0", type="none", level="false")
    add_text(center_lane, "roadMark", sOffset="0.0", type="broken", weight="standard", color="white", width="0.15")
    group = add_text(lane_section, "right")
    for lane_index in range(1, max(1, lanes_per_road) + 1):
        lane = add_text(group, "lane", id=str(-lane_index), type="driving", level="false")
        add_text(lane, "width", sOffset="0.0", a=f"{lane_width:.6f}", b="0.0", c="0.0", d="0.0")
        road_mark_type = "broken" if lane_index < max(1, lanes_per_road) else "solid"
        add_text(lane, "roadMark", sOffset="0.0", type=road_mark_type, weight="standard", color="white", width="0.15")
    return road


def add_lane_group(
    lane_section: ET.Element,
    side: str,
    lane_ids: Sequence[int],
    lane_widths: Sequence[float],
) -> None:
    group = add_text(lane_section, side)
    for index, (lane_id, width) in enumerate(zip(lane_ids, lane_widths), start=1):
        lane = add_text(group, "lane", id=str(lane_id), type="driving", level="false")
        add_text(lane, "width", sOffset="0.0", a=f"{width:.6f}", b="0.0", c="0.0", d="0.0")
        road_mark_type = "broken" if index < len(lane_ids) else "solid"
        add_text(lane, "roadMark", sOffset="0.0", type=road_mark_type, weight="standard", color="white", width="0.15")


def add_bidirectional_road(
    root: ET.Element,
    road_id: int,
    road_name: str,
    reference_points: Sequence[Point],
    right_widths: Sequence[float],
    left_widths: Sequence[float],
) -> ET.Element:
    road_length = sum(distance_2d(a, b) for a, b in zip(reference_points, reference_points[1:]))
    road = add_text(root, "road", name=road_name, length=f"{road_length:.6f}", id=str(road_id), junction="-1")
    add_text(road, "type", s="0.0", type="town")
    plan_view = add_text(road, "planView")
    road_s = 0.0
    for start, end in zip(reference_points, reference_points[1:]):
        length = distance_2d(start, end)
        if length <= 1e-6:
            continue
        hdg = math.atan2(end[1] - start[1], end[0] - start[0])
        geometry = add_text(
            plan_view,
            "geometry",
            s=f"{road_s:.6f}",
            x=f"{start[0]:.6f}",
            y=f"{start[1]:.6f}",
            hdg=f"{hdg:.12f}",
            length=f"{length:.6f}",
        )
        add_text(geometry, "line")
        road_s += length
    lanes = add_text(road, "lanes")
    lane_section = add_text(lanes, "laneSection", s="0.0")
    center = add_text(lane_section, "center")
    center_lane = add_text(center, "lane", id="0", type="none", level="false")
    add_text(center_lane, "roadMark", sOffset="0.0", type="solid", weight="standard", color="yellow", width="0.15")
    if left_widths:
        add_lane_group(lane_section, "left", list(range(1, len(left_widths) + 1)), left_widths)
    if right_widths:
        add_lane_group(lane_section, "right", [-idx for idx in range(1, len(right_widths) + 1)], right_widths)
    return road


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


def fill_center_n(fill: dict) -> float:
    n0, n1 = fill.get("n_range", [0.0, 0.0])
    return (float(n0) + float(n1)) * 0.5


def fill_length(fill: dict) -> float:
    s0, s1 = fill.get("s_range", [0.0, 0.0])
    return float(s1) - float(s0)


def choose_default_fills(fills: Sequence[dict]) -> list[dict]:
    selected = []
    for direction in ("with_pose_s", "against_pose_s"):
        candidates = [fill for fill in fills if fill.get("traffic_direction") == direction]
        if not candidates:
            continue
        selected.append(max(candidates, key=lambda fill: (-abs(fill_center_n(fill)), fill_length(fill))))
    if selected:
        return selected
    if not fills:
        raise ValueError("road network contains no road_fills")
    return [max(fills, key=fill_length)]


def choose_fills(data: dict, ids: Sequence[int]) -> list[dict]:
    fills = data.get("road_fills", [])
    if not ids:
        return choose_default_fills(fills)
    by_id = {int(fill["road_fill_id"]): fill for fill in fills}
    missing = [road_id for road_id in ids if road_id not in by_id]
    if missing:
        raise ValueError(f"missing road_fill_id(s): {missing}")
    return [by_id[road_id] for road_id in ids]


def prepare_road_points(fill: dict, args: argparse.Namespace, reverse_ids: set[int]) -> tuple[list[Point], float]:
    fill_id = int(fill["road_fill_id"])
    raw_points = fill.get(args.centerline_key) or fill.get("centerline_world")
    if not raw_points or len(raw_points) < 2:
        raise ValueError(f"road_fill_id={fill_id} has no usable {args.centerline_key}")
    points = [point_tuple(point) for point in raw_points]
    if fill_id in reverse_ids:
        points.reverse()
    points = extend_polyline(points, args.prepend_length, args.append_length)
    points = resample_polyline(points, args.max_segment_length)
    points = filter_short_segments(points, args.min_segment_length)
    lane_width = float(args.lane_width if args.lane_width is not None else fill.get("median_width", 3.6))
    reference = offset_centerline_to_reference(points, lane_width, args.lane_side)
    reference = filter_short_segments(reference, args.min_segment_length)
    if len(reference) < 2:
        raise ValueError(f"road_fill_id={fill_id} produced too few reference points")
    return reference, lane_width


def prepare_centerline_points(fill: dict, args: argparse.Namespace, reverse: bool = False) -> list[Point]:
    raw_points = fill.get(args.centerline_key) or fill.get("centerline_world")
    if not raw_points or len(raw_points) < 2:
        raise ValueError(f"road_fill_id={fill.get('road_fill_id')} has no usable {args.centerline_key}")
    points = [point_tuple(point) for point in raw_points]
    if reverse:
        points.reverse()
    points = extend_polyline(points, args.prepend_length, args.append_length)
    points = resample_polyline(points, args.max_segment_length)
    return filter_short_segments(points, args.min_segment_length)


def fill_width(fill: dict, args: argparse.Namespace) -> float:
    if args.lane_width is not None:
        return float(args.lane_width)
    if "median_width" in fill:
        return float(fill["median_width"])
    n0, n1 = fill.get("n_range", [0.0, 3.6])
    return max(0.1, abs(float(n1) - float(n0)))


def build_bidirectional_road(
    root: ET.Element,
    data: dict,
    source_path: Path,
    args: argparse.Namespace,
) -> list[tuple[dict, list[Point], float]]:
    fills = choose_fills(data, parse_id_list(args.road_fill_ids))
    with_fills = [fill for fill in fills if fill.get("traffic_direction") == "with_pose_s"]
    against_fills = [fill for fill in fills if fill.get("traffic_direction") == "against_pose_s"]
    if not with_fills or not against_fills:
        raise ValueError("--layout bidirectional-road requires both with_pose_s and against_pose_s road_fills")

    with_upper = max(float(fill.get("n_range", [0.0, 0.0])[1]) for fill in with_fills)
    against_lower = min(float(fill.get("n_range", [0.0, 0.0])[0]) for fill in against_fills)
    divider_n = 0.5 * (with_upper + against_lower)

    source_fill = min(
        with_fills,
        key=lambda fill: (abs(float(fill.get("n_range", [0.0, 0.0])[1]) - divider_n), -fill_length(fill)),
    )
    source_center_n = fill_center_n(source_fill)
    reference_points = prepare_centerline_points(source_fill, args)
    reference_points = offset_points(reference_points, divider_n - source_center_n)
    reference_points = filter_short_segments(reference_points, args.min_segment_length)
    if len(reference_points) < 2:
        raise ValueError("bidirectional-road reference line has too few points")

    right_fills = sorted(with_fills, key=lambda fill: float(fill.get("n_range", [0.0, 0.0])[1]), reverse=True)
    left_fills = sorted(against_fills, key=lambda fill: float(fill.get("n_range", [0.0, 0.0])[0]))
    right_widths = [fill_width(fill, args) for fill in right_fills]
    left_widths = [fill_width(fill, args) for fill in left_fills]

    add_bidirectional_road(root, 1, args.road_name, reference_points, right_widths, left_widths)
    road = root.find("road")
    if road is not None:
        metadata = data.get("metadata", {})
        user_data = add_text(road, "userData")
        add_text(
            user_data,
            "classlab",
            source=str(source_path),
            road_fill_ids=",".join(str(fill["road_fill_id"]) for fill in fills),
            right_road_fill_ids=",".join(str(fill["road_fill_id"]) for fill in right_fills),
            left_road_fill_ids=",".join(str(fill["road_fill_id"]) for fill in left_fills),
            pose_source=str(metadata.get("pose_source", "")),
            coordinate_frame=str(metadata.get("coordinate_frame", "")),
            centerline_key=args.centerline_key,
            divider_n=f"{divider_n:.6f}",
            sequence_origin_frame=str(metadata.get("sequence_origin_frame", args.sequence_origin_frame)),
        )
    debug_fill = {
        "road_fill_id": "bidirectional:" + ",".join(str(fill["road_fill_id"]) for fill in fills),
        "traffic_direction": "bidirectional",
        "right_road_fill_ids": [int(fill["road_fill_id"]) for fill in right_fills],
        "left_road_fill_ids": [int(fill["road_fill_id"]) for fill in left_fills],
        "right_widths": right_widths,
        "left_widths": left_widths,
    }
    return [(debug_fill, reference_points, sum(right_widths) + sum(left_widths))]


def run_mapping_pose_semantic(args: argparse.Namespace) -> Path:
    if args.tracking_output_dir is None:
        raise ValueError("--tracking-output-dir is required with --sequence-dir")
    # Fill defaults from the original parser so tracking thresholds and drawing
    # options remain identical to the standard ClassLab semantic script.
    original_argv = sys.argv
    try:
        sys.argv = [
            "track_classlab_roadlanes_bev_semantic_mapping_pose.py",
            "--sequence-dir",
            str(args.sequence_dir),
            "--output-dir",
            str(args.tracking_output_dir),
        ]
        default_args = mapping_semantic.parse_args()
    finally:
        sys.argv = original_argv
    semantic_args = argparse.Namespace(**vars(default_args))
    for key in ("lane_csv", "pose_file", "start_frame", "end_frame", "frame_step", "sequence_origin_frame"):
        setattr(semantic_args, key, getattr(args, key))
    if semantic_args.lane_csv is not None:
        semantic_args.lane_csv = str(semantic_args.lane_csv)
    if semantic_args.pose_file is not None:
        semantic_args.pose_file = str(semantic_args.pose_file)
    semantic_args.sequence_dir = str(args.sequence_dir)
    semantic_args.output_dir = str(args.tracking_output_dir)
    mapping_semantic.process_sequence_mapping_pose(semantic_args, args.sequence_dir, args.tracking_output_dir)
    return args.tracking_output_dir / "road_network_info_local.json"


def load_road_network(args: argparse.Namespace) -> tuple[dict, Path]:
    if args.road_network:
        path = args.road_network
    else:
        path = run_mapping_pose_semantic(args)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, path


def write_xodr(data: dict, source_path: Path, args: argparse.Namespace) -> list[tuple[dict, list[Point], float]]:
    if args.layout == "bidirectional-road":
        selected_fills = choose_fills(data, parse_id_list(args.road_fill_ids))
        all_points = [
            point_tuple(point)
            for fill in selected_fills
            for point in (fill.get(args.centerline_key) or fill.get("centerline_world") or [])
        ]
    else:
        fills = choose_fills(data, parse_id_list(args.road_fill_ids))
        reverse_ids = set(parse_id_list(args.reverse_road_fill_ids))
        roads = []
        for fill in fills:
            points, lane_width = prepare_road_points(fill, args, reverse_ids)
            roads.append((fill, points, lane_width))
        all_points = [point for _, points, _ in roads for point in points]

    root = ET.Element("OpenDRIVE")
    add_text(
        root,
        "header",
        revMajor="1",
        revMinor="4",
        name=args.road_name,
        version="1.00",
        date=datetime.utcnow().isoformat(timespec="seconds"),
        north=max(point[1] for point in all_points),
        south=min(point[1] for point in all_points),
        east=max(point[0] for point in all_points),
        west=min(point[0] for point in all_points),
        vendor="classlab",
    )
    if args.layout == "bidirectional-road":
        roads = build_bidirectional_road(root, data, source_path, args)
        indent_xml(root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(args.output, encoding="UTF-8", xml_declaration=True)
        return roads

    metadata = data.get("metadata", {})
    for index, (fill, points, lane_width) in enumerate(roads, start=1):
        fill_id = int(fill["road_fill_id"])
        road = add_road(root, index, f"{args.road_name}_{fill_id}", points, lane_width, args.lanes_per_road)
        user_data = add_text(road, "userData")
        add_text(
            user_data,
            "classlab",
            source=str(source_path),
            road_fill_ids=str(fill_id),
            pose_source=str(metadata.get("pose_source", "")),
            coordinate_frame=str(metadata.get("coordinate_frame", "")),
            centerline_key=args.centerline_key,
            traffic_direction=str(fill.get("traffic_direction", "")),
            lanes_per_road=str(args.lanes_per_road),
            sequence_origin_frame=str(metadata.get("sequence_origin_frame", args.sequence_origin_frame)),
        )
    indent_xml(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding="UTF-8", xml_declaration=True)
    return [(fill, points, lane_width) for fill, points, lane_width in roads]


def offset_points(points: Sequence[Point], offset: float) -> list[Point]:
    output: list[Point] = []
    for index, point in enumerate(points):
        prev_point = points[index - 1] if index > 0 else point
        next_point = points[index + 1] if index < len(points) - 1 else point
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            output.append(point)
            continue
        left_x = -dy / norm
        left_y = dx / norm
        output.append((point[0] + left_x * offset, point[1] + left_y * offset, point[2]))
    return output


def draw_debug_plot(path: Path, roads: Sequence[tuple[dict, list[Point], float]], data: dict, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#10161d")
    ax.set_facecolor("#10161d")
    colors = ["#56b4e9", "#e69f00", "#7fdbff", "#ffb000"]
    for index, fill in enumerate(data.get("road_fills", [])):
        center = fill.get("traffic_manager_centerline_world", [])
        if len(center) < 2:
            continue
        xs = [point[0] for point in center]
        ys = [point[1] for point in center]
        ax.plot(xs, ys, color="#59636e", linewidth=0.6, alpha=0.35)
    for index, (fill, points, lane_width) in enumerate(roads):
        color = colors[index % len(colors)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if fill.get("traffic_direction") == "bidirectional":
            right_width = sum(float(width) for width in fill.get("right_widths", []))
            left_width = sum(float(width) for width in fill.get("left_widths", []))
            right_edge = offset_points(points, -right_width)
            left_edge = offset_points(points, left_width)
            polygon = list(left_edge) + list(reversed(right_edge))
        else:
            total_width = lane_width * max(1, args.lanes_per_road)
            edge_offset = -total_width if args.lane_side == "right" else total_width
            edge = offset_points(points, edge_offset)
            polygon = list(points) + list(reversed(edge))
        ax.fill(
            [point[0] for point in polygon],
            [point[1] for point in polygon],
            color=color,
            alpha=0.28,
            linewidth=0,
        )
        if fill.get("traffic_direction") == "bidirectional":
            offset = 0.0
            for width in fill.get("right_widths", [])[:-1]:
                offset -= float(width)
                lane_edge = offset_points(points, offset)
                ax.plot([point[0] for point in lane_edge], [point[1] for point in lane_edge], color="#e8eef5", linewidth=0.9, linestyle=(0, (6, 6)), alpha=0.9)
            offset = 0.0
            for width in fill.get("left_widths", [])[:-1]:
                offset += float(width)
                lane_edge = offset_points(points, offset)
                ax.plot([point[0] for point in lane_edge], [point[1] for point in lane_edge], color="#e8eef5", linewidth=0.9, linestyle=(0, (6, 6)), alpha=0.9)
        else:
            for lane_index in range(1, max(1, args.lanes_per_road)):
                lane_edge = offset_points(points, edge_offset * lane_index / max(1, args.lanes_per_road))
                ax.plot(
                    [point[0] for point in lane_edge],
                    [point[1] for point in lane_edge],
                    color="#e8eef5",
                    linewidth=0.9,
                    linestyle=(0, (6, 6)),
                    alpha=0.9,
                )
        ax.plot(xs, ys, color=color, linewidth=2.5, label=f"xodr road_fill {fill['road_fill_id']}")
        stride = max(len(points) // 24, 1)
        for a, b in zip(points[::stride], points[1::stride]):
            ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3})
    ax.axis("equal")
    ax.grid(True, color="#26313b", linewidth=0.6)
    ax.set_xlabel(f"{data.get('metadata', {}).get('coordinate_frame', 'local')} x / m")
    ax.set_ylabel(f"{data.get('metadata', {}).get('coordinate_frame', 'local')} y / m")
    ax.set_title("Mapping-pose semantic road fills exported to XODR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data, source_path = load_road_network(args)
    roads = write_xodr(data, source_path, args)
    if args.debug_plot:
        draw_debug_plot(args.debug_plot, roads, data, args)
    selected_ids = ",".join(str(fill["road_fill_id"]) for fill, _, _ in roads)
    print(f"selected road_fill_ids={selected_ids}")
    print(f"roads={len(roads)} wrote: {args.output}")
    if args.debug_plot:
        print(f"debug plot: {args.debug_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
