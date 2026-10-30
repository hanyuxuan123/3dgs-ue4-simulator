#!/usr/bin/env python3
"""Generate ClassLab XODR by projecting road-fill s/n onto mapping_pose."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree as ET


Point = tuple[float, float, float]
Pose2 = tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road-network", required=True, type=Path, help="road_network_info_local.json")
    parser.add_argument("--mapping-pose", required=True, type=Path, help="raw result/mapping/mapping_pose.txt")
    parser.add_argument("--output", required=True, type=Path, help="Output .xodr")
    parser.add_argument("--debug-plot", type=Path, help="Optional debug PNG showing mapping trajectory and XODR roads.")
    parser.add_argument("--road-fill-ids", required=True, help="Comma-separated road_fill_id list, e.g. 2,0")
    parser.add_argument("--origin-frame", type=int, default=0, help="sequence_start_local origin frame")
    parser.add_argument(
        "--tracking-start-frame",
        type=int,
        help="Raw lidar frame corresponding to s=0 in road_fill centerline_sn. Defaults to metadata.local_to_enu.origin_frame.",
    )
    parser.add_argument(
        "--tracking-end-frame",
        type=int,
        help="Last raw lidar frame used to build the mapping trajectory. Defaults to metadata args/end_frame or max s coverage.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--prepend-length", type=float, default=10.0)
    parser.add_argument("--append-length", type=float, default=0.0)
    parser.add_argument("--max-segment-length", type=float, default=6.0)
    parser.add_argument("--min-segment-length", type=float, default=0.25)
    parser.add_argument("--lane-side", choices=("right", "left"), default="right")
    parser.add_argument("--road-name", default="classlab_mapping_pose_road")
    parser.add_argument("--lane-width", type=float, help="Override lane width.")
    parser.add_argument(
        "--reverse-road-fill-ids",
        help="Comma-separated selected road_fill_id list whose exported OpenDRIVE direction should be reversed.",
    )
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def parse_id_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
                "yaw": float(parts[index["yaw"]]),
            }
    return poses


def to_origin_local(x: float, y: float, yaw: float, origin: dict[str, float]) -> Pose2:
    dx = x - origin["x"]
    dy = y - origin["y"]
    c = math.cos(-origin["yaw"])
    s = math.sin(-origin["yaw"])
    return (c * dx - s * dy, s * dx + c * dy, normalize_angle(yaw - origin["yaw"]))


def build_mapping_path(
    poses: dict[int, dict[str, float]],
    origin_frame: int,
    start_frame: int,
    end_frame: int,
    frame_step: int,
) -> list[tuple[float, Pose2]]:
    if origin_frame not in poses:
        raise ValueError(f"origin frame {origin_frame} is not in mapping poses")
    origin = poses[origin_frame]
    frames = [frame for frame in sorted(poses) if start_frame <= frame <= end_frame and (frame - start_frame) % frame_step == 0]
    if len(frames) < 2:
        raise ValueError(f"Not enough mapping frames in [{start_frame}, {end_frame}]")
    path: list[tuple[float, Pose2]] = []
    prev: Pose2 | None = None
    distance = 0.0
    for frame in frames:
        pose = poses[frame]
        local = to_origin_local(pose["x"], pose["y"], pose["yaw"], origin)
        if prev is not None:
            distance += math.hypot(local[0] - prev[0], local[1] - prev[1])
        path.append((distance, local))
        prev = local
    return path


def interpolate_pose_by_s(path: Sequence[tuple[float, Pose2]], query_s: float) -> Pose2:
    if query_s <= path[0][0]:
        return path[0][1]
    if query_s >= path[-1][0]:
        return path[-1][1]
    for (s0, p0), (s1, p1) in zip(path, path[1:]):
        if s0 <= query_s <= s1:
            span = max(s1 - s0, 1e-9)
            t = (query_s - s0) / span
            yaw_delta = normalize_angle(p1[2] - p0[2])
            return (
                p0[0] + (p1[0] - p0[0]) * t,
                p0[1] + (p1[1] - p0[1]) * t,
                normalize_angle(p0[2] + yaw_delta * t),
            )
    return path[-1][1]


def sn_to_mapping_point(path: Sequence[tuple[float, Pose2]], s_value: float, n_value: float) -> Point:
    x, y, yaw = interpolate_pose_by_s(path, s_value)
    left_x = -math.sin(yaw)
    left_y = math.cos(yaw)
    return (x + left_x * n_value, y + left_y * n_value, 0.0)


def distance_2d(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


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


def add_road(root: ET.Element, road_id: int, road_name: str, points: Sequence[Point], lane_width: float) -> ET.Element:
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
    add_text(center_lane, "roadMark", sOffset="0.0", type="broken", weight="standard", color="white", width="0.15")
    right = add_text(lane_section, "right")
    lane = add_text(right, "lane", id="-1", type="driving", level="false")
    add_text(lane, "width", sOffset="0.0", a=f"{lane_width:.6f}", b="0.0", c="0.0", d="0.0")
    add_text(lane, "roadMark", sOffset="0.0", type="solid", weight="standard", color="white", width="0.15")
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


def choose_fills(data: dict, ids: Sequence[int]) -> list[dict]:
    by_id = {int(fill["road_fill_id"]): fill for fill in data.get("road_fills", [])}
    missing = [road_id for road_id in ids if road_id not in by_id]
    if missing:
        raise ValueError(f"missing road_fill_id(s): {missing}")
    return [by_id[road_id] for road_id in ids]


def points_from_fill_sn(fill: dict, path: Sequence[tuple[float, Pose2]], reverse: bool) -> list[Point]:
    sn_points = fill.get("centerline_sn") or []
    if reverse:
        sn_points = list(reversed(sn_points))
    return [sn_to_mapping_point(path, float(item[0]), float(item[1])) for item in sn_points]


def draw_debug_plot(path: Path, mapping_path: Sequence[tuple[float, Pose2]], roads: Sequence[tuple[int, list[Point]]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#10161d")
    ax.set_facecolor("#10161d")
    map_x = [pose[0] for _, pose in mapping_path]
    map_y = [pose[1] for _, pose in mapping_path]
    ax.plot(map_x, map_y, "--", color="#cccccc", linewidth=1.5, label="mapping_pose frame0-local")
    colors = ["#56b4e9", "#e69f00", "#7fdbff", "#ffb000"]
    for index, (fill_id, points) in enumerate(roads):
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        color = colors[index % len(colors)]
        ax.plot(xs, ys, color=color, linewidth=2.5, label=f"xodr road_fill {fill_id}")
        stride = max(len(points) // 20, 1)
        for a, b in zip(points[::stride], points[1::stride]):
            ax.annotate("", xy=(b[0], b[1]), xytext=(a[0], a[1]), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4})
    ax.axis("equal")
    ax.grid(True, color="#26313b", linewidth=0.6)
    ax.set_xlabel("sequence_start_local x / m")
    ax.set_ylabel("sequence_start_local y / m")
    ax.set_title("XODR roads projected from road_fill s/n onto mapping_pose")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data = json.loads(args.road_network.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    local_to_enu = metadata.get("local_to_enu", {})
    tracking_start = args.tracking_start_frame
    if tracking_start is None:
        tracking_start = int(local_to_enu.get("origin_frame", metadata.get("args", {}).get("start_frame", 0)))
    road_fill_ids = parse_id_list(args.road_fill_ids)
    reverse_ids = set(parse_id_list(args.reverse_road_fill_ids))
    fills = choose_fills(data, road_fill_ids)
    max_fill_s = max(float(point[0]) for fill in fills for point in fill.get("centerline_sn", []))
    tracking_end = args.tracking_end_frame
    if tracking_end is None:
        metadata_end = metadata.get("args", {}).get("end_frame")
        tracking_end = int(metadata_end) if metadata_end is not None else int(tracking_start + math.ceil(max_fill_s) + 10)

    mapping_poses = read_mapping_poses(args.mapping_pose)
    mapping_path = build_mapping_path(mapping_poses, args.origin_frame, tracking_start, tracking_end, args.frame_step)

    roads: list[tuple[dict, list[Point], float]] = []
    for fill in fills:
        fill_id = int(fill["road_fill_id"])
        centerline = points_from_fill_sn(fill, mapping_path, fill_id in reverse_ids)
        centerline = extend_polyline(centerline, args.prepend_length, args.append_length)
        centerline = resample_polyline(centerline, args.max_segment_length)
        centerline = filter_short_segments(centerline, args.min_segment_length)
        if len(centerline) < 2:
            raise ValueError(f"road_fill_id={fill_id} produced too few points")
        lane_width = float(args.lane_width if args.lane_width is not None else fill.get("median_width", 3.6))
        reference = offset_centerline_to_reference(centerline, lane_width, args.lane_side)
        reference = filter_short_segments(reference, args.min_segment_length)
        roads.append((fill, reference, lane_width))

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
    for road_index, (fill, points, lane_width) in enumerate(roads, start=1):
        fill_id = int(fill["road_fill_id"])
        road = add_road(root, road_index, f"{args.road_name}_{fill_id}", points, lane_width)
        user_data = add_text(road, "userData")
        add_text(
            user_data,
            "classlab",
            source=str(args.road_network),
            road_fill_ids=str(fill_id),
            projection="centerline_sn_on_mapping_pose",
            tracking_start_frame=str(tracking_start),
            tracking_end_frame=str(tracking_end),
            origin_frame=str(args.origin_frame),
            mapping_pose=str(args.mapping_pose),
            traffic_direction=str(fill.get("traffic_direction", "")),
        )
    indent_xml(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding="UTF-8", xml_declaration=True)
    if args.debug_plot:
        draw_debug_plot(args.debug_plot, mapping_path, [(int(fill["road_fill_id"]), points) for fill, points, _ in roads])
    print(f"tracking frames: {tracking_start}-{tracking_end}")
    print(f"selected road_fill_ids={','.join(str(fill['road_fill_id']) for fill, _, _ in roads)}")
    print(f"roads={len(roads)} wrote: {args.output}")
    if args.debug_plot:
        print(f"debug plot: {args.debug_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
