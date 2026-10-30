#!/usr/bin/env python3
"""Fuse per-frame ClassLab lane detections into global lane polylines.

The ClassLab lane.csv format stores lane markings in the ego-vehicle frame:
front is +x, left is +y, up is +z. This script aligns each lane row with
result/mapping/mapping_pose.txt by frameNum/lidar_frame, transforms sampled lane
points into the mapping/world frame, and merges nearby repeated observations
into global lane clusters for BEV inspection and later map conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float, float]


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence_dir",
        required=True,
        help="ClassLab sequence directory containing result/lane.csv and result/mapping/mapping_pose.txt.",
    )
    parser.add_argument("--lane_csv", default=None, help="Override lane.csv path.")
    parser.add_argument("--mapping_pose", default=None, help="Override mapping_pose.txt path.")
    parser.add_argument("--output_dir", required=True, help="Directory for fused lane outputs.")
    parser.add_argument("--start_frame", type=int, default=None, help="Inclusive frameNum/lidar_frame start.")
    parser.add_argument("--end_frame", type=int, default=None, help="Inclusive frameNum/lidar_frame end.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--sample_step", type=float, default=2.0, help="Lane sampling step in meters.")
    parser.add_argument("--min_forward", type=float, default=-5.0, help="Minimum ego-frame x to keep.")
    parser.add_argument("--max_forward", type=float, default=80.0, help="Maximum ego-frame x to keep.")
    parser.add_argument("--s_bin", type=float, default=1.0, help="Longitudinal fusion bin size in meters.")
    parser.add_argument(
        "--lateral_merge",
        type=float,
        default=0.8,
        help="Merge raw points inside one s-bin when lateral distance is below this many meters.",
    )
    parser.add_argument(
        "--lateral_link",
        type=float,
        default=1.2,
        help="Link lane nodes across neighboring s-bins when lateral distance is below this many meters.",
    )
    parser.add_argument(
        "--max_gap_bins",
        type=int,
        default=4,
        help="Maximum empty/skip bins allowed when linking lane nodes.",
    )
    parser.add_argument(
        "--min_cluster_points",
        type=int,
        default=20,
        help="Drop fused lane clusters with fewer output polyline points.",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=50,
        help="Drop fused lane clusters supported by fewer sampled observations.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=7,
        help="Odd moving-average window for fused lane polylines. Use 1 to disable.",
    )
    parser.add_argument(
        "--max_time_diff",
        type=float,
        default=0.2,
        help="Warn when lane timeStamp and mapping lidar_frame_time differ by more than this many seconds.",
    )
    parser.add_argument("--no_raw_points", action="store_true", help="Do not draw raw sampled points in BEV.")
    parser.add_argument("--dpi", type=int, default=220, help="Output plot DPI.")
    parser.add_argument("--fig_width", type=float, default=20.0, help="BEV figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=16.0, help="BEV figure height in inches.")
    parser.add_argument("--plot_margin", type=float, default=30.0, help="BEV plot margin around lane/road data in meters.")
    parser.add_argument("--xlim", nargs=2, type=float, default=None, help="Optional BEV x-axis limits.")
    parser.add_argument("--ylim", nargs=2, type=float, default=None, help="Optional BEV y-axis limits.")
    parser.add_argument("--road_edge_dir", default=None, help="Directory containing Road.pcd/fill_road.pcd/leftRoadEdge.pcd/rightRoadEdge.pcd.")
    parser.add_argument("--draw_road_points", action="store_true", help="Draw Road.pcd/fill_road.pcd drivable-area points.")
    parser.add_argument("--road_point_stride", type=int, default=5, help="Draw every Nth road/fill-road point.")
    parser.add_argument("--edge_point_stride", type=int, default=1, help="Draw every Nth road-edge point.")
    parser.add_argument("--draw_connectors", action="store_true", help="Draw candidate connector/gap links between fused lane endpoints.")
    parser.add_argument("--connector_max_distance", type=float, default=80.0, help="Maximum candidate connector distance in meters.")
    parser.add_argument("--connector_max_heading_deg", type=float, default=80.0, help="Maximum candidate connector heading mismatch.")
    parser.add_argument("--connector_max_lateral", type=float, default=8.0, help="Maximum lateral offset in endpoint local frame.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    lane_csv = Path(args.lane_csv) if args.lane_csv else sequence_dir / "result" / "lane.csv"
    mapping_pose = Path(args.mapping_pose) if args.mapping_pose else sequence_dir / "result" / "mapping" / "mapping_pose.txt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poses = read_mapping_poses(mapping_pose)
    add_trajectory_s(poses)
    raw_points, stats = read_lane_points(lane_csv, poses, args)
    if not raw_points:
        raise RuntimeError("No valid lane points were read. Check frame range and lane.csv validity.")

    nodes = build_lane_nodes(raw_points, args)
    fused_lanes = link_lane_nodes(nodes, args)
    fused_lanes = [lane for lane in fused_lanes if keep_lane(lane, args)]
    fused_lanes.sort(key=lambda lane: (lane["s_range"][0], lane["n_median"]))
    for new_id, lane in enumerate(fused_lanes):
        lane["global_lane_id"] = new_id

    road_layers = load_road_layers(args, sequence_dir)
    connector_candidates = build_connector_candidates(fused_lanes, args) if args.draw_connectors else []

    output = {
        "metadata": {
            "sequence_dir": str(sequence_dir),
            "lane_csv": str(lane_csv),
            "mapping_pose": str(mapping_pose),
            "frame_range": [args.start_frame, args.end_frame],
            "frame_stride": args.frame_stride,
            "sample_step": args.sample_step,
            "min_forward": args.min_forward,
            "max_forward": args.max_forward,
            "s_bin": args.s_bin,
            "lateral_merge": args.lateral_merge,
            "lateral_link": args.lateral_link,
            "max_gap_bins": args.max_gap_bins,
            "raw_point_count": len(raw_points),
            "node_count": len(nodes),
            "connector_candidate_count": len(connector_candidates),
            **stats,
        },
        "lanes": fused_lanes,
        "connector_candidates": connector_candidates,
    }

    json_path = output_dir / "global_lane_info.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    draw_bev(output_dir / "global_lane_bev.png", raw_points, fused_lanes, road_layers, connector_candidates, args)
    draw_sn(output_dir / "global_lane_sn.png", raw_points, fused_lanes, args)

    print(f"raw lane points: {len(raw_points)}")
    print(f"fusion nodes: {len(nodes)}")
    print(f"global lanes: {len(fused_lanes)}")
    print(f"wrote: {json_path}")
    print(f"wrote: {output_dir / 'global_lane_bev.png'}")
    print(f"wrote: {output_dir / 'global_lane_sn.png'}")
    return 0


def read_mapping_poses(path: Path) -> Dict[int, Dict]:
    poses: Dict[int, Dict] = {}
    with path.open("r", encoding="utf-8") as fp:
        header = fp.readline().strip().split()
        index = {name: idx for idx, name in enumerate(header)}
        required = ["lidar_frame", "x", "y", "z", "roll", "pitch", "yaw", "lidar_frame_time"]
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for line in fp:
            if not line.strip():
                continue
            parts = line.split()
            frame = int(parts[index["lidar_frame"]])
            x = float(parts[index["x"]])
            y = float(parts[index["y"]])
            z = float(parts[index["z"]])
            roll = float(parts[index["roll"]])
            pitch = float(parts[index["pitch"]])
            yaw = float(parts[index["yaw"]])
            poses[frame] = {
                "frame": frame,
                "time": float(parts[index["lidar_frame_time"]]),
                "translation": [x, y, z],
                "yaw": yaw,
                "transform": transform_from_xyz_rpy(x, y, z, roll, pitch, yaw),
                "s": 0.0,
            }
    if not poses:
        raise ValueError(f"No mapping poses found in {path}")
    return poses


def add_trajectory_s(poses: Dict[int, Dict]) -> None:
    previous_frame: Optional[int] = None
    cumulative = 0.0
    for frame in sorted(poses):
        if previous_frame is not None:
            prev = poses[previous_frame]["translation"]
            cur = poses[frame]["translation"]
            cumulative += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        poses[frame]["s"] = cumulative
        previous_frame = frame


def read_lane_points(lane_csv: Path, poses: Dict[int, Dict], args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    raw_points: List[Dict] = []
    stats = {
        "lane_rows": 0,
        "used_lane_rows": 0,
        "skipped_no_pose_rows": 0,
        "time_warning_rows": 0,
        "valid_lane_groups": 0,
        "skipped_lane_groups": 0,
    }

    with lane_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        prefixes = get_lane_prefixes(fieldnames)
        validate_lane_columns(fieldnames, prefixes, lane_csv)
        for row in reader:
            stats["lane_rows"] += 1
            try:
                frame = int(row["frameNum"])
            except (KeyError, ValueError):
                continue
            if args.start_frame is not None and frame < args.start_frame:
                continue
            if args.end_frame is not None and frame > args.end_frame:
                continue
            if args.frame_stride > 1 and args.start_frame is not None:
                if (frame - args.start_frame) % args.frame_stride != 0:
                    continue
            elif args.frame_stride > 1 and frame % args.frame_stride != 0:
                continue
            pose = poses.get(frame)
            if pose is None:
                stats["skipped_no_pose_rows"] += 1
                continue

            stats["used_lane_rows"] += 1
            try:
                lane_time = float(row["timeStamp"])
                if abs(lane_time - pose["time"]) > args.max_time_diff:
                    stats["time_warning_rows"] += 1
            except (KeyError, ValueError):
                pass

            for prefix in prefixes:
                lane = parse_lane(row, prefix, args.sample_step, args.min_forward, args.max_forward)
                if lane is None:
                    stats["skipped_lane_groups"] += 1
                    continue
                stats["valid_lane_groups"] += 1
                points = lane["points_vehicle"]
                headings = lane_point_headings(points)
                for point, heading in zip(points, headings):
                    world = transform_point(pose["transform"], point)
                    raw_points.append(
                        {
                            "frame": frame,
                            "name": prefix,
                            "lane_type": lane["lane_type"],
                            "lane_width": lane["lane_width"],
                            "x": world[0],
                            "y": world[1],
                            "z": world[2],
                            "s": pose["s"] + point[0],
                            "n": point[1],
                            "heading": normalize_angle(pose["yaw"] + heading),
                        }
                    )
    return raw_points, stats


def get_lane_prefixes(fieldnames: Iterable[str]) -> List[str]:
    return [name[: -len(".isValid")] for name in fieldnames if name.endswith(".isValid")]


def validate_lane_columns(fieldnames: Sequence[str], prefixes: Sequence[str], lane_csv: Path) -> None:
    if "frameNum" not in fieldnames or "timeStamp" not in fieldnames:
        raise ValueError(f"{lane_csv} must contain frameNum and timeStamp")
    if not prefixes:
        raise ValueError(f"{lane_csv} does not contain *.isValid lane groups")
    required_suffixes = [
        "isValid",
        "fitFarFlag",
        "zeroOrderCoeff",
        "firstOrderCoeff",
        "secondOrderCoeff",
        "thirdOrderCoeffNear",
        "thirdOrderCoeffFar",
        "dividingPoint",
        "rangeN",
        "rangeP",
        "laneWidth",
        "laneType",
    ]
    missing = []
    field_set = set(fieldnames)
    for prefix in prefixes:
        for suffix in required_suffixes:
            key = f"{prefix}.{suffix}"
            if key not in field_set:
                missing.append(key)
    if missing:
        raise ValueError(f"{lane_csv} missing lane columns, first missing: {missing[:20]}")


def parse_lane(row: Dict[str, str], prefix: str, sample_step: float, min_forward: float, max_forward: float) -> Optional[Dict]:
    if not safe_bool(row.get(f"{prefix}.isValid", "false")):
        return None
    try:
        fit_far_flag = safe_bool(row[f"{prefix}.fitFarFlag"])
        a = float(row[f"{prefix}.zeroOrderCoeff"])
        b = float(row[f"{prefix}.firstOrderCoeff"])
        c = float(row[f"{prefix}.secondOrderCoeff"])
        d_near = float(row[f"{prefix}.thirdOrderCoeffNear"])
        d_far = float(row[f"{prefix}.thirdOrderCoeffFar"])
        dividing_point = float(row[f"{prefix}.dividingPoint"])
        range_n = max(float(row[f"{prefix}.rangeN"]), min_forward)
        range_p = min(float(row[f"{prefix}.rangeP"]), max_forward)
        lane_width = float(row[f"{prefix}.laneWidth"])
        lane_type = int(float(row[f"{prefix}.laneType"]))
    except (KeyError, ValueError):
        return None
    if range_p <= range_n:
        return None

    count = max(2, int(math.ceil((range_p - range_n) / sample_step)) + 1)
    points = []
    for index in range(count):
        ratio = index / max(count - 1, 1)
        x = range_n + ratio * (range_p - range_n)
        y = eval_lane_y(x, a, b, c, d_near, d_far, dividing_point, fit_far_flag)
        points.append([x, y, 0.0])
    return {
        "lane_width": lane_width,
        "lane_type": lane_type,
        "points_vehicle": points,
    }


def eval_lane_y(
    x: float,
    a: float,
    b: float,
    c: float,
    d_near: float,
    d_far: float,
    dividing_point: float,
    fit_far_flag: bool,
) -> float:
    if (not fit_far_flag) or x < dividing_point:
        return a + b * x + c * x * x + d_near * x * x * x
    t = dividing_point
    return a + b * x + c * x * x + d_near * (t * t * t + 3.0 * x * (x - t) * t) + d_far * (x - t) ** 3


def lane_point_headings(points: Sequence[Sequence[float]]) -> List[float]:
    headings = []
    for index, point in enumerate(points):
        if index == 0:
            other = points[1]
            dx = other[0] - point[0]
            dy = other[1] - point[1]
        elif index == len(points) - 1:
            other = points[index - 1]
            dx = point[0] - other[0]
            dy = point[1] - other[1]
        else:
            prev_point = points[index - 1]
            next_point = points[index + 1]
            dx = next_point[0] - prev_point[0]
            dy = next_point[1] - prev_point[1]
        headings.append(math.atan2(dy, dx))
    return headings


def build_lane_nodes(raw_points: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    by_s_bin: Dict[int, List[Dict]] = defaultdict(list)
    for point in raw_points:
        s_bin = int(round(point["s"] / args.s_bin))
        by_s_bin[s_bin].append(point)

    nodes: List[Dict] = []
    for s_bin in sorted(by_s_bin):
        points = sorted(by_s_bin[s_bin], key=lambda item: item["n"])
        current: List[Dict] = []
        for point in points:
            if not current:
                current = [point]
                continue
            current_median_n = median([item["n"] for item in current])
            if abs(point["n"] - current_median_n) <= args.lateral_merge:
                current.append(point)
            else:
                nodes.append(make_node(s_bin, current))
                current = [point]
        if current:
            nodes.append(make_node(s_bin, current))
    return nodes


def make_node(s_bin: int, points: Sequence[Dict]) -> Dict:
    return {
        "s_bin": s_bin,
        "s": median([point["s"] for point in points]),
        "n": median([point["n"] for point in points]),
        "x": median([point["x"] for point in points]),
        "y": median([point["y"] for point in points]),
        "z": median([point["z"] for point in points]),
        "heading": circular_mean([point["heading"] for point in points]),
        "count": len(points),
        "names": Counter(point["name"] for point in points),
        "lane_types": Counter(str(point["lane_type"]) for point in points),
        "lane_widths": [point["lane_width"] for point in points if point["lane_width"] > 0.0],
        "frames": Counter(str(point["frame"]) for point in points),
    }


def link_lane_nodes(nodes: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    if not nodes:
        return []
    by_s_bin: Dict[int, List[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        by_s_bin[node["s_bin"]].append(index)

    tracks: List[Dict] = []
    max_heading = math.radians(35.0)
    for s_bin in sorted(by_s_bin):
        current_indices = sorted(by_s_bin[s_bin], key=lambda item: nodes[item]["n"])
        candidate_pairs = []
        for track_index, track in enumerate(tracks):
            gap = s_bin - track["last_s_bin"]
            if gap <= 0 or gap > args.max_gap_bins:
                continue
            last_node = track["nodes"][-1]
            for node_index in current_indices:
                node = nodes[node_index]
                dn = abs(node["n"] - last_node["n"])
                if dn > args.lateral_link:
                    continue
                d_heading = abs(angle_diff(node["heading"], last_node["heading"]))
                if d_heading > max_heading:
                    continue
                cost = dn + 0.35 * d_heading + 0.08 * (gap - 1)
                candidate_pairs.append((cost, track_index, node_index))

        used_tracks = set()
        used_nodes = set()
        for _cost, track_index, node_index in sorted(candidate_pairs):
            if track_index in used_tracks or node_index in used_nodes:
                continue
            tracks[track_index]["nodes"].append(nodes[node_index])
            tracks[track_index]["last_s_bin"] = s_bin
            used_tracks.add(track_index)
            used_nodes.add(node_index)

        for node_index in current_indices:
            if node_index in used_nodes:
                continue
            tracks.append({"nodes": [nodes[node_index]], "last_s_bin": s_bin})

    return [fused_lane_from_nodes(track["nodes"], args) for track in tracks]


def fused_lane_from_nodes(track_nodes: Sequence[Dict], args: argparse.Namespace) -> Dict:
    track_nodes = sorted(track_nodes, key=lambda item: item["s"])
    track_nodes = smooth_track_nodes(track_nodes, args.smooth_window)
    polyline = [[node["x"], node["y"], node["z"]] for node in track_nodes]
    polyline_sn = [[node["s"], node["n"], node["z"]] for node in track_nodes]
    count = sum(node["count"] for node in track_nodes)
    names = Counter()
    lane_types = Counter()
    frames = Counter()
    widths = []
    for node in track_nodes:
        names.update(node["names"])
        lane_types.update(node["lane_types"])
        frames.update(node["frames"])
        widths.extend(node["lane_widths"])
    s_values = [node["s"] for node in track_nodes]
    n_values = [node["n"] for node in track_nodes]
    return {
        "global_lane_id": -1,
        "polyline_world": polyline,
        "polyline_sn": polyline_sn,
        "point_count": len(polyline),
        "observation_count": count,
        "frame_count": len(frames),
        "s_range": [min(s_values), max(s_values)],
        "n_median": median(n_values),
        "n_range": [min(n_values), max(n_values)],
        "mean_lane_width": sum(widths) / len(widths) if widths else None,
        "name_hist": dict(names.most_common()),
        "lane_type_hist": dict(lane_types.most_common()),
    }


def smooth_track_nodes(track_nodes: Sequence[Dict], window: int) -> List[Dict]:
    if window <= 1 or len(track_nodes) < 3:
        return [dict(node) for node in track_nodes]
    if window % 2 == 0:
        window += 1
    radius = window // 2
    smoothed = []
    for index, node in enumerate(track_nodes):
        left = max(0, index - radius)
        right = min(len(track_nodes), index + radius + 1)
        neighbors = track_nodes[left:right]
        new_node = dict(node)
        # Keep s monotonic from the original node; smooth lateral/world noise.
        for key in ("x", "y", "z", "n"):
            new_node[key] = sum(item[key] for item in neighbors) / len(neighbors)
        new_node["heading"] = circular_mean([item["heading"] for item in neighbors])
        smoothed.append(new_node)
    return smoothed


def keep_lane(lane: Dict, args: argparse.Namespace) -> bool:
    if lane["point_count"] < args.min_cluster_points:
        return False
    if lane["observation_count"] < args.min_observations:
        return False
    if lane["s_range"][1] - lane["s_range"][0] < max(args.s_bin * args.min_cluster_points * 0.25, 8.0):
        return False
    return True


def load_road_layers(args: argparse.Namespace, sequence_dir: Path) -> Dict[str, List[Point]]:
    road_edge_dir = Path(args.road_edge_dir) if args.road_edge_dir else sequence_dir / "result" / "debug" / "roadEdge"
    layers: Dict[str, List[Point]] = {}
    if not road_edge_dir.is_dir():
        return layers

    files = {
        "road": "Road.pcd",
        "fill_road": "fill_road.pcd",
        "left_edge": "leftRoadEdge.pcd",
        "right_edge": "rightRoadEdge.pcd",
        "edge_z": "roadEdgez.pcd",
    }
    for key, filename in files.items():
        path = road_edge_dir / filename
        if path.exists():
            layers[key] = read_pcd_xyz(path)
    return layers


def read_pcd_xyz(path: Path) -> List[Point]:
    with path.open("rb") as fp:
        header_lines = []
        while True:
            line = fp.readline()
            if not line:
                raise ValueError(f"{path} does not contain a DATA line")
            decoded = line.decode("utf-8", errors="replace").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA"):
                data_type = decoded.split(maxsplit=1)[1].strip().lower()
                break

        fields = []
        for line in header_lines:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
                break
        if not {"x", "y", "z"}.issubset(fields):
            raise ValueError(f"{path} must contain x y z fields")
        x_index = fields.index("x")
        y_index = fields.index("y")
        z_index = fields.index("z")

        if data_type != "ascii":
            raise ValueError(f"{path} uses DATA {data_type}; only ascii PCD is supported here")

        points: List[Point] = []
        for raw_line in fp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) <= max(x_index, y_index, z_index):
                continue
            try:
                points.append((float(parts[x_index]), float(parts[y_index]), float(parts[z_index])))
            except ValueError:
                continue
        return points


def build_connector_candidates(lanes: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    endpoints = []
    for lane in lanes:
        polyline = lane["polyline_world"]
        if len(polyline) < 3:
            continue
        start_heading = heading_between(polyline[0], polyline[min(3, len(polyline) - 1)])
        end_heading = heading_between(polyline[max(0, len(polyline) - 4)], polyline[-1])
        endpoints.append(
            {
                "lane_id": lane["global_lane_id"],
                "kind": "start",
                "point": polyline[0],
                "heading": start_heading,
            }
        )
        endpoints.append(
            {
                "lane_id": lane["global_lane_id"],
                "kind": "end",
                "point": polyline[-1],
                "heading": end_heading,
            }
        )

    max_heading = math.radians(args.connector_max_heading_deg)
    candidates = []
    for incoming in endpoints:
        if incoming["kind"] != "end":
            continue
        for outgoing in endpoints:
            if outgoing["kind"] != "start" or outgoing["lane_id"] == incoming["lane_id"]:
                continue
            dx = outgoing["point"][0] - incoming["point"][0]
            dy = outgoing["point"][1] - incoming["point"][1]
            distance = math.hypot(dx, dy)
            if distance <= 1e-6 or distance > args.connector_max_distance:
                continue

            direction = math.atan2(dy, dx)
            departure_error = abs(angle_diff(direction, incoming["heading"]))
            arrival_error = abs(angle_diff(outgoing["heading"], direction))
            if departure_error > max_heading or arrival_error > max_heading:
                continue

            lateral = abs(-math.sin(incoming["heading"]) * dx + math.cos(incoming["heading"]) * dy)
            if lateral > args.connector_max_lateral:
                continue

            score = distance + 10.0 * (departure_error + arrival_error) + 2.0 * lateral
            candidates.append(
                {
                    "incoming_lane_id": incoming["lane_id"],
                    "outgoing_lane_id": outgoing["lane_id"],
                    "start": incoming["point"],
                    "end": outgoing["point"],
                    "distance": distance,
                    "departure_error_deg": math.degrees(departure_error),
                    "arrival_error_deg": math.degrees(arrival_error),
                    "lateral": lateral,
                    "score": score,
                }
            )

    candidates.sort(key=lambda item: item["score"])
    selected = []
    used_pairs = set()
    for candidate in candidates:
        key = (candidate["incoming_lane_id"], candidate["outgoing_lane_id"])
        if key in used_pairs:
            continue
        used_pairs.add(key)
        selected.append(candidate)
    return selected


def heading_between(start: Sequence[float], end: Sequence[float]) -> float:
    return math.atan2(end[1] - start[1], end[0] - start[0])


def draw_bev(
    path: Path,
    raw_points: Sequence[Dict],
    lanes: Sequence[Dict],
    road_layers: Dict[str, List[Point]],
    connector_candidates: Sequence[Dict],
    args: argparse.Namespace,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(args.fig_width, args.fig_height))
    draw_road_layers(road_layers, args)
    if not args.no_raw_points:
        xs = [point["x"] for point in raw_points]
        ys = [point["y"] for point in raw_points]
        plt.scatter(xs, ys, s=0.15, c="#7f8c8d", alpha=0.18, linewidths=0, label="raw lane samples")
    colors = matplotlib.colormaps["tab20"].resampled(max(len(lanes), 1))
    for index, lane in enumerate(lanes):
        polyline = lane["polyline_world"]
        xs = [point[0] for point in polyline]
        ys = [point[1] for point in polyline]
        color = colors(index % 20)
        label = f"id={lane['global_lane_id']} n={lane['n_median']:.1f} obs={lane['observation_count']}"
        plt.plot(xs, ys, "-", linewidth=2.0, color=color, label=label)
        if xs:
            mid = len(xs) // 2
            plt.text(xs[mid], ys[mid], str(lane["global_lane_id"]), color=color, fontsize=8)
    draw_connector_candidates(connector_candidates)
    plt.axis("equal")
    plt.grid(True, linewidth=0.3)
    plt.xlabel("world x / m")
    plt.ylabel("world y / m")
    plt.title("Fused ClassLab Global Lane BEV")
    if len(lanes) <= 30:
        plt.legend(loc="best", fontsize=6)
    apply_plot_limits(raw_points, lanes, road_layers, args)
    plt.tight_layout()
    plt.savefig(path, dpi=args.dpi)
    plt.close()


def draw_road_layers(road_layers: Dict[str, List[Point]], args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    if args.draw_road_points:
        if "fill_road" in road_layers:
            points = road_layers["fill_road"][::max(args.road_point_stride, 1)]
            plt.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=0.18,
                c="#d8dde2",
                alpha=0.22,
                linewidths=0,
                label="fill_road",
                zorder=0,
            )
        elif "road" in road_layers:
            points = road_layers["road"][::max(args.road_point_stride, 1)]
            plt.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=0.18,
                c="#d8dde2",
                alpha=0.22,
                linewidths=0,
                label="Road.pcd",
                zorder=0,
            )

    edge_specs = [
        ("left_edge", "#111111", "leftRoadEdge"),
        ("right_edge", "#444444", "rightRoadEdge"),
        ("edge_z", "#9b59b6", "roadEdgez"),
    ]
    for key, color, label in edge_specs:
        if key not in road_layers:
            continue
        points = road_layers[key][::max(args.edge_point_stride, 1)]
        plt.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            s=2.0 if key != "edge_z" else 0.8,
            c=color,
            alpha=0.75 if key != "edge_z" else 0.28,
            linewidths=0,
            label=label,
            zorder=1,
        )


def draw_connector_candidates(connector_candidates: Sequence[Dict]) -> None:
    if not connector_candidates:
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    for candidate in connector_candidates:
        start = candidate["start"]
        end = candidate["end"]
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linestyle="--",
            linewidth=1.2,
            color="#e74c3c",
            alpha=0.65,
            zorder=3,
        )
        mid_x = (start[0] + end[0]) * 0.5
        mid_y = (start[1] + end[1]) * 0.5
        radius = max(4.0, min(candidate["distance"] * 0.25, 18.0))
        circle = Circle((mid_x, mid_y), radius=radius, fill=True, color="#e74c3c", alpha=0.06, zorder=2)
        plt.gca().add_patch(circle)


def apply_plot_limits(
    raw_points: Sequence[Dict],
    lanes: Sequence[Dict],
    road_layers: Dict[str, List[Point]],
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    if args.xlim is not None:
        plt.xlim(args.xlim[0], args.xlim[1])
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    if args.xlim is not None and args.ylim is not None:
        return

    xs = []
    ys = []
    for lane in lanes:
        for point in lane["polyline_world"]:
            xs.append(point[0])
            ys.append(point[1])
    if not xs and raw_points:
        xs.extend(point["x"] for point in raw_points)
        ys.extend(point["y"] for point in raw_points)
    if not xs:
        for points in road_layers.values():
            xs.extend(point[0] for point in points)
            ys.extend(point[1] for point in points)
    if not xs:
        return

    margin = args.plot_margin
    if args.xlim is None:
        plt.xlim(min(xs) - margin, max(xs) + margin)
    if args.ylim is None:
        plt.ylim(min(ys) - margin, max(ys) + margin)


def draw_sn(path: Path, raw_points: Sequence[Dict], lanes: Sequence[Dict], args: argparse.Namespace) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 8))
    if not args.no_raw_points:
        ss = [point["s"] for point in raw_points]
        ns = [point["n"] for point in raw_points]
        plt.scatter(ss, ns, s=0.2, c="#b8b8b8", alpha=0.25, linewidths=0)
    colors = matplotlib.colormaps["tab20"].resampled(max(len(lanes), 1))
    for index, lane in enumerate(lanes):
        polyline = lane["polyline_sn"]
        ss = [point[0] for point in polyline]
        ns = [point[1] for point in polyline]
        color = colors(index % 20)
        plt.plot(ss, ns, "-", linewidth=2.0, color=color)
        if ss:
            mid = len(ss) // 2
            plt.text(ss[mid], ns[mid], str(lane["global_lane_id"]), color=color, fontsize=8)
    plt.grid(True, linewidth=0.3)
    plt.xlabel("trajectory s + lane x / m")
    plt.ylabel("ego-frame lateral y / m")
    plt.title("Fused ClassLab Lanes in Trajectory s-n Coordinates")
    plt.tight_layout()
    plt.savefig(path, dpi=args.dpi)
    plt.close()


def transform_from_xyz_rpy(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> List[List[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    rotation = matmul3(matmul3(rz, ry), rx)
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], x],
        [rotation[1][0], rotation[1][1], rotation[1][2], y],
        [rotation[2][0], rotation[2][1], rotation[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(matrix: Sequence[Sequence[float]], point: Sequence[float]) -> List[float]:
    x, y, z = point
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def matmul3(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) * 0.5)


def circular_mean(angles: Sequence[float]) -> float:
    if not angles:
        return 0.0
    sin_sum = sum(math.sin(angle) for angle in angles)
    cos_sum = sum(math.cos(angle) for angle in angles)
    return math.atan2(sin_sum, cos_sum)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(left: float, right: float) -> float:
    return normalize_angle(left - right)


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
