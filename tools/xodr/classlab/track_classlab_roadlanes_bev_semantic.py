#!/usr/bin/env python3
"""Build ClassLab roadlanes with laneType semantics and drivable road fills.

This keeps the original roadlane tracking pipeline in
track_classlab_roadlanes_bev.py, then adds:

* ClassLab laneType decoding for drawing and downstream map conversion.
* Type-colored BEV rendering independent of global_lane_id.
* Adjacent-lane fill polygons with right-hand-traffic direction hints.
"""

from __future__ import annotations

import math
import os
import csv
import copy
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import track_classlab_roadlanes_bev as base


Point3 = List[float]


YELLOW = "#f5a300"
WHITE = "#f7f7f2"
UNKNOWN = "#7f8c8d"
FILL_FORWARD = "#56b4e9"
FILL_REVERSE = "#e69f00"
DEFAULT_INFERRED_LANE_WIDTH = 3.6


def process_sequence(args, sequence_dir: Path, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    lane_csv = Path(args.lane_csv) if args.lane_csv else sequence_dir / "result" / "lane.csv"
    pose_file = Path(args.pose_file) if args.pose_file else sequence_dir / "result" / "mapping" / "mapping_pose.txt"

    poses = base.read_mapping_poses(pose_file)
    frames = base.select_frames(sorted(poses), args.start_frame, args.end_frame, args.frame_step)
    frame_lanes = base.read_lane_csv(lane_csv, set(args.use_lane_names), frames, args)
    initial_poses = base.make_initial_relative_poses(poses, frames)

    corrected, debug_rows = base.estimate_corrected_poses(frames, initial_poses, frame_lanes, args)
    observations, raw_points = base.build_corrected_lane_observations(frames, corrected, frame_lanes, args)
    roadlanes = base.track_global_lane_observations(observations, args)
    road_edge_csv = sequence_dir / "result" / "debug" / "roadEdge" / "road_edge.csv"
    road_edges = build_corrected_road_edges(road_edge_csv, frames, corrected, args) if road_edge_csv.exists() else []

    for lane_id, lane in enumerate(roadlanes):
        lane["global_lane_id"] = lane_id
        lane["inferred"] = False
        lane["dominant_lane_type"] = dominant_lane_type(lane.get("lane_type_hist", {}))
        lane["lane_type_semantic"] = decode_lane_type(lane["dominant_lane_type"])
        lane["is_direction_divider"] = lane["lane_type_semantic"]["color"] == "yellow"

    inferred_lanes = infer_missing_lane_markings(roadlanes, road_edges)
    map_lanes = list(roadlanes) + inferred_lanes
    road_fills = build_road_fills(map_lanes)
    road_connectors = build_road_connectors(road_fills)
    enu_transform = make_local_to_enu_transform(poses, frames)
    enu_map_lanes = transform_lanes_to_enu(map_lanes, enu_transform)
    enu_roadlanes = enu_map_lanes[: len(roadlanes)]
    enu_inferred_lanes = enu_map_lanes[len(roadlanes) :]
    enu_road_fills = transform_fills_to_enu(road_fills, enu_transform)
    enu_road_connectors = transform_connectors_to_enu(road_connectors, enu_transform)
    enu_corrected = transform_poses_to_enu(corrected, enu_transform)
    enu_raw_points = transform_point_records_to_enu(raw_points, enu_transform)
    fill_road_pcd = sequence_dir / "result" / "debug" / "roadEdge" / "fill_road.pcd"
    raw_road_polygon_points = read_ascii_pcd_xyz(fill_road_pcd) if fill_road_pcd.exists() else []
    raw_enu_poses = {frame: (poses[frame]["x"], poses[frame]["y"], poses[frame]["yaw"]) for frame in frames}
    road_polygon_points = align_road_polygon_to_corrected_poses(
        raw_road_polygon_points,
        raw_enu_poses,
        enu_corrected,
        frames,
    )
    road_instruction_regions = build_road_instruction_regions(
        road_polygon_points,
        enu_road_connectors,
        enu_road_fills,
    )

    metadata = {
        "sequence_dir": str(sequence_dir),
        "lane_csv": str(lane_csv),
        "pose_file": str(pose_file),
        "frame_count": len(frames),
        "raw_lane_points": len(raw_points),
        "lane_observation_count": len(observations),
        "roadlane_count": len(roadlanes),
        "inferred_lane_marking_count": len(inferred_lanes),
        "road_fill_count": len(road_fills),
        "road_connector_count": len(road_connectors),
        "road_polygon_point_count": len(road_polygon_points),
        "road_instruction_region_count": len(road_instruction_regions),
        "road_edge_csv": str(road_edge_csv) if road_edge_csv.exists() else None,
        "fill_road_pcd": str(fill_road_pcd) if fill_road_pcd.exists() else None,
        "road_edge_polyline_count": len(road_edges),
        "coordinate_frame": "enu",
        "road_polygon_alignment": {
            "method": "local_pose_correction_warp",
            "source_frame": "mapping_slam_enu",
            "target_frame": "corrected_roadlane_enu",
            "formula": "p_aligned = weighted_average_i(T_corrected_i * inverse(T_raw_mapping_i) * p_raw)",
        },
        "local_to_enu": enu_transform,
        "lane_type_rule": (
            "laneType = solid_or_dashed*100 + yellow_or_white*10 + "
            "special_type; yellow laneTypes are treated as direction dividers."
        ),
        "right_hand_traffic_rule": (
            "In corrected s/n coordinates, traffic on the lower-n/right side "
            "of the nearest yellow divider travels with increasing s; traffic "
            "on the higher-n/left side travels against increasing s."
        ),
        "args": vars(args),
    }

    local_metadata = dict(metadata)
    local_metadata["coordinate_frame"] = "corrected_local"
    base.write_pose_corrections(output_dir / "corrected_pose.csv", frames, initial_poses, corrected, debug_rows)
    write_pose_enu(output_dir / "corrected_pose_enu.csv", frames, enu_corrected, debug_rows)
    base.write_json(
        output_dir / "roadlane_info_local.json",
        {
            "metadata": local_metadata,
            "lanes": roadlanes,
            "inferred_lane_markings": inferred_lanes,
            "road_fills": road_fills,
            "road_connectors": road_connectors,
        },
    )
    base.write_json(output_dir / "road_fill_info_local.json", {"metadata": local_metadata, "road_fills": road_fills})
    base.write_json(
        output_dir / "road_network_info_local.json",
        {"metadata": local_metadata, "road_fills": road_fills, "road_connectors": road_connectors},
    )
    base.write_json(
        output_dir / "roadlane_info.json",
        {
            "metadata": metadata,
            "lanes": enu_roadlanes,
            "inferred_lane_markings": enu_inferred_lanes,
            "road_fills": enu_road_fills,
            "road_connectors": enu_road_connectors,
        },
    )
    base.write_json(output_dir / "road_fill_info.json", {"metadata": metadata, "road_fills": enu_road_fills})
    base.write_json(
        output_dir / "road_network_info.json",
        {"metadata": metadata, "road_fills": enu_road_fills, "road_connectors": enu_road_connectors},
    )
    base.write_json(
        output_dir / "road_polygon_info.json",
        {
            "metadata": metadata,
            "road_polygon_source": "fill_road.pcd",
            "coordinate_frame": "enu",
            "road_polygon_points": road_polygon_points,
        },
    )
    base.write_json(
        output_dir / "road_instruction_info.json",
        {
            "metadata": metadata,
            "coordinate_frame": "enu",
            "road_instruction_regions": road_instruction_regions,
        },
    )
    draw_id_bev(output_dir / "roadlane_id_bev.png", enu_raw_points, enu_corrected, enu_map_lanes, args)
    draw_semantic_bev(output_dir / "roadlane_bev.png", enu_raw_points, enu_corrected, enu_map_lanes, args)
    draw_road_fill_bev(
        output_dir / "road_fill_bev.png",
        enu_raw_points,
        enu_corrected,
        enu_map_lanes,
        enu_road_fills,
        args,
        enu_road_connectors,
    )
    draw_road_fill_bev(
        output_dir / "road_direction_bev.png",
        enu_raw_points,
        enu_corrected,
        enu_map_lanes,
        enu_road_fills,
        args,
        enu_road_connectors,
    )
    draw_road_instruction_bev(
        output_dir / "road_instruction_bev.png",
        road_polygon_points,
        road_instruction_regions,
        enu_raw_points,
        enu_corrected,
        enu_map_lanes,
        enu_road_fills,
        enu_road_connectors,
        args,
    )

    print(f"frames: {len(frames)}")
    print(f"raw lane points: {len(raw_points)}")
    print(f"lane observations: {len(observations)}")
    print(f"roadlanes: {len(roadlanes)}")
    print(f"inferred lane markings: {len(inferred_lanes)}")
    print(f"road fills: {len(road_fills)}")
    print(f"road connectors: {len(road_connectors)}")
    print(f"road polygon points: {len(road_polygon_points)}")
    print(f"road instruction regions: {len(road_instruction_regions)}")
    print(f"wrote: {output_dir / 'roadlane_info.json'}")
    print(f"wrote: {output_dir / 'road_fill_info.json'}")
    print(f"wrote: {output_dir / 'road_network_info.json'}")
    print(f"wrote: {output_dir / 'road_polygon_info.json'}")
    print(f"wrote: {output_dir / 'road_instruction_info.json'}")
    print(f"wrote: {output_dir / 'roadlane_bev.png'}")
    print(f"wrote: {output_dir / 'road_fill_bev.png'}")
    print(f"wrote: {output_dir / 'road_direction_bev.png'}")
    print(f"wrote: {output_dir / 'road_instruction_bev.png'}")
    print(f"wrote: {output_dir / 'roadlane_id_bev.png'}")
    print(f"wrote: {output_dir / 'corrected_pose.csv'}")
    return {
        "sequence_dir": sequence_dir,
        "output_dir": output_dir,
        "metadata": metadata,
        "lanes": enu_map_lanes,
        "road_fills": enu_road_fills,
        "road_connectors": enu_road_connectors,
        "road_polygon_points": road_polygon_points,
        "road_instruction_regions": road_instruction_regions,
        "poses": enu_corrected,
        "raw_points": enu_raw_points,
    }


def normalize_path_list(value) -> List[Path]:
    """Normalize argparse path arguments to a flat List[Path].

    Some versions of the base parser define --multi-sequence-dirs with
    nargs='+' and/or action='append'.  In that case argparse returns a nested
    list such as [['/seq_a', '/seq_b']] instead of a plain list.  Passing that
    inner list directly to pathlib.Path raises:

        TypeError: expected str, bytes or os.PathLike object, not list

    This helper accepts None, a single string/path, a flat list, or a nested
    list/tuple and returns a clean flat list of Path objects.
    """

    paths: List[Path] = []

    def visit(item) -> None:
        if item is None:
            return
        if isinstance(item, (str, bytes, os.PathLike)):
            paths.append(Path(item))
            return
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                visit(sub_item)
            return
        paths.append(Path(str(item)))

    visit(value)
    return paths


def main() -> int:
    args = base.parse_args()
    sequence_dirs = [Path(args.sequence_dir)] + normalize_path_list(getattr(args, "multi_sequence_dirs", []))
    max_sequences = int(getattr(args, "multi_max_sequences", 0) or 0)
    if max_sequences > 0:
        sequence_dirs = sequence_dirs[:max_sequences]

    output_dir = Path(args.output_dir)
    if len(sequence_dirs) == 1:
        process_sequence(args, sequence_dirs[0], output_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, sequence_dir in enumerate(sequence_dirs):
        sequence_output_dir = output_dir / f"{index:02d}_{sequence_dir.name}"
        print(f"processing sequence {index}: {sequence_dir}")
        results.append(process_sequence(args, sequence_dir, sequence_output_dir))

    base_index = max(0, min(int(getattr(args, "multi_base_index", 0) or 0), len(results) - 1))
    aligned_results = align_multi_sequence_results_by_gnss(results, base_index)
    draw_multi_sequence_bev(output_dir / "multi_sequence_bev.png", aligned_results, base_index, args)
    base.write_json(
        output_dir / "multi_sequence_info.json",
        {
            "metadata": {
                "sequence_count": len(aligned_results),
                "base_sequence_index": base_index,
                "base_sequence_dir": str(aligned_results[base_index]["sequence_dir"]),
                "coordinate_frame": "base_sequence_corrected_roadlane_enu",
                "multi_alignment": "rtk_gps_origin_offset",
                "args": vars(args),
            },
            "sequences": [
                {
                    "sequence_index": index,
                    "sequence_dir": str(result["sequence_dir"]),
                    "output_dir": str(result["output_dir"]),
                    "rtk_gps_origin": result.get("rtk_gps_origin"),
                    "multi_alignment_offset_xy": result.get("multi_alignment_offset_xy"),
                    "roadlane_count": len(result["lanes"]),
                    "road_fill_count": len(result["road_fills"]),
                    "road_connector_count": len(result["road_connectors"]),
                    "road_instruction_region_count": len(result["road_instruction_regions"]),
                }
                for index, result in enumerate(aligned_results)
            ],
        },
    )
    print(f"wrote: {output_dir / 'multi_sequence_bev.png'}")
    print(f"wrote: {output_dir / 'multi_sequence_info.json'}")
    return 0


def dominant_lane_type(hist: Dict) -> int:
    if not hist:
        return 0
    counter = Counter()
    for key, value in hist.items():
        try:
            counter[int(float(key))] += int(value)
        except (TypeError, ValueError):
            continue
    if not counter:
        return 0
    return counter.most_common(1)[0][0]


def decode_lane_type(value: int) -> Dict:
    code = int(value)
    abs_code = abs(code)
    solid_digit = (abs_code // 100) % 10
    color_digit = (abs_code // 10) % 10
    special_digit = abs_code % 10

    color = "yellow" if color_digit == 1 else "white"
    base_pattern = "solid" if solid_digit == 1 else "dashed"
    special = {
        0: "single",
        2: "double",
        4: "fishbone",
        6: "solid_dashed",
    }.get(special_digit, "single")

    if special == "solid_dashed":
        pattern = "solid_dashed"
    else:
        pattern = base_pattern
    chinese_name = lane_type_chinese_name(color, pattern, special)

    return {
        "code": code,
        "color": color,
        "pattern": pattern,
        "special": special,
        "chinese_name": chinese_name,
        "draw_color": YELLOW if color == "yellow" else WHITE,
        "is_yellow_divider": color == "yellow",
        "is_double": special == "double",
        "is_fishbone": special == "fishbone",
        "is_solid_dashed": special == "solid_dashed",
    }


def lane_type_chinese_name(color: str, pattern: str, special: str) -> str:
    color_name = "黄" if color == "yellow" else "白"
    if special == "fishbone":
        pattern_name = "鱼骨实线" if pattern == "solid" else "鱼骨虚线"
        return f"{color_name}{pattern_name}"
    if special == "solid_dashed":
        return f"{color_name}虚实线"
    count_name = "双" if special == "double" else "单"
    pattern_name = "虚线" if pattern == "dashed" else "实线"
    return f"{color_name}{count_name}{pattern_name}"


def build_corrected_road_edges(
    path: Path,
    frames: Sequence[int],
    corrected_poses: Dict[int, base.Pose2],
    args,
) -> List[Dict]:
    if not path.exists():
        return []
    pose_s = base.build_pose_s(frames, corrected_poses)
    selected = set(frames)
    edge_points = {"left_edge": [], "righ_edge": []}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                frame = int(row["frameNum"])
            except (KeyError, ValueError):
                continue
            if frame not in selected or frame not in corrected_poses:
                continue
            pose = corrected_poses[frame]
            for prefix in ("left_edge", "righ_edge"):
                lane = parse_road_edge_row(row, prefix, args)
                if lane is None:
                    continue
                for point, heading_vehicle in zip(lane["points_vehicle"], lane["headings_vehicle"]):
                    world = base.transform_point_2d(pose, point)
                    edge_points[prefix].append(
                        {
                            "frame": frame,
                            "side": prefix,
                            "ego_x": point[0],
                            "ego_y": point[1],
                            "s": pose_s[frame] + point[0],
                            "n": point[1],
                            "x": world[0],
                            "y": world[1],
                            "heading": base.normalize_angle(pose[2] + heading_vehicle),
                        }
                    )
    return [edge for edge in (fuse_road_edge_points(name, points, args) for name, points in edge_points.items()) if edge]


def parse_road_edge_row(row: Dict[str, str], prefix: str, args) -> Optional[Dict]:
    if not base.safe_bool(row.get(f"{prefix}.isValid", "false")):
        return None
    try:
        fit_far_flag = base.safe_bool(row[f"{prefix}.fitFarFlag"])
        coeff = (
            float(row[f"{prefix}.zeroOrderCoeff"]),
            float(row[f"{prefix}.firstOrderCoeff"]),
            float(row[f"{prefix}.secondOrderCoeff"]),
            float(row[f"{prefix}.thirdOrderCoeffNear"]),
            float(row[f"{prefix}.thirdOrderCoeffFar"]),
            float(row[f"{prefix}.dividingPoint"]),
        )
        range_n = float(row[f"{prefix}.rangeN"])
        range_p = float(row[f"{prefix}.rangeP"])
    except (KeyError, ValueError):
        return None
    start = max(range_n, args.min_forward)
    end = min(range_p, args.max_forward)
    if end <= start:
        return None
    points = base.sample_lane_points(coeff, fit_far_flag, start, end, max(args.sample_step, 1.0))
    if len(points) < 2:
        return None
    return {"points_vehicle": points, "headings_vehicle": base.lane_headings(points)}


def fuse_road_edge_points(name: str, points: Sequence[Dict], args) -> Optional[Dict]:
    if not points:
        return None
    by_s_bin: Dict[int, List[Dict]] = {}
    for point in points:
        s_bin = int(round(point["s"] / max(args.s_bin, 1e-3)))
        by_s_bin.setdefault(s_bin, []).append(point)
    nodes = []
    for s_bin in sorted(by_s_bin):
        group = by_s_bin[s_bin]
        if len(group) < 3:
            continue
        nodes.append(
            {
                "s_bin": s_bin,
                "s": base.median([point["s"] for point in group]),
                "n": base.median([point["n"] for point in group]),
                "x": base.median([point["x"] for point in group]),
                "y": base.median([point["y"] for point in group]),
                "heading": base.circular_mean([point["heading"] for point in group]),
                "count": len(group),
            }
        )
    nodes = base.smooth_track_nodes(nodes, 9)
    if len(nodes) < 8:
        return None
    return {
        "edge_id": name,
        "polyline_world": [[node["x"], node["y"], 0.0] for node in nodes],
        "polyline_sn": [[node["s"], node["n"], 0.0] for node in nodes],
        "s_range": [nodes[0]["s"], nodes[-1]["s"]],
        "n_median": base.median([node["n"] for node in nodes]),
        "point_count": len(nodes),
    }


def infer_missing_lane_markings(lanes: Sequence[Dict], road_edges: Sequence[Dict]) -> List[Dict]:
    inferred = []
    real_lanes = list(lanes)
    for lane in real_lanes:
        semantic = lane.get("lane_type_semantic", {})
        if semantic.get("color") != "white" or semantic.get("pattern") != "dashed":
            continue
        yellow = nearest_yellow_for_white_dashed(lane, real_lanes)
        if yellow is None:
            continue
        side_away_from_yellow = 1 if float(lane["n_median"]) > float(yellow["n_median"]) else -1
        edge = road_edge_on_side(lane, road_edges, side_away_from_yellow)
        if edge is None:
            continue
        candidate = make_inferred_from_road_edge(lane, yellow, edge, side_away_from_yellow, len(inferred))
        if candidate is None:
            continue
        clear_segments = split_candidate_clear_segments(candidate, real_lanes + inferred, len(inferred))
        inferred.extend(clear_segments)
    return inferred


def nearest_yellow_for_white_dashed(white: Dict, lanes: Sequence[Dict]) -> Optional[Dict]:
    candidates = []
    for other in lanes:
        semantic = other.get("lane_type_semantic", {})
        if semantic.get("color") != "yellow" or semantic.get("pattern") != "solid":
            continue
        overlap = s_overlap(white, other)
        if overlap < 25.0:
            continue
        dn = abs(float(other.get("n_median", 0.0)) - float(white.get("n_median", 0.0)))
        if 2.0 <= dn <= 6.8:
            candidates.append((dn, -overlap, other))
    if not candidates:
        return None
    return sorted(candidates)[0][2]


def road_edge_on_side(white: Dict, road_edges: Sequence[Dict], side: int) -> Optional[Dict]:
    candidates = []
    n0 = float(white.get("n_median", 0.0))
    for edge in road_edges:
        overlap = s_overlap(white, edge)
        if overlap < 25.0:
            continue
        dn = float(edge.get("n_median", 0.0)) - n0
        if side * dn <= 0.0:
            continue
        if 2.0 <= abs(dn) <= 12.0:
            candidates.append((abs(dn - side * DEFAULT_INFERRED_LANE_WIDTH), -overlap, edge))
    if not candidates:
        return None
    return sorted(candidates)[0][2]


def make_inferred_from_road_edge(white: Dict, yellow: Dict, edge: Dict, side: int, index: int) -> Optional[Dict]:
    s0 = max(float(white["s_range"][0]), float(yellow["s_range"][0]), float(edge["s_range"][0]))
    s1 = min(float(white["s_range"][1]), float(yellow["s_range"][1]), float(edge["s_range"][1]))
    if s1 - s0 < 25.0:
        return None
    points_world = []
    points_sn = []
    count = max(int(math.floor((s1 - s0) / 2.0)) + 1, 2)
    for sample_index in range(count):
        s = s0 + (s1 - s0) * sample_index / max(count - 1, 1)
        white_sample = interpolate_lane(white, s)
        edge_sample = interpolate_lane(edge, s)
        if white_sample is None or edge_sample is None:
            continue
        dn = edge_sample[1] - white_sample[1]
        if side * dn <= 0.0 or not (2.0 <= abs(dn) <= 12.0):
            continue
        ratio = min(DEFAULT_INFERRED_LANE_WIDTH / abs(dn), 0.95)
        n = white_sample[1] + ratio * dn
        x = white_sample[2][0] + ratio * (edge_sample[2][0] - white_sample[2][0])
        y = white_sample[2][1] + ratio * (edge_sample[2][1] - white_sample[2][1])
        points_sn.append([s, n, 0.0])
        points_world.append([x, y, 0.0])
    if len(points_world) < 8:
        return None
    if not curvature_compatible(white, points_sn):
        return None
    n_values = [point[1] for point in points_sn]
    semantic = decode_lane_type(100)
    return {
        "global_lane_id": f"inferred_{index}",
        "source_track_id": f"inferred_road_edge_{white['global_lane_id']}_{yellow['global_lane_id']}",
        "inferred": True,
        "inferred_from_lane_ids": [white["global_lane_id"], yellow["global_lane_id"], edge["edge_id"]],
        "inferred_side": "left" if side > 0 else "right",
        "inference_reason": "white dashed lane with nearby yellow solid and road edge support; inserted missing white solid",
        "dominant_lane_type": 100,
        "lane_type_semantic": semantic,
        "is_direction_divider": False,
        "polyline_world": points_world,
        "polyline_sn": points_sn,
        "point_count": len(points_world),
        "observation_count": 0,
        "frame_count": 0,
        "s_range": [points_sn[0][0], points_sn[-1][0]],
        "n_median": base.median(n_values),
        "n_range": [min(n_values), max(n_values)],
        "mean_lane_width": None,
        "name_hist": {"inferred_road_edge_white_solid": 1},
        "lane_type_hist": {"100": 1},
    }


def split_candidate_clear_segments(candidate: Dict, lanes: Sequence[Dict], start_index: int) -> List[Dict]:
    segments = []
    current_world = []
    current_sn = []
    for world, sn in zip(candidate["polyline_world"], candidate["polyline_sn"]):
        if existing_marking_near_sample(sn, lanes):
            if current_world:
                segments.extend(make_candidate_segments(candidate, current_world, current_sn, start_index + len(segments)))
                current_world = []
                current_sn = []
            continue
        current_world.append(world)
        current_sn.append(sn)
    if current_world:
        segments.extend(make_candidate_segments(candidate, current_world, current_sn, start_index + len(segments)))
    return segments


def make_candidate_segments(template: Dict, points_world: Sequence[Point3], points_sn: Sequence[Point3], index: int) -> List[Dict]:
    if len(points_world) < 8:
        return []
    if float(points_sn[-1][0]) - float(points_sn[0][0]) < 18.0:
        return []
    lane = dict(template)
    lane["global_lane_id"] = f"inferred_{index}"
    lane["polyline_world"] = [list(point) for point in points_world]
    lane["polyline_sn"] = [list(point) for point in points_sn]
    lane["point_count"] = len(points_world)
    lane["s_range"] = [points_sn[0][0], points_sn[-1][0]]
    n_values = [point[1] for point in points_sn]
    lane["n_median"] = base.median(n_values)
    lane["n_range"] = [min(n_values), max(n_values)]
    return [lane]


def existing_marking_near_sample(sample_sn: Point3, lanes: Sequence[Dict]) -> bool:
    s = float(sample_sn[0])
    n = float(sample_sn[1])
    for lane in lanes:
        other = interpolate_lane(lane, s)
        if other is None:
            continue
        if abs(other[1] - n) < 1.15:
            return True
    return False


def curvature_compatible(reference: Dict, candidate_sn: Sequence[Point3]) -> bool:
    checks = []
    for sample in candidate_sn[:: max(len(candidate_sn) // 16, 1)]:
        ref_sample = interpolate_lane(reference, sample[0])
        if ref_sample is None:
            continue
        checks.append(abs(local_slope(reference, sample[0]) - local_slope_from_sn(candidate_sn, sample[0])) < 0.55)
    return bool(checks) and sum(checks) / len(checks) >= 0.65


def local_slope(lane: Dict, s: float) -> float:
    sn = lane.get("polyline_sn", [])
    return local_slope_from_sn(sn, s)


def local_slope_from_sn(sn: Sequence[Point3], s: float) -> float:
    if len(sn) < 2:
        return 0.0
    nearest = min(range(len(sn)), key=lambda index: abs(float(sn[index][0]) - s))
    left = max(nearest - 2, 0)
    right = min(nearest + 2, len(sn) - 1)
    ds = float(sn[right][0]) - float(sn[left][0])
    if abs(ds) < 1e-6:
        return 0.0
    return (float(sn[right][1]) - float(sn[left][1])) / ds


def has_existing_marking_near(candidate: Dict, lanes: Sequence[Dict]) -> bool:
    n0 = float(candidate.get("n_median", 0.0))
    for other in lanes:
        if s_overlap(candidate, other) < 25.0:
            continue
        if abs(float(other.get("n_median", 0.0)) - n0) < 1.25:
            return True
    return False


def has_lateral_neighbor(lane: Dict, lanes: Sequence[Dict], side: int) -> bool:
    n0 = float(lane.get("n_median", 0.0))
    for other in lanes:
        if other is lane:
            continue
        overlap = s_overlap(lane, other)
        if overlap < 25.0:
            continue
        dn = float(other.get("n_median", 0.0)) - n0
        if side * dn <= 0.0:
            continue
        if 2.0 <= abs(dn) <= 4.9:
            return True
    return False


def make_inferred_solid_boundary(source: Dict, side: int, side_name: str, index: int) -> Dict:
    width = DEFAULT_INFERRED_LANE_WIDTH
    semantic = decode_lane_type(100)
    polyline_world = offset_polyline(source["polyline_world"], side * width)
    polyline_sn = [[point[0], point[1] + side * width, point[2]] for point in source["polyline_sn"]]
    n_values = [point[1] for point in polyline_sn]
    return {
        "global_lane_id": f"inferred_{index}",
        "source_track_id": f"inferred_from_{source['global_lane_id']}",
        "inferred": True,
        "inferred_from_lane_id": source["global_lane_id"],
        "inferred_side": side_name,
        "inference_reason": "white dashed lane marking has no adjacent boundary on this side",
        "dominant_lane_type": 100,
        "lane_type_semantic": semantic,
        "is_direction_divider": False,
        "polyline_world": polyline_world,
        "polyline_sn": polyline_sn,
        "point_count": len(polyline_world),
        "observation_count": 0,
        "frame_count": 0,
        "s_range": list(source["s_range"]),
        "n_median": base.median(n_values),
        "n_range": [min(n_values), max(n_values)],
        "mean_lane_width": None,
        "name_hist": {"inferred_white_solid_boundary": 1},
        "lane_type_hist": {"100": 1},
    }


def build_road_fills(lanes: Sequence[Dict]) -> List[Dict]:
    candidates = []
    sorted_lanes = sorted(lanes, key=lambda lane: lane.get("n_median", 0.0))
    yellow_lanes = [lane for lane in sorted_lanes if lane.get("lane_type_semantic", {}).get("color") == "yellow"]
    center_dividers = central_yellow_dividers(sorted_lanes, yellow_lanes)

    for left_index, left in enumerate(sorted_lanes):
        for right in sorted_lanes[left_index + 1 :]:
            fill = build_fill_between(left, right, center_dividers)
            if fill is not None:
                candidates.append(fill)

    candidates.sort(
        key=lambda fill: (
            fill["s_range"][0],
            fill["n_range"][0],
            [str(value) for value in fill["boundary_lane_ids"]],
        )
    )
    for index, fill in enumerate(candidates):
        fill["road_fill_id"] = index
    return candidates


def build_road_connectors(fills: Sequence[Dict]) -> List[Dict]:
    """Connect short gaps between drivable road fills for map-graph export.

    The fill centerlines are already oriented as CARLA Traffic Manager should
    drive them.  A connector is only created when the end of one directed fill
    is close to the start of another fill with the same traffic direction and
    the headings agree.  This keeps the visual lane marking inference separate
    from the road-level graph needed by CARLA/nuscenes conversion.
    """

    candidates = []
    for source in fills:
        source_center = source.get("traffic_manager_centerline_world", [])
        if len(source_center) < 2:
            continue
        source_end = source_center[-1]
        source_heading = polyline_end_heading(source_center)
        for target in fills:
            if source is target:
                continue
            if source.get("traffic_direction") != target.get("traffic_direction"):
                continue
            target_center = target.get("traffic_manager_centerline_world", [])
            if len(target_center) < 2:
                continue
            target_start = target_center[0]
            target_heading = polyline_start_heading(target_center)
            gap = distance_xy(source_end, target_start)
            if gap < 2.0 or gap > 85.0:
                continue
            gap_heading = math.atan2(target_start[1] - source_end[1], target_start[0] - source_end[0])
            source_angle = abs(base.normalize_angle(gap_heading - source_heading))
            target_angle = abs(base.normalize_angle(target_heading - gap_heading))
            lateral_gap = abs(
                -(target_start[0] - source_end[0]) * math.sin(source_heading)
                + (target_start[1] - source_end[1]) * math.cos(source_heading)
            )
            if source_angle > math.radians(75.0) or target_angle > math.radians(75.0):
                continue
            if lateral_gap > 11.0:
                continue
            score = gap + 18.0 * source_angle + 18.0 * target_angle + 1.5 * lateral_gap
            candidates.append((score, gap, source_angle, target_angle, source, target))

    candidates.sort(key=lambda item: item[0])
    used_sources = set()
    used_targets = set()
    connectors = []
    for _score, gap, source_angle, target_angle, source, target in candidates:
        source_id = source["road_fill_id"]
        target_id = target["road_fill_id"]
        if source_id in used_sources or target_id in used_targets:
            continue
        center = make_connector_centerline(
            source["traffic_manager_centerline_world"],
            target["traffic_manager_centerline_world"],
        )
        if len(center) < 2:
            continue
        connectors.append(
            {
                "road_connector_id": len(connectors),
                "from_road_fill_id": source_id,
                "to_road_fill_id": target_id,
                "traffic_direction": source["traffic_direction"],
                "traffic_rule": "same_direction_gap_connector",
                "source": "road_fill_endpoint_geometry",
                "gap_length": gap,
                "source_heading_error_deg": math.degrees(source_angle),
                "target_heading_error_deg": math.degrees(target_angle),
                "centerline_world": center,
                "traffic_manager_centerline_world": center,
                "carla_role": "road_connector_centerline",
                "nuscenes_role": "lane_connector",
            }
        )
        used_sources.add(source_id)
        used_targets.add(target_id)
    return connectors


def polyline_start_heading(points: Sequence[Point3]) -> float:
    return math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])


def polyline_end_heading(points: Sequence[Point3]) -> float:
    return math.atan2(points[-1][1] - points[-2][1], points[-1][0] - points[-2][0])


def distance_xy(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def make_connector_centerline(source_center: Sequence[Point3], target_center: Sequence[Point3]) -> List[Point3]:
    start = source_center[-1]
    end = target_center[0]
    gap = distance_xy(start, end)
    if gap < 1e-6:
        return []
    h0 = polyline_end_heading(source_center)
    h1 = polyline_start_heading(target_center)
    tangent = min(max(gap * 0.45, 3.0), 22.0)
    m0 = (math.cos(h0) * tangent, math.sin(h0) * tangent)
    m1 = (math.cos(h1) * tangent, math.sin(h1) * tangent)
    count = max(6, min(18, int(math.ceil(gap / 4.0)) + 3))
    points = []
    for index in range(count):
        t = index / max(count - 1, 1)
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        x = h00 * start[0] + h10 * m0[0] + h01 * end[0] + h11 * m1[0]
        y = h00 * start[1] + h10 * m0[1] + h01 * end[1] + h11 * m1[1]
        points.append([x, y, 0.0])
    return points


def read_ascii_pcd_xyz(path: Path) -> List[Point3]:
    points = []
    data_started = False
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            if data_started:
                parts = text.split()
                if len(parts) < 3:
                    continue
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
            elif text.lower().startswith("data"):
                if text.lower() != "data ascii":
                    raise ValueError(f"Only ASCII PCD is supported for road polygon: {path}")
                data_started = True
    return points


def align_road_polygon_to_corrected_poses(
    points: Sequence[Point3],
    raw_poses: Dict[int, base.Pose2],
    corrected_poses: Dict[int, base.Pose2],
    frames: Sequence[int],
    pose_stride: int = 5,
    search_radius: float = 150.0,
    neighbor_count: int = 4,
) -> List[Point3]:
    if not points:
        return []
    pose_samples = []
    for frame in frames[:: max(pose_stride, 1)]:
        if frame not in raw_poses or frame not in corrected_poses:
            continue
        raw_pose = raw_poses[frame]
        corrected_pose = corrected_poses[frame]
        pose_samples.append(
            {
                "frame": frame,
                "raw": raw_pose,
                "corrected": corrected_pose,
            }
        )
    if not pose_samples:
        return [list(point) for point in points]

    cell_size = max(search_radius * 0.55, 1.0)
    pose_index: Dict[Tuple[int, int], List[int]] = {}
    for index, sample in enumerate(pose_samples):
        key = spatial_key(sample["raw"], cell_size)
        pose_index.setdefault(key, []).append(index)

    aligned = []
    for point in points:
        neighbors = nearest_pose_corrections(point, pose_samples, pose_index, cell_size, search_radius, neighbor_count)
        if not neighbors:
            aligned.append(list(point))
            continue
        sum_x = 0.0
        sum_y = 0.0
        sum_w = 0.0
        for distance, sample in neighbors:
            warped = warp_point_by_pose_correction(point, sample["raw"], sample["corrected"])
            weight = 1.0 / (distance * distance + 100.0)
            sum_x += warped[0] * weight
            sum_y += warped[1] * weight
            sum_w += weight
        aligned.append([sum_x / sum_w, sum_y / sum_w, float(point[2]) if len(point) > 2 else 0.0])
    return aligned


def nearest_pose_corrections(
    point: Sequence[float],
    pose_samples: Sequence[Dict],
    pose_index: Dict[Tuple[int, int], List[int]],
    cell_size: float,
    search_radius: float,
    count: int,
) -> List[Tuple[float, Dict]]:
    key_x, key_y = spatial_key(point, cell_size)
    candidates = []
    for ix in range(key_x - 2, key_x + 3):
        for iy in range(key_y - 2, key_y + 3):
            for sample_index in pose_index.get((ix, iy), []):
                sample = pose_samples[sample_index]
                raw_pose = sample["raw"]
                distance = math.hypot(float(point[0]) - raw_pose[0], float(point[1]) - raw_pose[1])
                if distance <= search_radius:
                    candidates.append((distance, sample))
    candidates.sort(key=lambda item: item[0])
    return candidates[: max(count, 1)]


def warp_point_by_pose_correction(point: Sequence[float], raw_pose: base.Pose2, corrected_pose: base.Pose2) -> Point3:
    raw_x, raw_y, raw_yaw = raw_pose
    corrected_x, corrected_y, corrected_yaw = corrected_pose
    c0 = math.cos(-raw_yaw)
    s0 = math.sin(-raw_yaw)
    dx = float(point[0]) - raw_x
    dy = float(point[1]) - raw_y
    local_x = c0 * dx - s0 * dy
    local_y = s0 * dx + c0 * dy
    c1 = math.cos(corrected_yaw)
    s1 = math.sin(corrected_yaw)
    z = float(point[2]) if len(point) > 2 else 0.0
    return [corrected_x + c1 * local_x - s1 * local_y, corrected_y + s1 * local_x + c1 * local_y, z]


def build_road_instruction_regions(
    road_polygon_points: Sequence[Point3],
    connectors: Sequence[Dict],
    fills: Sequence[Dict] = (),
    radius: float = 8.0,
    min_points: int = 20,
) -> List[Dict]:
    if not road_polygon_points:
        return []
    radius = max(radius, 10.0)
    index = build_spatial_point_index(road_polygon_points, radius)
    regions = []
    for connector in connectors:
        center = connector.get("traffic_manager_centerline_world", [])
        if len(center) < 2:
            continue
        matched_indices = set()
        for sample in resample_polyline(center, step=2.0):
            for point_index in query_spatial_point_index(index, sample, radius):
                point = road_polygon_points[point_index]
                if point_to_polyline_distance(point, center) <= radius:
                    matched_indices.add(point_index)
        matched_indices = filter_instruction_indices_outside_fills(matched_indices, road_polygon_points, fills)
        if len(matched_indices) < min_points:
            continue
        region_points = [road_polygon_points[index] for index in sorted(matched_indices)]
        xs = [point[0] for point in region_points]
        ys = [point[1] for point in region_points]
        zs = [point[2] for point in region_points]
        regions.append(
            {
                "road_instruction_id": len(regions),
                "source_connector_id": connector["road_connector_id"],
                "from_road_fill_id": connector["from_road_fill_id"],
                "to_road_fill_id": connector["to_road_fill_id"],
                "traffic_direction": connector["traffic_direction"],
                "instruction_type": "road_polygon_supports_lane_gap",
                "reason": (
                    "lane/road fill is disconnected in roadlane geometry, while fill_road.pcd "
                    "contains drivable-area evidence around the connector; points already inside "
                    "tracked road fills are clipped by fill endpoint stop-lines"
                ),
                "clip_rule": "exclude_points_inside_existing_road_fill_polygons",
                "point_count": len(region_points),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "z_median": base.median(zs),
                "connector_centerline_world": connector.get("traffic_manager_centerline_world", []),
                "road_polygon_points": region_points,
            }
        )
    regions.extend(
        build_endpoint_gap_instruction_regions(
            road_polygon_points,
            index,
            fills,
            connectors,
            radius=max(radius, 10.0),
            min_points=max(min_points, 80),
            start_region_id=len(regions),
        )
    )
    return regions


def build_endpoint_gap_instruction_regions(
    road_polygon_points: Sequence[Point3],
    point_index: Dict[Tuple[int, int], List[int]],
    fills: Sequence[Dict],
    connectors: Sequence[Dict],
    radius: float,
    min_points: int,
    start_region_id: int,
) -> List[Dict]:
    if not fills:
        return []
    existing_pairs = {(connector["from_road_fill_id"], connector["to_road_fill_id"]) for connector in connectors}
    candidates = []
    for source in fills:
        source_center = source.get("traffic_manager_centerline_world", [])
        if len(source_center) < 2:
            continue
        for target in fills:
            if source is target:
                continue
            source_id = source["road_fill_id"]
            target_id = target["road_fill_id"]
            if (source_id, target_id) in existing_pairs:
                continue
            if source.get("traffic_direction") != target.get("traffic_direction"):
                continue
            target_center = target.get("traffic_manager_centerline_world", [])
            if len(target_center) < 2:
                continue
            gap = distance_xy(source_center[-1], target_center[0])
            if gap < 14.0 or gap > 95.0:
                continue
            connector_center = make_connector_centerline(source_center, target_center)
            if len(connector_center) < 2:
                continue
            matched_indices = matched_road_polygon_indices(
                road_polygon_points,
                point_index,
                connector_center,
                radius,
            )
            matched_indices = filter_instruction_indices_outside_fills(matched_indices, road_polygon_points, fills)
            if len(matched_indices) < min_points:
                continue
            if not has_endpoint_gap_support(road_polygon_points, matched_indices, source_center[-1], target_center[0]):
                continue
            candidates.append((gap - 0.01 * len(matched_indices), gap, source, target, connector_center, matched_indices))

    candidates.sort(key=lambda item: item[0])
    used_sources = {connector["from_road_fill_id"] for connector in connectors}
    used_targets = {connector["to_road_fill_id"] for connector in connectors}
    regions = []
    for _score, gap, source, target, connector_center, matched_indices in candidates:
        source_id = source["road_fill_id"]
        target_id = target["road_fill_id"]
        if source_id in used_sources or target_id in used_targets:
            continue
        region_points = [road_polygon_points[index] for index in sorted(matched_indices)]
        xs = [point[0] for point in region_points]
        ys = [point[1] for point in region_points]
        zs = [point[2] for point in region_points]
        regions.append(
            {
                "road_instruction_id": start_region_id + len(regions),
                "source_connector_id": None,
                "from_road_fill_id": source_id,
                "to_road_fill_id": target_id,
                "traffic_direction": source["traffic_direction"],
                "instruction_type": "road_polygon_supports_endpoint_gap",
                "reason": (
                    "two directed road fill endpoints are not connected by tracked lane geometry, "
                    "but fill_road.pcd contains drivable-area evidence between them; points already "
                    "inside tracked road fills are clipped by fill endpoint stop-lines"
                ),
                "clip_rule": "exclude_points_inside_existing_road_fill_polygons",
                "gap_length": gap,
                "point_count": len(region_points),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "z_median": base.median(zs),
                "connector_centerline_world": connector_center,
                "road_polygon_points": region_points,
            }
        )
        used_sources.add(source_id)
        used_targets.add(target_id)
    return regions


def matched_road_polygon_indices(
    road_polygon_points: Sequence[Point3],
    point_index: Dict[Tuple[int, int], List[int]],
    centerline: Sequence[Point3],
    radius: float,
) -> set:
    matched_indices = set()
    for sample in resample_polyline(centerline, step=2.0):
        for point_index_value in query_spatial_point_index(point_index, sample, radius):
            point = road_polygon_points[point_index_value]
            if point_to_polyline_distance(point, centerline) <= radius:
                matched_indices.add(point_index_value)
    return matched_indices


def filter_instruction_indices_outside_fills(
    matched_indices: Sequence[int],
    road_polygon_points: Sequence[Point3],
    fills: Sequence[Dict],
) -> set:
    if not matched_indices or not fills:
        return set(matched_indices)
    fill_polygons = []
    for fill in fills:
        polygon = fill.get("polygon_world", [])
        if len(polygon) < 3:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        fill_polygons.append((polygon, (min(xs), min(ys), max(xs), max(ys))))

    output = set()
    for point_index_value in matched_indices:
        point = road_polygon_points[point_index_value]
        inside_existing_fill = False
        for polygon, bbox in fill_polygons:
            if point[0] < bbox[0] or point[0] > bbox[2] or point[1] < bbox[1] or point[1] > bbox[3]:
                continue
            if point_in_polygon(point, polygon):
                inside_existing_fill = True
                break
        if not inside_existing_fill:
            output.add(point_index_value)
    return output


def point_in_polygon(point: Sequence[float], polygon: Sequence[Point3]) -> bool:
    x = float(point[0])
    y = float(point[1])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def has_endpoint_gap_support(
    road_polygon_points: Sequence[Point3],
    matched_indices: Sequence[int],
    start: Point3,
    end: Point3,
) -> bool:
    gap = distance_xy(start, end)
    if gap < 1e-6:
        return False
    near_start = 0
    near_end = 0
    spans = []
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    denom = vx * vx + vy * vy
    for point_index_value in matched_indices:
        point = road_polygon_points[point_index_value]
        t = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / denom
        spans.append(t)
        if distance_xy(point, start) <= 16.0:
            near_start += 1
        if distance_xy(point, end) <= 16.0:
            near_end += 1
    if near_start < 12 or near_end < 12:
        return False
    return min(spans) <= 0.25 and max(spans) >= 0.75


def build_spatial_point_index(points: Sequence[Point3], cell_size: float) -> Dict[Tuple[int, int], List[int]]:
    index: Dict[Tuple[int, int], List[int]] = {}
    for point_index, point in enumerate(points):
        key = spatial_key(point, cell_size)
        index.setdefault(key, []).append(point_index)
    return index


def spatial_key(point: Sequence[float], cell_size: float) -> Tuple[int, int]:
    return (int(math.floor(float(point[0]) / cell_size)), int(math.floor(float(point[1]) / cell_size)))


def query_spatial_point_index(index: Dict[Tuple[int, int], List[int]], point: Sequence[float], radius: float) -> List[int]:
    cx, cy = spatial_key(point, radius)
    output = []
    for ix in range(cx - 1, cx + 2):
        for iy in range(cy - 1, cy + 2):
            output.extend(index.get((ix, iy), []))
    return output


def resample_polyline(points: Sequence[Point3], step: float) -> List[Point3]:
    if len(points) < 2:
        return list(points)
    samples = [points[0]]
    carry = 0.0
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        distance = step - carry
        while distance < length:
            ratio = distance / length
            samples.append([start[0] + dx * ratio, start[1] + dy * ratio, 0.0])
            distance += step
        carry = max(length - (distance - step), 0.0)
    samples.append(points[-1])
    return samples


def point_to_polyline_distance(point: Sequence[float], polyline: Sequence[Point3]) -> float:
    best = float("inf")
    for index in range(len(polyline) - 1):
        best = min(best, point_to_segment_distance(point, polyline[index], polyline[index + 1]))
    return best


def point_to_segment_distance(point: Sequence[float], start: Sequence[float], end: Sequence[float]) -> float:
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    wx = point[0] - start[0]
    wy = point[1] - start[1]
    denom = vx * vx + vy * vy
    if denom < 1e-9:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    px = start[0] + t * vx
    py = start[1] + t * vy
    return math.hypot(point[0] - px, point[1] - py)


def central_yellow_dividers(lanes: Sequence[Dict], yellow_lanes: Sequence[Dict]) -> List[Dict]:
    dividers = []
    for yellow in yellow_lanes:
        has_lower = has_lane_on_side(yellow, lanes, -1)
        has_upper = has_lane_on_side(yellow, lanes, 1)
        if has_lower and has_upper:
            dividers.append(yellow)
    return dividers


def has_lane_on_side(lane: Dict, lanes: Sequence[Dict], side: int) -> bool:
    n0 = float(lane.get("n_median", 0.0))
    for other in lanes:
        if other is lane:
            continue
        if s_overlap(lane, other) < 25.0:
            continue
        dn = float(other.get("n_median", 0.0)) - n0
        if side * dn <= 0.0:
            continue
        if 1.8 <= abs(dn) <= 7.2:
            return True
    return False


def build_fill_between(left: Dict, right: Dict, center_dividers: Sequence[Dict]) -> Optional[Dict]:
    n_left = float(left.get("n_median", 0.0))
    n_right = float(right.get("n_median", 0.0))
    n_gap = abs(n_right - n_left)
    has_inferred_boundary = bool(left.get("inferred")) or bool(right.get("inferred"))
    if not has_inferred_boundary and (n_gap < 2.0 or n_gap > 6.8):
        return None
    if has_inferred_boundary and n_gap > 8.5:
        return None

    left_sem = left.get("lane_type_semantic", {})
    right_sem = right.get("lane_type_semantic", {})
    if left_sem.get("color") == "yellow" and right_sem.get("color") == "yellow" and n_gap < 2.2:
        return None

    s0 = max(float(left["s_range"][0]), float(right["s_range"][0]))
    s1 = min(float(left["s_range"][1]), float(right["s_range"][1]))
    if s1 - s0 < 18.0:
        return None

    samples = sample_pair(left, right, s0, s1, step=2.0)
    if len(samples) < 8:
        return None

    widths = [abs(sample["right"][1] - sample["left"][1]) for sample in samples]
    median_width = base.median(widths)
    if median_width < 2.0 or median_width > 6.8:
        return None

    lower_boundary = left if n_left <= n_right else right
    upper_boundary = right if n_left <= n_right else left
    polygon = [sample["lower_world"] for sample in samples] + [sample["upper_world"] for sample in reversed(samples)]
    centerline = [sample["center_world"] for sample in samples]
    centerline_sn = [[sample["s"], sample["mid_n"], 0.0] for sample in samples]
    mid_n = base.median([sample["mid_n"] for sample in samples])
    divider = nearest_yellow_divider(center_dividers, mid_n, s0, s1)
    direction = direction_from_divider(mid_n, divider)
    traffic_rule = "right_hand_two_way_by_center_yellow" if divider is not None else "one_way_no_opposing_lane_scanned"
    centerline_for_tm = list(centerline if direction == "with_pose_s" else reversed(centerline))

    return {
        "road_fill_id": -1,
        "boundary_lane_ids": [lower_boundary["global_lane_id"], upper_boundary["global_lane_id"]],
        "lower_n_boundary_lane_id": lower_boundary["global_lane_id"],
        "upper_n_boundary_lane_id": upper_boundary["global_lane_id"],
        "s_range": [s0, s1],
        "n_range": [min(n_left, n_right), max(n_left, n_right)],
        "median_width": median_width,
        "polygon_world": polygon,
        "centerline_world": centerline,
        "centerline_sn": centerline_sn,
        "traffic_manager_centerline_world": centerline_for_tm,
        "traffic_direction": direction,
        "traffic_rule": traffic_rule,
        "right_hand_traffic_basis": {
            "nearest_yellow_lane_id": None if divider is None else divider["global_lane_id"],
            "nearest_yellow_n_median": None if divider is None else divider["n_median"],
            "rule": (
                "central yellow with lanes on both sides -> two-way right-hand traffic; "
                "otherwise treat scanned carriageway as one-way with_pose_s"
            ),
        },
    }


def s_overlap(left: Dict, right: Dict) -> float:
    return min(float(left["s_range"][1]), float(right["s_range"][1])) - max(
        float(left["s_range"][0]), float(right["s_range"][0])
    )


def sample_pair(left: Dict, right: Dict, s0: float, s1: float, step: float) -> List[Dict]:
    count = max(int(math.floor((s1 - s0) / step)) + 1, 2)
    samples = []
    for index in range(count):
        s = s0 + (s1 - s0) * index / max(count - 1, 1)
        left_sample = interpolate_lane(left, s)
        right_sample = interpolate_lane(right, s)
        if left_sample is None or right_sample is None:
            continue
        lower, upper = (left_sample, right_sample) if left_sample[1] <= right_sample[1] else (right_sample, left_sample)
        center = [
            (left_sample[2][0] + right_sample[2][0]) * 0.5,
            (left_sample[2][1] + right_sample[2][1]) * 0.5,
            0.0,
        ]
        samples.append(
            {
                "s": s,
                "left": left_sample,
                "right": right_sample,
                "mid_n": (left_sample[1] + right_sample[1]) * 0.5,
                "lower_world": lower[2],
                "upper_world": upper[2],
                "center_world": center,
            }
        )
    return samples


def interpolate_lane(lane: Dict, s: float) -> Optional[Tuple[float, float, Point3]]:
    sn = lane.get("polyline_sn", [])
    world = lane.get("polyline_world", [])
    if len(sn) < 2 or len(sn) != len(world):
        return None
    if s < sn[0][0] or s > sn[-1][0]:
        return None
    for index in range(len(sn) - 1):
        s_a = float(sn[index][0])
        s_b = float(sn[index + 1][0])
        if s_a <= s <= s_b or s_b <= s <= s_a:
            denom = s_b - s_a
            ratio = 0.0 if abs(denom) < 1e-6 else (s - s_a) / denom
            n = float(sn[index][1]) + ratio * (float(sn[index + 1][1]) - float(sn[index][1]))
            x = float(world[index][0]) + ratio * (float(world[index + 1][0]) - float(world[index][0]))
            y = float(world[index][1]) + ratio * (float(world[index + 1][1]) - float(world[index][1]))
            return (s, n, [x, y, 0.0])
    return None


def nearest_yellow_divider(yellow_lanes: Sequence[Dict], mid_n: float, s0: float, s1: float) -> Optional[Dict]:
    candidates = []
    for lane in yellow_lanes:
        overlap = min(s1, float(lane["s_range"][1])) - max(s0, float(lane["s_range"][0]))
        if overlap <= 0.0:
            continue
        candidates.append((abs(float(lane.get("n_median", 0.0)) - mid_n), -overlap, lane))
    if not candidates:
        return None
    return sorted(candidates)[0][2]


def direction_from_divider(mid_n: float, divider: Optional[Dict]) -> str:
    if divider is None:
        return "with_pose_s"
    return "with_pose_s" if mid_n < float(divider.get("n_median", 0.0)) else "against_pose_s"


def make_local_to_enu_transform(poses: Dict[int, Dict[str, float]], frames: Sequence[int]) -> Dict:
    first = poses[frames[0]]
    yaw0 = float(first["yaw"])
    return {
        "type": "local_to_enu_from_initial_gnss_pose",
        "origin_frame": int(frames[0]),
        "origin_enu_x": float(first["x"]),
        "origin_enu_y": float(first["y"]),
        "origin_yaw": yaw0,
        "cos_yaw": math.cos(yaw0),
        "sin_yaw": math.sin(yaw0),
        "formula": "enu_xy = origin_xy + R(origin_yaw) * corrected_local_xy",
    }


def local_point_to_enu(point: Sequence[float], transform: Dict) -> Point3:
    c = float(transform["cos_yaw"])
    s = float(transform["sin_yaw"])
    x0 = float(transform["origin_enu_x"])
    y0 = float(transform["origin_enu_y"])
    x = float(point[0])
    y = float(point[1])
    z = float(point[2]) if len(point) > 2 else 0.0
    return [x0 + c * x - s * y, y0 + s * x + c * y, z]


def local_pose_to_enu(pose: base.Pose2, transform: Dict) -> base.Pose2:
    point = local_point_to_enu((pose[0], pose[1], 0.0), transform)
    return (point[0], point[1], base.normalize_angle(float(pose[2]) + float(transform["origin_yaw"])))


def transform_lanes_to_enu(lanes: Sequence[Dict], transform: Dict) -> List[Dict]:
    output = []
    for lane in lanes:
        item = copy.deepcopy(lane)
        item["polyline_world"] = [local_point_to_enu(point, transform) for point in lane.get("polyline_world", [])]
        item["coordinate_frame"] = "enu"
        output.append(item)
    return output


def transform_fills_to_enu(fills: Sequence[Dict], transform: Dict) -> List[Dict]:
    output = []
    for fill in fills:
        item = copy.deepcopy(fill)
        for key in ("polygon_world", "centerline_world", "traffic_manager_centerline_world"):
            item[key] = [local_point_to_enu(point, transform) for point in fill.get(key, [])]
        item["coordinate_frame"] = "enu"
        output.append(item)
    return output


def transform_connectors_to_enu(connectors: Sequence[Dict], transform: Dict) -> List[Dict]:
    output = []
    for connector in connectors:
        item = copy.deepcopy(connector)
        for key in ("centerline_world", "traffic_manager_centerline_world"):
            item[key] = [local_point_to_enu(point, transform) for point in connector.get(key, [])]
        item["coordinate_frame"] = "enu"
        output.append(item)
    return output


def transform_poses_to_enu(poses: Dict[int, base.Pose2], transform: Dict) -> Dict[int, base.Pose2]:
    return {frame: local_pose_to_enu(pose, transform) for frame, pose in poses.items()}


def transform_point_records_to_enu(points: Sequence[Dict], transform: Dict) -> List[Dict]:
    output = []
    for point in points:
        item = dict(point)
        enu = local_point_to_enu((point["x"], point["y"], 0.0), transform)
        item["x"] = enu[0]
        item["y"] = enu[1]
        item["heading"] = base.normalize_angle(float(point["heading"]) + float(transform["origin_yaw"]))
        output.append(item)
    return output


def align_multi_sequence_results_by_gnss(results: Sequence[Dict], base_index: int) -> List[Dict]:
    if not results:
        return []
    aligned = [copy.deepcopy(result) for result in results]
    gps_origins = [read_rtk_gps_origin(Path(result["sequence_dir"]) / "raw" / "rtk_gps.txt") for result in aligned]
    base_gps = gps_origins[base_index]
    if base_gps is None:
        for result, gps_origin in zip(aligned, gps_origins):
            result["rtk_gps_origin"] = gps_origin
            result["multi_alignment_offset_xy"] = [0.0, 0.0]
        return aligned

    base_origin = result_map_origin_xy(aligned[base_index])
    for index, result in enumerate(aligned):
        gps_origin = gps_origins[index]
        own_origin = result_map_origin_xy(result)
        gps_offset = (0.0, 0.0) if gps_origin is None else gnss_offset_meters(base_gps, gps_origin)
        translation = [base_origin[0] + gps_offset[0] - own_origin[0], base_origin[1] + gps_offset[1] - own_origin[1]]
        translate_result_xy(result, translation[0], translation[1])
        result["rtk_gps_origin"] = gps_origin
        result["multi_alignment_offset_xy"] = translation
        result["multi_alignment_basis"] = {
            "method": "rtk_gps_origin_offset",
            "base_sequence_dir": str(aligned[base_index]["sequence_dir"]),
            "gps_offset_east_north_m": [gps_offset[0], gps_offset[1]],
            "translation_applied_xy_m": translation,
        }
    return aligned


def read_rtk_gps_origin(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) < 4:
                continue
            try:
                return {
                    "timestamp": int(float(parts[0])),
                    "longitude": float(parts[1]),
                    "latitude": float(parts[2]),
                    "altitude": float(parts[3]),
                    "status": int(float(parts[4])) if len(parts) > 4 else None,
                    "source": str(path),
                }
            except ValueError:
                continue
    return None


def gnss_offset_meters(base_gps: Dict, target_gps: Dict) -> Tuple[float, float]:
    radius = 6378137.0
    lat0 = math.radians(float(base_gps["latitude"]))
    lon0 = math.radians(float(base_gps["longitude"]))
    lat1 = math.radians(float(target_gps["latitude"]))
    lon1 = math.radians(float(target_gps["longitude"]))
    east = (lon1 - lon0) * math.cos((lat0 + lat1) * 0.5) * radius
    north = (lat1 - lat0) * radius
    return (east, north)


def result_map_origin_xy(result: Dict) -> Tuple[float, float]:
    transform = result.get("metadata", {}).get("local_to_enu", {})
    if "origin_enu_x" in transform and "origin_enu_y" in transform:
        return (float(transform["origin_enu_x"]), float(transform["origin_enu_y"]))
    poses = result.get("poses", {})
    if poses:
        first_key = sorted(poses)[0]
        pose = poses[first_key]
        return (float(pose[0]), float(pose[1]))
    return (0.0, 0.0)


def translate_result_xy(result: Dict, dx: float, dy: float) -> None:
    for lane in result.get("lanes", []):
        translate_points_xy(lane.get("polyline_world", []), dx, dy)
    for fill in result.get("road_fills", []):
        for key in ("polygon_world", "centerline_world", "traffic_manager_centerline_world"):
            translate_points_xy(fill.get(key, []), dx, dy)
    for connector in result.get("road_connectors", []):
        for key in ("centerline_world", "traffic_manager_centerline_world"):
            translate_points_xy(connector.get(key, []), dx, dy)
    translate_points_xy(result.get("road_polygon_points", []), dx, dy)
    for region in result.get("road_instruction_regions", []):
        translate_points_xy(region.get("road_polygon_points", []), dx, dy)
        translate_points_xy(region.get("connector_centerline_world", []), dx, dy)
        if "bbox" in region and len(region["bbox"]) == 4:
            region["bbox"] = [region["bbox"][0] + dx, region["bbox"][1] + dy, region["bbox"][2] + dx, region["bbox"][3] + dy]
    for point in result.get("raw_points", []):
        if "x" in point:
            point["x"] += dx
        if "y" in point:
            point["y"] += dy
    translated_poses = {}
    for frame, pose in result.get("poses", {}).items():
        translated_poses[frame] = (pose[0] + dx, pose[1] + dy, pose[2])
    result["poses"] = translated_poses


def translate_points_xy(points: Sequence[Point3], dx: float, dy: float) -> None:
    for point in points:
        point[0] += dx
        point[1] += dy


def write_pose_enu(path: Path, frames: Sequence[int], poses: Dict[int, base.Pose2], debug: Dict[int, Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frame", "enu_x", "enu_y", "enu_yaw", "match_status", "match_count", "match_score"])
        for frame in frames:
            pose = poses[frame]
            info = debug.get(frame, {})
            writer.writerow(
                [
                    frame,
                    pose[0],
                    pose[1],
                    pose[2],
                    info.get("status", ""),
                    info.get("matches", 0),
                    info.get("score", ""),
                ]
            )


def draw_semantic_bev(path: Path, raw_points: Sequence[Dict], poses: Dict, lanes: Sequence[Dict], args) -> None:
    setup_matplotlib()
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    apply_dark_axes(ax)
    if not args.no_raw_points and raw_points:
        plt.scatter(
            [point["x"] for point in raw_points],
            [point["y"] for point in raw_points],
            s=0.10,
            c="#8b949e",
            alpha=0.10,
            linewidths=0,
            label="corrected lane samples",
            zorder=0,
        )

    draw_poses(poses)
    for lane in lanes:
        draw_lane_semantic(lane, annotate=False)

    legend_items = semantic_legend_items(lanes, mlines)
    style_legend(plt.legend(handles=legend_items, loc="best", fontsize=10))
    finish_plot("ClassLab RoadLane BEV by laneType", raw_points, lanes, args)
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close()


def draw_road_fill_bev(
    path: Path,
    raw_points: Sequence[Dict],
    poses: Dict,
    lanes: Sequence[Dict],
    fills: Sequence[Dict],
    args,
    connectors: Sequence[Dict] = (),
) -> None:
    setup_matplotlib()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    apply_dark_axes(ax)
    for fill in fills:
        polygon = fill["polygon_world"]
        color = FILL_FORWARD if fill["traffic_direction"] == "with_pose_s" else FILL_REVERSE
        plt.fill([p[0] for p in polygon], [p[1] for p in polygon], color=color, alpha=0.28, linewidth=0, zorder=1)
        center = fill["traffic_manager_centerline_world"]
        if len(center) >= 2:
            xs = [p[0] for p in center]
            ys = [p[1] for p in center]
            plt.plot(xs, ys, color=color, linewidth=1.4, alpha=0.85, zorder=2)
            draw_direction_arrows(center, color)

    for connector in connectors:
        center = connector.get("traffic_manager_centerline_world", [])
        if len(center) < 2:
            continue
        color = FILL_FORWARD if connector.get("traffic_direction") == "with_pose_s" else FILL_REVERSE
        xs = [p[0] for p in center]
        ys = [p[1] for p in center]
        plt.plot(
            xs,
            ys,
            color=color,
            linestyle=(0, (5, 5)),
            linewidth=2.3,
            alpha=0.96,
            zorder=5,
        )
        draw_direction_arrows(center, color)

    if not args.no_raw_points and raw_points:
        plt.scatter(
            [point["x"] for point in raw_points],
            [point["y"] for point in raw_points],
            s=0.08,
            c="#8b949e",
            alpha=0.08,
            linewidths=0,
            zorder=0,
        )
    draw_poses(poses)
    for lane in lanes:
        draw_lane_semantic(lane, annotate=False, linewidth=1.8)

    legend_items = [
        mpatches.Patch(color=FILL_FORWARD, alpha=0.32, label="行驶方向: pose s 正向"),
        mpatches.Patch(color=FILL_REVERSE, alpha=0.32, label="行驶方向: pose s 反向"),
        mpatches.Patch(color="#c9d1d9", alpha=0.32, label="虚线中心线: road/lane connector"),
    ]
    style_legend(plt.legend(handles=legend_items, loc="best", fontsize=8))
    finish_plot("ClassLab Road Fill Candidates for CARLA/XODR", raw_points, lanes, args)
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close()



def collect_context_polylines(
    lanes: Sequence[Dict] = (),
    fills: Sequence[Dict] = (),
    connectors: Sequence[Dict] = (),
    poses: Optional[Dict] = None,
) -> List[List[Point3]]:
    """Collect reliable road/lane geometry used as the local map context.

    The raw fill_road.pcd may contain points from a larger SLAM/road-edge map.
    For visualization we only use points that are close to lane markings,
    drivable-fill centerlines, connector centerlines, or the corrected ego path.
    This prevents road-instruction support points from expanding the BEV canvas
    or appearing far away from the road-lane graph.
    """

    polylines: List[List[Point3]] = []
    for lane in lanes or []:
        points = lane.get("polyline_world", [])
        if len(points) >= 2:
            polylines.append(points)
    for fill in fills or []:
        for key in ("traffic_manager_centerline_world", "centerline_world"):
            points = fill.get(key, [])
            if len(points) >= 2:
                polylines.append(points)
                break
        polygon = fill.get("polygon_world", [])
        if len(polygon) >= 3:
            polylines.append(polygon + [polygon[0]])
    for connector in connectors or []:
        for key in ("traffic_manager_centerline_world", "centerline_world"):
            points = connector.get(key, [])
            if len(points) >= 2:
                polylines.append(points)
                break
    if poses:
        traj = [[pose[0], pose[1], 0.0] for _frame, pose in sorted(poses.items())]
        if len(traj) >= 2:
            polylines.append(traj)
    return polylines


def context_sample_points(polylines: Sequence[Sequence[Point3]], step: float = 3.0) -> List[Point3]:
    samples: List[Point3] = []
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        samples.extend(resample_polyline(polyline, step=step))
    return samples


def filter_points_near_context(
    points: Sequence[Point3],
    context_points: Sequence[Point3],
    radius: float = 28.0,
) -> List[Point3]:
    """Keep only points close to the trusted road/lane context."""

    if not points or not context_points:
        return list(points)
    point_index = build_spatial_point_index(context_points, radius)
    kept: List[Point3] = []
    radius_sq = radius * radius
    for point in points:
        keep = False
        for context_index in query_spatial_point_index(point_index, point, radius):
            context = context_points[context_index]
            dx = float(point[0]) - float(context[0])
            dy = float(point[1]) - float(context[1])
            if dx * dx + dy * dy <= radius_sq:
                keep = True
                break
        if keep:
            kept.append(point)
    return kept


def clipped_region_points(
    region: Dict,
    context_points: Sequence[Point3],
    radius: float = 28.0,
    min_keep: int = 6,
) -> List[Point3]:
    """Return instruction support points clipped to the road-lane context.

    If a region was generated from distant road-polygon outliers, the returned
    list is empty; the caller can still draw the connector centerline as a red
    instruction cue.
    """

    points = region.get("road_polygon_points", [])
    if not points:
        return []
    filtered = filter_points_near_context(points, context_points, radius=radius)
    return filtered if len(filtered) >= min_keep else []


def collect_plot_support_points(
    raw_points: Sequence[Dict] = (),
    lanes: Sequence[Dict] = (),
    fills: Sequence[Dict] = (),
    connectors: Sequence[Dict] = (),
    instruction_regions: Sequence[Dict] = (),
    poses: Optional[Dict] = None,
) -> List[Point3]:
    """Collect points that should control BEV plot limits.

    This intentionally excludes the full raw road polygon cloud; only clipped
    instruction support and trusted lane/road geometry are used for axis bounds.
    """

    support: List[Point3] = []
    for point in raw_points or []:
        if "x" in point and "y" in point:
            support.append([point["x"], point["y"], 0.0])
    for polyline in collect_context_polylines(lanes, fills, connectors, poses):
        support.extend(polyline)
    context_points = context_sample_points(collect_context_polylines(lanes, fills, connectors, poses), step=3.0)
    for region in instruction_regions or []:
        support.extend(clipped_region_points(region, context_points, radius=30.0, min_keep=3))
        center = region.get("connector_centerline_world", [])
        if len(center) >= 2:
            support.extend(center)
    return support

def draw_road_instruction_bev(
    path: Path,
    road_polygon_points: Sequence[Point3],
    instruction_regions: Sequence[Dict],
    raw_points: Sequence[Dict],
    poses: Dict,
    lanes: Sequence[Dict],
    fills: Sequence[Dict],
    connectors: Sequence[Dict],
    args,
) -> None:
    setup_matplotlib()
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    apply_dark_axes(ax)

    context_polylines = collect_context_polylines(lanes, fills, connectors, poses)
    context_points = context_sample_points(context_polylines, step=3.0)
    visible_road_polygon_points = filter_points_near_context(
        road_polygon_points,
        context_points,
        radius=35.0,
    )

    if visible_road_polygon_points:
        stride = max(len(visible_road_polygon_points) // 65000, 1)
        show_points = visible_road_polygon_points[::stride]
        plt.scatter(
            [point[0] for point in show_points],
            [point[1] for point in show_points],
            s=0.9,
            marker="s",
            c="#8b949e",
            alpha=0.20,
            linewidths=0,
            zorder=0,
        )

    for fill in fills:
        polygon = fill["polygon_world"]
        color = FILL_FORWARD if fill["traffic_direction"] == "with_pose_s" else FILL_REVERSE
        plt.fill([p[0] for p in polygon], [p[1] for p in polygon], color=color, alpha=0.12, linewidth=0, zorder=1)

    for region in instruction_regions:
        points = clipped_region_points(region, context_points, radius=30.0, min_keep=6)
        if points:
            plt.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=4.5,
                marker="s",
                c="#ff3b30",
                alpha=0.70,
                linewidths=0,
                zorder=4,
            )
        center = region.get("connector_centerline_world", [])
        if len(center) >= 2:
            xs = [p[0] for p in center]
            ys = [p[1] for p in center]
            plt.plot(xs, ys, color="#ff6b5f", linestyle=(0, (4, 4)), linewidth=2.8, alpha=0.98, zorder=6)
            draw_direction_arrows(center, "#ff6b5f")

    for connector in connectors:
        center = connector.get("traffic_manager_centerline_world", [])
        if len(center) < 2:
            continue
        color = FILL_FORWARD if connector.get("traffic_direction") == "with_pose_s" else FILL_REVERSE
        plt.plot(
            [point[0] for point in center],
            [point[1] for point in center],
            color=color,
            linestyle=(0, (5, 5)),
            linewidth=1.6,
            alpha=0.78,
            zorder=5,
        )
        draw_direction_arrows(center, color)

    if not args.no_raw_points and raw_points:
        plt.scatter(
            [point["x"] for point in raw_points],
            [point["y"] for point in raw_points],
            s=0.08,
            c="#8b949e",
            alpha=0.06,
            linewidths=0,
            zorder=0,
        )
    draw_poses(poses)
    for lane in lanes:
        draw_lane_semantic(lane, annotate=False, linewidth=1.7)

    legend_items = [
        mlines.Line2D([], [], color="#8b949e", marker="s", linestyle="", markersize=5, alpha=0.5, label="clipped fill_road.pcd road polygon"),
        mpatches.Patch(color="#ff3b30", alpha=0.70, label="红色: road instruction / lane gap support"),
        mlines.Line2D([], [], color=FILL_FORWARD, linestyle=(0, (5, 5)), linewidth=2.0, label="connector: pose s 正向"),
        mlines.Line2D([], [], color=FILL_REVERSE, linestyle=(0, (5, 5)), linewidth=2.0, label="connector: pose s 反向"),
    ]
    style_legend(plt.legend(handles=legend_items, loc="best", fontsize=8))
    support_points = collect_plot_support_points(
        raw_points=raw_points,
        lanes=lanes,
        fills=fills,
        connectors=connectors,
        instruction_regions=instruction_regions,
        poses=poses,
    )
    finish_plot_with_extra_points(
        "ClassLab RoadLane + Clipped Road Instructions",
        raw_points,
        lanes,
        support_points,
        args,
    )
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close()


def draw_multi_sequence_bev(path: Path, results: Sequence[Dict], base_index: int, args) -> None:
    setup_matplotlib()
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    apply_dark_axes(ax)
    sequence_colors = ["#56b4e9", "#e69f00", "#009e73", "#cc79a7", "#f0e442", "#0072b2"]

    clipped_polygon_points_by_seq: Dict[int, List[Point3]] = {}
    clipped_instruction_points_by_seq: Dict[int, List[Point3]] = {}

    for index, result in enumerate(results):
        color = sequence_colors[index % len(sequence_colors)]
        context_polylines = collect_context_polylines(
            result.get("lanes", []),
            result.get("road_fills", []),
            result.get("road_connectors", []),
            result.get("poses", {}),
        )
        context_points = context_sample_points(context_polylines, step=3.0)
        polygon_points = result.get("road_polygon_points", [])
        visible_polygon_points = filter_points_near_context(polygon_points, context_points, radius=35.0)
        clipped_polygon_points_by_seq[index] = visible_polygon_points
        if visible_polygon_points:
            stride = max(len(visible_polygon_points) // 45000, 1)
            show_points = visible_polygon_points[::stride]
            plt.scatter(
                [point[0] for point in show_points],
                [point[1] for point in show_points],
                s=0.6,
                marker="s",
                c="#8b949e" if index == base_index else color,
                alpha=0.16 if index == base_index else 0.10,
                linewidths=0,
                zorder=0,
            )

        for fill in result.get("road_fills", []):
            polygon = fill.get("polygon_world", [])
            if len(polygon) >= 3:
                plt.fill(
                    [point[0] for point in polygon],
                    [point[1] for point in polygon],
                    color=color,
                    alpha=0.10 if index == base_index else 0.06,
                    linewidth=0,
                    zorder=1,
                )
            center = fill.get("traffic_manager_centerline_world", [])
            if len(center) >= 2:
                plt.plot(
                    [point[0] for point in center],
                    [point[1] for point in center],
                    color=color,
                    linewidth=1.0 if index == base_index else 0.8,
                    alpha=0.48 if index == base_index else 0.28,
                    zorder=2,
                )

        for connector in result.get("road_connectors", []):
            center = connector.get("traffic_manager_centerline_world", [])
            if len(center) < 2:
                continue
            plt.plot(
                [point[0] for point in center],
                [point[1] for point in center],
                color=color,
                linestyle=(0, (5, 5)),
                linewidth=2.0 if index == base_index else 1.2,
                alpha=0.82 if index == base_index else 0.48,
                zorder=5,
            )

        for lane in result.get("lanes", []):
            points = lane.get("polyline_world", [])
            if len(points) < 2:
                continue
            semantic = lane.get("lane_type_semantic", {})
            lane_color = semantic.get("draw_color", color)
            linestyle = linestyle_for_semantic(semantic)
            plt.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=lane_color,
                linestyle=linestyle,
                linewidth=2.0 if index == base_index else 1.2,
                alpha=0.95 if index == base_index else 0.55,
                zorder=4,
            )

        clipped_instruction_points: List[Point3] = []
        for region in result.get("road_instruction_regions", []):
            points = clipped_region_points(region, context_points, radius=30.0, min_keep=6)
            clipped_instruction_points.extend(points)
            if points:
                plt.scatter(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    s=3.8 if index == base_index else 2.4,
                    marker="s",
                    c="#ff3b30",
                    alpha=0.72 if index == base_index else 0.42,
                    linewidths=0,
                    zorder=7,
                )
            center = region.get("connector_centerline_world", [])
            if len(center) >= 2:
                plt.plot(
                    [point[0] for point in center],
                    [point[1] for point in center],
                    color="#ff6b5f",
                    linestyle=(0, (4, 4)),
                    linewidth=2.8 if index == base_index else 1.8,
                    alpha=0.98 if index == base_index else 0.62,
                    zorder=8,
                )
                clipped_instruction_points.extend(center)
        clipped_instruction_points_by_seq[index] = clipped_instruction_points

    all_raw_points = []
    all_lanes = []
    extra_points = []
    for index, result in enumerate(results):
        all_raw_points.extend(result.get("raw_points", []))
        all_lanes.extend(result.get("lanes", []))
        extra_points.extend(clipped_polygon_points_by_seq.get(index, []))
        extra_points.extend(clipped_instruction_points_by_seq.get(index, []))
        extra_points.extend(
            collect_plot_support_points(
                raw_points=[],
                lanes=result.get("lanes", []),
                fills=result.get("road_fills", []),
                connectors=result.get("road_connectors", []),
                instruction_regions=[],
                poses=result.get("poses", {}),
            )
        )

    legend_items = [
        mlines.Line2D([], [], color=sequence_colors[index % len(sequence_colors)], linewidth=2.0, label=f"seq {index}: {result['sequence_dir'].name}")
        for index, result in enumerate(results)
    ]
    legend_items.append(mpatches.Patch(color="#ff3b30", alpha=0.62, label="road instruction, clipped to road/lane context"))
    style_legend(plt.legend(handles=legend_items, loc="best", fontsize=8))
    finish_plot_with_extra_points("ClassLab Multi-sequence RoadLane / Road Instruction BEV", all_raw_points, all_lanes, extra_points, args)
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close()


def draw_id_bev(path: Path, raw_points: Sequence[Dict], poses: Dict, lanes: Sequence[Dict], args) -> None:
    setup_matplotlib()
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
    apply_dark_axes(ax)
    if not args.no_raw_points and raw_points:
        plt.scatter(
            [point["x"] for point in raw_points],
            [point["y"] for point in raw_points],
            s=0.10,
            c="#8b949e",
            alpha=0.08,
            linewidths=0,
            zorder=0,
        )
    draw_poses(poses)
    colors = matplotlib.colormaps["tab20"].resampled(max(len(lanes), 1))
    for index, lane in enumerate(lanes):
        points = lane["polyline_world"]
        if len(points) < 2:
            continue
        color = colors(index % 20)
        plt.plot([p[0] for p in points], [p[1] for p in points], "-", linewidth=2.2, color=color, zorder=3)
    finish_plot("Corrected ClassLab RoadLane BEV", raw_points, lanes, args)
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close()


def setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    from matplotlib import font_manager

    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def draw_poses(poses: Dict) -> None:
    import matplotlib.pyplot as plt

    traj_x = [pose[0] for _frame, pose in sorted(poses.items())]
    traj_y = [pose[1] for _frame, pose in sorted(poses.items())]
    plt.plot(traj_x, traj_y, "-", color="#c9d1d9", linewidth=1.0, alpha=0.38, label="corrected ego trajectory", zorder=2)


def draw_lane_semantic(lane: Dict, annotate: bool, linewidth: float = 2.6) -> None:
    import matplotlib.pyplot as plt

    points = lane.get("polyline_world", [])
    if len(points) < 2:
        return
    semantic = lane.get("lane_type_semantic", {})
    color = semantic.get("draw_color", UNKNOWN)
    style = linestyle_for_semantic(semantic)
    draw_points = points
    lane_width = linewidth * (0.82 if lane.get("inferred") else 1.0)

    if semantic.get("is_double"):
        for amount in (-0.18, 0.18):
            offset = offset_polyline(points, amount)
            plot_marking(offset, color, style, lane_width)
    elif semantic.get("is_solid_dashed"):
        solid = offset_polyline(points, -0.16)
        dashed = offset_polyline(points, 0.16)
        plot_marking(solid, color, "-", lane_width)
        plot_marking(dashed, color, (0, (7, 6)), lane_width)
    elif semantic.get("is_fishbone"):
        plot_marking(draw_points, color, style, lane_width + 0.5)
        xs = [point[0] for point in draw_points[:: max(len(draw_points) // 28, 1)]]
        ys = [point[1] for point in draw_points[:: max(len(draw_points) // 28, 1)]]
        plt.scatter(xs, ys, s=8, color=color, edgecolors="#333333", linewidths=0.35, zorder=5)
    else:
        plot_marking(draw_points, color, style, lane_width)

    if annotate:
        mid = len(points) // 2
        label = semantic.get("chinese_name", "未知线型")
        plt.text(
            points[mid][0],
            points[mid][1],
            label,
            color="#111111",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffffff", "edgecolor": "#444444", "alpha": 0.78},
            zorder=6,
        )


def semantic_legend_items(lanes: Sequence[Dict], mlines) -> List:
    seen = {}
    for lane in lanes:
        semantic = lane.get("lane_type_semantic", {})
        code = semantic.get("code")
        if code in seen:
            continue
        seen[code] = semantic

    items = []
    for code, semantic in sorted(seen.items(), key=lambda item: str(item[0])):
        color = semantic.get("draw_color", UNKNOWN)
        linestyle = linestyle_for_semantic(semantic)
        label = semantic.get("chinese_name", "未知线型")
        if color == WHITE:
            items.append(mlines.Line2D([], [], color=WHITE, linestyle=linestyle, linewidth=3.2, label=label))
        else:
            items.append(mlines.Line2D([], [], color=color, linestyle=linestyle, linewidth=3.0, label=label))
    return items


def plot_marking(points: Sequence[Point3], color, linestyle, linewidth: float) -> None:
    import matplotlib.pyplot as plt

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if color == WHITE:
        plt.plot(xs, ys, color="#0b0f14", linestyle=linestyle, linewidth=linewidth + 1.2, alpha=0.95, zorder=3)
    plt.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.96, zorder=4)


def draw_direction_arrows(centerline: Sequence[Point3], color: str) -> None:
    import matplotlib.pyplot as plt

    if len(centerline) < 6:
        return
    step = max(len(centerline) // 6, 8)
    for arrow_index in range(step, len(centerline) - step, step):
        start_index = max(arrow_index - 2, 0)
        end_index = min(arrow_index + 3, len(centerline) - 1)
        start = centerline[start_index]
        end = centerline[end_index]
        plt.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 2.0,
                "mutation_scale": 13,
                "alpha": 0.95,
            },
            zorder=6,
        )


def linestyle_for_semantic(semantic: Dict):
    pattern = semantic.get("pattern", "solid")
    if pattern == "dashed":
        return (0, (7, 6))
    if pattern == "solid_dashed":
        return "-."
    return "-"


def offset_polyline(points: Sequence[Point3], amount: float) -> List[Point3]:
    output = []
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
        length = math.hypot(dx, dy)
        if length < 1e-6:
            output.append([point[0], point[1], 0.0])
            continue
        nx = -dy / length
        ny = dx / length
        output.append([point[0] + nx * amount, point[1] + ny * amount, 0.0])
    return output


def finish_plot(title: str, raw_points: Sequence[Dict], lanes: Sequence[Dict], args) -> None:
    import matplotlib.pyplot as plt

    plt.axis("equal")
    plt.grid(True, linewidth=0.35, alpha=0.35, color="#30363d")
    plt.xlabel("corrected map x / m")
    plt.ylabel("corrected map y / m")
    plt.title(title)
    base.apply_plot_limits(raw_points, lanes, args)
    plt.tight_layout()


def finish_plot_with_extra_points(
    title: str,
    raw_points: Sequence[Dict],
    lanes: Sequence[Dict],
    extra_points: Sequence[Point3],
    args,
) -> None:
    import matplotlib.pyplot as plt

    plt.axis("equal")
    plt.grid(True, linewidth=0.35, alpha=0.35, color="#30363d")
    plt.xlabel("corrected map x / m")
    plt.ylabel("corrected map y / m")
    plt.title(title)

    if args.xlim is not None:
        plt.xlim(args.xlim[0], args.xlim[1])
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    if args.xlim is not None and args.ylim is not None:
        plt.tight_layout()
        return

    xs = [point["x"] for point in raw_points]
    ys = [point["y"] for point in raw_points]
    for lane in lanes:
        for point in lane.get("polyline_world", []):
            xs.append(point[0])
            ys.append(point[1])
    for point in extra_points:
        xs.append(point[0])
        ys.append(point[1])
    if not xs or not ys:
        plt.tight_layout()
        return
    margin = args.plot_margin
    if args.xlim is None:
        plt.xlim(min(xs) - margin, max(xs) + margin)
    if args.ylim is None:
        plt.ylim(min(ys) - margin, max(ys) + margin)
    plt.tight_layout()


def apply_dark_axes(ax) -> None:
    fig = ax.figure
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_color("#6e7681")
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#f0f6fc")


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("#161b22")
    frame.set_edgecolor("#6e7681")
    frame.set_alpha(0.92)
    for text in legend.get_texts():
        text.set_color("#f0f6fc")


if __name__ == "__main__":
    raise SystemExit(main())
