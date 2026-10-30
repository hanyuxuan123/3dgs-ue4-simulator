#!/usr/bin/env python3
"""Build ClassLab semantic roadlanes using mapping_pose as the fixed trajectory.

Unlike track_classlab_roadlanes_bev_semantic.py, this script does not run
lane-matching pose correction.  It projects lane observations with poses from
mapping_pose.txt, expressed in sequence_start_local coordinates using a fixed
origin frame.  This keeps the generated map in the same pose family as
DriveStudio/3DGS processed data.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Sequence

import track_classlab_roadlanes_bev as base
import track_classlab_roadlanes_bev_semantic as semantic


def parse_args():
    origin_frame = 0
    cleaned_argv = [sys.argv[0]]
    index = 1
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg in ("--sequence_origin_frame", "--sequence-origin-frame"):
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{arg} needs a value")
            origin_frame = int(sys.argv[index + 1])
            index += 2
            continue
        cleaned_argv.append(arg)
        index += 1
    original_argv = sys.argv
    try:
        sys.argv = cleaned_argv
        args = base.parse_args()
    finally:
        sys.argv = original_argv
    args.sequence_origin_frame = origin_frame
    return args


def make_mapping_sequence_local_poses(
    poses: Dict[int, Dict[str, float]],
    frames: Sequence[int],
    origin_frame: int,
) -> Dict[int, base.Pose2]:
    if origin_frame not in poses:
        raise ValueError(f"sequence origin frame {origin_frame} is missing from mapping_pose")
    origin = poses[origin_frame]
    x0 = float(origin["x"])
    y0 = float(origin["y"])
    yaw0 = float(origin["yaw"])
    c = math.cos(-yaw0)
    s = math.sin(-yaw0)
    output: Dict[int, base.Pose2] = {}
    for frame in frames:
        pose = poses[frame]
        dx = float(pose["x"]) - x0
        dy = float(pose["y"]) - y0
        output[frame] = (
            c * dx - s * dy,
            s * dx + c * dy,
            base.normalize_angle(float(pose["yaw"]) - yaw0),
        )
    return output


def make_sequence_local_to_enu_transform(poses: Dict[int, Dict[str, float]], origin_frame: int) -> Dict:
    if origin_frame not in poses:
        raise ValueError(f"sequence origin frame {origin_frame} is missing from mapping_pose")
    origin = poses[origin_frame]
    yaw0 = float(origin["yaw"])
    return {
        "type": "sequence_start_local_to_enu_from_mapping_pose",
        "origin_frame": int(origin_frame),
        "origin_enu_x": float(origin["x"]),
        "origin_enu_y": float(origin["y"]),
        "origin_yaw": yaw0,
        "cos_yaw": math.cos(yaw0),
        "sin_yaw": math.sin(yaw0),
        "formula": "enu_xy = origin_xy + R(origin_yaw) * sequence_start_local_xy",
    }


def write_pose_csv(path: Path, frames: Sequence[int], poses: Dict[int, base.Pose2], debug_rows: Sequence[Dict]) -> None:
    debug_by_frame = {int(row["frame"]): row for row in debug_rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["frame", "corrected_x", "corrected_y", "corrected_yaw", "match_status", "match_count", "match_score"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            pose = poses[frame]
            debug = debug_by_frame.get(frame, {})
            writer.writerow(
                {
                    "frame": frame,
                    "corrected_x": pose[0],
                    "corrected_y": pose[1],
                    "corrected_yaw": pose[2],
                    "match_status": debug.get("status", "mapping_pose"),
                    "match_count": debug.get("matches", 0),
                    "match_score": debug.get("score", ""),
                }
            )


def write_pose_enu_csv(path: Path, frames: Sequence[int], poses: Dict[int, base.Pose2], debug_rows: Sequence[Dict]) -> None:
    debug_by_frame = {int(row["frame"]): row for row in debug_rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["frame", "enu_x", "enu_y", "enu_yaw", "match_status", "match_count", "match_score"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for frame in frames:
            pose = poses[frame]
            debug = debug_by_frame.get(frame, {})
            writer.writerow(
                {
                    "frame": frame,
                    "enu_x": pose[0],
                    "enu_y": pose[1],
                    "enu_yaw": pose[2],
                    "match_status": debug.get("status", "mapping_pose"),
                    "match_count": debug.get("matches", 0),
                    "match_score": debug.get("score", ""),
                }
            )


def process_sequence_mapping_pose(args, sequence_dir: Path, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    lane_csv = Path(args.lane_csv) if args.lane_csv else sequence_dir / "result" / "lane.csv"
    pose_file = Path(args.pose_file) if args.pose_file else sequence_dir / "result" / "mapping" / "mapping_pose.txt"

    poses = base.read_mapping_poses(pose_file)
    frames = base.select_frames(sorted(poses), args.start_frame, args.end_frame, args.frame_step)
    frame_lanes = base.read_lane_csv(lane_csv, set(args.use_lane_names), frames, args)
    sequence_local_poses = make_mapping_sequence_local_poses(poses, frames, args.sequence_origin_frame)
    debug_rows = [
        {"frame": frame, "status": "mapping_pose", "matches": 0, "score": ""}
        for frame in frames
    ]

    observations, raw_points = base.build_corrected_lane_observations(frames, sequence_local_poses, frame_lanes, args)
    roadlanes = base.track_global_lane_observations(observations, args)
    road_edge_csv = sequence_dir / "result" / "debug" / "roadEdge" / "road_edge.csv"
    road_edges = semantic.build_corrected_road_edges(road_edge_csv, frames, sequence_local_poses, args) if road_edge_csv.exists() else []

    for lane_id, lane in enumerate(roadlanes):
        lane["global_lane_id"] = lane_id
        lane["inferred"] = False
        lane["dominant_lane_type"] = semantic.dominant_lane_type(lane.get("lane_type_hist", {}))
        lane["lane_type_semantic"] = semantic.decode_lane_type(lane["dominant_lane_type"])
        lane["is_direction_divider"] = lane["lane_type_semantic"]["color"] == "yellow"

    inferred_lanes = semantic.infer_missing_lane_markings(roadlanes, road_edges)
    map_lanes = list(roadlanes) + inferred_lanes
    road_fills = semantic.build_road_fills(map_lanes)
    road_connectors = semantic.build_road_connectors(road_fills)

    local_to_enu = make_sequence_local_to_enu_transform(poses, args.sequence_origin_frame)
    enu_map_lanes = semantic.transform_lanes_to_enu(map_lanes, local_to_enu)
    enu_roadlanes = enu_map_lanes[: len(roadlanes)]
    enu_inferred_lanes = enu_map_lanes[len(roadlanes) :]
    enu_road_fills = semantic.transform_fills_to_enu(road_fills, local_to_enu)
    enu_road_connectors = semantic.transform_connectors_to_enu(road_connectors, local_to_enu)
    enu_mapping_poses = semantic.transform_poses_to_enu(sequence_local_poses, local_to_enu)
    enu_raw_points = semantic.transform_point_records_to_enu(raw_points, local_to_enu)

    fill_road_pcd = sequence_dir / "result" / "debug" / "roadEdge" / "fill_road.pcd"
    raw_road_polygon_points = semantic.read_ascii_pcd_xyz(fill_road_pcd) if fill_road_pcd.exists() else []
    raw_enu_poses = {frame: (poses[frame]["x"], poses[frame]["y"], poses[frame]["yaw"]) for frame in frames}
    road_polygon_points = semantic.align_road_polygon_to_corrected_poses(
        raw_road_polygon_points,
        raw_enu_poses,
        enu_mapping_poses,
        frames,
    )
    road_instruction_regions = semantic.build_road_instruction_regions(
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
        "pose_source": "mapping_pose_no_lane_matching_correction",
        "sequence_origin_frame": int(args.sequence_origin_frame),
        "local_to_enu": local_to_enu,
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
    local_metadata["coordinate_frame"] = "sequence_start_local"

    write_pose_csv(output_dir / "corrected_pose.csv", frames, sequence_local_poses, debug_rows)
    write_pose_enu_csv(output_dir / "corrected_pose_enu.csv", frames, enu_mapping_poses, debug_rows)
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

    semantic.draw_id_bev(output_dir / "roadlane_id_bev.png", enu_raw_points, enu_mapping_poses, enu_map_lanes, args)
    semantic.draw_semantic_bev(output_dir / "roadlane_bev.png", enu_raw_points, enu_mapping_poses, enu_map_lanes, args)
    semantic.draw_road_fill_bev(
        output_dir / "road_fill_bev.png",
        enu_raw_points,
        enu_mapping_poses,
        enu_map_lanes,
        enu_road_fills,
        args,
        enu_road_connectors,
    )
    semantic.draw_road_fill_bev(
        output_dir / "road_direction_bev.png",
        enu_raw_points,
        enu_mapping_poses,
        enu_map_lanes,
        enu_road_fills,
        args,
        enu_road_connectors,
    )
    semantic.draw_road_instruction_bev(
        output_dir / "road_instruction_bev.png",
        road_polygon_points,
        road_instruction_regions,
        enu_raw_points,
        enu_mapping_poses,
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
    print(f"pose source: mapping_pose, no lane-matching correction")
    print(f"sequence origin frame: {args.sequence_origin_frame}")
    print(f"wrote: {output_dir / 'road_network_info_local.json'}")
    print(f"wrote: {output_dir / 'road_fill_bev.png'}")
    print(f"wrote: {output_dir / 'road_direction_bev.png'}")
    return {
        "sequence_dir": sequence_dir,
        "output_dir": output_dir,
        "metadata": metadata,
        "lanes": enu_map_lanes,
        "road_fills": enu_road_fills,
        "road_connectors": enu_road_connectors,
        "poses": enu_mapping_poses,
        "raw_points": enu_raw_points,
    }


def main() -> int:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)
    process_sequence_mapping_pose(args, sequence_dir, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
