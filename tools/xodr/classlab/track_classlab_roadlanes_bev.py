#!/usr/bin/env python3
"""Fuse ClassLab lane.csv into a BEV road-lane map with local lane tracking.

This script treats mapping_pose.txt as a weak prior.  Each frame starts from the
relative GNSS/INS pose, then is locally corrected by matching the current
ego-frame lane observations against a recent lane map.  The final lane tracks
are generated from the corrected poses, not directly from raw GNSS projection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


Point2 = Tuple[float, float]
Pose2 = Tuple[float, float, float]


LANE_NAMES = (
    "leftHost",
    "rightHost",
    "leftNeighbor",
    "rightNeighbor",
    "leftThird",
    "rightThird",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a corrected BEV road-lane map from ClassLab lane.csv."
    )
    parser.add_argument("--sequence_dir", "--sequence-dir", required=True, help="Raw ClassLab sequence directory.")
    parser.add_argument("--output_dir", "--output-dir", required=True, help="Directory for BEV outputs.")
    parser.add_argument("--lane_csv", "--lane-csv", default=None, help="Defaults to <sequence_dir>/result/lane.csv.")
    parser.add_argument(
        "--pose_file",
        "--pose-file",
        default=None,
        help="Defaults to <sequence_dir>/result/mapping/mapping_pose.txt.",
    )
    parser.add_argument("--start_frame", "--start-frame", type=int, default=None)
    parser.add_argument("--end_frame", "--end-frame", type=int, default=None)
    parser.add_argument("--frame_step", "--frame-step", type=int, default=1)

    parser.add_argument("--sample_step", "--sample-step", type=float, default=1.0)
    parser.add_argument("--track_sample_step", "--track-sample-step", type=float, default=2.0)
    parser.add_argument("--min_forward", "--min-forward", type=float, default=0.0)
    parser.add_argument("--max_forward", "--max-forward", type=float, default=70.0)
    parser.add_argument(
        "--use_lane_names",
        "--use-lane-names",
        nargs="*",
        default=list(LANE_NAMES),
        help="Lane groups used from lane.csv. Defaults to common road-lane fields.",
    )

    parser.add_argument(
        "--match_radius",
        "--match-radius",
        type=float,
        default=1.25,
        help="Maximum point-to-map distance accepted by lane scan matching.",
    )
    parser.add_argument(
        "--min_match_points",
        "--min-match-points",
        type=int,
        default=60,
        help="Minimum accepted correspondences before applying a pose correction.",
    )
    parser.add_argument(
        "--max_match_points",
        "--max-match-points",
        type=int,
        default=160,
        help="Deterministically downsample each frame to this many points for pose matching. Full points are still used for final BEV fusion.",
    )
    parser.add_argument(
        "--map_window_frames",
        "--map-window-frames",
        type=int,
        default=90,
        help="Number of recent corrected frames kept in the local matching map.",
    )
    parser.add_argument(
        "--search_xy",
        "--search-xy",
        type=float,
        default=1.5,
        help="Initial +/- xy correction search radius in meters.",
    )
    parser.add_argument(
        "--search_yaw_deg",
        "--search-yaw-deg",
        type=float,
        default=2.0,
        help="Initial +/- yaw correction search radius in degrees.",
    )
    parser.add_argument(
        "--search_levels",
        "--search-levels",
        type=int,
        default=3,
        help="Coarse-to-fine local search levels.",
    )
    parser.add_argument(
        "--prior_weight",
        "--prior-weight",
        type=float,
        default=0.05,
        help="Weak cost that keeps corrections near the pose prior.",
    )
    parser.add_argument(
        "--max_correction_xy",
        "--max-correction-xy",
        type=float,
        default=4.0,
        help="Reject local matching correction larger than this value.",
    )
    parser.add_argument(
        "--max_correction_yaw_deg",
        "--max-correction-yaw-deg",
        type=float,
        default=6.0,
        help="Reject local matching yaw correction larger than this value.",
    )

    parser.add_argument("--s_bin", "--s-bin", type=float, default=1.0)
    parser.add_argument("--lateral_merge", "--lateral-merge", type=float, default=0.8)
    parser.add_argument("--lateral_link", "--lateral-link", type=float, default=1.2)
    parser.add_argument("--max_gap_bins", "--max-gap-bins", type=int, default=8)
    parser.add_argument("--min_track_nodes", "--min-track-nodes", type=int, default=35)
    parser.add_argument("--min_observations", "--min-observations", type=int, default=400)
    parser.add_argument("--smooth_window", "--smooth-window", type=int, default=9)
    parser.add_argument(
        "--global_track_radius",
        "--global-track-radius",
        type=float,
        default=1.15,
        help="Maximum world-space distance for associating a lane observation to an existing global roadlane.",
    )
    parser.add_argument(
        "--global_min_match_ratio",
        "--global-min-match-ratio",
        type=float,
        default=0.25,
        help="Minimum ratio of sampled observation points that must match a global roadlane track.",
    )
    parser.add_argument(
        "--global_min_match_points",
        "--global-min-match-points",
        type=int,
        default=8,
        help="Minimum number of matched sampled points for global roadlane association.",
    )
    parser.add_argument(
        "--global_max_heading_deg",
        "--global-max-heading-deg",
        type=float,
        default=35.0,
        help="Maximum heading difference for global roadlane association.",
    )
    parser.add_argument(
        "--global_max_frame_gap",
        "--global-max-frame-gap",
        type=int,
        default=120,
        help="Maximum frame gap for continuing an inactive global roadlane track.",
    )
    parser.add_argument(
        "--global_recent_points",
        "--global-recent-points",
        type=int,
        default=900,
        help="Recent points per roadlane track used for association.",
    )
    parser.add_argument(
        "--global_bbox_margin",
        "--global-bbox-margin",
        type=float,
        default=5.0,
        help="Fast bbox prefilter margin in meters before detailed global roadlane association.",
    )

    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--fig_width", "--fig-width", type=float, default=30.0)
    parser.add_argument("--fig_height", "--fig-height", type=float, default=24.0)
    parser.add_argument("--plot_margin", "--plot-margin", type=float, default=40.0)
    parser.add_argument("--no_raw_points", "--no-raw-points", action="store_true")
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--progress_interval", "--progress-interval", type=int, default=200)
    parser.add_argument(
        "--multi_sequence_dirs",
        "--multi-sequence-dirs",
        nargs="*",
        action="append",
        default=[],
        help="Additional ClassLab sequence directories to process and overlay with --sequence_dir.",
    )
    parser.add_argument(
        "--multi_max_sequences",
        "--multi-max-sequences",
        type=int,
        default=0,
        help="Limit number of sequences in multi-sequence mode. 0 means use all provided sequences.",
    )
    parser.add_argument(
        "--multi_base_index",
        "--multi-base-index",
        type=int,
        default=0,
        help="Sequence index used as the visual/reference base in multi-sequence metadata.",
    )
    parser.add_argument(
        "--multi_alignment",
        "--multi-alignment",
        choices=("gnss", "endpoint_chain"),
        default="endpoint_chain",
        help=(
            "Multi-sequence alignment strategy. 'gnss' applies only RTK origin translation; "
            "'endpoint_chain' uses the GNSS result as an initial placement, then rigidly "
            "closes chronological sequence endpoints around --multi-base-index."
        ),
    )
    parser.add_argument(
        "--multi_cross_connector_max_gap",
        "--multi-cross-connector-max-gap",
        type=float,
        default=95.0,
        help="Maximum endpoint gap in meters for cross-sequence road fill connectors.",
    )
    parser.add_argument(
        "--multi_cross_lane_max_gap",
        "--multi-cross-lane-max-gap",
        type=float,
        default=70.0,
        help="Maximum endpoint gap in meters for cross-sequence roadlane marking links.",
    )
    parser.add_argument(
        "--multi_cross_lane_lateral_gap",
        "--multi-cross-lane-lateral-gap",
        type=float,
        default=4.0,
        help="Maximum lateral endpoint mismatch in meters for cross-sequence roadlane marking links.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lane_csv = Path(args.lane_csv) if args.lane_csv else sequence_dir / "result" / "lane.csv"
    pose_file = Path(args.pose_file) if args.pose_file else sequence_dir / "result" / "mapping" / "mapping_pose.txt"

    poses = read_mapping_poses(pose_file)
    frames = select_frames(sorted(poses), args.start_frame, args.end_frame, args.frame_step)
    frame_lanes = read_lane_csv(lane_csv, set(args.use_lane_names), frames, args)
    initial_poses = make_initial_relative_poses(poses, frames)

    corrected, debug_rows = estimate_corrected_poses(frames, initial_poses, frame_lanes, args)
    observations, raw_points = build_corrected_lane_observations(frames, corrected, frame_lanes, args)
    roadlanes = track_global_lane_observations(observations, args)

    for lane_id, lane in enumerate(roadlanes):
        lane["global_lane_id"] = lane_id

    metadata = {
        "sequence_dir": str(sequence_dir),
        "lane_csv": str(lane_csv),
        "pose_file": str(pose_file),
        "frame_count": len(frames),
        "raw_lane_points": len(raw_points),
        "lane_observation_count": len(observations),
        "roadlane_count": len(roadlanes),
        "args": vars(args),
    }

    write_pose_corrections(output_dir / "corrected_pose.csv", frames, initial_poses, corrected, debug_rows)
    write_json(output_dir / "roadlane_info.json", {"metadata": metadata, "lanes": roadlanes})
    draw_bev(output_dir / "roadlane_bev.png", raw_points, corrected, roadlanes, args)

    print(f"frames: {len(frames)}")
    print(f"raw lane points: {len(raw_points)}")
    print(f"lane observations: {len(observations)}")
    print(f"roadlanes: {len(roadlanes)}")
    print(f"wrote: {output_dir / 'roadlane_info.json'}")
    print(f"wrote: {output_dir / 'roadlane_bev.png'}")
    print(f"wrote: {output_dir / 'corrected_pose.csv'}")
    return 0


def read_mapping_poses(path: Path) -> Dict[int, Dict[str, float]]:
    poses: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fp:
        header = fp.readline().strip().split()
        index = {name: idx for idx, name in enumerate(header)}
        required = ["lidar_frame", "x", "y", "z", "roll", "pitch", "yaw", "lidar_frame_time"]
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
                "timestamp_sec": float(parts[index["lidar_frame_time"]]),
            }
    if not poses:
        raise ValueError(f"No mapping poses in {path}")
    return poses


def select_frames(
    all_frames: Sequence[int],
    start_frame: Optional[int],
    end_frame: Optional[int],
    frame_step: int,
) -> List[int]:
    start = all_frames[0] if start_frame is None else start_frame
    end = all_frames[-1] if end_frame is None else end_frame
    step = max(frame_step, 1)
    return [frame for frame in all_frames if start <= frame <= end and (frame - start) % step == 0]


def make_initial_relative_poses(poses: Dict[int, Dict[str, float]], frames: Sequence[int]) -> Dict[int, Pose2]:
    first = poses[frames[0]]
    x0, y0, yaw0 = first["x"], first["y"], first["yaw"]
    cos0 = math.cos(-yaw0)
    sin0 = math.sin(-yaw0)
    output: Dict[int, Pose2] = {}
    for frame in frames:
        pose = poses[frame]
        dx = pose["x"] - x0
        dy = pose["y"] - y0
        x = cos0 * dx - sin0 * dy
        y = sin0 * dx + cos0 * dy
        yaw = normalize_angle(pose["yaw"] - yaw0)
        output[frame] = (x, y, yaw)
    return output


def read_lane_csv(
    path: Path,
    allowed_names: set,
    selected_frames: Sequence[int],
    args: argparse.Namespace,
) -> Dict[int, List[Dict]]:
    selected = set(selected_frames)
    by_frame: Dict[int, List[Dict]] = {frame: [] for frame in selected_frames}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        prefixes = [name[: -len(".isValid")] for name in fieldnames if name.endswith(".isValid")]
        prefixes = [prefix for prefix in prefixes if prefix in allowed_names]
        if not prefixes:
            raise ValueError(f"No allowed lane groups found in {path}")
        for row in reader:
            try:
                frame = int(row["frameNum"])
            except (KeyError, ValueError):
                continue
            if frame not in selected:
                continue
            for prefix in prefixes:
                lane = parse_lane(row, prefix, args)
                if lane is not None:
                    by_frame[frame].append(lane)
    return by_frame


def parse_lane(row: Dict[str, str], prefix: str, args: argparse.Namespace) -> Optional[Dict]:
    if not safe_bool(row.get(f"{prefix}.isValid", "false")):
        return None
    try:
        fit_far_flag = safe_bool(row[f"{prefix}.fitFarFlag"])
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
        lane_width = safe_float(row.get(f"{prefix}.laneWidth"), 0.0)
        lane_type = int(safe_float(row.get(f"{prefix}.laneType"), 0.0))
    except (KeyError, ValueError):
        return None

    start = max(range_n, args.min_forward)
    end = min(range_p, args.max_forward)
    if end <= start:
        return None
    points = sample_lane_points(coeff, fit_far_flag, start, end, args.sample_step)
    track_points = sample_lane_points(coeff, fit_far_flag, start, end, args.track_sample_step)
    if len(points) < 2 or len(track_points) < 2:
        return None
    headings = lane_headings(points)
    return {
        "name": prefix,
        "lane_type": lane_type,
        "lane_width": lane_width,
        "points_vehicle": points,
        "track_points_vehicle": track_points,
        "headings_vehicle": headings,
    }


def sample_lane_points(
    coeff: Tuple[float, float, float, float, float, float],
    fit_far_flag: bool,
    start: float,
    end: float,
    sample_step: float,
) -> List[Point2]:
    count = max(int(math.ceil((end - start) / max(sample_step, 1e-3))) + 1, 2)
    points = []
    for index in range(count):
        ratio = index / max(count - 1, 1)
        x = start + ratio * (end - start)
        y = eval_lane_y(x, coeff, fit_far_flag)
        if math.isfinite(x) and math.isfinite(y) and abs(y) < 30.0:
            points.append((x, y))
    return points


def eval_lane_y(x: float, coeff: Tuple[float, float, float, float, float, float], fit_far_flag: bool) -> float:
    a, b, c, d_near, d_far, dividing_point = coeff
    if (not fit_far_flag) or x < dividing_point:
        return a + b * x + c * x * x + d_near * x * x * x
    t = dividing_point
    return a + b * x + c * x * x + d_near * (t * t * t + 3.0 * x * (x - t) * t) + d_far * (x - t) ** 3


def lane_headings(points: Sequence[Point2]) -> List[float]:
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


def estimate_corrected_poses(
    frames: Sequence[int],
    initial_poses: Dict[int, Pose2],
    frame_lanes: Dict[int, List[Dict]],
    args: argparse.Namespace,
) -> Tuple[Dict[int, Pose2], Dict[int, Dict]]:
    corrected: Dict[int, Pose2] = {}
    debug: Dict[int, Dict] = {}
    local_map: Deque[Tuple[int, List[Point2]]] = deque()
    local_points: List[Point2] = []

    prev_frame: Optional[int] = None
    for frame_index, frame in enumerate(frames):
        current_points = tracking_points_for_frame(frame_lanes.get(frame, []))
        current_points = downsample_points(current_points, args.max_match_points)
        if prev_frame is None:
            prior = initial_poses[frame]
        else:
            rel = relative_pose(initial_poses[prev_frame], initial_poses[frame])
            prior = compose_pose(corrected[prev_frame], rel)

        if len(local_points) < args.min_match_points or len(current_points) < 10:
            best_pose = prior
            info = {"status": "prior_only", "matches": 0, "score": None}
        else:
            grid = SpatialHash(local_points, cell_size=args.match_radius)
            best_pose, info = refine_pose_by_lane_matching(prior, current_points, grid, args)
            dx = best_pose[0] - prior[0]
            dy = best_pose[1] - prior[1]
            dyaw = angle_diff(best_pose[2], prior[2])
            if math.hypot(dx, dy) > args.max_correction_xy or abs(dyaw) > math.radians(args.max_correction_yaw_deg):
                best_pose = prior
                info["status"] = "rejected_large_correction"

        corrected[frame] = best_pose
        debug[frame] = info

        transformed = [transform_point_2d(best_pose, point) for point in current_points]
        local_map.append((frame, transformed))
        while len(local_map) > args.map_window_frames:
            local_map.popleft()
        local_points = [point for _frame, points in local_map for point in points]
        prev_frame = frame
        if args.progress_interval > 0 and (frame_index + 1) % args.progress_interval == 0:
            print(
                f"matched {frame_index + 1}/{len(frames)} frames, "
                f"frame={frame}, status={info.get('status')}, matches={info.get('matches')}"
            )

    return corrected, debug


def tracking_points_for_frame(lanes: Sequence[Dict]) -> List[Point2]:
    points: List[Point2] = []
    for lane in lanes:
        points.extend(lane["track_points_vehicle"])
    return points


def downsample_points(points: Sequence[Point2], max_points: int) -> List[Point2]:
    if max_points <= 0 or len(points) <= max_points:
        return list(points)
    if max_points == 1:
        return [points[len(points) // 2]]
    step = (len(points) - 1) / float(max_points - 1)
    return [points[int(round(index * step))] for index in range(max_points)]


def refine_pose_by_lane_matching(
    prior: Pose2,
    current_points: Sequence[Point2],
    grid: "SpatialHash",
    args: argparse.Namespace,
) -> Tuple[Pose2, Dict]:
    best = prior
    best_score, best_matches = score_pose(best, prior, current_points, grid, args)
    xy_step = args.search_xy
    yaw_step = math.radians(args.search_yaw_deg)

    for _level in range(max(args.search_levels, 1)):
        candidates = []
        for dx in (-xy_step, 0.0, xy_step):
            for dy in (-xy_step, 0.0, xy_step):
                for dyaw in (-yaw_step, 0.0, yaw_step):
                    candidates.append((best[0] + dx, best[1] + dy, normalize_angle(best[2] + dyaw)))
        for candidate in candidates:
            score, matches = score_pose(candidate, prior, current_points, grid, args)
            if matches >= args.min_match_points and score < best_score:
                best = candidate
                best_score = score
                best_matches = matches
        xy_step *= 0.5
        yaw_step *= 0.5

    status = "matched" if best_matches >= args.min_match_points else "weak_match_prior"
    if best_matches < args.min_match_points:
        return prior, {"status": status, "matches": best_matches, "score": best_score}
    return best, {"status": status, "matches": best_matches, "score": best_score}


def score_pose(
    pose: Pose2,
    prior: Pose2,
    points_vehicle: Sequence[Point2],
    grid: "SpatialHash",
    args: argparse.Namespace,
) -> Tuple[float, int]:
    distances = []
    for point in points_vehicle:
        world_point = transform_point_2d(pose, point)
        nearest_d2 = grid.nearest_distance_sq(world_point, max_distance=args.match_radius)
        if nearest_d2 is not None:
            distances.append(nearest_d2)
    if not distances:
        return float("inf"), 0
    distances.sort()
    keep = max(int(len(distances) * 0.8), 1)
    robust_mean = sum(distances[:keep]) / keep
    dx = pose[0] - prior[0]
    dy = pose[1] - prior[1]
    dyaw = angle_diff(pose[2], prior[2])
    prior_cost = args.prior_weight * (dx * dx + dy * dy + 4.0 * dyaw * dyaw)
    return robust_mean + prior_cost, len(distances)


class SpatialHash:
    def __init__(self, points: Sequence[Point2], cell_size: float):
        self.cell_size = max(cell_size, 1e-3)
        self.cells: Dict[Tuple[int, int], List[Point2]] = defaultdict(list)
        for point in points:
            self.cells[self.key(point)].append(point)

    def key(self, point: Point2) -> Tuple[int, int]:
        return (int(math.floor(point[0] / self.cell_size)), int(math.floor(point[1] / self.cell_size)))

    def nearest_distance_sq(self, point: Point2, max_distance: float) -> Optional[float]:
        key_x, key_y = self.key(point)
        max_d2 = max_distance * max_distance
        best = max_d2
        found = False
        for ix in range(key_x - 1, key_x + 2):
            for iy in range(key_y - 1, key_y + 2):
                for other in self.cells.get((ix, iy), []):
                    dx = point[0] - other[0]
                    dy = point[1] - other[1]
                    d2 = dx * dx + dy * dy
                    if d2 < best:
                        best = d2
                        found = True
        return best if found else None


def build_corrected_lane_observations(
    frames: Sequence[int],
    corrected_poses: Dict[int, Pose2],
    frame_lanes: Dict[int, List[Dict]],
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict]]:
    pose_s = build_pose_s(frames, corrected_poses)
    observations = []
    raw_points = []
    for frame in frames:
        pose = corrected_poses[frame]
        for lane in frame_lanes.get(frame, []):
            points = lane["points_vehicle"]
            headings = lane["headings_vehicle"]
            observation_points = []
            for point, heading_vehicle in zip(points, headings):
                world = transform_point_2d(pose, point)
                point_record = (
                    {
                        "frame": frame,
                        "name": lane["name"],
                        "lane_type": lane["lane_type"],
                        "lane_width": lane["lane_width"],
                        "ego_x": point[0],
                        "ego_y": point[1],
                        "s": pose_s[frame] + point[0],
                        "n": point[1],
                        "x": world[0],
                        "y": world[1],
                        "heading": normalize_angle(pose[2] + heading_vehicle),
                    }
                )
                observation_points.append(point_record)
                raw_points.append(point_record)
            if len(observation_points) < 2:
                continue
            observations.append(
                {
                    "frame": frame,
                    "name": lane["name"],
                    "lane_type": lane["lane_type"],
                    "lane_width": lane["lane_width"],
                    "points": observation_points,
                    "heading": circular_mean([point["heading"] for point in observation_points]),
                    "bbox": point_record_bbox(observation_points),
                    "s_range": [
                        min(point["s"] for point in observation_points),
                        max(point["s"] for point in observation_points),
                    ],
                }
            )
    return observations, raw_points


def track_global_lane_observations(observations: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    tracks: List[Dict] = []
    observations_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    for observation in observations:
        observations_by_frame[observation["frame"]].append(observation)

    next_track_id = 0
    for frame in sorted(observations_by_frame):
        frame_observations = observations_by_frame[frame]
        candidate_pairs = []
        for obs_index, observation in enumerate(frame_observations):
            for track_index, track in enumerate(tracks):
                if frame - track["last_frame"] > args.global_max_frame_gap:
                    continue
                score = score_observation_to_track(observation, track, args)
                if score is not None:
                    candidate_pairs.append((score["cost"], obs_index, track_index, score))

        used_observations = set()
        used_tracks = set()
        for _cost, obs_index, track_index, score in sorted(candidate_pairs):
            if obs_index in used_observations or track_index in used_tracks:
                continue
            add_observation_to_track(tracks[track_index], frame_observations[obs_index], score)
            used_observations.add(obs_index)
            used_tracks.add(track_index)

        for obs_index, observation in enumerate(frame_observations):
            if obs_index in used_observations:
                continue
            tracks.append(make_global_track(next_track_id, observation, args))
            next_track_id += 1

    lanes = []
    for track in tracks:
        lane = fused_lane_from_global_track(track, args)
        if keep_lane(lane, args):
            lane["global_lane_id"] = len(lanes)
            lanes.append(lane)
    return lanes


def make_global_track(track_id: int, observation: Dict, args: argparse.Namespace) -> Dict:
    track = {
        "track_id": track_id,
        "points": [],
        "recent_points": [],
        "last_frame": observation["frame"],
        "frame_count": 0,
        "names": Counter(),
        "lane_types": Counter(),
        "lane_widths": [],
        "heading": observation["heading"],
        "grid": None,
        "bbox": None,
        "max_recent": args.global_recent_points,
    }
    add_observation_to_track(track, observation, score=None)
    return track


def add_observation_to_track(track: Dict, observation: Dict, score: Optional[Dict]):
    track["points"].extend(observation["points"])
    track["recent_points"].extend(observation["points"])
    max_recent = max(int(track.get("max_recent", 0)), 0)
    if max_recent > 0 and len(track["recent_points"]) > max_recent:
        track["recent_points"] = track["recent_points"][-max_recent:]
    track["bbox"] = point_record_bbox(track["recent_points"])
    track["last_frame"] = observation["frame"]
    track["frame_count"] += 1
    track["names"].update([observation["name"]])
    track["lane_types"].update([str(observation["lane_type"])])
    if observation["lane_width"] > 0.0:
        track["lane_widths"].append(observation["lane_width"])
    old_weight = max(track["frame_count"] - 1, 1)
    track["heading"] = circular_mean([track["heading"]] * old_weight + [observation["heading"]])
    track["grid"] = None
    if score is not None:
        track.setdefault("association_scores", []).append(score)


def score_observation_to_track(observation: Dict, track: Dict, args: argparse.Namespace) -> Optional[Dict]:
    d_heading = abs(angle_diff(observation["heading"], track["heading"]))
    if d_heading > math.radians(args.global_max_heading_deg):
        return None
    if track.get("bbox") is not None:
        if bbox_distance(observation["bbox"], track["bbox"]) > args.global_bbox_margin:
            return None

    recent_points = track["recent_points"][-args.global_recent_points :]
    if not recent_points:
        return None
    if track.get("grid") is None:
        track["grid"] = SpatialHash(
            [(point["x"], point["y"]) for point in recent_points],
            cell_size=args.global_track_radius,
        )
    grid = track["grid"]

    sampled_points = downsample_point_records(observation["points"], max_points=60)
    distances = []
    for point in sampled_points:
        d2 = grid.nearest_distance_sq((point["x"], point["y"]), max_distance=args.global_track_radius)
        if d2 is not None:
            distances.append(math.sqrt(d2))

    match_count = len(distances)
    match_ratio = match_count / max(len(sampled_points), 1)
    if match_count < args.global_min_match_points or match_ratio < args.global_min_match_ratio:
        return None

    distances.sort()
    keep = max(int(len(distances) * 0.8), 1)
    robust_distance = sum(distances[:keep]) / keep
    frame_gap = max(observation["frame"] - track["last_frame"], 0)
    cost = robust_distance + 0.45 * d_heading + 0.002 * frame_gap
    return {
        "cost": cost,
        "distance": robust_distance,
        "heading_diff": d_heading,
        "match_count": match_count,
        "match_ratio": match_ratio,
    }


def point_record_bbox(points: Sequence[Dict]) -> Tuple[float, float, float, float]:
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_distance(left: Tuple[float, float, float, float], right: Tuple[float, float, float, float]) -> float:
    left_min_x, left_min_y, left_max_x, left_max_y = left
    right_min_x, right_min_y, right_max_x, right_max_y = right
    dx = max(right_min_x - left_max_x, left_min_x - right_max_x, 0.0)
    dy = max(right_min_y - left_max_y, left_min_y - right_max_y, 0.0)
    return math.hypot(dx, dy)


def downsample_point_records(points: Sequence[Dict], max_points: int) -> List[Dict]:
    if max_points <= 0 or len(points) <= max_points:
        return list(points)
    if max_points == 1:
        return [points[len(points) // 2]]
    step = (len(points) - 1) / float(max_points - 1)
    return [points[int(round(index * step))] for index in range(max_points)]


def fused_lane_from_global_track(track: Dict, args: argparse.Namespace) -> Dict:
    by_s_bin: Dict[int, List[Dict]] = defaultdict(list)
    for point in track["points"]:
        s_bin = int(round(point["s"] / args.s_bin))
        by_s_bin[s_bin].append(point)

    nodes = []
    for s_bin in sorted(by_s_bin):
        points = by_s_bin[s_bin]
        nodes.append(
            {
                "s_bin": s_bin,
                "s": median([point["s"] for point in points]),
                "n": median([point["ego_y"] for point in points]),
                "x": median([point["x"] for point in points]),
                "y": median([point["y"] for point in points]),
                "heading": circular_mean([point["heading"] for point in points]),
                "count": len(points),
            }
        )

    nodes = smooth_track_nodes(nodes, args.smooth_window)
    s_values = [node["s"] for node in nodes]
    n_values = [node["n"] for node in nodes]
    widths = track["lane_widths"]
    return {
        "global_lane_id": -1,
        "source_track_id": track["track_id"],
        "polyline_world": [[node["x"], node["y"], 0.0] for node in nodes],
        "polyline_sn": [[node["s"], node["n"], 0.0] for node in nodes],
        "point_count": len(nodes),
        "observation_count": sum(node["count"] for node in nodes),
        "frame_count": track["frame_count"],
        "s_range": [min(s_values), max(s_values)] if s_values else [0.0, 0.0],
        "n_median": median(n_values),
        "n_range": [min(n_values), max(n_values)] if n_values else [0.0, 0.0],
        "mean_lane_width": sum(widths) / len(widths) if widths else None,
        "name_hist": dict(track["names"].most_common()),
        "lane_type_hist": dict(track["lane_types"].most_common()),
    }


def build_pose_s(frames: Sequence[int], poses: Dict[int, Pose2]) -> Dict[int, float]:
    output: Dict[int, float] = {}
    total = 0.0
    prev: Optional[Pose2] = None
    for frame in frames:
        pose = poses[frame]
        if prev is not None:
            total += math.hypot(pose[0] - prev[0], pose[1] - prev[1])
        output[frame] = total
        prev = pose
    return output


def build_lane_nodes(raw_points: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    by_s_bin: Dict[int, List[Dict]] = defaultdict(list)
    for point in raw_points:
        s_bin = int(round(point["s"] / args.s_bin))
        by_s_bin[s_bin].append(point)

    nodes = []
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
        "heading": circular_mean([point["heading"] for point in points]),
        "count": len(points),
        "names": Counter(point["name"] for point in points),
        "lane_types": Counter(str(point["lane_type"]) for point in points),
        "lane_widths": [point["lane_width"] for point in points if point["lane_width"] > 0.0],
        "frames": Counter(str(point["frame"]) for point in points),
    }


def link_lane_nodes(nodes: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
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
    nodes = smooth_track_nodes(sorted(track_nodes, key=lambda item: item["s"]), args.smooth_window)
    names = Counter()
    lane_types = Counter()
    frames = Counter()
    widths = []
    observation_count = 0
    for node in nodes:
        names.update(node["names"])
        lane_types.update(node["lane_types"])
        frames.update(node["frames"])
        widths.extend(node["lane_widths"])
        observation_count += node["count"]
    s_values = [node["s"] for node in nodes]
    n_values = [node["n"] for node in nodes]
    return {
        "global_lane_id": -1,
        "polyline_world": [[node["x"], node["y"], 0.0] for node in nodes],
        "polyline_sn": [[node["s"], node["n"], 0.0] for node in nodes],
        "point_count": len(nodes),
        "observation_count": observation_count,
        "frame_count": len(frames),
        "s_range": [min(s_values), max(s_values)] if s_values else [0.0, 0.0],
        "n_median": median(n_values),
        "n_range": [min(n_values), max(n_values)] if n_values else [0.0, 0.0],
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
    output = []
    for index, node in enumerate(track_nodes):
        left = max(0, index - radius)
        right = min(len(track_nodes), index + radius + 1)
        neighbors = track_nodes[left:right]
        new_node = dict(node)
        for key in ("x", "y", "n"):
            new_node[key] = sum(item[key] for item in neighbors) / len(neighbors)
        new_node["heading"] = circular_mean([item["heading"] for item in neighbors])
        output.append(new_node)
    return output


def keep_lane(lane: Dict, args: argparse.Namespace) -> bool:
    if lane["point_count"] < args.min_track_nodes:
        return False
    if lane["observation_count"] < args.min_observations:
        return False
    if lane["s_range"][1] - lane["s_range"][0] < 10.0:
        return False
    return True


def draw_bev(path: Path, raw_points: Sequence[Dict], poses: Dict[int, Pose2], lanes: Sequence[Dict], args: argparse.Namespace):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(args.fig_width, args.fig_height))
    if not args.no_raw_points and raw_points:
        plt.scatter(
            [point["x"] for point in raw_points],
            [point["y"] for point in raw_points],
            s=0.12,
            c="#888888",
            alpha=0.10,
            linewidths=0,
            label="corrected lane samples",
            zorder=0,
        )

    traj_x = [pose[0] for _frame, pose in sorted(poses.items())]
    traj_y = [pose[1] for _frame, pose in sorted(poses.items())]
    plt.plot(traj_x, traj_y, "-", color="#222222", linewidth=1.0, alpha=0.35, label="corrected ego trajectory")

    colors = matplotlib.colormaps["tab20"].resampled(max(len(lanes), 1))
    for index, lane in enumerate(lanes):
        points = lane["polyline_world"]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        color = colors(index % 20)
        plt.plot(xs, ys, "-", linewidth=2.4, color=color, zorder=3)
        if xs:
            mid = len(xs) // 2
            plt.text(xs[mid], ys[mid], str(lane["global_lane_id"]), color=color, fontsize=8)

    plt.axis("equal")
    plt.grid(True, linewidth=0.3, alpha=0.55)
    plt.xlabel("corrected map x / m")
    plt.ylabel("corrected map y / m")
    plt.title("Corrected ClassLab RoadLane BEV")
    apply_plot_limits(raw_points, lanes, args)
    plt.tight_layout()
    plt.savefig(path, dpi=args.dpi)
    plt.close()


def apply_plot_limits(raw_points: Sequence[Dict], lanes: Sequence[Dict], args: argparse.Namespace):
    import matplotlib.pyplot as plt

    if args.xlim is not None:
        plt.xlim(args.xlim[0], args.xlim[1])
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    if args.xlim is not None and args.ylim is not None:
        return

    xs = [point["x"] for point in raw_points]
    ys = [point["y"] for point in raw_points]
    for lane in lanes:
        for point in lane["polyline_world"]:
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return
    margin = args.plot_margin
    if args.xlim is None:
        plt.xlim(min(xs) - margin, max(xs) + margin)
    if args.ylim is None:
        plt.ylim(min(ys) - margin, max(ys) + margin)


def write_pose_corrections(
    path: Path,
    frames: Sequence[int],
    initial: Dict[int, Pose2],
    corrected: Dict[int, Pose2],
    debug: Dict[int, Dict],
):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "frame",
                "initial_x",
                "initial_y",
                "initial_yaw",
                "corrected_x",
                "corrected_y",
                "corrected_yaw",
                "delta_x",
                "delta_y",
                "delta_yaw",
                "match_status",
                "match_count",
                "match_score",
            ]
        )
        for frame in frames:
            init = initial[frame]
            corr = corrected[frame]
            info = debug.get(frame, {})
            writer.writerow(
                [
                    frame,
                    init[0],
                    init[1],
                    init[2],
                    corr[0],
                    corr[1],
                    corr[2],
                    corr[0] - init[0],
                    corr[1] - init[1],
                    angle_diff(corr[2], init[2]),
                    info.get("status", ""),
                    info.get("matches", 0),
                    info.get("score", ""),
                ]
            )


def write_json(path: Path, data: Dict):
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def transform_point_2d(pose: Pose2, point: Point2) -> Point2:
    x, y, yaw = pose
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (x + c * point[0] - s * point[1], y + s * point[0] + c * point[1])


def compose_pose(left: Pose2, right: Pose2) -> Pose2:
    x, y = transform_point_2d(left, (right[0], right[1]))
    return (x, y, normalize_angle(left[2] + right[2]))


def inverse_pose(pose: Pose2) -> Pose2:
    x, y, yaw = pose
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (-c * x - s * y, s * x - c * y, normalize_angle(-yaw))


def relative_pose(base: Pose2, target: Pose2) -> Pose2:
    return compose_pose(inverse_pose(base), target)


def safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    return math.atan2(sum(math.sin(angle) for angle in angles), sum(math.cos(angle) for angle in angles))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_diff(left: float, right: float) -> float:
    return normalize_angle(left - right)


if __name__ == "__main__":
    raise SystemExit(main())
