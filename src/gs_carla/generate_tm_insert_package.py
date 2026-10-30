#!/usr/bin/env python3
"""Generate a complete Traffic Manager insert package.

This is a convenience wrapper around:

  1. generate_tm_instance_trajectories.py
  2. convert_tm_instances_to_sequence.py

It writes the full insert directory layout:

  insert/
    tm_raw/
    carla_aligned/
    sequence_local/
    3dgs_origin/
    info.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--insert-dir", required=True, type=Path, help="Root insert output directory.")
    parser.add_argument("--xodr", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=60.0)

    parser.add_argument("--frames", type=int, default=101)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--num-actors", type=int, default=8)
    parser.add_argument("--target-instances", type=int, default=2)
    parser.add_argument("--ego-start-index", type=int, default=0)
    parser.add_argument("--task-start-frame", type=int, default=15)
    parser.add_argument("--task-end-frame", type=int, default=90)
    parser.add_argument("--spawn-near-ego-min", type=float, default=12.0)
    parser.add_argument("--spawn-near-ego-max", type=float, default=85.0)
    parser.add_argument("--ego-clearance", type=float, default=7.0)
    parser.add_argument("--actor-collision-margin", type=float, default=0.75)
    parser.add_argument("--min-track-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--spawn-source",
        choices=("auto", "spawn-points", "generated-waypoints", "both"),
        default="auto",
    )
    parser.add_argument("--spawn-waypoint-spacing", type=float, default=4.0)

    parser.add_argument("--mapping-pose", required=True, type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--processed-origin-frame", type=int, default=0)
    parser.add_argument("--processed-origin-mapping-frame", type=int, default=300)
    parser.add_argument("--sequence-origin-frame", type=int, default=0)
    parser.add_argument("--insert-origin-sequence-frame", type=int, default=310)
    parser.add_argument(
        "--3dgs-origin-sequence-frame",
        dest="g3d_origin_sequence_frame",
        type=int,
        help="Sequence frame used as 3DGS local origin. Defaults to --insert-origin-sequence-frame.",
    )
    parser.add_argument("--id-prefix", default="insert_")

    parser.add_argument("--vehicle-filter", default="vehicle.*")
    parser.add_argument("--ignore-lights-percentage", type=float, default=30.0)
    parser.add_argument("--global-distance-to-leading", type=float, default=4.0)
    parser.add_argument("--global-speed-difference", type=float, default=15.0)
    parser.add_argument("--draw-debug", action="store_true")
    parser.add_argument("--keep-actors", action="store_true")

    parser.add_argument(
        "--bev-output",
        type=Path,
        help="Write an XODR BEV PNG with lane waypoints and generated/manual insert tracks.",
    )
    parser.add_argument(
        "--bev-metadata-output",
        type=Path,
        help="Write BEV projection metadata JSON. Defaults to --bev-output with .json suffix.",
    )
    parser.add_argument(
        "--bev-only",
        action="store_true",
        help="Only render the XODR BEV template, then exit without generating/converting inserts.",
    )
    parser.add_argument(
        "--manual-trajectory",
        type=Path,
        help="Manual trajectory JSON. If set, validate/export it instead of running Traffic Manager generation.",
    )
    parser.add_argument(
        "--manual-fill-to-target",
        action="store_true",
        help="With --manual-trajectory, keep manual tracks and use Traffic Manager to fill up to --target-instances.",
    )
    parser.add_argument(
        "--manual-coordinate-frame",
        choices=("auto", "pixel", "carla"),
        default="auto",
        help="Coordinate frame for manual points. auto reads coordinate_frame from JSON, defaulting to pixel.",
    )
    parser.add_argument("--manual-max-lane-distance", type=float, default=2.5)
    parser.add_argument("--bev-waypoint-spacing", type=float, default=2.0)
    parser.add_argument("--bev-image-size", type=int, default=1800)
    parser.add_argument("--bev-padding", type=float, default=20.0)
    parser.add_argument("--vertex-distance", type=float, default=2.0)
    parser.add_argument("--max-road-length", type=float, default=500.0)
    parser.add_argument("--wall-height", type=float, default=0.0)
    parser.add_argument("--additional-width", type=float, default=0.6)
    parser.add_argument("--no-smooth-junctions", action="store_true")
    parser.add_argument("--no-mesh-visibility", action="store_true")

    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def count_tm_instances(tm_raw_dir: Path) -> int:
    info_path = tm_raw_dir / "instances" / "instances_info.json"
    if not info_path.is_file():
        return 0
    return len(read_json(info_path))


def merge_tm_raw_packages(output_dir: Path, sources: list[Path], frames: int, fps: float) -> None:
    instances_info = {}
    frame_instances: dict[str, list[int]] = {str(frame): [] for frame in range(frames)}
    raw_tracks = {}
    next_id = 0
    source_summaries = []

    for source in sources:
        source_info_path = source / "instances" / "instances_info.json"
        if not source_info_path.is_file():
            continue
        source_info = read_json(source_info_path)
        source_frames_path = source / "instances" / "frame_instances.json"
        source_frames = read_json(source_frames_path) if source_frames_path.is_file() else {}
        source_tracks_path = source / "actor_trajectories_carla.json"
        source_tracks = read_json(source_tracks_path) if source_tracks_path.is_file() else {}

        id_map: dict[str, int] = {}
        for old_key, item in sorted(source_info.items(), key=lambda kv: int(kv[0])):
            new_key = str(next_id)
            id_map[str(old_key)] = next_id
            instances_info[new_key] = item
            if str(old_key) in source_tracks:
                raw_tracks[new_key] = source_tracks[str(old_key)]
            next_id += 1

        for frame_key, ids in source_frames.items():
            merged_ids = frame_instances.setdefault(str(frame_key), [])
            for old_id in ids:
                mapped = id_map.get(str(old_id))
                if mapped is not None:
                    merged_ids.append(mapped)

        summary_path = source / "generation_summary.json"
        source_summaries.append(read_json(summary_path) if summary_path.is_file() else {"source_dir": str(source)})

    write_json(output_dir / "instances" / "instances_info.json", instances_info)
    write_json(output_dir / "instances" / "frame_instances.json", frame_instances)
    write_json(output_dir / "actor_trajectories_carla.json", raw_tracks)
    write_json(
        output_dir / "generation_summary.json",
        {"source": "merged", "num_instances": len(instances_info), "frames": frames, "fps": fps, "inputs": source_summaries},
    )


def generate_command(args: argparse.Namespace, tools_dir: Path, out_dir: Path, target_instances: int) -> list[str]:
    return [
        sys.executable,
        "-m", "gs_carla.generate_tm_instance_trajectories",
        "--xodr",
        str(args.xodr),
        "--out-dir",
        str(out_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tm-port",
        str(args.tm_port),
        "--timeout",
        str(args.timeout),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--num-actors",
        str(args.num_actors),
        "--target-instances",
        str(target_instances),
        "--ego-start-index",
        str(args.ego_start_index),
        "--task-start-frame",
        str(args.task_start_frame),
        "--task-end-frame",
        str(args.task_end_frame),
        "--spawn-near-ego-min",
        str(args.spawn_near_ego_min),
        "--spawn-near-ego-max",
        str(args.spawn_near_ego_max),
        "--ego-clearance",
        str(args.ego_clearance),
        "--actor-collision-margin",
        str(args.actor_collision_margin),
        "--min-track-frames",
        str(args.min_track_frames),
        "--seed",
        str(args.seed),
        "--spawn-source",
        args.spawn_source,
        "--spawn-waypoint-spacing",
        str(args.spawn_waypoint_spacing),
        "--vehicle-filter",
        args.vehicle_filter,
        "--ignore-lights-percentage",
        str(args.ignore_lights_percentage),
        "--global-distance-to-leading",
        str(args.global_distance_to_leading),
        "--global-speed-difference",
        str(args.global_speed_difference),
    ]


def bev_command(
    args: argparse.Namespace,
    tools_dir: Path,
    tm_raw_dir: Path,
    overlay_instances: Path | None,
    include_manual: bool = True,
) -> list[str]:
    image_output = args.bev_output or (args.insert_dir / "xodr_bev.png")
    cmd = [
        sys.executable,
        "-m", "gs_carla.xodr_bev_trajectory_tool",
        "--xodr",
        str(args.xodr),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
        "--vertex-distance",
        str(args.vertex_distance),
        "--max-road-length",
        str(args.max_road_length),
        "--wall-height",
        str(args.wall_height),
        "--additional-width",
        str(args.additional_width),
        "--waypoint-spacing",
        str(args.bev_waypoint_spacing),
        "--image-output",
        str(image_output),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--max-lane-distance",
        str(args.manual_max_lane_distance),
        "--image-size",
        str(args.bev_image_size),
        "--padding",
        str(args.bev_padding),
    ]
    if args.bev_metadata_output is not None:
        cmd += ["--metadata-output", str(args.bev_metadata_output)]
    if args.no_smooth_junctions:
        cmd.append("--no-smooth-junctions")
    if args.no_mesh_visibility:
        cmd.append("--no-mesh-visibility")
    if overlay_instances is not None:
        cmd += ["--overlay-instances", str(overlay_instances)]
    if include_manual and args.manual_trajectory is not None:
        cmd += [
            "--manual-trajectory",
            str(args.manual_trajectory),
            "--manual-coordinate-frame",
            args.manual_coordinate_frame,
            "--manual-output-dir",
            str(tm_raw_dir),
        ]
    return cmd


def main() -> int:
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    tm_raw_dir = args.insert_dir / "tm_raw"

    generate_cmd = generate_command(args, tools_dir, tm_raw_dir, args.target_instances)
    if args.draw_debug:
        generate_cmd.append("--draw-debug")
    if args.keep_actors:
        generate_cmd.append("--keep-actors")

    convert_cmd = [
        sys.executable,
        "-m", "gs_carla.convert_tm_instances_to_sequence",
        "--tm-out-dir",
        str(tm_raw_dir),
        "--output-dir",
        str(args.insert_dir),
        "--mapping-pose",
        str(args.mapping_pose),
        "--processed-scene",
        str(args.processed_scene),
        "--camera",
        str(args.camera),
        "--processed-origin-frame",
        str(args.processed_origin_frame),
        "--processed-origin-mapping-frame",
        str(args.processed_origin_mapping_frame),
        "--sequence-origin-frame",
        str(args.sequence_origin_frame),
        "--insert-origin-sequence-frame",
        str(args.insert_origin_sequence_frame),
        "--3dgs-origin-sequence-frame",
        str(args.g3d_origin_sequence_frame if args.g3d_origin_sequence_frame is not None else args.insert_origin_sequence_frame),
        "--id-prefix",
        args.id_prefix,
    ]

    args.insert_dir.mkdir(parents=True, exist_ok=True)
    if args.bev_only:
        run_command(bev_command(args, tools_dir, tm_raw_dir, overlay_instances=None))
        print(f"BEV template ready: {args.bev_output or (args.insert_dir / 'xodr_bev.png')}", flush=True)
        return 0

    if args.manual_trajectory is not None:
        if args.manual_fill_to_target:
            manual_dir = args.insert_dir / "tm_manual"
            auto_dir = args.insert_dir / "tm_auto_fill"
            run_command(bev_command(args, tools_dir, manual_dir, overlay_instances=None))
            manual_count = count_tm_instances(manual_dir)
            remaining = max(args.target_instances - manual_count, 0)
            sources = [manual_dir]
            if remaining > 0:
                fill_cmd = generate_command(args, tools_dir, auto_dir, remaining)
                if args.draw_debug:
                    fill_cmd.append("--draw-debug")
                if args.keep_actors:
                    fill_cmd.append("--keep-actors")
                run_command(fill_cmd)
                sources.append(auto_dir)
            merge_tm_raw_packages(tm_raw_dir, sources, args.frames, args.fps)
            run_command(
                bev_command(
                    args,
                    tools_dir,
                    tm_raw_dir,
                    overlay_instances=tm_raw_dir / "instances" / "instances_info.json",
                    include_manual=False,
                )
            )
        else:
            run_command(bev_command(args, tools_dir, tm_raw_dir, overlay_instances=None))
    else:
        run_command(generate_cmd)
        run_command(bev_command(args, tools_dir, tm_raw_dir, overlay_instances=tm_raw_dir / "instances" / "instances_info.json"))
    run_command(convert_cmd)
    print(f"Insert package ready: {args.insert_dir}", flush=True)
    print(f"Bridge instances_info: {args.insert_dir / 'sequence_local' / 'instances_info.json'}", flush=True)
    print(f"XODR BEV: {args.bev_output or (args.insert_dir / 'xodr_bev.png')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
