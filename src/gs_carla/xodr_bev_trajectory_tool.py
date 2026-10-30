#!/usr/bin/env python3
"""Render XODR BEV maps and validate/export manual insert trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from .load_xodr_town_and_dump_poses import import_carla, load_xodr_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--vertex-distance", type=float, default=2.0)
    parser.add_argument("--max-road-length", type=float, default=500.0)
    parser.add_argument("--wall-height", type=float, default=0.0)
    parser.add_argument("--additional-width", type=float, default=0.6)
    parser.add_argument("--no-smooth-junctions", action="store_true")
    parser.add_argument("--no-mesh-visibility", action="store_true")
    parser.add_argument("--waypoint-spacing", type=float, default=2.0)
    parser.add_argument("--image-output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--overlay-instances", type=Path, help="instances_info.json to draw on the BEV.")
    parser.add_argument("--manual-trajectory", type=Path, help="Manual trajectory JSON to validate and optionally export.")
    parser.add_argument("--manual-coordinate-frame", choices=("auto", "pixel", "carla"), default="auto")
    parser.add_argument("--manual-output-dir", type=Path, help="Write tm_raw-compatible manual trajectories here when validation passes.")
    parser.add_argument("--frames", type=int, default=101)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-lane-distance", type=float, default=2.5)
    parser.add_argument("--vehicle-box-size", nargs=3, type=float, default=(4.8, 2.0, 1.7))
    parser.add_argument("--image-size", type=int, default=1800)
    parser.add_argument("--padding", type=float, default=20.0)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def transform_matrix(x: float, y: float, z: float, yaw_deg: float) -> list[list[float]]:
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]


def load_waypoints(carla_map, spacing: float) -> list:
    return carla_map.generate_waypoints(max(spacing, 0.5))


def bounds_from_waypoints(waypoints: list, padding: float) -> tuple[float, float, float, float]:
    xs = [float(w.transform.location.x) for w in waypoints]
    ys = [float(w.transform.location.y) for w in waypoints]
    return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding


def make_projection(bounds: tuple[float, float, float, float], image_size: int) -> dict[str, float]:
    min_x, max_x, min_y, max_y = bounds
    scale = image_size / max(max_x - min_x, max_y - min_y, 1e-6)
    width = int(math.ceil((max_x - min_x) * scale))
    height = int(math.ceil((max_y - min_y) * scale))
    return {"min_x": min_x, "max_y": max_y, "scale": scale, "width": width, "height": height}


def world_to_pixel(x: float, y: float, meta: dict[str, float]) -> tuple[int, int]:
    return int(round((x - meta["min_x"]) * meta["scale"])), int(round((meta["max_y"] - y) * meta["scale"]))


def pixel_to_world(x: float, y: float, meta: dict[str, float]) -> tuple[float, float]:
    return float(x) / meta["scale"] + meta["min_x"], meta["max_y"] - float(y) / meta["scale"]


def draw_waypoints(draw: ImageDraw.ImageDraw, waypoints: list, meta: dict[str, float]) -> None:
    colors = {}
    palette = [(60, 90, 220), (40, 160, 80), (220, 120, 40), (180, 60, 200), (60, 180, 180), (200, 80, 80)]
    grouped: dict[tuple[int, int], list] = {}
    for wp in waypoints:
        grouped.setdefault((int(wp.road_id), int(wp.lane_id)), []).append(wp)
    for index, (key, lane_wps) in enumerate(sorted(grouped.items())):
        lane_wps.sort(key=lambda w: float(w.s))
        color = colors.setdefault(key, palette[index % len(palette)])
        pts = [world_to_pixel(float(w.transform.location.x), float(w.transform.location.y), meta) for w in lane_wps]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)
        for p in pts[:: max(1, len(pts) // 20)]:
            draw.ellipse((p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2), fill=color)


def normalize_manual_tracks(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [{"id": "manual_0", "class_name": "vehicle.car", "points": data}]
    if "tracks" in data:
        return data["tracks"]
    if "points" in data:
        return [{"id": data.get("id", "manual_0"), "class_name": data.get("class_name", "vehicle.car"), "points": data["points"]}]
    raise ValueError("manual trajectory JSON needs tracks, points, or a list of points")


def point_xy(point: Any) -> tuple[float, float]:
    if isinstance(point, dict):
        return float(point.get("x", point.get("u"))), float(point.get("y", point.get("v")))
    return float(point[0]), float(point[1])


def manual_frame(data: dict[str, Any], cli_frame: str) -> str:
    if cli_frame != "auto":
        return cli_frame
    return data.get("coordinate_frame", "pixel")


def validate_track(carla, carla_map, points: list[tuple[float, float]], max_distance: float) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for idx, (x, y) in enumerate(points):
        loc = carla.Location(x=x, y=y, z=0.0)
        wp = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            errors.append(f"point {idx}: no driving lane near ({x:.2f}, {y:.2f})")
            continue
        wloc = wp.transform.location
        dist = math.hypot(float(wloc.x) - x, float(wloc.y) - y)
        if dist > max_distance:
            errors.append(f"point {idx}: {dist:.2f}m from nearest lane, over {max_distance:.2f}m")
        rows.append(
            {
                "x": float(wloc.x),
                "y": float(wloc.y),
                "z": float(wloc.z) + 0.35,
                "yaw": float(wp.transform.rotation.yaw),
                "road_id": int(wp.road_id),
                "lane_id": int(wp.lane_id),
                "s": float(wp.s),
                "distance_to_input": dist,
            }
        )
    return rows, errors


def resample_track(track: list[dict[str, Any]], frames: int, fps: float) -> list[dict[str, Any]]:
    if len(track) < 2:
        raise ValueError("manual track needs at least two valid points")
    arcs = [0.0]
    for a, b in zip(track, track[1:]):
        arcs.append(arcs[-1] + math.hypot(a["x"] - b["x"], a["y"] - b["y"]))
    out = []
    for frame in range(frames):
        target = arcs[-1] * frame / max(frames - 1, 1)
        j = 0
        while j < len(arcs) - 2 and arcs[j + 1] < target:
            j += 1
        span = max(arcs[j + 1] - arcs[j], 1e-6)
        r = (target - arcs[j]) / span
        a, b = track[j], track[min(j + 1, len(track) - 1)]
        yaw = math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))
        row = {key: float(a[key]) * (1.0 - r) + float(b[key]) * r for key in ("x", "y", "z")}
        row.update({"yaw": yaw, "frame": frame, "elapsed_time": frame / fps})
        out.append(row)
    return out


def write_tm_raw(out_dir: Path, tracks: list[dict[str, Any]], frames: int, fps: float, box_size: tuple[float, float, float]) -> None:
    instances_dir = out_dir / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    instances_info = {}
    frame_instances: dict[str, list[int]] = {str(frame): [] for frame in range(frames)}
    raw_tracks = {}
    for idx, track in enumerate(tracks):
        sampled = resample_track(track["validated"], frames, fps)
        frame_idx = [row["frame"] for row in sampled]
        for frame in frame_idx:
            frame_instances[str(frame)].append(idx)
        instances_info[str(idx)] = {
            "id": track.get("id", f"manual_{idx}"),
            "class_name": track.get("class_name", "vehicle.car"),
            "frame_annotations": {
                "frame_idx": frame_idx,
                "obj_to_world": [transform_matrix(row["x"], row["y"], row["z"], row["yaw"]) for row in sampled],
                "box_size": [list(box_size) for _ in sampled],
            },
        }
        raw_tracks[str(idx)] = {"actor_id": track.get("id", f"manual_{idx}"), "frames": sampled}
    (instances_dir / "instances_info.json").write_text(json.dumps(instances_info, indent=2) + "\n", encoding="utf-8")
    (instances_dir / "frame_instances.json").write_text(json.dumps(frame_instances, indent=2) + "\n", encoding="utf-8")
    (out_dir / "actor_trajectories_carla.json").write_text(json.dumps(raw_tracks, indent=2) + "\n", encoding="utf-8")
    (out_dir / "generation_summary.json").write_text(
        json.dumps({"source": "manual", "num_instances": len(tracks), "frames": frames, "fps": fps}, indent=2) + "\n",
        encoding="utf-8",
    )


def draw_instances(path: Path, draw: ImageDraw.ImageDraw, meta: dict[str, float], color: tuple[int, int, int], width: int) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data.values():
        ann = item.get("frame_annotations", {})
        pts = []
        for mat in ann.get("obj_to_world", []):
            pts.append(world_to_pixel(float(mat[0][3]), float(mat[1][3]), meta))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)


def main() -> int:
    args = parse_args()
    carla = import_carla()
    world_args = SimpleNamespace(**vars(args))
    _, world = load_xodr_world(carla, world_args)
    carla_map = world.get_map()
    waypoints = load_waypoints(carla_map, args.waypoint_spacing)
    if not waypoints:
        raise RuntimeError("No driving waypoints generated from XODR")
    meta = make_projection(bounds_from_waypoints(waypoints, args.padding), args.image_size)
    image = Image.new("RGB", (int(meta["width"]), int(meta["height"])), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw_waypoints(draw, waypoints, meta)
    if args.overlay_instances and args.overlay_instances.exists():
        draw_instances(args.overlay_instances, draw, meta, (220, 30, 30), 4)

    validation = {"tracks": [], "errors": []}
    if args.manual_trajectory:
        raw = json.loads(args.manual_trajectory.read_text(encoding="utf-8"))
        coord_frame = manual_frame(raw if isinstance(raw, dict) else {}, args.manual_coordinate_frame)
        manual_meta = raw.get("bev_metadata", meta) if isinstance(raw, dict) else meta
        valid_tracks = []
        for track in normalize_manual_tracks(raw):
            pts_in = [point_xy(p) for p in track.get("points", [])]
            pts = [pixel_to_world(x, y, manual_meta) for x, y in pts_in] if coord_frame == "pixel" else pts_in
            validated, errors = validate_track(carla, carla_map, pts, args.max_lane_distance)
            validation["tracks"].append({**track, "validated": validated, "num_errors": len(errors)})
            validation["errors"].extend([f"{track.get('id', 'manual')}: {err}" for err in errors])
            pix = [world_to_pixel(row["x"], row["y"], meta) for row in validated]
            if len(pix) >= 2:
                draw.line(pix, fill=(255, 0, 180), width=5)
            valid_tracks.append({**track, "validated": validated})
        if validation["errors"]:
            print("[manual] validation failed:")
            for err in validation["errors"]:
                print(f"  {err}")
        elif args.manual_output_dir:
            write_tm_raw(args.manual_output_dir, valid_tracks, args.frames, args.fps, tuple(args.vehicle_box_size))
            print(f"Wrote manual tm_raw package: {args.manual_output_dir}")

    if args.title:
        draw.text((10, 10), args.title, fill=(0, 0, 0))
    args.image_output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.image_output)
    metadata_path = args.metadata_output or args.image_output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "bev_metadata": meta,
                "xodr": str(args.xodr),
                "waypoint_spacing": args.waypoint_spacing,
                "manual_validation": validation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.image_output}")
    print(f"Wrote {metadata_path}")
    return 1 if validation["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
