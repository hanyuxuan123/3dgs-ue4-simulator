#!/usr/bin/env python3
"""Replay Bench2Drive-Real scheduled actors in a live CARLA world and record actual poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-trajectories", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tick-world", action="store_true", help="Tick CARLA here. Disable when another synchronous runtime owns ticking.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--draw-debug", action="store_true")
    parser.add_argument("--debug-life-time", type=float, default=0.08)
    return parser.parse_args()


def import_carla():
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise RuntimeError("CARLA PythonAPI is required for live traffic replay") from exc
    return carla


def blueprint_names(class_name: str) -> list[str]:
    if class_name.startswith("human.pedestrian"):
        return ["walker.pedestrian.0001", "walker.pedestrian.0002", "walker.pedestrian.0007"]
    if "bicycle" in class_name:
        return ["vehicle.diamondback.century"]
    if "truck" in class_name:
        return ["vehicle.carlamotors.carlacola", "vehicle.tesla.cybertruck"]
    return ["vehicle.audi.a2", "vehicle.lincoln.mkz_2017", "vehicle.toyota.prius", "vehicle.tesla.model3"]


def find_blueprint(world, class_name: str):
    library = world.get_blueprint_library()
    for name in blueprint_names(class_name):
        try:
            bp = library.find(name)
            if bp:
                return bp
        except Exception:
            pass
    pattern = "walker.pedestrian.*" if class_name.startswith("human.pedestrian") else "vehicle.*"
    candidates = list(library.filter(pattern))
    if not candidates:
        raise RuntimeError(f"No CARLA blueprint found for {class_name}")
    return candidates[0]


def transform_from_row(carla, row: dict[str, Any], z_offset: float = 0.35):
    return carla.Transform(
        carla.Location(x=float(row["x"]), y=float(row["y"]), z=float(row.get("z", 0.0)) + z_offset),
        carla.Rotation(yaw=float(row["yaw"])),
    )


def spawn_actor(carla, world, track: dict[str, Any]):
    first = track["trajectory"][0]
    bp = find_blueprint(world, track["class_name"])
    actor = world.try_spawn_actor(bp, transform_from_row(carla, first))
    if actor is None:
        actor = world.spawn_actor(bp, transform_from_row(carla, first, z_offset=1.0))
    if hasattr(actor, "set_simulate_physics"):
        actor.set_simulate_physics(False)
    return actor, getattr(bp, "id", "")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    tracks = json.loads(args.actor_trajectories.read_text(encoding="utf-8"))
    carla = import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    debug_color = carla.Color(255, 80, 40, 255)
    actors = []
    rows: list[dict[str, Any]] = []
    try:
        for track in tracks:
            actor, blueprint_id = spawn_actor(carla, world, track)
            actors.append({"track": track, "actor": actor, "blueprint_id": blueprint_id})
        max_len = max((len(item["track"]["trajectory"]) for item in actors), default=0)
        for step in range(max_len):
            for item in actors:
                trajectory = item["track"]["trajectory"]
                actor = item["actor"]
                track = item["track"]
                if step >= len(trajectory):
                    actor.set_transform(carla.Transform(carla.Location(z=-500.0)))
                    continue
                target = trajectory[step]
                actor.set_transform(transform_from_row(carla, target))
                actual = actor.get_transform()
                rows.append(
                    {
                        "step": step,
                        "elapsed_time": step * args.control_dt,
                        "frame": int(target["frame"]),
                        "track_id": track["track_id"],
                        "scenario_type": track["scenario_type"],
                        "class_name": track["class_name"],
                        "blueprint_id": item["blueprint_id"],
                        "actor_id": int(actor.id),
                        "target_x": float(target["x"]),
                        "target_y": float(target["y"]),
                        "target_z": float(target.get("z", 0.0)),
                        "target_yaw": float(target["yaw"]),
                        "actual_x": float(actual.location.x),
                        "actual_y": float(actual.location.y),
                        "actual_z": float(actual.location.z),
                        "actual_yaw": float(actual.rotation.yaw),
                    }
                )
                if args.draw_debug:
                    world.debug.draw_point(actual.location, size=0.12, color=debug_color, life_time=args.debug_life_time)
            if args.tick_world:
                world.tick()
            if args.real_time and args.control_dt > 0:
                time.sleep(args.control_dt)
    finally:
        for item in actors:
            try:
                item["actor"].destroy()
            except Exception:
                pass
    write_csv(args.output_csv, rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"frames": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_csv}")
    if args.output_json:
        print(f"Wrote {args.output_json}")
    print(f"actors={len(actors)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
