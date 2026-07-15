#!/usr/bin/env python3
"""Load replayed traffic actors in CARLA and export their bounding boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import carla_xodr_live_3dgs_bridge_with_instances as base
from .load_xodr_town_with_instances import (
    load_instance_tracks,
    spawn_instance_actors,
    update_instance_actors,
    write_instance_csv,
)


class TrafficBBoxCarlaLoader:
    """Replay instance traffic as CARLA actors and collect bbox records."""

    def __init__(self, carla, world, carla_map, args: argparse.Namespace, t_seq_from_instance, tracks: list[dict[str, Any]]):
        self.carla = carla
        self.world = world
        self.carla_map = carla_map
        self.args = args
        self.t_seq_from_instance = t_seq_from_instance
        self.actors = spawn_instance_actors(carla, world, carla_map, tracks, t_seq_from_instance, args) if tracks else []
        self.instance_rows: list[dict[str, Any]] = []
        self.bbox_frames: list[dict[str, Any]] = []

    @classmethod
    def from_instances_info(cls, carla, world, carla_map, args: argparse.Namespace, t_seq_from_instance):
        tracks = []
        if args.instances_info is not None:
            tracks = load_instance_tracks(args.instances_info, args.instance_class_prefix, args.max_instance_actors)
            print(f"[traffic] loaded tracks={len(tracks)} from {args.instances_info}")
        return cls(carla, world, carla_map, args, t_seq_from_instance, tracks)

    def update(self, sequence_frame: int, step: int) -> list[dict[str, Any]]:
        rows = (
            update_instance_actors(
                self.carla,
                self.carla_map,
                self.actors,
                sequence_frame,
                self.t_seq_from_instance,
                self.args,
            )
            if self.actors
            else []
        )
        self.instance_rows.extend(rows)
        objects = [self._bbox_record(entry, sequence_frame, step) for entry in self.actors]
        objects = [item for item in objects if item is not None]
        if objects:
            self.bbox_frames.append({"step": step, "sequence_frame": sequence_frame, "objects": objects})
            if getattr(self.args, "draw_traffic_bboxes", False):
                self.draw_boxes(objects)
        return rows

    def write_outputs(self) -> None:
        if self.args.instance_output:
            write_instance_csv(self.args.instance_output, self.instance_rows)
            print(f"Wrote {self.args.instance_output}")
        if self.args.traffic_bbox_output:
            self.args.traffic_bbox_output.parent.mkdir(parents=True, exist_ok=True)
            self.args.traffic_bbox_output.write_text(
                json.dumps(
                    {
                        "coordinate_frame": "carla",
                        "bbox_convention": "center + half extent; actor bounding box transformed by CARLA actor transform",
                        "frames": self.bbox_frames,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {self.args.traffic_bbox_output}")

    def draw_boxes(self, objects: list[dict[str, Any]]) -> None:
        color = self.carla.Color(255, 64, 64, 255)
        for item in objects:
            center = item["center"]
            extent = item["extent"]
            box = self.carla.BoundingBox(
                self.carla.Location(center["x"], center["y"], center["z"]),
                self.carla.Vector3D(extent["x"], extent["y"], extent["z"]),
            )
            self.world.debug.draw_box(
                box,
                self.carla.Rotation(yaw=item["yaw_deg"]),
                thickness=0.06,
                color=color,
                life_time=self.args.traffic_bbox_life_time,
            )

    def _bbox_record(self, entry: dict[str, Any], sequence_frame: int, step: int) -> dict[str, Any] | None:
        actor = entry["actor"]
        transform = actor.get_transform()
        if transform.location.z <= self.args.hidden_actor_z + 10.0:
            return None
        bbox = actor.bounding_box
        center = transform.transform(bbox.location)
        extent = bbox.extent
        return {
            "step": step,
            "sequence_frame": sequence_frame,
            "track_id": entry["track"]["id"],
            "class_name": entry["track"]["class_name"],
            "actor_id": int(actor.id),
            "blueprint_id": entry.get("blueprint_id", ""),
            "center": {"x": float(center.x), "y": float(center.y), "z": float(center.z)},
            "extent": {"x": float(extent.x), "y": float(extent.y), "z": float(extent.z)},
            "yaw_deg": float(transform.rotation.yaw),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw an exported traffic bbox JSON in an already-running CARLA world.")
    parser.add_argument("--bbox-json", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--life-time", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    carla = base.import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    data = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    color = carla.Color(255, 64, 64, 255)
    for frame in data.get("frames", []):
        for item in frame.get("objects", []):
            center = item["center"]
            extent = item["extent"]
            box = carla.BoundingBox(
                carla.Location(center["x"], center["y"], center["z"]),
                carla.Vector3D(extent["x"], extent["y"], extent["z"]),
            )
            world.debug.draw_box(box, carla.Rotation(yaw=item["yaw_deg"]), thickness=0.08, color=color, life_time=args.life_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
