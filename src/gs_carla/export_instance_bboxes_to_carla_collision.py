#!/usr/bin/env python3
"""Export DriveStudio instance bboxes as CARLA-space collision specs.

CARLA Python cannot create arbitrary new collision meshes without a matching
blueprint/static mesh. This script therefore produces a deterministic collision
spec JSON and can optionally draw the boxes in CARLA for debugging. Downstream
control can use the JSON for software collision checks, or a UE blueprint can
consume the same extents/transforms to create real collision actors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--classes", default="vehicle.", help="Comma-separated class prefixes; empty means all.")
    parser.add_argument("--calibration", type=Path, help="JSON containing T_drivestudio_from_carla or T_carla_from_drivestudio.")
    parser.add_argument("--coordinate-frame", choices=("drivestudio", "carla"), default="carla")
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--extent-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint to summarize GS instance tensors.")
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--draw-in-carla", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--draw-life-time", type=float, default=30.0)
    return parser.parse_args()


def load_matrix_json(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float32)
    data = json.loads(path.read_text())
    if "T_carla_from_drivestudio" in data:
        mat = np.asarray(data["T_carla_from_drivestudio"], dtype=np.float32)
    elif "T_drivestudio_from_carla" in data:
        mat = np.linalg.inv(np.asarray(data["T_drivestudio_from_carla"], dtype=np.float32))
    elif "matrix" in data:
        mat = np.asarray(data["matrix"], dtype=np.float32)
    else:
        raise ValueError(f"{path} must contain T_carla_from_drivestudio, T_drivestudio_from_carla, or matrix")
    if mat.size != 16:
        raise ValueError(f"{path} matrix must contain 16 values")
    return mat.reshape(4, 4).astype(np.float32)


def load_instances(scene: Path) -> tuple[dict, dict]:
    root = scene / "instances"
    info = json.loads((root / "instances_info.json").read_text())
    frame_instances = json.loads((root / "frame_instances.json").read_text())
    return info, frame_instances


def class_allowed(class_name: str, prefixes: list[str]) -> bool:
    if not prefixes:
        return True
    return any(class_name.startswith(prefix) for prefix in prefixes)


def yaw_from_matrix(mat: np.ndarray) -> float:
    return math.degrees(math.atan2(float(mat[1, 0]), float(mat[0, 0])))


def transform_to_record(mat: np.ndarray, size: np.ndarray, z_offset: float, extent_scale: float) -> dict:
    center = mat[:3, 3].astype(float)
    center[2] += z_offset
    extent = (size.astype(float) * 0.5 * extent_scale).tolist()
    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "extent": {"x": float(extent[0]), "y": float(extent[1]), "z": float(extent[2])},
        "yaw_deg": yaw_from_matrix(mat),
        "matrix": mat.astype(float).tolist(),
    }


def annotation_at_frame(instance: dict, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
    anns = instance.get("frame_annotations", {})
    frames = anns.get("frame_idx", [])
    try:
        idx = frames.index(frame)
    except ValueError:
        return None
    mat = np.asarray(anns["obj_to_world"][idx], dtype=np.float32)
    size = np.asarray(anns["box_size"][idx], dtype=np.float32)
    return mat, size


def summarize_checkpoint(path: Path) -> dict:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    models = ckpt.get("models", ckpt) if isinstance(ckpt, dict) else {}
    out: dict[str, dict] = {}
    for name in ("RigidNodes", "DeformableNodes", "SMPLNodes"):
        model = models.get(name) if isinstance(models, dict) else None
        if not isinstance(model, dict):
            continue
        item: dict[str, object] = {}
        for key in ("instances_trans", "instances_quats", "instances_size", "instances_fv"):
            value = model.get(key)
            if hasattr(value, "shape"):
                item[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        out[name] = item
    return out


def import_carla():
    try:
        import carla  # type: ignore

        return carla
    except ImportError:
        dist = Path(__file__).resolve().parents[2] / "PythonAPI" / "carla" / "dist"
        candidates = sorted(dist.glob(f"carla-*-cp{sys.version_info.major}{sys.version_info.minor}-*.whl"))
        if candidates:
            sys.path.insert(0, str(candidates[-1]))
            import carla  # type: ignore

            return carla
        raise


def draw_boxes_in_carla(args: argparse.Namespace, frames: list[dict]) -> None:
    carla = import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    for frame in frames:
        for obj in frame["objects"]:
            center = obj["center"]
            extent = obj["extent"]
            box = carla.BoundingBox(
                carla.Location(center["x"], center["y"], center["z"]),
                carla.Vector3D(extent["x"], extent["y"], extent["z"]),
            )
            yaw = obj["yaw_deg"]
            color = carla.Color(255, 64, 64, 255) if obj["class_name"].startswith("vehicle.") else carla.Color(64, 180, 255, 255)
            world.debug.draw_box(
                box,
                carla.Rotation(yaw=yaw),
                thickness=0.08,
                color=color,
                life_time=args.draw_life_time,
            )


def main() -> None:
    args = parse_args()
    instances, frame_instances = load_instances(args.processed_scene)
    prefixes = [item for item in args.classes.split(",") if item]
    t_carla_from_ds = load_matrix_json(args.calibration)
    use_carla = args.coordinate_frame == "carla"
    if use_carla and args.calibration is None:
        print("WARNING: --coordinate-frame carla without --calibration uses identity transform")

    available_frames = sorted(int(frame) for frame in frame_instances.keys())
    start = available_frames[0] if args.frame_start is None else args.frame_start
    end = available_frames[-1] if args.frame_end is None else args.frame_end

    frames_out: list[dict] = []
    for frame in range(start, end + 1):
        objects = []
        for instance_key in frame_instances.get(str(frame), []):
            instance = instances[str(instance_key)]
            class_name = instance.get("class_name", "")
            if not class_allowed(class_name, prefixes):
                continue
            ann = annotation_at_frame(instance, frame)
            if ann is None:
                continue
            mat, size = ann
            if use_carla:
                mat = t_carla_from_ds @ mat
            record = transform_to_record(mat, size, args.z_offset, args.extent_scale)
            record.update(
                {
                    "instance_key": int(instance_key),
                    "instance_id": instance.get("id", str(instance_key)),
                    "class_name": class_name,
                }
            )
            objects.append(record)
        frames_out.append({"frame": frame, "objects": objects})

    output = {
        "coordinate_frame": args.coordinate_frame,
        "source_processed_scene": str(args.processed_scene),
        "frame_start": start,
        "frame_end": end,
        "class_prefixes": prefixes,
        "bbox_convention": "center + half extent; matrix is object_to_world in coordinate_frame",
        "frames": frames_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"frames={len(frames_out)} objects={sum(len(f['objects']) for f in frames_out)}")

    if args.checkpoint is not None:
        summary = summarize_checkpoint(args.checkpoint)
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"Wrote {args.summary_output}")
        else:
            print(json.dumps(summary, indent=2))

    if args.draw_in_carla:
        draw_boxes_in_carla(args, frames_out)


if __name__ == "__main__":
    main()
