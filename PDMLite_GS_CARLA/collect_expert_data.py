#!/usr/bin/env python3
"""Run an expert over the CARLA-3DGS control socket and save aligned samples."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import socket
import time
from pathlib import Path
from typing import Any


DEFAULT_ENTRYPOINT = "PDMLite_GS_CARLA.team_code.pdmlite_adapter:create_expert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expert-config", required=True, type=Path)
    parser.add_argument("--expert-entrypoint", default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--camera-map", type=Path, default=Path(__file__).with_name("camera_map_nuscenes.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=29001)
    parser.add_argument("--connect-timeout", type=float, default=300.0)
    parser.add_argument("--retry-interval", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=1, help="Save every Nth new render while still controlling every tick")
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def load_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"entrypoint must be module:object, got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


def connect_with_retry(host: str, port: int, timeout: float, interval: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return socket.create_connection((host, port), timeout=30.0)
        except OSError as exc:
            last_error = exc
            time.sleep(max(0.05, interval))
    raise TimeoutError(f"could not connect to {host}:{port}: {last_error}")


def normalize_control(result: Any) -> tuple[dict[str, float], dict[str, Any]]:
    if isinstance(result, dict):
        source = result.get("control", result)
        annotations = result.get("annotations", {})
    else:
        source = result
        annotations = {}
    try:
        values = {
            "throttle": float(source["throttle"] if isinstance(source, dict) else source.throttle),
            "steer": float(source["steer"] if isinstance(source, dict) else source.steer),
            "brake": float(source["brake"] if isinstance(source, dict) else source.brake),
        }
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("expert must return control with throttle, steer and brake") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"expert returned non-finite control: {values}")
    values["throttle"] = max(0.0, min(1.0, values["throttle"]))
    values["steer"] = max(-1.0, min(1.0, values["steer"]))
    values["brake"] = max(0.0, min(1.0, values["brake"]))
    if values["brake"] > 0.05:
        values["throttle"] = 0.0
    return values, annotations


def json_default(value: Any) -> Any:
    """Convert common tensor/array/scalar annotations without taking a NumPy dependency."""
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        return tolist()
    item = getattr(value, "item", None)
    if item is not None:
        return item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def copy_sensor_images(packet: dict[str, Any], camera_map: dict[str, int], output_dir: Path, sample_id: int) -> dict[str, str]:
    images = packet.get("images", {})
    copied: dict[str, str] = {}
    for name, camera_id in camera_map.items():
        source_value = images.get(str(camera_id))
        if source_value is None:
            raise FileNotFoundError(f"sensor packet has no camera {camera_id} ({name})")
        source = Path(source_value)
        if not source.is_file():
            raise FileNotFoundError(f"rendered image disappeared before collection: {source}")
        destination = output_dir / "images" / name / f"{sample_id:06d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".png.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        copied[name] = str(destination.relative_to(output_dir))
    return copied


def main() -> int:
    args = parse_args()
    for path in (args.scene_package, args.expert_config, args.camera_map):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expert_config = json.loads(args.expert_config.read_text(encoding="utf-8"))
    camera_map = {name: int(value) for name, value in json.loads(args.camera_map.read_text(encoding="utf-8")).items()}
    context = {
        "scene_package": str(args.scene_package.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "camera_map": camera_map,
        "expert_config": expert_config,
    }
    expert = load_symbol(args.expert_entrypoint)(context)
    metadata = {
        "format": "carla_3dgs_expert_v1",
        "scene_package": str(args.scene_package.resolve()),
        "expert_entrypoint": args.expert_entrypoint,
        "expert_config": expert_config,
        "camera_map": camera_map,
        "sample_every": args.sample_every,
        "alignment": "observation uses sensor_packet.step; control is expert output for that packet",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    sample_count = 0
    last_render_index: int | None = None
    frames_path = args.output_dir / "frames.jsonl"
    try:
        sock = connect_with_retry(args.host, args.control_port, args.connect_timeout, args.retry_interval)
        print(f"[expert-collector] connected to {args.host}:{args.control_port}", flush=True)
        with sock, sock.makefile("rwb") as fp, frames_path.open("a", encoding="utf-8") as frames:
            while True:
                line = fp.readline()
                if not line:
                    break
                observation = json.loads(line.decode("utf-8"))
                packet = observation.get("sensor_packet")
                if packet is None:
                    control = {"throttle": 0.0, "steer": 0.0, "brake": 1.0}
                    state = "warmup"
                else:
                    control, annotations = normalize_control(expert.run_step(observation))
                    render_index = int(packet["render_index"])
                    is_new_render = render_index != last_render_index
                    if is_new_render:
                        last_render_index = render_index
                    if is_new_render and render_index % max(1, args.sample_every) == 0:
                        copied = copy_sensor_images(packet, camera_map, args.output_dir, sample_count)
                        record = {
                            "sample_id": sample_count,
                            "sensor_packet_step": packet.get("step"),
                            "render_index": render_index,
                            "images": copied,
                            "camera_calibration": packet.get("cameras", []),
                            "observation": {key: value for key, value in observation.items() if key != "sensor_packet"},
                            "expert_control": control,
                            "expert_annotations": annotations,
                        }
                        frames.write(json.dumps(record, default=json_default) + "\n")
                        frames.flush()
                        sample_count += 1
                    state = "expert"
                fp.write((json.dumps(control) + "\n").encode("utf-8"))
                fp.flush()
                print(f"[expert-collector] step={observation['step']} state={state} samples={sample_count} control={control}", flush=True)
                if args.max_samples > 0 and sample_count >= args.max_samples:
                    break
    finally:
        close = getattr(expert, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
