#!/usr/bin/env python3
"""Run the live 3DGS CARLA bridge from a scene package JSON manifest."""

from __future__ import annotations

import argparse
import json
import runpy
import shlex
import sys
from pathlib import Path


ASSET_ARGS = {
    "xodr": "--xodr",
    "instances_info": "--instances-info",
    "background": "--background",
    "sky": "--sky",
    "processed_scene": "--processed-scene",
    "mapping_pose": "--mapping-pose",
    "core_pose_csv": "--core-pose-csv",
    "sequence_origin_pose_file": "--sequence-origin-pose-file",
}
FRAME_ARGS = {
    "instance_origin_sequence_frame": "--instance-origin-sequence-frame",
    "processed_origin_frame": "--processed-origin-frame",
    "processed_origin_mapping_frame": "--processed-origin-mapping-frame",
    "sequence_origin_frame": "--sequence-origin-frame",
    "core_start_frame": "--core-start-frame",
    "core_end_frame": "--core-end-frame",
}
VALUE_ARGS = {
    "camera": "--camera",
    "instance_transform_mode": "--instance-transform-mode",
    "core_pose_coordinate_frame": "--core-pose-coordinate-frame",
}
WRAPPER_OPTIONS_WITH_VALUES = {"--scene-package"}
WRAPPER_FLAGS = {"--print-expanded-command"}


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--scene-package", required=True, type=Path)
    parser.add_argument(
        "--print-expanded-command",
        action="store_true",
        help="Print the expanded command before running the underlying bridge.",
    )
    return parser.parse_known_args(argv)


def provided_options(argv: list[str]) -> set[str]:
    return {item.split("=", 1)[0] for item in argv if item.startswith("--")}


def package_path(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else base / path)


def add_default_arg(expanded: list[str], option: str, value, provided: set[str]) -> None:
    if value is None or option in provided:
        return
    expanded.extend([option, str(value)])


def remove_wrapper_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        option = item.split("=", 1)[0]
        if option in WRAPPER_FLAGS:
            i += 1
            continue
        if option in WRAPPER_OPTIONS_WITH_VALUES:
            i += 1 if "=" in item else 2
            continue
        cleaned.append(item)
        i += 1
    return cleaned


def expanded_args(package_file: Path, passthrough: list[str]) -> list[str]:
    package_file = package_file.expanduser()
    data = json.loads(package_file.read_text(encoding="utf-8"))
    base = package_file.parent
    assets = data.get("assets", {})
    frames = data.get("frames", {})
    values = data.get("values", {})
    provided = provided_options(passthrough)
    expanded: list[str] = []
    for key, option in ASSET_ARGS.items():
        add_default_arg(expanded, option, package_path(base, assets.get(key)), provided)
    for key, option in FRAME_ARGS.items():
        add_default_arg(expanded, option, frames.get(key), provided)
    for key, option in VALUE_ARGS.items():
        add_default_arg(expanded, option, values.get(key), provided)
    expanded.extend(passthrough)
    return expanded


def main() -> int:
    wrapper_args, _ = parse_wrapper_args(sys.argv[1:])
    passthrough = remove_wrapper_args(sys.argv[1:])
    args = expanded_args(wrapper_args.scene_package, passthrough)
    target = Path(__file__).with_name("carla_xodr_live_3dgs_bridge_with_instances.py")
    if wrapper_args.print_expanded_command:
        print("python " + " ".join(shlex.quote(item) for item in [str(target), *args]))
    sys.argv = [str(target), *args]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
