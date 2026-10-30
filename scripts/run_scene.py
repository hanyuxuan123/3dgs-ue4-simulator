#!/usr/bin/env python3
"""Select the nuScenes or ClassLab CARLA/3DGS workflow from one command."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("xodr", "export-background", "prepare", "run", "runtime", "traffic", "control")
MODULES = {
    ("nuscenes", "prepare"): "gs_carla.nuscenes_global_runtime.prepare_nuscenes_global_scene",
    ("nuscenes", "run"): "gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene",
    ("classlab", "prepare"): "gs_carla.make_3dgs_carla_scene_package",
    ("classlab", "run"): "gs_carla.scene_runtime_external_control",
    ("nuscenes", "runtime"): "gs_carla.scene_runtime_external_control",
    ("classlab", "runtime"): "gs_carla.scene_runtime_external_control",
    ("nuscenes", "export-background"): "gs_carla.export_drivestudio_background",
    ("classlab", "export-background"): "gs_carla.export_drivestudio_background",
    ("nuscenes", "traffic"): "gs_carla.generate_tm_insert_package",
    ("classlab", "traffic"): "gs_carla.generate_tm_insert_package",
    ("nuscenes", "control"): "gs_carla.nuscenes_global_runtime.send_control",
    ("classlab", "control"): "gs_carla.nuscenes_global_runtime.send_control",
}
SCRIPTS = {
    ("nuscenes", "xodr"): ROOT / "tools/xodr/nuscenes_to_xodr.py",
    ("classlab", "xodr"): ROOT / "tools/xodr/classlab/classlab_mapping_pose_semantic_to_xodr_with_connectors.py",
}


def build_command(dataset: str, stage: str, forwarded: list[str]) -> list[str]:
    """Map dataset and stage to the existing implementation without rewriting it."""
    target = (dataset, stage)
    if target in MODULES:
        return [sys.executable, "-m", MODULES[target], *forwarded]
    if target in SCRIPTS:
        return [sys.executable, str(SCRIPTS[target]), *forwarded]
    raise ValueError(f"unsupported dataset/stage: {dataset}/{stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Pass the selected tool's arguments after --. Example: "
        "python scripts/run_scene.py --dataset nuscenes --stage run -- --xodr map.xodr ...",
    )
    parser.add_argument("--dataset", required=True, choices=("nuscenes", "classlab"))
    parser.add_argument("--stage", default="run", choices=STAGES)
    parser.add_argument("--dry-run", action="store_true", help="Print the selected command and exit.")
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    command = build_command(args.dataset, args.stage, forwarded)
    print(f"[scene] {args.dataset}/{args.stage}: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
