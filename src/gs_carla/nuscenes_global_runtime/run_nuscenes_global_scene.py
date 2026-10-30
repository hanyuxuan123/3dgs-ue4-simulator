#!/usr/bin/env python3
"""Prepare and launch a global nuScenes XODR + 3DGS CARLA runtime."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xodr", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=48)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", type=int)
    parser.add_argument("--export-background", action="store_true")
    parser.add_argument("--force-export-background", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--control-port", type=int, default=29001)
    parser.add_argument("--disable-control-server", action="store_true")
    parser.add_argument("--motion-mode", choices=("kinematic", "control", "path-follow"), default="kinematic")
    parser.add_argument("--render-pose-source", choices=("actor", "reference"), default=None)
    parser.add_argument("--pygame", action="store_true")
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument(
        "--publish-rendered-observations",
        action="store_true",
        help="Expose rendered camera image paths/calibration to an external agent over the control socket.",
    )
    parser.add_argument("--publish-expert-state", action="store_true")
    parser.add_argument("--expert-route-points", type=int, default=80)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--runtime-output-dir", type=Path)
    parser.add_argument("--max-render-frames", type=int, default=0)
    parser.add_argument("--exit-after-max-render-frames", action="store_true")
    parser.add_argument("--render-every-n", type=int, default=1)
    parser.add_argument("--follow-speed-kmh", type=float, default=12.0)
    parser.add_argument("--follow-lookahead-distance", type=float, default=10.0)
    parser.add_argument("--steer-gain", type=float, default=0.35)
    parser.add_argument("--max-steer", type=float, default=0.30)
    parser.add_argument("--follow-max-seconds", type=float, default=0.0)
    parser.add_argument("--control-dt", type=float, default=0.05)
    parser.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="Stop on route completion/collision and write episode_result.json.",
    )
    parser.add_argument("--evaluation-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--collision-impulse-threshold", type=float, default=0.0)
    parser.add_argument("--finish-distance", type=float, default=3.0)
    parser.add_argument("--downscale", type=float, default=2.0)
    parser.add_argument("--max-gaussians", type=int, default=2_000_000)
    parser.add_argument("--crop-radius", type=float, default=180.0)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--spectator-follow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-follow-map-z", action="store_true")
    parser.add_argument("--no-mesh-visibility", action="store_true")
    parser.add_argument(
        "--render-dynamic-gaussians",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render checkpoint RigidNodes dynamic Gaussians in 3DGS; live CARLA instances remain enabled either way.",
    )
    parser.add_argument(
        "--instances-info",
        type=Path,
        help="Optional generated instances_info.json to attach to the prepared scene_package before launching.",
    )
    parser.add_argument(
        "--instance-transform-mode",
        default="mapping-absolute",
        choices=("mapping-absolute", "sequence-anchor", "raw-local"),
        help="Coordinate convention for --instances-info.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runtime-start-timeout", type=float, default=300.0)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_for_runtime_control_server(runtime: subprocess.Popen, timeout: float, log_path: Path) -> bool:
    deadline = time.monotonic() + max(timeout, 1.0)
    while time.monotonic() < deadline:
        if "[control-server] listening" in tail_text(log_path, max_lines=200):
            return True
        code = runtime.poll()
        if code is not None:
            print(f"[run] runtime exited before opening control port; returncode={code}")
            print(f"[run] runtime log tail:\n{tail_text(log_path)}")
            return False
        time.sleep(1.0)
    print("[run] timed out waiting for runtime control server")
    print(f"[run] runtime log tail:\n{tail_text(log_path)}")
    return False


def attach_instances_to_scene_package(scene_package: Path, instances_info: Path, instance_transform_mode: str) -> None:
    package = json.loads(scene_package.read_text(encoding="utf-8"))
    package.setdefault("assets", {})["instances_info"] = str(instances_info.resolve())
    package.setdefault("values", {})["instance_transform_mode"] = instance_transform_mode
    package.setdefault("metadata", {})["attached_instances"] = {
        "instances_info": str(instances_info.resolve()),
        "instance_transform_mode": instance_transform_mode,
        "note": "Attached by run_nuscenes_global_scene.py before launching scene_runtime_external_control.py.",
    }
    scene_package.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    print(f"[run] attached instances_info to scene package: {instances_info}")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    prepare_cmd = [
        args.python,
        "-m", "gs_carla.nuscenes_global_runtime.prepare_nuscenes_global_scene",
        "--xodr",
        str(args.xodr),
        "--checkpoint",
        str(args.checkpoint),
        "--processed-scene",
        str(args.processed_scene),
        "--output-dir",
        str(args.output_dir),
        "--start-frame",
        str(args.start_frame),
        "--end-frame",
        str(args.end_frame),
        "--camera",
        str(args.camera),
        "--python",
        args.python,
    ]
    if args.config:
        prepare_cmd += ["--config", str(args.config)]
    if args.export_background:
        prepare_cmd.append("--export-background")
    if args.force_export_background:
        prepare_cmd.append("--force-export-background")
    subprocess.run(prepare_cmd, check=True)

    scene_package = args.output_dir / "scene_package.json"
    if args.instances_info:
        attach_instances_to_scene_package(scene_package, args.instances_info, args.instance_transform_mode)

    runtime_output = args.runtime_output_dir or args.output_dir / "runtime_frames"
    pose_output = args.output_dir / "runtime_pose.csv"
    path_json = args.output_dir / "runtime_path.json"
    collision_output = args.output_dir / "runtime_collisions.csv"
    episode_result_output = args.output_dir / "episode_result.json"
    runtime_log = args.output_dir / "runtime.log"
    controller_log = args.output_dir / "controller.log"
    runtime_cmd = [
        args.python,
        "-m", "gs_carla.scene_runtime_external_control",
        "--scene-package",
        str(scene_package),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--control-port",
        str(args.control_port),
        "--core-start-frame",
        str(args.start_frame),
        "--core-end-frame",
        str(args.end_frame),
        "--source-frame-mode",
        "nearest" if args.motion_mode == "control" else "timeline",
        "--source-frame-fps",
        "10.0",
        "--follow-speed-kmh",
        str(args.follow_speed_kmh),
        "--follow-max-seconds",
        str(args.evaluation_timeout_seconds if args.evaluation_mode else args.follow_max_seconds),
        "--follow-finish-distance",
        str(args.finish_distance),
        "--control-dt",
        str(args.control_dt),
        "--render-every-n",
        str(args.render_every_n),
        "--downscale",
        str(args.downscale),
        "--max-gaussians",
        str(args.max_gaussians),
        "--crop-radius",
        str(args.crop_radius),
        "--opacity-threshold",
        str(args.opacity_threshold),
        "--device",
        args.device,
        "--output-dir",
        str(runtime_output),
        "--pose-output",
        str(pose_output),
        "--path-json-output",
        str(path_json),
    ]
    render_pose_source = args.render_pose_source or "actor"
    runtime_cmd += ["--render-pose-source", render_pose_source]
    cameras = args.cameras if args.cameras is not None else [args.camera]
    runtime_cmd += ["--cameras", *[str(camera) for camera in cameras]]
    if args.disable_control_server:
        runtime_cmd.append("--disable-control-server")
    if args.motion_mode == "kinematic":
        runtime_cmd += ["--kinematic-core-replay", "--disable-control-server", "--no-follow-map-z"]
    if args.motion_mode == "control":
        runtime_cmd.append("--wait-for-control-client")
        if not args.evaluation_mode:
            runtime_cmd.append("--no-exit-on-path-end")
    if args.evaluation_mode:
        runtime_cmd += [
            "--terminate-on-collision",
            "--collision-impulse-threshold",
            str(args.collision_impulse_threshold),
            "--collision-output",
            str(collision_output),
            "--episode-result-output",
            str(episode_result_output),
        ]
    if args.pygame:
        runtime_cmd.append("--pygame")
    if args.no_save_frames:
        runtime_cmd.append("--no-save-frames")
    if args.publish_rendered_observations:
        runtime_cmd.append("--publish-rendered-observations")
    if args.publish_expert_state:
        runtime_cmd += ["--publish-expert-state", "--expert-route-points", str(args.expert_route_points)]
    if args.video_output:
        runtime_cmd += ["--video-output", str(args.video_output)]
    if args.max_render_frames > 0:
        runtime_cmd += ["--max-render-frames", str(args.max_render_frames)]
    if args.exit_after_max_render_frames:
        runtime_cmd.append("--exit-after-max-render-frames")
    if args.spectator_follow:
        runtime_cmd.append("--spectator-follow")
    if args.no_follow_map_z:
        runtime_cmd.append("--no-follow-map-z")
    if args.no_mesh_visibility:
        runtime_cmd.append("--no-mesh-visibility")
    if not args.render_dynamic_gaussians:
        runtime_cmd.append("--no-render-dynamic-gaussians")

    if args.disable_control_server or render_pose_source == "reference" or args.motion_mode == "kinematic":
        print("[run] launching CARLA/3DGS runtime")
        return subprocess.run(runtime_cmd).returncode

    if args.motion_mode == "control":
        print("[run] launching CARLA/3DGS runtime; waiting for external controls on "
              f"127.0.0.1:{args.control_port}")
        print("[run] send JSON lines like {\"throttle\":0.25,\"steer\":0.0,\"brake\":0.0}")
        return subprocess.run(runtime_cmd).returncode

    controller_cmd = [
        args.python,
        "-m", "gs_carla.pose_control_client",
        "--scene-package",
        str(scene_package),
        "--control-host",
        "127.0.0.1",
        "--control-port",
        str(args.control_port),
        "--core-start-frame",
        str(args.start_frame),
        "--core-end-frame",
        str(args.end_frame),
        "--control-mode",
        "path",
        "--follow-speed-kmh",
        str(args.follow_speed_kmh),
        "--follow-lookahead-distance",
        str(args.follow_lookahead_distance),
        "--control-dt",
        str(args.control_dt),
        "--steer-gain",
        str(args.steer_gain),
        "--max-steer",
        str(args.max_steer),
    ]

    print("[run] launching CARLA/3DGS runtime")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_fp = runtime_log.open("w", encoding="utf-8")
    controller_fp = None
    runtime = subprocess.Popen(runtime_cmd, stdout=runtime_fp, stderr=subprocess.STDOUT, text=True)
    controller = None
    try:
        if not wait_for_runtime_control_server(runtime, args.runtime_start_timeout, runtime_log):
            return runtime.poll() if runtime.poll() is not None else 1
        print("[run] launching pose controller")
        controller_fp = controller_log.open("w", encoding="utf-8")
        controller = subprocess.Popen(controller_cmd, stdout=controller_fp, stderr=subprocess.STDOUT, text=True)
        runtime_code = runtime.wait()
        if controller.poll() is None:
            controller.terminate()
            controller.wait(timeout=10)
        print(f"[run] runtime log: {runtime_log}")
        print(f"[run] controller log: {controller_log}")
        return runtime_code
    except KeyboardInterrupt:
        print("[run] interrupted; stopping runtime/controller")
        for proc in (controller, runtime):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in (controller, runtime):
            if proc is not None and proc.poll() is None:
                proc.wait(timeout=10)
        return 130
    finally:
        if controller_fp is not None:
            controller_fp.close()
        runtime_fp.close()


if __name__ == "__main__":
    raise SystemExit(main())
