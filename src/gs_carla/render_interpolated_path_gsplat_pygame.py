#!/usr/bin/env python3
"""Render 3DGS along an interpolated DriveStudio camera path.

This is intentionally separate from render_background_gsplat.py: it reuses the
same gsplat renderer, but feeds interpolated camera_to_world matrices instead of
integer processed frame poses.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from .render_background_gsplat import (
    Profiler,
    load_background,
    load_intrinsics_from_processed,
    load_local_cam_to_world,
    load_sky,
    render,
    resolve_device,
    write_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--sky", type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-pattern", default="path_{index:04d}_cam{camera}.png")
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--pygame", action="store_true", help="Show rendered frames in a pygame window.")
    parser.add_argument("--display-fps", type=float, default=10.0)
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-gaussians", type=int, default=2_000_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--crop-radius", type=float, default=180.0)
    parser.add_argument("--near", type=float, default=0.2)
    parser.add_argument("--far", type=float, default=250.0)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--downscale", type=float, default=2.0)
    parser.add_argument("--background-rgb", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--sh-degree", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def keyframe_ids(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("--frame-step must be positive")
    if end < start:
        raise ValueError("--frame-end must be >= --frame-start")
    ids = list(range(start, end + 1, step))
    if ids[-1] != end:
        ids.append(end)
    return ids


def interpolate_camera_path(poses: list[np.ndarray], samples: int) -> list[np.ndarray]:
    if samples < 2:
        raise ValueError("--samples must be >= 2")
    if len(poses) < 2:
        raise ValueError("Need at least two keyframe poses to interpolate")

    key_t = np.linspace(0.0, 1.0, len(poses), dtype=np.float64)
    out_t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    translations = np.stack([pose[:3, 3] for pose in poses], axis=0)
    interp_trans = np.stack([np.interp(out_t, key_t, translations[:, axis]) for axis in range(3)], axis=1)

    rotations = Rotation.from_matrix(np.stack([pose[:3, :3] for pose in poses], axis=0))
    interp_rots = Slerp(key_t, rotations)(out_t).as_matrix()

    out: list[np.ndarray] = []
    for rot, trans in zip(interp_rots, interp_trans):
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = rot.astype(np.float32)
        mat[:3, 3] = trans.astype(np.float32)
        out.append(mat)
    return out


def maybe_init_pygame(enabled: bool, width: int, height: int):
    if not enabled:
        return None, None
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DriveStudio 3DGS interpolated path")
    return pygame, screen


def show_pygame_frame(pygame, screen, image: np.ndarray, delay_s: float) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    if delay_s > 0:
        time.sleep(delay_s)
    return True


def main() -> None:
    args = parse_args()
    if args.no_save_frames and args.video_output is None and not args.pygame:
        raise ValueError("--no-save-frames needs --video-output or --pygame")

    ids = keyframe_ids(args.frame_start, args.frame_end, args.frame_step)
    key_poses = [load_local_cam_to_world(args.processed_scene, frame, args.camera) for frame in ids]
    path = interpolate_camera_path(key_poses, args.samples)

    device = resolve_device(args.device)
    profiler = Profiler(args.profile, device)
    k, width, height = load_intrinsics_from_processed(args.processed_scene, args.camera, args.downscale)
    profiler.mark("load path camera")
    bg = load_background(args.background, device)
    profiler.mark("load background")
    sky = load_sky(args.sky, device)
    profiler.mark("load sky")

    pygame, screen = maybe_init_pygame(args.pygame, width, height)
    display_delay = 1.0 / args.display_fps if args.display_fps > 0 else 0.0
    video_frames: list[np.ndarray] = []

    start_time = time.perf_counter()
    for index, cam_to_world in enumerate(path):
        image, _ = render(bg, sky, cam_to_world, k, width, height, args, device, profiler)
        if args.video_output is not None:
            video_frames.append(image)
        if args.output_dir is not None and not args.no_save_frames:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out = args.output_dir / args.output_pattern.format(index=index, camera=args.camera)
            Image.fromarray(image).save(out)
        if pygame is not None:
            if not show_pygame_frame(pygame, screen, image, display_delay):
                break
        elapsed = time.perf_counter() - start_time
        print(
            f"[path] sample {index + 1}/{len(path)} "
            f"avg_render_fps={(index + 1) / max(elapsed, 1e-9):.2f}"
        )

    render_elapsed = time.perf_counter() - start_time
    print(
        f"[path] keyframes={ids[0]}-{ids[-1]} keyframe_count={len(ids)} "
        f"samples={len(path)} render_elapsed={render_elapsed:.4f}s "
        f"avg_render_fps={len(path) / max(render_elapsed, 1e-9):.2f}"
    )
    if args.video_output is not None:
        write_video(args.video_output, video_frames, args.video_fps)
        print(f"Wrote {args.video_output}")
    profiler.print()

    if pygame is not None:
        pygame.quit()


if __name__ == "__main__":
    main()
