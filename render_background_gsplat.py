#!/usr/bin/env python3
"""Render exported DriveStudio background gaussians with gsplat."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

python_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")

from gsplat.rendering import rasterization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--sky", type=Path, help="Optional DriveStudio EnvLight cubemap exported as sky_envlight.pth.")
    parser.add_argument("--processed-scene", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--frame-start", type=int, help="First frame for batch rendering.")
    parser.add_argument("--frame-end", type=int, help="Last frame for batch rendering, inclusive.")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-pose", type=Path, help="JSON file containing camera_to_world.")
    parser.add_argument("--intrinsics", type=Path, help="JSON file containing fx, fy, cx, cy, width, height.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Directory for batch-rendered frames.")
    parser.add_argument("--output-pattern", default="frame{frame:03d}_cam{camera}_gsplat.png")
    parser.add_argument("--video-output", type=Path, help="Optional mp4/gif path written from batch-rendered frames.")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--no-save-frames", action="store_true", help="In batch mode, write only --video-output.")
    parser.add_argument("--overlay-output", type=Path)
    parser.add_argument(
        "--render-output-only",
        action="store_true",
        help="Only save --output; skip sky-output and overlay-output even if their paths are provided.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-gaussians", type=int, default=2_000_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--crop-radius", type=float, default=180.0)
    parser.add_argument("--near", type=float, default=0.2)
    parser.add_argument("--far", type=float, default=250.0)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--downscale", type=float, default=2.0)
    parser.add_argument("--background-rgb", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--sky-output", type=Path, help="Optional path to save the rendered sky layer.")
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=None,
        help="Spherical harmonics degree for exported gaussian colors. Defaults to auto-detect from features_rest.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true", help="Print per-stage render timing.")
    return parser.parse_args()


class Profiler:
    def __init__(self, enabled: bool, device: torch.device | None = None) -> None:
        self.enabled = enabled
        self.device = device
        self.start = time.perf_counter()
        self.last = self.start
        self.rows: list[tuple[str, float, float]] = []

    def set_device(self, device: torch.device) -> None:
        self.device = device

    def sync(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        self.sync()
        now = time.perf_counter()
        self.rows.append((name, now - self.last, now - self.start))
        self.last = now

    def print(self) -> None:
        if not self.enabled:
            return
        self.sync()
        now = time.perf_counter()
        total = now - self.start
        print("[profile] timing seconds:")
        for name, delta, cumulative in self.rows:
            print(f"[profile] {name:28s} step={delta:.4f} total={cumulative:.4f}")
        print(f"[profile] {'total':28s} step={total:.4f} total={total:.4f}")


def load_intrinsics_from_processed(scene: Path, cam_id: int, downscale: float) -> tuple[np.ndarray, int, int]:
    values = np.loadtxt(scene / "intrinsics" / f"{cam_id}.txt").reshape(-1)
    fx, fy, cx, cy = values[:4]
    image_path = scene / "images" / f"000_{cam_id}.jpg"
    width, height = Image.open(image_path).size
    return scale_intrinsics(fx, fy, cx, cy, width, height, downscale)


def load_intrinsics_json(path: Path, downscale: float) -> tuple[np.ndarray, int, int]:
    data = json.loads(path.read_text())
    return scale_intrinsics(
        float(data["fx"]),
        float(data["fy"]),
        float(data["cx"]),
        float(data["cy"]),
        int(data["width"]),
        int(data["height"]),
        downscale,
    )


def scale_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    downscale: float,
) -> tuple[np.ndarray, int, int]:
    if downscale <= 0:
        raise ValueError("--downscale must be positive")
    render_w = int(round(width / downscale))
    render_h = int(round(height / downscale))
    sx = render_w / width
    sy = render_h / height
    k = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return k, render_w, render_h


def load_local_cam_to_world(scene: Path, frame: int, cam_id: int) -> np.ndarray:
    first_front = np.loadtxt(scene / "extrinsics" / "000_0.txt").reshape(4, 4)
    raw = np.loadtxt(scene / "extrinsics" / f"{frame:03d}_{cam_id}.txt").reshape(4, 4)
    return np.linalg.inv(first_front) @ raw


def load_camera_pose_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    matrix = data.get("camera_to_world", data.get("camtoworld", data.get("matrix")))
    if matrix is None:
        raise ValueError(f"{path} must contain camera_to_world, camtoworld, or matrix")
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.size != 16:
        raise ValueError(f"{path} camera matrix must contain 16 values")
    return arr.reshape(4, 4)


def batch_frames(args: argparse.Namespace) -> list[int]:
    if args.frame_start is None and args.frame_end is None:
        return [args.frame]
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    start = args.frame if args.frame_start is None else args.frame_start
    end = start if args.frame_end is None else args.frame_end
    if end < start:
        raise ValueError("--frame-end must be >= --frame-start")
    return list(range(start, end + 1, args.frame_step))


def is_batch(args: argparse.Namespace, frames: list[int]) -> bool:
    return len(frames) > 1 or args.output_dir is not None or args.video_output is not None


def output_path_for_frame(args: argparse.Namespace, frame: int, batch: bool) -> Path:
    if not batch:
        if args.output is None:
            raise ValueError("--output is required for single-frame rendering")
        return args.output
    if args.output_dir is not None:
        return args.output_dir / args.output_pattern.format(frame=frame, camera=args.camera)
    if args.output is not None:
        suffix = args.output.suffix or ".png"
        return args.output.with_name(f"{args.output.stem}_{frame:03d}{suffix}")
    raise ValueError("Batch rendering needs --output-dir, --output, or --video-output with --no-save-frames")


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
        return
    except Exception as imageio_error:
        try:
            import cv2

            if not frames:
                raise ValueError("No frames to write")
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if path.suffix.lower() == ".mp4" else "MJPG"))
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not open video writer for {path}")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return
        except Exception as cv2_error:
            raise RuntimeError(
                "Cannot write video in this Python environment. Install one backend, for example:\n"
                "  pip install imageio-ffmpeg\n"
                "or:\n"
                "  pip install 'imageio[ffmpeg]'\n"
                "Optional fallback:\n"
                "  pip install opencv-python\n"
                f"imageio error: {imageio_error}\n"
                f"opencv error: {cv2_error}"
            ) from imageio_error


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise RuntimeError(
            "gsplat rasterization requires CUDA in this environment. "
            "Run this script from a session where torch.cuda.is_available() is true."
        )
    return device


def load_background(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        bg = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        bg = torch.load(path, map_location="cpu")
    return {
        "means": bg["means"].float().to(device),
        "scales": torch.exp(bg["scales_log"].float()).to(device),
        "quats": torch.nn.functional.normalize(bg["quats_raw"].float(), dim=-1).to(device),
        "opacities": torch.sigmoid(bg["opacities_logit"].float()).squeeze(-1).to(device),
        "colors": torch.cat([bg["features_dc"].float()[:, None, :], bg["features_rest"].float()], dim=1).to(device),
    }


def load_sky(path: Path | None, device: torch.device) -> torch.Tensor | None:
    if path is None:
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    base = data["base"] if isinstance(data, dict) else data
    if base.shape[0] != 6 or base.shape[-1] != 3:
        raise ValueError(f"{path} must contain a cubemap tensor shaped [6, R, R, 3], got {tuple(base.shape)}")
    return base.float().to(device).clamp(0.0, 1.0)


def infer_sh_degree(colors: torch.Tensor, requested_degree: int | None) -> int | None:
    if colors.ndim != 3:
        return None
    num_bases = colors.shape[-2]
    if requested_degree is not None:
        if requested_degree < 0:
            raise ValueError("--sh-degree must be non-negative")
        required_bases = (requested_degree + 1) ** 2
        if required_bases > num_bases:
            raise ValueError(
                f"--sh-degree {requested_degree} needs {required_bases} SH bases, "
                f"but gaussian colors only contain {num_bases}"
            )
        return requested_degree

    degree = int(round(num_bases**0.5)) - 1
    if (degree + 1) ** 2 != num_bases:
        raise ValueError(
            f"Cannot infer SH degree from {num_bases} color bases; pass --sh-degree explicitly "
            "or export colors as [N, 3]."
        )
    return degree


def pixel_viewdirs(cam_to_world: torch.Tensor, k: torch.Tensor, width: int, height: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=cam_to_world.device),
        torch.arange(width, dtype=torch.float32, device=cam_to_world.device),
        indexing="ij",
    )
    dirs_cam = torch.stack(
        [
            (xs - k[0, 2] + 0.5) / k[0, 0],
            (ys - k[1, 2] + 0.5) / k[1, 1],
            torch.ones_like(xs),
        ],
        dim=-1,
    )
    dirs_world = dirs_cam.reshape(-1, 3) @ cam_to_world[:3, :3].T
    dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1)
    return dirs_world.reshape(height, width, 3)


def sample_cubemap(sky_base: torch.Tensor, dirs_world: torch.Tensor) -> torch.Tensor:
    """Approximate nvdiffrast cube sampling used by DriveStudio EnvLight."""
    to_opengl = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
        device=dirs_world.device,
    )
    dirs = dirs_world @ to_opengl.T
    x, y, z = dirs.unbind(-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()
    major = torch.maximum(torch.maximum(ax, ay), az).clamp_min(1e-8)
    face_idx = torch.zeros_like(x, dtype=torch.long)
    sc = torch.zeros_like(x)
    tc = torch.zeros_like(x)

    masks = [
        (ax >= ay) & (ax >= az) & (x > 0),
        (ax >= ay) & (ax >= az) & (x <= 0),
        (ay > ax) & (ay >= az) & (y > 0),
        (ay > ax) & (ay >= az) & (y <= 0),
        (az > ax) & (az > ay) & (z > 0),
        (az > ax) & (az > ay) & (z <= 0),
    ]
    values = [
        (0, -z / major, -y / major),
        (1, z / major, -y / major),
        (2, x / major, z / major),
        (3, x / major, -z / major),
        (4, x / major, -y / major),
        (5, -x / major, -y / major),
    ]
    for mask, (face, s, t) in zip(masks, values):
        face_idx[mask] = face
        sc[mask] = s[mask]
        tc[mask] = t[mask]

    sky_chw = sky_base.permute(0, 3, 1, 2).contiguous()
    out = torch.empty((*dirs.shape[:-1], 3), dtype=sky_base.dtype, device=sky_base.device)
    for face in range(6):
        mask = face_idx == face
        if not mask.any():
            continue
        grid = torch.stack([sc[mask], tc[mask]], dim=-1).view(1, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            sky_chw[face : face + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        out[mask] = sampled[0, :, :, 0].T
    return out.clamp(0.0, 1.0)


def filter_gaussians(
    bg: dict[str, torch.Tensor],
    cam_to_world: torch.Tensor,
    opacity_threshold: float,
    crop_radius: float,
    max_gaussians: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    keep = bg["opacities"] > opacity_threshold
    if crop_radius > 0:
        cam_center = cam_to_world[:3, 3]
        keep &= torch.linalg.norm(bg["means"] - cam_center[None, :], dim=-1) < crop_radius
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    if idx.numel() > max_gaussians:
        gen = torch.Generator(device=idx.device).manual_seed(seed)
        idx = idx[torch.randperm(idx.numel(), generator=gen, device=idx.device)[:max_gaussians]]
    return {k: v[idx] for k, v in bg.items()}


@torch.no_grad()
def render(
    bg: dict[str, torch.Tensor],
    sky: torch.Tensor | None,
    cam_to_world_np: np.ndarray,
    k_np: np.ndarray,
    width: int,
    height: int,
    args: argparse.Namespace,
    device: torch.device,
    profiler: Profiler | None = None,
    return_sky: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    cam_to_world = torch.from_numpy(cam_to_world_np).float().to(device)
    k = torch.from_numpy(k_np).float().to(device)
    if profiler:
        profiler.mark("camera tensors")
    subset = filter_gaussians(
        bg,
        cam_to_world,
        args.opacity_threshold,
        args.crop_radius,
        args.max_gaussians,
        args.seed,
    )
    if profiler:
        profiler.mark("filter gaussians")
    if subset["means"].numel() == 0:
        raise RuntimeError("No gaussians left after opacity/distance filtering")
    sh_degree = infer_sh_degree(subset["colors"], args.sh_degree)
    if profiler:
        profiler.mark("infer sh degree")
    renders, alphas, info = rasterization(
        means=subset["means"],
        quats=subset["quats"],
        scales=subset["scales"],
        opacities=subset["opacities"],
        colors=subset["colors"],
        sh_degree=sh_degree,
        viewmats=torch.linalg.inv(cam_to_world)[None, ...],
        Ks=k[None, ...],
        width=width,
        height=height,
        near_plane=args.near,
        far_plane=args.far,
        radius_clip=args.radius_clip,
        backgrounds=None,
        packed=True,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    if profiler:
        profiler.mark("gsplat rasterization")
    rgb_gaussians = renders[0]
    alpha = alphas[0]
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    if sky is not None:
        sky_rgb = sample_cubemap(sky, pixel_viewdirs(cam_to_world, k, width, height))
    else:
        sky_rgb = torch.tensor(args.background_rgb, dtype=torch.float32, device=device).view(1, 1, 3).expand(height, width, 3)
    if profiler:
        profiler.mark("sky/background layer")
    image = (rgb_gaussians + sky_rgb * (1.0 - alpha)).clamp(0.0, 1.0)
    tiles = info.get("tiles_per_gauss")
    tiles_desc = f"shape={tuple(tiles.shape)}" if torch.is_tensor(tiles) else str(tiles)
    print(f"Rendered {subset['means'].shape[0]} gaussians; alpha mean={alpha.mean().item():.4f}; tiles={tiles_desc}")
    image_np = (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
    sky_np = (sky_rgb.detach().cpu().numpy() * 255.0).astype(np.uint8) if return_sky and sky is not None else None
    if profiler:
        profiler.mark("gpu to cpu")
    return image_np, sky_np


def main() -> None:
    args = parse_args()
    frames = batch_frames(args)
    batch = is_batch(args, frames)
    if args.no_save_frames and args.video_output is None:
        raise ValueError("--no-save-frames requires --video-output")
    if batch and (args.sky_output or args.overlay_output) and not args.render_output_only:
        raise ValueError("Batch rendering only supports main frame outputs; pass --render-output-only or omit sky/overlay outputs")
    if not args.processed_scene and batch:
        raise ValueError("Batch rendering currently requires --processed-scene")
    profiler = Profiler(args.profile)
    device = resolve_device(args.device)
    profiler.set_device(device)
    profiler.mark("resolve device")
    if args.processed_scene:
        k, width, height = load_intrinsics_from_processed(args.processed_scene, args.camera, args.downscale)
        cam_to_world = load_local_cam_to_world(args.processed_scene, args.frame, args.camera)
    else:
        if not args.camera_pose or not args.intrinsics:
            raise ValueError("Use either --processed-scene or both --camera-pose and --intrinsics")
        k, width, height = load_intrinsics_json(args.intrinsics, args.downscale)
        cam_to_world = load_camera_pose_json(args.camera_pose)
    profiler.mark("load camera")

    bg = load_background(args.background, device)
    profiler.mark("load background")
    sky = load_sky(args.sky, device)
    profiler.mark("load sky")

    video_frames: list[np.ndarray] = []
    frame_start_time = time.perf_counter()
    last_frame_time = frame_start_time
    for index, frame in enumerate(frames):
        if args.processed_scene:
            cam_to_world = load_local_cam_to_world(args.processed_scene, frame, args.camera)
        needs_sky_output = not batch and args.sky_output is not None and not args.render_output_only
        image, sky_image = render(bg, sky, cam_to_world, k, width, height, args, device, profiler, return_sky=needs_sky_output)
        if args.video_output is not None:
            video_frames.append(image)
        if not args.no_save_frames:
            output_path = output_path_for_frame(args, frame, batch)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(output_path)
            profiler.mark(f"save render frame {frame}")
            print(f"Wrote {output_path}")

        if not batch:
            if args.sky_output and sky_image is not None and not args.render_output_only:
                args.sky_output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(sky_image).save(args.sky_output)
                profiler.mark("save sky")
                print(f"Wrote {args.sky_output}")

            if args.overlay_output and not args.render_output_only:
                if not args.processed_scene:
                    raise ValueError("--overlay-output requires --processed-scene")
                src_path = args.processed_scene / "images" / f"{args.frame:03d}_{args.camera}.jpg"
                src = Image.open(src_path).convert("RGB").resize((width, height))
                src_arr = np.asarray(src).astype(np.float32)
                overlay = 0.5 * src_arr + 0.5 * image.astype(np.float32)
                args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(args.overlay_output)
                profiler.mark("save overlay")
                print(f"Wrote {args.overlay_output}")
        if batch:
            now = time.perf_counter()
            frame_elapsed = now - last_frame_time
            elapsed = now - frame_start_time
            last_frame_time = now
            print(
                f"[batch] frame {index + 1}/{len(frames)} "
                f"frame_time={frame_elapsed:.4f}s "
                f"avg_render_fps={(index + 1) / max(elapsed, 1e-9):.2f}"
            )

    render_elapsed = time.perf_counter() - frame_start_time
    if batch:
        duration = len(frames) / args.video_fps if args.video_fps > 0 else 0.0
        print(
            f"[batch] rendered_frames={len(frames)} "
            f"frame_range={frames[0]}-{frames[-1]} "
            f"render_elapsed={render_elapsed:.4f}s "
            f"avg_render_fps={len(frames) / max(render_elapsed, 1e-9):.2f} "
            f"video_fps={args.video_fps:.2f} "
            f"video_duration={duration:.2f}s"
        )
    if args.video_output is not None:
        write_video(args.video_output, video_frames, args.video_fps)
        profiler.mark("write video")
        print(f"Wrote {args.video_output}")
    profiler.print()


if __name__ == "__main__":
    main()
