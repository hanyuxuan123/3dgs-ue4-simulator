#!/usr/bin/env python3
"""Coarse background point projection demo for DriveStudio coordinate checks.

This is not a Gaussian splatting renderer. It projects Gaussian centers with a
depth buffer so we can validate camera/world alignment before wiring gsplat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True, type=Path)
    parser.add_argument("--processed-scene", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overlay-output", type=Path)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.02)
    parser.add_argument("--near", type=float, default=0.2)
    parser.add_argument("--far", type=float, default=200.0)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_intrinsics(scene: Path, cam_id: int, downscale: float) -> tuple[np.ndarray, int, int]:
    values = np.loadtxt(scene / "intrinsics" / f"{cam_id}.txt")
    fx, fy, cx, cy = values[:4]
    image_path = scene / "images" / f"000_{cam_id}.jpg"
    width, height = Image.open(image_path).size
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


def sample_background(data: dict[str, torch.Tensor], max_points: int, opacity_threshold: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = data["means"].float()
    opacities = torch.sigmoid(data["opacities_logit"].float()).squeeze(-1)
    colors = torch.clamp(data["features_dc"].float() * C0 + 0.5, 0.0, 1.0)
    keep = torch.nonzero(opacities > opacity_threshold, as_tuple=False).squeeze(-1)
    if keep.numel() > max_points:
        gen = torch.Generator().manual_seed(seed)
        keep = keep[torch.randperm(keep.numel(), generator=gen)[:max_points]]
    return (
        means[keep].cpu().numpy().astype(np.float32),
        colors[keep].cpu().numpy().astype(np.float32),
        opacities[keep].cpu().numpy().astype(np.float32),
    )


def render_points(
    xyz: np.ndarray,
    colors: np.ndarray,
    cam_to_world: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
    radius: int,
) -> np.ndarray:
    world_to_cam = np.linalg.inv(cam_to_world).astype(np.float32)
    xyz1 = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ xyz1.T).T[:, :3]
    z = cam[:, 2]
    valid = (z > near) & (z < far)
    cam = cam[valid]
    colors = colors[valid]
    z = z[valid]
    uv = (k @ cam.T).T
    u = np.rint(uv[:, 0] / uv[:, 2]).astype(np.int32)
    v = np.rint(uv[:, 1] / uv[:, 2]).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, colors = u[valid], v[valid], z[valid], colors[valid]

    order = np.argsort(z)[::-1]
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu = u[order] + du
            vv = v[order] + dv
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            canvas[vv[inside], uu[inside]] = colors[order][inside]
    return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    bg = torch.load(args.background, map_location="cpu")
    xyz, colors, _ = sample_background(bg, args.max_points, args.opacity_threshold, args.seed)
    k, width, height = load_intrinsics(args.processed_scene, args.camera, args.downscale)
    cam_to_world = load_local_cam_to_world(args.processed_scene, args.frame, args.camera)
    image = render_points(
        xyz,
        colors,
        cam_to_world,
        k,
        width,
        height,
        args.near,
        args.far,
        args.radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f"Wrote {args.output}")

    if args.overlay_output:
        src_path = args.processed_scene / "images" / f"{args.frame:03d}_{args.camera}.jpg"
        src = Image.open(src_path).convert("RGB").resize((width, height))
        src_arr = np.asarray(src).astype(np.float32)
        mask = image.sum(axis=-1, keepdims=True) > 0
        overlay = np.where(mask, 0.55 * src_arr + 0.45 * image.astype(np.float32), src_arr)
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(args.overlay_output)
        print(f"Wrote {args.overlay_output}")


if __name__ == "__main__":
    main()
