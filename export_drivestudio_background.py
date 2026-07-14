#!/usr/bin/env python3
"""Export DriveStudio Background gaussians for CARLA/GS bridge experiments."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--processed-scene", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--ply",
        type=Path,
        help="Optional binary PLY output. Use --ply-max-points to keep this small.",
    )
    parser.add_argument("--ply-max-points", type=int, default=500_000)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--no-sky", action="store_true", help="Do not export models['Sky'] even when it exists.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sh0_to_rgb(features_dc: torch.Tensor) -> torch.Tensor:
    return torch.clamp(features_dc * C0 + 0.5, 0.0, 1.0)


def write_binary_ply(path: Path, data: dict[str, torch.Tensor], max_points: int, seed: int) -> None:
    xyz = data["means"].float().cpu()
    rgb = sh0_to_rgb(data["features_dc"].float().cpu())
    opacity = torch.sigmoid(data["opacities"].float().cpu()).squeeze(-1)
    keep = opacity > data["opacity_threshold"]
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    if idx.numel() > max_points:
        gen = torch.Generator().manual_seed(seed)
        idx = idx[torch.randperm(idx.numel(), generator=gen)[:max_points]]
    xyz = xyz[idx]
    rgb = (rgb[idx] * 255.0).byte()
    opacity = opacity[idx].float()
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float opacity\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        for p, c, a in zip(xyz.tolist(), rgb.tolist(), opacity.tolist()):
            f.write(struct.pack("<fffBBBf", p[0], p[1], p[2], c[0], c[1], c[2], a))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    bg = ckpt["models"]["Background"]
    exported = {
        "means": bg["_means"].contiguous(),
        "scales_log": bg["_scales"].contiguous(),
        "quats_raw": bg["_quats"].contiguous(),
        "features_dc": bg["_features_dc"].contiguous(),
        "features_rest": bg["_features_rest"].contiguous(),
        "opacities_logit": bg["_opacities"].contiguous(),
    }
    out_pth = args.output_dir / "background_gaussians.pth"
    torch.save(exported, out_pth)
    sky_path = None
    if not args.no_sky and "Sky" in ckpt["models"]:
        sky = ckpt["models"]["Sky"]
        if "base" in sky:
            sky_path = args.output_dir / "sky_envlight.pth"
            torch.save({"base": sky["base"].contiguous()}, sky_path)

    means = exported["means"].float()
    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "source_config": str(args.config) if args.config else None,
        "processed_scene": str(args.processed_scene) if args.processed_scene else None,
        "step": int(ckpt.get("step", -1)),
        "num_background_gaussians": int(means.shape[0]),
        "means_min": [float(v) for v in means.min(dim=0).values],
        "means_max": [float(v) for v in means.max(dim=0).values],
        "color_model": "spherical_harmonics_degree_3",
        "scale_activation": "exp(scales_log)",
        "opacity_activation": "sigmoid(opacities_logit)",
        "quaternion_activation": "normalize(quats_raw)",
        "sky_model": "EnvLight cubemap" if sky_path else None,
        "sky_file": str(sky_path) if sky_path else None,
        "coordinate_note": (
            "DriveStudio local world. For ClassLab/NuScenes loaders this is aligned "
            "by inv(extrinsics/<start>_0.txt) @ raw_cam_to_world."
        ),
    }
    if args.processed_scene:
        start_cam = args.processed_scene / "extrinsics" / "000_0.txt"
        if start_cam.exists():
            metadata["T_classlab_camera0_start_to_classlab_world_file"] = str(start_cam)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if args.ply:
        write_binary_ply(
            args.ply,
            {
                "means": exported["means"],
                "features_dc": exported["features_dc"],
                "opacities": exported["opacities_logit"],
                "opacity_threshold": torch.tensor(args.opacity_threshold),
            },
            args.ply_max_points,
            args.seed,
        )
    print(f"Wrote {out_pth}")
    if sky_path:
        print(f"Wrote {sky_path}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")
    if args.ply:
        print(f"Wrote {args.ply}")


if __name__ == "__main__":
    main()
