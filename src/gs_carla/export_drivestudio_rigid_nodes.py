#!/usr/bin/env python3
"""Export DriveStudio RigidNodes tensors for the lightweight CARLA/gsplat runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    models = ckpt.get("models", {})
    if "RigidNodes" not in models:
        raise KeyError(f"{args.checkpoint} does not contain models['RigidNodes']")
    rigid = models["RigidNodes"]
    point_ids = rigid.get("point_ids", rigid.get("points_ids"))
    if point_ids is None:
        raise KeyError("RigidNodes checkpoint has neither point_ids nor points_ids")
    exported = {
        "means": rigid["_means"].contiguous(),
        "scales_log": rigid["_scales"].contiguous(),
        "quats_raw": rigid["_quats"].contiguous(),
        "features_dc": rigid["_features_dc"].contiguous(),
        "features_rest": rigid["_features_rest"].contiguous(),
        "opacities_logit": rigid["_opacities"].contiguous(),
        "point_ids": point_ids.long().contiguous(),
        "instances_quats": rigid["instances_quats"].contiguous(),
        "instances_trans": rigid["instances_trans"].contiguous(),
        "instances_fv": rigid["instances_fv"].bool().contiguous(),
        "instances_size": rigid.get("instances_size", torch.empty(0)).contiguous(),
        "source_checkpoint_step": torch.tensor(int(ckpt.get("step", -1))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, args.output)
    metadata = {
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_step": int(ckpt.get("step", -1)),
        "num_rigid_gaussians": int(exported["means"].shape[0]),
        "num_frames": int(exported["instances_fv"].shape[0]),
        "num_instances": int(exported["instances_fv"].shape[1]),
        "frame_index_note": "runtime uses source_frame - processed_origin_mapping_frame, clamped to [0, num_frames-1]",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
