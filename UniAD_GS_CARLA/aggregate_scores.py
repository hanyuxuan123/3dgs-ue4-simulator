#!/usr/bin/env python3
"""Aggregate multiple CARLA-3DGS episode score JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes = []
    for path in args.inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        score = json.loads(path.read_text(encoding="utf-8"))
        episodes.append(
            {
                "path": str(path.resolve()),
                "success": bool(score["success"]),
                "route_completion_percent": float(score["route_completion_percent"]),
                "driving_score": float(score["driving_score"]),
                "collision_count": int(score["collision_count"]),
                "termination_reason": score.get("termination_reason", "unknown"),
            }
        )
    result = {
        "schema": "carla_3dgs_agent_aggregate_v1",
        "episode_count": len(episodes),
        "success_rate_percent": 100.0 * sum(item["success"] for item in episodes) / len(episodes),
        "mean_route_completion_percent": statistics.fmean(item["route_completion_percent"] for item in episodes),
        "mean_driving_score": statistics.fmean(item["driving_score"] for item in episodes),
        "total_collision_count": sum(item["collision_count"] for item in episodes),
        "definition": "aggregate of lightweight CARLA-3DGS scores; not official Bench2Drive",
        "episodes": episodes,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
