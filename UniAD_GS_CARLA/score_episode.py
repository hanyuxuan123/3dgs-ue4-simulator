#!/usr/bin/env python3
"""Score one CARLA-3DGS agent episode from runtime termination facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-result", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=0.0,
        help="Multiplicative infraction penalty after any qualifying collision (default: 0).",
    )
    return parser.parse_args()


def score_episode(facts: dict[str, Any], collision_penalty: float = 0.0) -> dict[str, Any]:
    completion = max(0.0, min(100.0, float(facts.get("route_completion_percent", 0.0))))
    collision_count = max(0, int(facts.get("collision_count", 0)))
    penalty = 1.0 if collision_count == 0 else max(0.0, min(1.0, float(collision_penalty)))
    success = bool(facts.get("success", False)) and completion >= 99.999 and collision_count == 0
    return {
        "schema": "carla_3dgs_agent_score_v1",
        "success": success,
        "termination_reason": facts.get("termination_reason", "unknown"),
        "route_completion_percent": completion,
        "infraction_penalty": penalty,
        "driving_score": completion * penalty,
        "collision_count": collision_count,
        "elapsed_time_s": float(facts.get("elapsed_time_s", 0.0)),
        "definition": "lightweight CARLA-3DGS score; not the official Bench2Drive metric",
        "episode_facts": facts,
    }


def main() -> int:
    args = parse_args()
    if not args.episode_result.is_file():
        raise FileNotFoundError(args.episode_result)
    facts = json.loads(args.episode_result.read_text(encoding="utf-8"))
    score = score_episode(facts, args.collision_penalty)
    text = json.dumps(score, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text, end="")
    return 0 if score["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
