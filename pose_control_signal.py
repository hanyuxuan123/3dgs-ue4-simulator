#!/usr/bin/env python3
"""Pose-path controller that outputs CARLA VehicleControl commands."""

from __future__ import annotations

import argparse
from typing import Any

import carla_xodr_live_3dgs_bridge_with_instances as base
from load_xodr_town_and_dump_poses import nearest_path_index


class PoseControlSignal:
    """Read the current hero pose and output one control signal."""

    def __init__(self, carla, args: argparse.Namespace, core_poses: list[dict]):
        self.carla = carla
        self.args = args
        self.core_poses = core_poses
        self.nearest_index = 0
        self.pid_state: dict[str, float] = {}

    def step(self, hero) -> dict[str, Any]:
        transform = hero.get_transform()
        self.nearest_index, nearest_dist = nearest_path_index(
            self.core_poses,
            transform.location.x,
            transform.location.y,
            self.nearest_index,
        )
        target_index = base.lookahead_path_index(
            self.core_poses,
            self.nearest_index,
            self.args.follow_lookahead_distance,
        )
        target = self.core_poses[target_index]
        control, speed, lateral_error = base.make_path_follow_control(
            self.carla,
            self.args,
            hero,
            target,
            self.args.control_dt,
            self.pid_state,
        )
        control, base_steer, steer_bias = base.apply_control_mode(self.carla, self.args, control, self.args.max_steer)
        return {
            "control": control,
            "speed": speed,
            "lateral_error": lateral_error,
            "nearest_dist": nearest_dist,
            "nearest_index": self.nearest_index,
            "target_index": target_index,
            "target": target,
            "base_steer": base_steer,
            "steer_bias": steer_bias,
            "source_frame": int(self.core_poses[self.nearest_index]["frame"]),
        }

