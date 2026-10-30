"""Thin adapter between the repository's stable expert API and a PDM-Lite backend.

The upstream/source-specific backend is deliberately injected by entrypoint because
"PDM-Lite" is not a single checkpoint/API in this repository.  See README.md.
"""

from __future__ import annotations

import importlib
from typing import Any


def _load_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"entrypoint must be module:object, got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), object_name)


class PDMLiteAdapter:
    def __init__(self, context: dict[str, Any]):
        config = context.get("expert_config", {})
        spec = config.get("backend_entrypoint")
        if not spec:
            raise ValueError("expert config must define backend_entrypoint=module:factory")
        factory = _load_symbol(str(spec))
        self.backend = factory(context=context, **config.get("backend_kwargs", {}))

    def run_step(self, observation: dict[str, Any]) -> Any:
        packet = observation.get("sensor_packet") or {}
        privileged = observation.get("expert_state") or {}
        expert_input = {
            "step": observation["step"],
            "timestamp_s": float(observation["step"]) * float(observation["dt"]),
            "ego_state": {
                "pose": observation["pose"],
                "speed_mps": observation["speed_mps"],
                "acceleration_mps2": observation.get("acceleration_mps2"),
                "angular_velocity_dps": observation.get("angular_velocity_dps"),
                "compass_rad": observation.get("compass_rad"),
            },
            "route": privileged.get("route", []),
            "tracked_objects": privileged.get("traffic_objects", []),
            "cameras": packet.get("cameras", []),
            "image_paths": packet.get("images", {}),
            "source_frame": observation.get("source_frame"),
        }
        return self.backend.run_step(expert_input)

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()


def create_expert(context: dict[str, Any]) -> PDMLiteAdapter:
    return PDMLiteAdapter(context)
