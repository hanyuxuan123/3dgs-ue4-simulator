"""Adapt Bench2DriveZoo's UniAD agent to externally supplied sensor packets."""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Iterable

import numpy as np

from Bench2DriveZoo.team_code.planner import RoutePlanner
from Bench2DriveZoo.team_code.uniad_b2d_agent import UniadAgent


EARTH_RADIUS = 6378137.0


class RoadCommand(IntEnum):
    """Subset of CARLA RoadOption values used by the UniAD route planner."""

    LANEFOLLOW = 4


def carla_xy_to_synthetic_gps(x: float, y: float) -> dict[str, float]:
    """Encode CARLA XY as WGS84-like coordinates with a (0, 0) reference.

    This is the exact inverse of the Mercator conversion used by the original
    Bench2Drive UniAD agent at lat_ref=lon_ref=0.
    """

    lon = math.degrees(float(x) / EARTH_RADIUS)
    lat = math.degrees(2.0 * math.atan(math.exp(-float(y) / EARTH_RADIUS)) - math.pi / 2.0)
    return {"lat": lat, "lon": lon, "z": 0.0}


class GSUniadAgent(UniadAgent):
    """UniAD inference core without Leaderboard sensor ownership.

    The original model setup and ``run_step`` remain unchanged. Only route
    initialization and metric collection are replaced because CARLA/3DGS owns
    the hero actor and sends kinematics over the control socket.
    """

    def __init__(self, config_and_checkpoint: str, route_xy: Iterable[tuple[float, float]]) -> None:
        # Do not call AutonomousAgent.__init__: it queries CarlaDataProvider,
        # while this process deliberately does not own the CARLA world.
        self._global_plan_world_coord = []
        self._global_plan = [
            (carla_xy_to_synthetic_gps(x, y), RoadCommand.LANEFOLLOW)
            for x, y in route_xy
        ]
        if len(self._global_plan) < 2:
            raise ValueError("UniAD needs at least two route points")
        self._external_metric_info: dict[str, Any] = {}
        self.wallclock_t0 = None
        self.sensor_interface = None
        self.hero_actor = None
        self.setup(config_and_checkpoint)

    def _init(self) -> None:
        self.lat_ref = 0.0
        self.lon_ref = 0.0
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=0.0, lon_ref=0.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def set_external_observation(self, observation: dict[str, Any]) -> None:
        self._external_metric_info = {
            "step": int(observation["step"]),
            "speed_mps": float(observation["speed_mps"]),
            "pose": observation["pose"],
            "acceleration_mps2": observation.get("acceleration_mps2", [0.0, 0.0, 0.0]),
            "angular_velocity_dps": observation.get("angular_velocity_dps", [0.0, 0.0, 0.0]),
        }

    def get_metric_info(self) -> dict[str, Any]:
        return dict(self._external_metric_info)


def observation_to_input_data(
    observation: dict[str, Any],
    camera_map: dict[str, int],
    cv2_module: Any,
    image_size: tuple[int, int] = (1600, 900),
) -> dict[str, tuple[int, Any]]:
    """Build the Leaderboard-style input dictionary expected by UniAD."""

    packet = observation.get("sensor_packet")
    if not packet:
        raise ValueError("observation has no rendered sensor_packet yet")
    paths = packet.get("images", {})
    result: dict[str, tuple[int, Any]] = {}
    for sensor_name, camera_id in camera_map.items():
        path = paths.get(str(camera_id))
        if path is None:
            raise KeyError(f"sensor packet has no camera {camera_id} for {sensor_name}")
        image = cv2_module.imread(path, cv2_module.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"failed to read rendered camera image: {path}")
        if (image.shape[1], image.shape[0]) != image_size:
            image = cv2_module.resize(image, image_size, interpolation=cv2_module.INTER_LINEAR)
        result[sensor_name] = (int(packet["render_index"]), image)

    pose = observation["pose"]
    gps = carla_xy_to_synthetic_gps(float(pose["x"]), float(pose["y"]))
    acceleration = np.asarray(observation.get("acceleration_mps2", [0.0, 0.0, 0.0]), dtype=np.float64)
    angular_dps = np.asarray(observation.get("angular_velocity_dps", [0.0, 0.0, 0.0]), dtype=np.float64)
    imu = np.concatenate(
        [
            acceleration,
            np.deg2rad(angular_dps),
            np.asarray([float(observation["compass_rad"])], dtype=np.float64),
        ]
    )
    frame = int(observation["step"])
    result["GPS"] = (frame, np.asarray([gps["lat"], gps["lon"], gps["z"]], dtype=np.float64))
    # The released PID logger calls speed.astype(np.float64), so preserve the
    # NumPy scalar type produced by Leaderboard's speedometer sensor.
    result["SPEED"] = (frame, {"speed": np.float64(observation["speed_mps"])})
    result["IMU"] = (frame, imu)
    # The released UniAD agent reads BEV only for optional visualization/saving.
    result["bev"] = (frame, np.zeros((512, 512, 3), dtype=np.uint8))
    return result
