"""CPU smoke tests for the two scene-package routes."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def run_scene(dataset: str, stage: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_scene.py"), "--dataset", dataset,
         "--stage", stage, "--", *args],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


class SceneIntegrationTest(unittest.TestCase):
    def test_single_entrypoint_routes_both_datasets(self) -> None:
        expected = {
            ("nuscenes", "xodr"): "tools/xodr/nuscenes_to_xodr.py",
            ("classlab", "xodr"): "tools/xodr/classlab/classlab_mapping_pose_semantic_to_xodr_with_connectors.py",
            ("nuscenes", "run"): "gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene",
            ("classlab", "run"): "gs_carla.scene_runtime_external_control",
        }
        for (dataset, stage), target in expected.items():
            with self.subTest(dataset=dataset, stage=stage):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/run_scene.py"), "--dataset", dataset,
                     "--stage", stage, "--dry-run", "--", "--example", "a b"],
                    cwd=ROOT, check=True, capture_output=True, text=True,
                )
                self.assertIn(target, result.stdout)
                self.assertIn("--example 'a b'", result.stdout)

    def test_classlab_road_network_exports_driving_xodr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            network = root / "road_network_info_local.json"
            network.write_text(json.dumps({
                "road_fills": [{
                    "road_fill_id": 0,
                    "traffic_direction": "with_pose_s",
                    "median_width": 3.6,
                    "centerline_world": [[0, 0, 0], [20, 0, 0], [40, 0, 0]],
                }],
                "metadata": {"coordinate_frame": "test"},
            }))
            output = root / "map.xodr"
            run_scene("classlab", "xodr", "--road-network", str(network), "--output", str(output))
            xml = ET.parse(output).getroot()
            self.assertEqual(xml.tag, "OpenDRIVE")
            self.assertEqual(len(xml.findall("road")), 1)
            self.assertEqual(len(xml.findall(".//lane[@type='driving']")), 2)

    def test_nuscenes_global_prepare_keeps_mapping_and_carla_frames_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            poses = processed / "lidar_pose"
            poses.mkdir(parents=True)
            for frame, x in [(0, 10.0), (1, 12.0)]:
                matrix = np.eye(4)
                matrix[:3, 3] = (x, 5.0, 1.5)
                np.savetxt(poses / f"{frame:03d}.txt", matrix)
            xodr = root / "map.xodr"
            checkpoint = root / "model.pth"
            xodr.write_text("<OpenDRIVE/>")
            checkpoint.touch()
            output = root / "output"
            run_scene(
                "nuscenes", "prepare",
                "--xodr", str(xodr), "--checkpoint", str(checkpoint),
                "--processed-scene", str(processed), "--output-dir", str(output),
                "--start-frame", "0", "--end-frame", "1",
            )
            package = json.loads((output / "scene_package.json").read_text())
            self.assertEqual(package["values"]["core_pose_coordinate_frame"], "carla")
            self.assertEqual(package["values"]["carla_pose_frame"], "opendrive-global")
            with (output / "core_pose_global.csv").open() as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual([float(row["x"]) for row in rows], [10.0, 12.0])
            self.assertEqual([float(row["y"]) for row in rows], [-5.0, -5.0])
            self.assertEqual([float(row["yaw"]) for row in rows], [0.0, 0.0])
            self.assertIn(" 5.0000000000000000e+00 ", (output / "mapping_pose.txt").read_text())

    def test_classlab_package_accepts_traffic_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_path = root / "scene.json"
            run_scene(
                "classlab", "prepare",
                "--output", str(package_path), "--xodr", str(root / "map.xodr"),
                "--background", str(root / "background.pth"),
                "--processed-scene", str(root / "processed"),
                "--mapping-pose", str(root / "mapping_pose.txt"),
                "--core-pose-csv", str(root / "core.csv"),
                "--processed-origin-mapping-frame", "300",
                "--instance-origin-sequence-frame", "315",
                "--instances-info", str(root / "instances_info.json"),
            )
            package = json.loads(package_path.read_text())
            self.assertEqual(package["schema"], "carla_3dgs_scene_package_v1")
            self.assertEqual(package["frames"]["processed_origin_mapping_frame"], 300)
            self.assertEqual(package["frames"]["instance_origin_sequence_frame"], 315)
            self.assertEqual(package["assets"]["instances_info"], str(root / "instances_info.json"))


if __name__ == "__main__":
    unittest.main()
