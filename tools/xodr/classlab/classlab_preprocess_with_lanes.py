import ast
import bisect
import csv
import json
import math
import os
import re
import shutil
import struct
from typing import Dict, Iterable, List, Optional, Tuple

from datasets.tools.multiprocess_utils import track_parallel_progress


# ===== LANE ADD START: support lane.csv preprocessing =====
SUPPORTED_PROCESS_KEYS = {
    "images",
    "lidar",
    "calib",
    "objects",
    "dynamic_masks",
    "lanes",
}
# ===== LANE ADD END =====

# ClassLab GT type ids mapped to the class names accepted by NuScenesPixelSource.
CLASSLAB_TYPE_TO_NUSCENES_CLASS = {
    1: "human.pedestrian.adult",
    2: "vehicle.bicycle",
    3: "vehicle.car",
    4: "vehicle.truck",
    5: "vehicle.bus.rigid",
    6: "vehicle.truck",
    7: "vehicle.construction",
    8: "vehicle.bicycle",
    10: "animal",
    13: "vehicle.truck",
}

CLASSLAB_NONRIGID_DYNAMIC_CLASSES = {
    "animal",
    "human.pedestrian.adult",
    "vehicle.bicycle",
}

CLASSLAB_RIGID_DYNAMIC_CLASSES = {
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus.rigid",
    "vehicle.construction",
}

CLASSLAB_DYNAMIC_CLASSES = (
    CLASSLAB_NONRIGID_DYNAMIC_CLASSES | CLASSLAB_RIGID_DYNAMIC_CLASSES
)

# Output camera ids follow the NuScenes processed-data convention.
# These defaults can be overridden from datasets/preprocess.py via --camera_topics.
DEFAULT_CAMERA_TOPICS = [
    "front_left_camera",   # 0: front
    "left_camera",         # 1: front_left
    "right_front_camera",  # 2: front_right
    "left_back_camera",    # 3: back_left
    "right_camera",        # 4: back_right
    "back_right_camera",   # 5: back
]

class ClassLabProcessor:
    """Convert ClassLab calibrated sequences to the local NuScenes-style format.

    The converter writes images, lidar, intrinsics, extrinsics, lidar poses, and
    optional tracking instances. Dynamic masks stay out until their schema is
    wired in explicitly.
    """

    def __init__(
        self,
        load_dir: str,
        save_dir: str,
        sequence: Optional[str] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        process_keys: Optional[Iterable[str]] = None,
        workers: int = 1,
        calib_file: Optional[str] = None,
        camera_topics: Optional[List[str]] = None,
        max_sync_time_diff_ns: int = 20_000_000,
        tracking_file: Optional[str] = None,
        tracking_coordinate_frame: str = "auto",
        # ===== LANE ADD START: optional lane.csv path and sampling step =====
        lane_file: Optional[str] = None,
        lane_sample_step: float = 1.0,
        # ===== LANE ADD END =====
        undistort_images: bool = False,
        undistort_alpha: float = 0.0,
    ):
        self.load_dir = os.path.abspath(load_dir)
        self.sequence = sequence
        self.sequence_dir = self._resolve_sequence_dir(self.load_dir, sequence)
        self.sequence_name = os.path.basename(self.sequence_dir.rstrip(os.sep))
        self.save_dir = os.path.abspath(save_dir)
        self.scene_save_dir = os.path.join(self.save_dir, self.sequence_name)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.workers = int(workers)
        self.undistort_images = bool(undistort_images)
        self.undistort_alpha = float(undistort_alpha)
        if not 0.0 <= self.undistort_alpha <= 1.0:
            raise ValueError("undistort_alpha must be in [0, 1]")
        self.calib_file = calib_file or os.path.join(
            self.sequence_dir, "car__2024-12-12.yaml"
        )
        self.tracking_file = tracking_file or os.path.join(
            self.sequence_dir, "result", "global_GT.csv"
        )
        # ===== LANE ADD START: default lane.csv path =====
        self.lane_file = lane_file or os.path.join(
            self.sequence_dir, "result", "lane.csv"
        )
        self.lane_sample_step = float(lane_sample_step)
        if self.lane_sample_step <= 0.0:
            raise ValueError("lane_sample_step must be positive")
        # ===== LANE ADD END =====
        self.tracking_coordinate_frame = _resolve_tracking_coordinate_frame(
            self.tracking_file,
            tracking_coordinate_frame,
        )
        self.camera_topics = [
            _normalize_topic(topic)
            for topic in (camera_topics or DEFAULT_CAMERA_TOPICS)
        ]
        self.max_sync_time_diff_ns = int(max_sync_time_diff_ns)

        if len(self.camera_topics) != 6:
            raise ValueError("--camera_topics must provide exactly 6 topics")

        requested = set(process_keys or ["images", "lidar", "calib"])
        unsupported = requested - SUPPORTED_PROCESS_KEYS
        if unsupported:
            raise ValueError(
                "ClassLab preprocessing currently supports only "
                f"{sorted(SUPPORTED_PROCESS_KEYS)}; unsupported keys: "
                f"{sorted(unsupported)}"
            )
        self.process_keys = requested

        self.raw_dir = os.path.join(self.sequence_dir, "raw")
        self.mapping_pose_path = os.path.join(
            self.sequence_dir, "result", "mapping", "mapping_pose.txt"
        )
        self.sync_path = os.path.join(self.raw_dir, "sync_time_list.txt")
        self.frame_records = self._build_frame_records()

        self.calibration = None
        needs_calibration = (
            "calib" in self.process_keys
            or "lidar" in self.process_keys
            or "dynamic_masks" in self.process_keys
            or (self.undistort_images and "images" in self.process_keys)
        )
        if needs_calibration:
            self.calibration = _load_classlab_calibration(self.calib_file)
            missing_calib = [
                topic for topic in self.camera_topics
                if topic not in self.calibration["cameras"]
            ]
            if missing_calib:
                raise ValueError(
                    f"Calibration is missing camera topics: {missing_calib}"
                )
        self._undistorted_intrinsics = {}
        if (
            ("objects" in self.process_keys or "dynamic_masks" in self.process_keys)
            and not os.path.exists(self.tracking_file)
        ):
            raise FileNotFoundError(f"ClassLab tracking csv not found: {self.tracking_file}")

        # ===== LANE ADD START: lane.csv existence check =====
        if "lanes" in self.process_keys and not os.path.exists(self.lane_file):
            raise FileNotFoundError(f"ClassLab lane csv not found: {self.lane_file}")
        # ===== LANE ADD END =====

    def convert(self):
        print(
            f"Start converting ClassLab sequence {self.sequence_name}: "
            f"{len(self.frame_records)} frames"
        )
        self.create_folder()

        if "calib" in self.process_keys:
            self.save_intrinsics()
        if "objects" in self.process_keys:
            self.save_objects()
        # ===== LANE ADD START: build lane jsons from lane.csv =====
        if "lanes" in self.process_keys:
            self.save_lanes()
        # ===== LANE ADD END =====

        if self.workers > 1 and len(self.frame_records) > 1:
            track_parallel_progress(
                self.convert_frame, list(range(len(self.frame_records))), self.workers
            )
        else:
            for output_frame_idx in range(len(self.frame_records)):
                self.convert_frame(output_frame_idx)

        if "dynamic_masks" in self.process_keys:
            self.save_dynamic_masks()
        print(f"Finished ClassLab conversion: {self.scene_save_dir}")

    def convert_one(self, _=None):
        self.convert()

    def create_folder(self):
        os.makedirs(self.scene_save_dir, exist_ok=True)
        for key, subdir in (
            ("images", "images"),
            ("lidar", "lidar"),
            ("lidar", "lidar_pose"),
            ("calib", "intrinsics"),
            ("calib", "extrinsics"),
            ("objects", "instances"),
            # ===== LANE ADD START: lane output folders =====
            ("lanes", "lanes"),
            # ===== LANE ADD END =====
        ):
            if key in self.process_keys:
                os.makedirs(os.path.join(self.scene_save_dir, subdir), exist_ok=True)
        # ===== LANE ADD START: one compact json per processed frame =====
        if "lanes" in self.process_keys:
            os.makedirs(
                os.path.join(self.scene_save_dir, "lanes", "per_frame"),
                exist_ok=True,
            )
        # ===== LANE ADD END =====
        if "dynamic_masks" in self.process_keys:
            for class_valid in ("all", "human", "vehicle"):
                os.makedirs(
                    os.path.join(self.scene_save_dir, "dynamic_masks", class_valid),
                    exist_ok=True,
                )

    def save_intrinsics(self):
        for cam_idx, topic in enumerate(self.camera_topics):
            camera = self.calibration["cameras"][topic]
            intrinsics, distortion = self._output_camera_model(topic, camera)
            # NuScenesCameraData reads fx, fy, cx, cy, then k1, k2, p1, p2, k3.
            values = [
                intrinsics[0],
                intrinsics[4],
                intrinsics[2],
                intrinsics[5],
                *distortion[:5],
            ]
            _save_vector(
                os.path.join(self.scene_save_dir, "intrinsics", f"{cam_idx}.txt"),
                values,
            )

    def convert_frame(self, output_frame_idx: int):
        record = self.frame_records[output_frame_idx]
        if "images" in self.process_keys:
            self.save_images(record, output_frame_idx)
        if "lidar" in self.process_keys:
            self.save_lidar(record, output_frame_idx)
        if "calib" in self.process_keys:
            self.save_calib(record, output_frame_idx)

    def save_images(self, record: Dict, output_frame_idx: int):
        for cam_idx, topic in enumerate(self.camera_topics):
            src = self._raw_image_path(record, topic)
            dst = os.path.join(
                self.scene_save_dir,
                "images",
                f"{output_frame_idx:03d}_{cam_idx}.jpg",
            )
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing ClassLab image: {src}")
            if self.undistort_images:
                self._save_undistorted_image(src, dst, topic)
            else:
                shutil.copyfile(src, dst)

    def _raw_image_path(self, record: Dict, topic: str) -> str:
        image_ts = record["camera_timestamps"][topic]
        return os.path.join(self.raw_dir, "image", topic, f"{image_ts}.jpeg")

    def _output_camera_model(self, topic: str, camera: Dict) -> Tuple[List[float], List[float]]:
        if not self.undistort_images:
            return camera["intrinsics"], camera["distortion"][:5]

        intrinsic_matrix = self._undistorted_intrinsic_matrix(topic, camera)
        return [
            intrinsic_matrix[0][0],
            0.0,
            intrinsic_matrix[0][2],
            0.0,
            intrinsic_matrix[1][1],
            intrinsic_matrix[1][2],
            0.0,
            0.0,
            1.0,
        ], [0.0, 0.0, 0.0, 0.0, 0.0]

    def _undistorted_intrinsic_matrix(self, topic: str, camera: Dict) -> List[List[float]]:
        if topic not in self._undistorted_intrinsics:
            sample_image = self._raw_image_path(self.frame_records[0], topic)
            image_size = _read_image_size(sample_image)
            self._undistorted_intrinsics[topic] = _compute_undistorted_intrinsic_matrix(
                camera["intrinsics"],
                camera["distortion"],
                image_size,
                self.undistort_alpha,
            )
        return self._undistorted_intrinsics[topic]

    def _save_undistorted_image(self, src: str, dst: str, topic: str):
        import cv2

        camera = self.calibration["cameras"][topic]
        image = cv2.imread(src, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read ClassLab image: {src}")

        camera_matrix = _intrinsic_matrix(camera["intrinsics"])
        distortion = _distortion_array(camera["distortion"])
        new_camera_matrix = self._undistorted_intrinsic_matrix(topic, camera)
        undistorted = cv2.undistort(
            image,
            camera_matrix,
            distortion,
            None,
            new_camera_matrix,
        )
        if not cv2.imwrite(dst, undistorted):
            raise ValueError(f"Unable to write undistorted image: {dst}")

    def save_lidar(self, record: Dict, output_frame_idx: int):
        pcd_path = os.path.join(
            self.raw_dir, "pcd", "middle", f"{record['lidar_timestamp_ns']}.pcd"
        )
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Missing ClassLab lidar pcd: {pcd_path}")

        lidar_points = _read_pcd_xyzi(pcd_path)
        lidar_points.tofile(
            os.path.join(self.scene_save_dir, "lidar", f"{output_frame_idx:03d}.bin")
        )
        lidar_to_world = _matmul(
            record["vehicle_to_world"],
            self.calibration["lidar_to_vehicle"],
        )
        _save_matrix(
            os.path.join(self.scene_save_dir, "lidar_pose", f"{output_frame_idx:03d}.txt"),
            lidar_to_world,
        )

    def save_calib(self, record: Dict, output_frame_idx: int):
        vehicle_to_world = record["vehicle_to_world"]

        for cam_idx, topic in enumerate(self.camera_topics):
            camera_to_vehicle = self.calibration["cameras"][topic]["camera_to_vehicle"]
            # mapping_pose is the vehicle/rear-axle trajectory. Compose camera
            # sensor-to-vehicle calibration directly with it. The calibration
            # camera axes already match the pinhole projection used downstream.
            camera_to_world = _matmul(vehicle_to_world, camera_to_vehicle)
            _save_matrix(
                os.path.join(
                    self.scene_save_dir,
                    "extrinsics",
                    f"{output_frame_idx:03d}_{cam_idx}.txt",
                ),
                camera_to_world,
            )

    def save_objects(self):
        instances_info, frame_instances, stats = _build_instances_from_tracking_csv(
            self.tracking_file,
            {
                record["lidar_frame"]: output_idx
                for output_idx, record in enumerate(self.frame_records)
            },
            {
                record["lidar_frame"]: record["vehicle_to_world"]
                for record in self.frame_records
            },
            self.tracking_coordinate_frame,
        )
        instances_dir = os.path.join(self.scene_save_dir, "instances")
        with open(os.path.join(instances_dir, "instances_info.json"), "w", encoding="utf-8") as fp:
            json.dump(instances_info, fp, indent=4)
        with open(os.path.join(instances_dir, "frame_instances.json"), "w", encoding="utf-8") as fp:
            json.dump(frame_instances, fp, indent=4)
        print(
            "ClassLab objects: "
            f"{len(instances_info)} instances, "
            f"{stats['kept_rows']} boxes kept, "
            f"{stats['skipped_rows']} rows skipped, "
            f"{self.tracking_coordinate_frame} tracking coordinates"
        )


    # ===== LANE ADD START: save lane.csv as processed lane jsons =====
    def save_lanes(self):
        lane_info, stats = _build_lanes_from_lane_csv(
            lane_file=self.lane_file,
            lidar_frame_to_output_idx={
                record["lidar_frame"]: output_idx
                for output_idx, record in enumerate(self.frame_records)
            },
            lidar_frame_to_vehicle_to_world={
                record["lidar_frame"]: record["vehicle_to_world"]
                for record in self.frame_records
            },
            sample_step=self.lane_sample_step,
        )

        lanes_dir = os.path.join(self.scene_save_dir, "lanes")
        per_frame_dir = os.path.join(lanes_dir, "per_frame")
        os.makedirs(per_frame_dir, exist_ok=True)

        with open(os.path.join(lanes_dir, "lane_info.json"), "w", encoding="utf-8") as fp:
            json.dump(lane_info, fp, indent=4)

        for frame_idx_str, frame_lanes in lane_info.items():
            frame_idx = int(frame_idx_str)
            frame_json = {
                "frame_idx": frame_idx,
                "lanes": frame_lanes,
            }
            with open(
                os.path.join(per_frame_dir, f"{frame_idx:03d}.json"),
                "w",
                encoding="utf-8",
            ) as fp:
                json.dump(frame_json, fp, indent=4)

        print(
            "ClassLab lanes: "
            f"{stats['rows']} rows read, "
            f"{stats['kept_lanes']} lanes kept, "
            f"{stats['skipped_lanes']} lanes skipped, "
            f"sample_step={self.lane_sample_step}"
        )
    # ===== LANE ADD END =====

    def save_dynamic_masks(self):
        for class_valid in ("all", "human", "vehicle"):
            self.save_dynamic_mask(class_valid=class_valid)
        print("Processed ClassLab dynamic masks")

    def save_dynamic_mask(self, class_valid: str = "all"):
        if class_valid not in {"all", "human", "vehicle"}:
            raise ValueError("class_valid must be one of all, human, vehicle")

        instances_info = self._load_or_build_instances_info_for_masks()

        frame_objects = {
            output_idx: []
            for output_idx in range(len(self.frame_records))
        }

        for obj_id, obj_info in instances_info.items():
            class_name = obj_info["class_name"]
            if not _classlab_class_valid_for_dynamic_mask(class_name, class_valid):
                continue

            ann = obj_info["frame_annotations"]
            frame_indices = ann["frame_idx"]
            obj_to_world_list = ann["obj_to_world"]
            box_size_list = ann["box_size"]

            for frame_idx, obj_to_world, box_size in zip(
                frame_indices, obj_to_world_list, box_size_list
            ):
                frame_idx = int(frame_idx)
                if frame_idx not in frame_objects:
                    continue

                frame_objects[frame_idx].append(
                    {
                        "obj_id": obj_id,
                        "class_name": class_name,
                        "obj_to_world": obj_to_world,
                        "box_size": box_size,
                    }
                )

        mask_dir = os.path.join(self.scene_save_dir, "dynamic_masks", class_valid)
        os.makedirs(mask_dir, exist_ok=True)

        for output_frame_idx, record in enumerate(self.frame_records):
            for cam_idx, topic in enumerate(self.camera_topics):
                img_path = os.path.join(
                    self.scene_save_dir,
                    "images",
                    f"{output_frame_idx:03d}_{cam_idx}.jpg",
                )
                if not os.path.exists(img_path):
                    raise FileNotFoundError(
                        f"Dynamic mask requires saved image: {img_path}"
                    )

                import cv2
                import numpy as np

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Unable to read image for dynamic mask: {img_path}")

                height, width = img.shape[:2]
                dynamic_mask = np.zeros((height, width), dtype=np.float32)

                camera = self.calibration["cameras"][topic]
                intrinsics, _ = self._output_camera_model(topic, camera)
                camera_matrix = _intrinsic_matrix(intrinsics)

                camera_to_vehicle = camera["camera_to_vehicle"]
                camera_to_world = _matmul(record["vehicle_to_world"], camera_to_vehicle)
                world_to_camera = _invert_transform(camera_to_world)

                for obj in frame_objects[output_frame_idx]:
                    rect = _project_classlab_box_to_image_rect(
                        obj_to_world=obj["obj_to_world"],
                        box_size=obj["box_size"],
                        world_to_camera=world_to_camera,
                        camera_matrix=camera_matrix,
                        image_width=width,
                        image_height=height,
                    )

                    if rect is None:
                        continue

                    x0, y0, x1, y1 = rect

                    dynamic_mask[y0:y1, x0:x1] = np.maximum(
                        dynamic_mask[y0:y1, x0:x1],
                        1.0,
                    )

                dynamic_mask = np.clip((dynamic_mask > 0.0) * 255, 0, 255).astype(np.uint8)

                save_path = os.path.join(
                    mask_dir,
                    f"{output_frame_idx:03d}_{cam_idx}.png",
                )

                ok = cv2.imwrite(save_path, dynamic_mask)
                if not ok:
                    raise ValueError(f"Unable to write dynamic mask: {save_path}")

    def _load_or_build_instances_info_for_masks(self):
        instances_info_path = os.path.join(
            self.scene_save_dir,
            "instances",
            "instances_info.json",
        )

        if os.path.exists(instances_info_path):
            with open(instances_info_path, "r", encoding="utf-8") as fp:
                return json.load(fp)

        instances_info, _, _ = _build_instances_from_tracking_csv(
            self.tracking_file,
            {
                record["lidar_frame"]: output_idx
                for output_idx, record in enumerate(self.frame_records)
            },
            {
                record["lidar_frame"]: record["vehicle_to_world"]
                for record in self.frame_records
            },
            self.tracking_coordinate_frame,
        )

        return instances_info

    def _build_frame_records(self) -> List[Dict]:
        poses = _read_mapping_poses(self.mapping_pose_path)
        sync_rows = _read_sync_rows(self.sync_path)
        sync_timestamps = [row["middle"] for row in sync_rows]

        if not poses:
            raise ValueError(f"No mapping poses found in {self.mapping_pose_path}")
        if not sync_rows:
            raise ValueError(f"No sync rows found in {self.sync_path}")

        frame_ids = sorted(poses)
        start = self.start_frame if self.start_frame is not None else frame_ids[0]
        end = self.end_frame if self.end_frame is not None else frame_ids[-1]
        if end < start:
            raise ValueError("end_frame must be greater than or equal to start_frame")

        records = []
        for lidar_frame in range(start, end + 1):
            if lidar_frame not in poses:
                raise ValueError(
                    f"Requested lidar_frame {lidar_frame} is missing from "
                    f"{self.mapping_pose_path}"
                )
            pose = poses[lidar_frame]
            lidar_ns = int(round(pose["timestamp_sec"] * 1_000_000_000))
            sync_idx = _find_closest_index(sync_timestamps, lidar_ns)
            sync_row = sync_rows[sync_idx]
            sync_diff = abs(sync_row["middle"] - lidar_ns)
            if sync_diff > self.max_sync_time_diff_ns:
                raise ValueError(
                    f"Closest sync row for lidar_frame {lidar_frame} is "
                    f"{sync_diff} ns away, above {self.max_sync_time_diff_ns} ns"
                )

            missing_topics = [
                topic for topic in self.camera_topics if topic not in sync_row
            ]
            if missing_topics:
                raise ValueError(
                    f"sync_time_list.txt is missing camera topics {missing_topics}"
                )

            records.append(
                {
                    "lidar_frame": lidar_frame,
                    "lidar_timestamp_ns": sync_row["middle"],
                    "camera_timestamps": {
                        topic: sync_row[topic] for topic in self.camera_topics
                    },
                    "vehicle_to_world": _transform_from_xyz_rpy(
                        pose["x"],
                        pose["y"],
                        pose["z"],
                        pose["roll"],
                        pose["pitch"],
                        pose["yaw"],
                    ),
                }
            )
        return records

    @staticmethod
    def _resolve_sequence_dir(load_dir: str, sequence: Optional[str]) -> str:
        if sequence:
            sequence_dir = os.path.join(load_dir, sequence)
        else:
            sequence_dir = load_dir
        if not os.path.isdir(sequence_dir):
            raise FileNotFoundError(f"ClassLab sequence directory not found: {sequence_dir}")
        return sequence_dir

def _classlab_class_valid_for_dynamic_mask(class_name: str, class_valid: str) -> bool:
    if class_valid == "all":
        return class_name in CLASSLAB_DYNAMIC_CLASSES
    if class_valid == "human":
        return class_name in CLASSLAB_NONRIGID_DYNAMIC_CLASSES
    if class_valid == "vehicle":
        return class_name in CLASSLAB_RIGID_DYNAMIC_CLASSES
    raise ValueError(f"Invalid class_valid: {class_valid}")


def _project_classlab_box_to_image_rect(
    obj_to_world,
    box_size,
    world_to_camera,
    camera_matrix,
    image_width: int,
    image_height: int,
):
    import numpy as np

    obj_to_world = np.asarray(obj_to_world, dtype=np.float64).reshape(4, 4)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    length, width, height = [float(v) for v in box_size]

    corners_obj = _classlab_box_corners(length, width, height)

    ones = np.ones((8, 1), dtype=np.float64)
    corners_obj_h = np.concatenate([corners_obj, ones], axis=1)

    corners_world_h = corners_obj_h @ obj_to_world.T
    corners_cam_h = corners_world_h @ world_to_camera.T
    corners_cam = corners_cam_h[:, :3]

    near_z = 0.1

    # 3D bbox 的 12 条边，corner 顺序来自 _classlab_box_corners:
    # 0: +x +y +z
    # 1: +x +y -z
    # 2: +x -y +z
    # 3: +x -y -z
    # 4: -x +y +z
    # 5: -x +y -z
    # 6: -x -y +z
    # 7: -x -y -z
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    project_points = []

    # 保留近平面前方的角点。
    front_mask = corners_cam[:, 2] > near_z
    if np.any(front_mask):
        project_points.append(corners_cam[front_mask])

    # 如果 bbox 边穿过近平面，加入交点。
    for i, j in edges:
        p0 = corners_cam[i]
        p1 = corners_cam[j]
        z0 = p0[2]
        z1 = p1[2]

        crosses_near_plane = (z0 - near_z) * (z1 - near_z) < 0
        if not crosses_near_plane:
            continue

        t = (near_z - z0) / (z1 - z0)
        p = p0 + t * (p1 - p0)
        project_points.append(p.reshape(1, 3))

    if not project_points:
        return None

    project_points = np.concatenate(project_points, axis=0)

    z = project_points[:, 2]
    valid = z > near_z
    if not np.any(valid):
        return None

    project_points = project_points[valid]
    z = project_points[:, 2]

    u = camera_matrix[0, 0] * project_points[:, 0] / z + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * project_points[:, 1] / z + camera_matrix[1, 2]

    x0_raw = float(np.min(u))
    x1_raw = float(np.max(u))
    y0_raw = float(np.min(v))
    y1_raw = float(np.max(v))

    # 2D bbox 与图像没有交集。
    if x1_raw < 0:
        return None
    if x0_raw >= image_width:
        return None
    if y1_raw < 0:
        return None
    if y0_raw >= image_height:
        return None

    # clip 到图像范围。
    x0 = int(np.clip(np.floor(x0_raw), 0, image_width - 1))
    x1 = int(np.clip(np.ceil(x1_raw), 0, image_width - 1))
    y0 = int(np.clip(np.floor(y0_raw), 0, image_height - 1))
    y1 = int(np.clip(np.ceil(y1_raw), 0, image_height - 1))

    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1 + 1, y1 + 1


def _classlab_box_corners(length: float, width: float, height: float):
    import numpy as np

    x = length * 0.5
    y = width * 0.5
    z = height * 0.5

    return np.asarray(
        [
            [ x,  y,  z],
            [ x,  y, -z],
            [ x, -y,  z],
            [ x, -y, -z],
            [-x,  y,  z],
            [-x,  y, -z],
            [-x, -y,  z],
            [-x, -y, -z],
        ],
        dtype=np.float64,
    )

def _build_instances_from_tracking_csv(
    tracking_file: str,
    lidar_frame_to_output_idx: Dict[int, int],
    lidar_frame_to_vehicle_to_world: Dict[int, List[List[float]]],
    coordinate_frame: str,
) -> Tuple[Dict, Dict, Dict[str, int]]:
    frame_instances = {
        str(output_idx): []
        for output_idx in range(len(lidar_frame_to_output_idx))
    }
    candidates = {}
    stats = {
        "accepted_rows": 0,
        "duplicate_rows": 0,
        "kept_rows": 0,
        "skipped_rows": 0,
    }

    with open(tracking_file, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        _validate_tracking_columns(reader.fieldnames or [], tracking_file)
        for row in reader:
            candidate = _tracking_row_to_candidate(
                row,
                lidar_frame_to_output_idx,
                lidar_frame_to_vehicle_to_world,
                coordinate_frame,
            )
            if candidate is None:
                stats["skipped_rows"] += 1
                continue
            stats["accepted_rows"] += 1
            key = (candidate["frame_idx"], candidate["track_id"])
            previous = candidates.get(key)
            if previous is not None:
                stats["duplicate_rows"] += 1
            if previous is None or candidate["score"] > previous["score"]:
                candidates[key] = candidate

    instances_by_track = {}
    for candidate in candidates.values():
        instances_by_track.setdefault(candidate["track_id"], []).append(candidate)

    sorted_track_ids = sorted(
        instances_by_track,
        key=lambda track_id: (
            min(item["frame_idx"] for item in instances_by_track[track_id]),
            _numeric_sort_value(track_id),
        ),
    )

    instances_info = {}
    for simplified_id, track_id in enumerate(sorted_track_ids):
        track_candidates = sorted(
            instances_by_track[track_id],
            key=lambda item: item["frame_idx"],
        )
        class_name = _majority_class_name(track_candidates)
        frame_annotations = {
            "frame_idx": [],
            "obj_to_world": [],
            "box_size": [],
        }
        for candidate in track_candidates:
            frame_idx = candidate["frame_idx"]
            frame_annotations["frame_idx"].append(frame_idx)
            frame_annotations["obj_to_world"].append(candidate["obj_to_world"])
            frame_annotations["box_size"].append(candidate["box_size"])
            frame_instances[str(frame_idx)].append(simplified_id)

        instances_info[str(simplified_id)] = {
            "id": f"classlab_track_{track_id}",
            "class_name": class_name,
            "frame_annotations": frame_annotations,
        }

    for frame_idx in frame_instances:
        frame_instances[frame_idx].sort()
    stats["kept_rows"] = sum(
        len(instance["frame_annotations"]["frame_idx"])
        for instance in instances_info.values()
    )
    return instances_info, frame_instances, stats



# ===== LANE ADD START: lane.csv parsing and sampling helpers =====
def _build_lanes_from_lane_csv(
    lane_file: str,
    lidar_frame_to_output_idx: Dict[int, int],
    lidar_frame_to_vehicle_to_world: Dict[int, List[List[float]]],
    sample_step: float = 1.0,
) -> Tuple[Dict, Dict[str, int]]:
    """Build processed lane annotations from ClassLab result/lane.csv.

    The output frame index follows the same convention as objects:
    source lidar frame start_frame -> processed frame 000.
    """
    lane_info = {
        str(output_idx): []
        for output_idx in range(len(lidar_frame_to_output_idx))
    }
    stats = {
        "rows": 0,
        "kept_lanes": 0,
        "skipped_lanes": 0,
    }

    with open(lane_file, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        _validate_lane_columns(fieldnames, lane_file)
        lane_prefixes = _get_lane_prefixes(fieldnames)

        for row in reader:
            stats["rows"] += 1
            try:
                source_frame = int(row["frameNum"])
            except (KeyError, ValueError):
                continue

            if source_frame not in lidar_frame_to_output_idx:
                continue

            output_idx = lidar_frame_to_output_idx[source_frame]
            vehicle_to_world = lidar_frame_to_vehicle_to_world[source_frame]

            for prefix in lane_prefixes:
                lane = _parse_lane_group(
                    row=row,
                    prefix=prefix,
                    source_frame=source_frame,
                    output_frame_idx=output_idx,
                    vehicle_to_world=vehicle_to_world,
                    sample_step=sample_step,
                )
                if lane is None:
                    stats["skipped_lanes"] += 1
                    continue
                lane_info[str(output_idx)].append(lane)
                stats["kept_lanes"] += 1

    return lane_info, stats


def _get_lane_prefixes(fieldnames: List[str]) -> List[str]:
    prefixes = []
    for name in fieldnames:
        if name.endswith(".isValid"):
            prefixes.append(name[: -len(".isValid")])
    return prefixes


def _validate_lane_columns(fieldnames: List[str], lane_file: str):
    required_base = ["frameNum", "timeStamp"]
    missing_base = [name for name in required_base if name not in fieldnames]
    if missing_base:
        raise ValueError(f"{lane_file} is missing lane columns: {missing_base}")

    prefixes = _get_lane_prefixes(fieldnames)
    if not prefixes:
        raise ValueError(f"{lane_file} does not contain any *.isValid lane columns")

    required_suffixes = [
        "isValid",
        "fitFarFlag",
        "zeroOrderCoeff",
        "firstOrderCoeff",
        "secondOrderCoeff",
        "thirdOrderCoeffNear",
        "thirdOrderCoeffFar",
        "dividingPoint",
        "rangeN",
        "rangeP",
        "laneWidth",
        "laneType",
    ]
    missing = []
    for prefix in prefixes:
        for suffix in required_suffixes:
            key = f"{prefix}.{suffix}"
            if key not in fieldnames:
                missing.append(key)

    if missing:
        raise ValueError(
            f"{lane_file} is missing lane columns, first missing fields: {missing[:20]}"
        )


def _parse_lane_group(
    row: Dict[str, str],
    prefix: str,
    source_frame: int,
    output_frame_idx: int,
    vehicle_to_world: List[List[float]],
    sample_step: float,
) -> Optional[Dict]:
    if not _safe_bool(row.get(f"{prefix}.isValid", "false")):
        return None

    try:
        fit_far_flag = _safe_bool(row[f"{prefix}.fitFarFlag"])
        a = float(row[f"{prefix}.zeroOrderCoeff"])
        b = float(row[f"{prefix}.firstOrderCoeff"])
        c = float(row[f"{prefix}.secondOrderCoeff"])
        d_near = float(row[f"{prefix}.thirdOrderCoeffNear"])
        d_far = float(row[f"{prefix}.thirdOrderCoeffFar"])
        dividing_point = float(row[f"{prefix}.dividingPoint"])
        range_n = float(row[f"{prefix}.rangeN"])
        range_p = float(row[f"{prefix}.rangeP"])
        lane_width = float(row[f"{prefix}.laneWidth"])
        lane_type = int(float(row[f"{prefix}.laneType"]))
        timestamp = float(row["timeStamp"])
    except (KeyError, ValueError):
        return None

    if range_p <= range_n:
        return None

    points_vehicle = _sample_classlab_lane_points(
        a=a,
        b=b,
        c=c,
        d_near=d_near,
        d_far=d_far,
        dividing_point=dividing_point,
        fit_far_flag=fit_far_flag,
        range_n=range_n,
        range_p=range_p,
        sample_step=sample_step,
    )
    if len(points_vehicle) < 2:
        return None

    points_world = [_transform_point(vehicle_to_world, point) for point in points_vehicle]

    return {
        "name": prefix,
        "source_frame": source_frame,
        "frame_idx": output_frame_idx,
        "time_stamp": timestamp,
        "is_valid": True,
        "fit_far_flag": fit_far_flag,
        "coefficients": {
            "zero_order": a,
            "first_order": b,
            "second_order": c,
            "third_order_near": d_near,
            "third_order_far": d_far,
            "dividing_point": dividing_point,
        },
        "range": [range_n, range_p],
        "lane_width": lane_width,
        "lane_type": lane_type,
        "points_vehicle": points_vehicle,
        "points_world": points_world,
    }


def _sample_classlab_lane_points(
    a: float,
    b: float,
    c: float,
    d_near: float,
    d_far: float,
    dividing_point: float,
    fit_far_flag: bool,
    range_n: float,
    range_p: float,
    sample_step: float,
) -> List[List[float]]:
    num = max(2, int(math.ceil((range_p - range_n) / sample_step)) + 1)
    num = min(num, 512)

    points = []
    for idx in range(num):
        ratio = idx / max(num - 1, 1)
        x = range_n + ratio * (range_p - range_n)
        y = _eval_classlab_lane_y(
            x=x,
            a=a,
            b=b,
            c=c,
            d_near=d_near,
            d_far=d_far,
            dividing_point=dividing_point,
            fit_far_flag=fit_far_flag,
        )
        points.append([float(x), float(y), 0.0])
    return points


def _eval_classlab_lane_y(
    x: float,
    a: float,
    b: float,
    c: float,
    d_near: float,
    d_far: float,
    dividing_point: float,
    fit_far_flag: bool,
) -> float:
    if (not fit_far_flag) or x < dividing_point:
        return a + b * x + c * x * x + d_near * x * x * x

    t = dividing_point
    return (
        a
        + b * x
        + c * x * x
        + d_near * (t * t * t + 3.0 * x * (x - t) * t)
        + d_far * (x - t) * (x - t) * (x - t)
    )


def _transform_point(matrix: List[List[float]], point: List[float]) -> List[float]:
    x, y, z = point
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
# ===== LANE ADD END =====

def _resolve_tracking_coordinate_frame(
    tracking_file: str,
    coordinate_frame: str,
) -> str:
    coordinate_frame = coordinate_frame.lower()
    if coordinate_frame in {"global", "vehicle"}:
        return coordinate_frame
    if coordinate_frame != "auto":
        raise ValueError(
            "tracking_coordinate_frame must be one of auto, global, vehicle"
        )

    basename = os.path.basename(tracking_file)
    if basename.startswith("global_"):
        return "global"
    if basename in {"GT.csv", "GT_ROI.csv"}:
        return "vehicle"
    raise ValueError(
        "Unable to infer tracking coordinate frame from "
        f"{tracking_file}; pass --tracking_coordinate_frame global or vehicle"
    )


def _validate_tracking_columns(fieldnames: List[str], tracking_file: str):
    required = [
        "frame_num",
        "track_id",
        "center.x",
        "center.y",
        "center.z",
        "obj_yaw",
        "height",
        "length",
        "width",
        "type",
    ]
    missing = [field for field in required if field not in fieldnames]
    if missing:
        raise ValueError(f"{tracking_file} is missing tracking columns: {missing}")


def _tracking_row_to_candidate(
    row: Dict[str, str],
    lidar_frame_to_output_idx: Dict[int, int],
    lidar_frame_to_vehicle_to_world: Dict[int, List[List[float]]],
    coordinate_frame: str,
) -> Optional[Dict]:
    try:
        source_frame = int(row["frame_num"])
    except ValueError:
        return None
    if source_frame not in lidar_frame_to_output_idx:
        return None

    class_name = _classlab_type_to_class_name(row["type"])
    if class_name is None:
        return None

    box_size = _read_box_size(row)
    if box_size is None:
        return None

    try:
        center = [
            float(row["center.x"]),
            float(row["center.y"]),
            float(row["center.z"]),
        ]
        yaw = float(row["obj_yaw"])
    except ValueError:
        return None

    track_id = row.get("track_id", "").strip()
    if not track_id:
        track_id = f"{source_frame}_{row.get('id', '').strip()}"

    obj_to_source = _yaw_transform(center, yaw)
    if coordinate_frame == "global":
        obj_to_world = obj_to_source
    elif coordinate_frame == "vehicle":
        obj_to_world = _matmul(
            lidar_frame_to_vehicle_to_world[source_frame],
            obj_to_source,
        )
    else:
        raise ValueError(f"Unsupported tracking coordinate frame: {coordinate_frame}")

    return {
        "frame_idx": lidar_frame_to_output_idx[source_frame],
        "track_id": track_id,
        "class_name": class_name,
        "box_size": box_size,
        "obj_to_world": obj_to_world,
        "score": _safe_float(row.get("type_confidence"), 0.0),
    }


def _classlab_type_to_class_name(type_value: str) -> Optional[str]:
    try:
        type_id = int(type_value)
    except ValueError:
        return None
    return CLASSLAB_TYPE_TO_NUSCENES_CLASS.get(type_id)


def _read_box_size(row: Dict[str, str]) -> Optional[List[float]]:
    try:
        length = float(row["length"])
        width = float(row["width"])
        height = float(row["height"])
    except ValueError:
        return None
    if length <= 0.0 or width <= 0.0 or height <= 0.0:
        return None
    return [length, width, height]


def _yaw_transform(center: List[float], yaw: float) -> List[List[float]]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return [
        [cos_yaw, -sin_yaw, 0.0, center[0]],
        [sin_yaw, cos_yaw, 0.0, center[1]],
        [0.0, 0.0, 1.0, center[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _safe_float(value: Optional[str], default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric_sort_value(value: str) -> Tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _majority_class_name(track_candidates: List[Dict]) -> str:
    counts = {}
    first_seen = {}
    for index, candidate in enumerate(track_candidates):
        class_name = candidate["class_name"]
        counts[class_name] = counts.get(class_name, 0) + 1
        first_seen.setdefault(class_name, index)
    return max(counts, key=lambda name: (counts[name], -first_seen[name]))


def _read_mapping_poses(path: str) -> Dict[int, Dict[str, float]]:
    poses = {}
    with open(path, "r", encoding="utf-8") as fp:
        header = fp.readline().strip().split()
        index = {name: idx for idx, name in enumerate(header)}
        required = [
            "lidar_frame",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "lidar_frame_time",
        ]
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        for line in fp:
            if not line.strip():
                continue
            parts = line.split()
            frame = int(parts[index["lidar_frame"]])
            poses[frame] = {
                "x": float(parts[index["x"]]),
                "y": float(parts[index["y"]]),
                "z": float(parts[index["z"]]),
                "roll": float(parts[index["roll"]]),
                "pitch": float(parts[index["pitch"]]),
                "yaw": float(parts[index["yaw"]]),
                "timestamp_sec": float(parts[index["lidar_frame_time"]]),
            }
    return poses


def _read_sync_rows(path: str) -> List[Dict[str, int]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fp:
        header_line = fp.readline().strip()
        if header_line.startswith("#"):
            header_line = header_line[1:]
        columns = [_normalize_topic(name) for name in header_line.split()]
        for line in fp:
            if not line.strip():
                continue
            values = [int(item) for item in line.split()]
            rows.append(dict(zip(columns, values)))
    return rows


def _load_classlab_calibration(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ClassLab calibration yaml not found: {path}")

    text = open(path, "r", encoding="utf-8").read()
    lidar_block = _extract_list_item_block(text, "lidar", 0)
    lidar_to_vehicle = _transform_from_xyz_rpy(
        *_extract_xyz_rpy(lidar_block)
    )

    cameras = {}
    camera_blocks = _extract_list_blocks(text, "camera")
    for block in camera_blocks:
        topic_match = re.search(r"^\s*-\s*topic:\s*(\S+)", block, re.MULTILINE)
        if not topic_match:
            continue
        topic = _normalize_topic(topic_match.group(1))
        intrinsics = _extract_inline_list(block, "CameraIntMat")
        distortion = _extract_inline_list(block, "DistCoeff")
        camera_to_vehicle = _transform_from_xyz_rpy(*_extract_xyz_rpy(block))
        cameras[topic] = {
            "intrinsics": intrinsics,
            "distortion": distortion,
            "camera_to_vehicle": camera_to_vehicle,
        }

    return {
        "lidar_to_vehicle": lidar_to_vehicle,
        "cameras": cameras,
    }


def _extract_list_item_block(text: str, section: str, index: int) -> str:
    blocks = _extract_list_blocks(text, section)
    if index >= len(blocks):
        raise ValueError(f"Unable to find item {index} in yaml section {section}")
    return blocks[index]


def _extract_list_blocks(text: str, section: str) -> List[str]:
    lines = text.splitlines()
    start = None
    section_indent = None
    for idx, line in enumerate(lines):
        match = re.match(r"^(\s*)" + re.escape(section) + r":\s*$", line)
        if match:
            start = idx + 1
            section_indent = len(match.group(1))
            break
    if start is None:
        raise ValueError(f"Unable to find yaml section {section}")

    end = len(lines)
    for idx in range(start, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= section_indent and not line.lstrip().startswith("-"):
            end = idx
            break

    blocks = []
    current = []
    for line in lines[start:end]:
        if re.match(r"^\s*-\s+", line):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _extract_xyz_rpy(block: str) -> Tuple[float, float, float, float, float, float]:
    return (
        _extract_float(block, "x"),
        _extract_float(block, "y"),
        _extract_float(block, "z"),
        _extract_float(block, "roll"),
        _extract_float(block, "pitch"),
        _extract_float(block, "yaw"),
    )


def _extract_float(block: str, key: str) -> float:
    match = re.search(
        r"^\s*" + re.escape(key) + r":\s*([-+0-9.eE]+)\s*$",
        block,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Unable to find numeric key {key} in calibration block")
    return float(match.group(1))


def _extract_inline_list(block: str, key: str) -> List[float]:
    match = re.search(
        r"^\s*" + re.escape(key) + r":\s*(\[[^\]]+\])\s*$",
        block,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Unable to find list key {key} in calibration block")
    return [float(value) for value in ast.literal_eval(match.group(1))]


def _intrinsic_matrix(intrinsics: List[float]):
    import numpy as np

    return np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)


def _distortion_array(distortion: List[float]):
    import numpy as np

    return np.asarray(distortion, dtype=np.float64).reshape(-1, 1)


def _read_image_size(path: str) -> Tuple[int, int]:
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read ClassLab image: {path}")
    height, width = image.shape[:2]
    return width, height


def _compute_undistorted_intrinsic_matrix(
    intrinsics: List[float],
    distortion: List[float],
    image_size: Tuple[int, int],
    alpha: float,
):
    import cv2
    import numpy as np

    camera_matrix = _intrinsic_matrix(intrinsics)
    distortion_coeffs = _distortion_array(distortion)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coeffs,
        image_size,
        float(alpha),
        image_size,
    )
    return np.asarray(new_camera_matrix, dtype=np.float64)


def _read_pcd_xyzi(path: str):
    import numpy as np

    with open(path, "rb") as fp:
        header_lines = []
        while True:
            line = fp.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA line: {path}")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA"):
                data_type = decoded.split()[1]
                break
        payload = fp.read()

    header = _parse_pcd_header(header_lines)
    points = int(header["POINTS"][0])
    fields = header["FIELDS"]
    sizes = [int(v) for v in header["SIZE"]]
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
    types = header["TYPE"]

    if data_type == "ascii":
        data = np.loadtxt(path, comments="#", skiprows=len(header_lines))
        field_index = {name: idx for idx, name in enumerate(fields)}
        xyzi = data[:, [
            field_index["x"],
            field_index["y"],
            field_index["z"],
            field_index["intensity"],
        ]]
        return _filter_finite_points(xyzi.astype(np.float32, copy=False))

    if data_type == "binary":
        xyzi = _read_binary_pcd_interleaved(payload, fields, sizes, counts, types, points)
        return _filter_finite_points(xyzi)

    if data_type == "binary_compressed":
        if len(payload) < 8:
            raise ValueError(f"Compressed PCD payload is too short: {path}")
        compressed_size, uncompressed_size = struct.unpack_from("<II", payload, 0)
        compressed = payload[8:8 + compressed_size]
        raw = _lzf_decompress(compressed, uncompressed_size)
        xyzi = _read_binary_pcd_fieldwise(raw, fields, sizes, counts, types, points)
        return _filter_finite_points(xyzi)

    raise ValueError(f"Unsupported PCD DATA type {data_type}: {path}")


def _parse_pcd_header(header_lines: List[str]) -> Dict[str, List[str]]:
    header = {}
    for line in header_lines:
        if not line or line.startswith("#") or line.startswith("DATA"):
            continue
        parts = line.split()
        header[parts[0]] = parts[1:]
    for key in ("FIELDS", "SIZE", "TYPE", "POINTS"):
        if key not in header:
            raise ValueError(f"PCD header is missing {key}")
    return header


def _read_binary_pcd_interleaved(payload, fields, sizes, counts, types, points):
    import numpy as np

    dtype_fields = []
    for field, size, count, pcd_type in zip(fields, sizes, counts, types):
        dtype = _numpy_dtype(size, pcd_type)
        dtype_fields.append((field, dtype, count) if count > 1 else (field, dtype))
    data = np.frombuffer(payload, dtype=np.dtype(dtype_fields), count=points)
    return np.stack(
        [data["x"], data["y"], data["z"], data["intensity"]], axis=1
    ).astype(np.float32, copy=False)


def _read_binary_pcd_fieldwise(payload, fields, sizes, counts, types, points):
    import numpy as np

    arrays = {}
    offset = 0
    for field, size, count, pcd_type in zip(fields, sizes, counts, types):
        byte_count = size * count * points
        raw = memoryview(payload)[offset:offset + byte_count]
        dtype = _numpy_dtype(size, pcd_type)
        values = np.frombuffer(raw, dtype=dtype)
        if count > 1:
            values = values.reshape(points, count)
        arrays[field] = values
        offset += byte_count
    return np.stack(
        [arrays["x"], arrays["y"], arrays["z"], arrays["intensity"]], axis=1
    ).astype(np.float32, copy=False)


def _numpy_dtype(size: int, pcd_type: str):
    import numpy as np

    if pcd_type == "F":
        return np.dtype(f"<f{size}")
    if pcd_type == "U":
        return np.dtype(f"<u{size}")
    if pcd_type == "I":
        return np.dtype(f"<i{size}")
    raise ValueError(f"Unsupported PCD field type {pcd_type}")


def _filter_finite_points(points):
    import numpy as np

    finite_mask = np.isfinite(points[:, :3]).all(axis=1)
    return points[finite_mask].astype(np.float32, copy=False)


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    output = bytearray()
    i = 0
    while i < len(data):
        ctrl = data[i]
        i += 1
        if ctrl < 32:
            length = ctrl + 1
            output.extend(data[i:i + length])
            i += length
        else:
            length = ctrl >> 5
            ref_offset = (ctrl & 0x1F) << 8
            if length == 7:
                length += data[i]
                i += 1
            ref_offset += data[i]
            i += 1
            ref_pos = len(output) - ref_offset - 1
            if ref_pos < 0:
                raise ValueError("Invalid LZF back reference")
            length += 2
            for _ in range(length):
                output.append(output[ref_pos])
                ref_pos += 1

    if len(output) != expected_size:
        raise ValueError(
            f"LZF decompressed size mismatch: got {len(output)}, expected {expected_size}"
        )
    return bytes(output)


def _find_closest_index(sorted_values: List[int], target: int) -> int:
    idx = bisect.bisect_left(sorted_values, target)
    if idx == 0:
        return 0
    if idx == len(sorted_values):
        return len(sorted_values) - 1
    before = sorted_values[idx - 1]
    after = sorted_values[idx]
    return idx - 1 if abs(target - before) <= abs(after - target) else idx


def _transform_from_xyz_rpy(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> List[List[float]]:
    import math

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    rotation = _matmul3(_matmul3(rz, ry), rx)
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], x],
        [rotation[1][0], rotation[1][1], rotation[1][2], y],
        [rotation[2][0], rotation[2][1], rotation[2][2], z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def _matmul3(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _invert_transform(matrix: List[List[float]]) -> List[List[float]]:
    rotation = [[matrix[row][col] for col in range(3)] for row in range(3)]
    translation = [matrix[row][3] for row in range(3)]
    rotation_t = [[rotation[col][row] for col in range(3)] for row in range(3)]
    inverse_translation = [
        -sum(rotation_t[row][k] * translation[k] for k in range(3))
        for row in range(3)
    ]
    return [
        [rotation_t[0][0], rotation_t[0][1], rotation_t[0][2], inverse_translation[0]],
        [rotation_t[1][0], rotation_t[1][1], rotation_t[1][2], inverse_translation[1]],
        [rotation_t[2][0], rotation_t[2][1], rotation_t[2][2], inverse_translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _save_matrix(path: str, matrix: List[List[float]]):
    with open(path, "w", encoding="utf-8") as fp:
        for row in matrix:
            fp.write(" ".join(f"{value:.18e}" for value in row) + "\n")


def _save_vector(path: str, values: List[float]):
    with open(path, "w", encoding="utf-8") as fp:
        for value in values:
            fp.write(f"{value:.18e}\n")


def _normalize_topic(topic: str) -> str:
    topic = topic.strip()
    if topic.startswith("/"):
        topic = topic[1:]
    if topic.endswith("/compressed"):
        topic = topic[: -len("/compressed")]
    return topic
