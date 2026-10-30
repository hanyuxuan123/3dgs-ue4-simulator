from typing import Dict, List
import json
import logging
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.geometry import get_corners, project_camera_points_to_image


logger = logging.getLogger()

# ClassLab processed data follows the local NuScenes-style 6-camera layout.
CAMERA_LIST = [0, 1, 2, 3, 4, 5]
SMPLNODE_CLASSES = [
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
]

# ClassLab camera extrinsics are saved in the OpenCV pinhole camera convention.
OPENCV2DATASET = np.eye(4, dtype=np.float32)


def project_human_boxes(
    scene_dir: str,
    camera_list: List[int],
    save_temp: bool = True,
    verbose: bool = False,
    narrow_width_ratio: float = 0.2,
    fps: int = 12,
) -> Dict[int, Dict]:
    """Project ClassLab 3D pedestrian boxes to each processed camera image.

    The returned collector format matches datasets.tools.postprocess, so the
    common 4D-Humans matching pipeline can be reused for ClassLab scenes.
    """
    images_dir = os.path.join(scene_dir, "images")
    extrinsics_dir = os.path.join(scene_dir, "extrinsics")
    intrinsics_dir = os.path.join(scene_dir, "intrinsics")
    instances_dir = os.path.join(scene_dir, "instances")
    for path in [images_dir, extrinsics_dir, intrinsics_dir, instances_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Path {path} does not exist. Run ClassLab preprocess first."
            )

    save_dir = os.path.join(scene_dir, "humanpose", "temp", "Pedes_GTTracks")
    os.makedirs(save_dir, exist_ok=True)

    if verbose:
        video_dir = os.path.join(save_dir, "vis")
        per_human_img_dir = os.path.join(video_dir, "images")
        os.makedirs(per_human_img_dir, exist_ok=True)

    with open(os.path.join(instances_dir, "frame_instances.json"), "r") as fp:
        frame_infos = json.load(fp)
    with open(os.path.join(instances_dir, "instances_info.json"), "r") as fp:
        instances_meta = json.load(fp)

    frame_items = sorted(frame_infos.items(), key=lambda item: int(item[0]))
    collector_all = {}
    for cam_id in camera_list:
        pkl_path = os.path.join(save_dir, f"{cam_id}.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "r") as fp:
                collector_all[cam_id] = json.load(fp)
            logger.info("Results for camera %s already exists at %s", cam_id, pkl_path)
            continue

        if verbose:
            per_cam_vis_dir = os.path.join(per_human_img_dir, f"{cam_id}")
            os.makedirs(per_cam_vis_dir, exist_ok=True)

        collector = {}
        frames = []
        for frame_id_str, frame_ins_list in frame_items:
            frame_id = int(frame_id_str)
            frame_collector = {
                "gt_bbox": [],
                "extra_data": {
                    "gt_track_id": [],
                    "gt_class": [],
                },
            }

            cam2world = np.loadtxt(
                os.path.join(extrinsics_dir, f"{frame_id:03d}_{cam_id}.txt")
            )
            cam2world = cam2world @ OPENCV2DATASET

            intrinsics = np.loadtxt(os.path.join(intrinsics_dir, f"{cam_id}.txt"))
            fx, fy, cx, cy = intrinsics[:4]
            intrinsic = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )

            image_path = os.path.join(images_dir, f"{frame_id:03d}_{cam_id}.jpg")
            ori_image = cv2.imread(image_path)
            if ori_image is None:
                raise FileNotFoundError(f"Unable to read ClassLab image: {image_path}")
            image = ori_image.copy()
            height, width = image.shape[:2]

            if len(frame_ins_list) > 0:
                image_plotted = image.copy()
                for instance_id in frame_ins_list:
                    instance = instances_meta[str(instance_id)]
                    if instance["class_name"] not in SMPLNODE_CLASSES:
                        continue

                    annotations = instance["frame_annotations"]
                    if frame_id not in annotations["frame_idx"]:
                        continue
                    annotation_index = annotations["frame_idx"].index(frame_id)
                    obj_to_world = np.array(
                        annotations["obj_to_world"][annotation_index]
                    )
                    length, box_width, box_height = annotations["box_size"][
                        annotation_index
                    ]

                    corners = get_corners(length, box_width, box_height)
                    corners_world = (
                        obj_to_world[:3, :3] @ corners + obj_to_world[:3, 3:4]
                    )
                    world2cam = np.linalg.inv(cam2world)
                    corners_cam = (
                        world2cam[:3, :3] @ corners_world + world2cam[:3, 3:4]
                    )
                    cam_points, depth = project_camera_points_to_image(
                        corners_cam.T, intrinsic
                    )

                    x_min, y_min = np.min(cam_points, axis=0)
                    x_max, y_max = np.max(cam_points, axis=0)
                    if narrow_width_ratio > 0.0:
                        projected_width = x_max - x_min
                        x_min += projected_width * narrow_width_ratio
                        x_max -= projected_width * narrow_width_ratio

                    original_area = (x_max - x_min) * (y_max - y_min)
                    x_min, x_max = np.clip(x_min, 0, width), np.clip(x_max, 0, width)
                    y_min, y_max = np.clip(y_min, 0, height), np.clip(y_max, 0, height)
                    clipped_area = (x_max - x_min) * (y_max - y_min)

                    behind = depth.max() < 0
                    too_small = clipped_area < width * height * (0.03) ** 2
                    too_large = clipped_area > width * height / 1.1
                    too_far = (
                        np.linalg.norm(obj_to_world[:3, 3] - cam2world[:3, 3]) > 40
                    )
                    clip_large = (
                        original_area <= 0 or clipped_area / original_area < 1 / 3
                    )
                    if behind or too_small or too_large or too_far or clip_large:
                        continue

                    gt_box = [x_min, y_min, x_max - x_min, y_max - y_min]
                    frame_collector["gt_bbox"].append(gt_box)
                    frame_collector["extra_data"]["gt_track_id"].append(instance_id)
                    frame_collector["extra_data"]["gt_class"].append([0])

                    if verbose:
                        raw_image = cv2.rectangle(
                            ori_image.copy(),
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (0, 255, 0),
                            2,
                        )
                        raw_image_path = os.path.join(
                            per_cam_vis_dir, f"{frame_id}_{instance_id}.jpg"
                        )
                        cv2.imwrite(raw_image_path, raw_image)
                        image_plotted = cv2.rectangle(
                            image,
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (0, 255, 0),
                            2,
                        )

                if verbose:
                    frames.append(image_plotted)
            elif verbose:
                frames.append(ori_image)

            collector[frame_id] = frame_collector

        if verbose and frames:
            output_path = os.path.join(video_dir, f"cam_{cam_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0])
            )
            for frame in tqdm(frames, desc=f"Writing video for camera {cam_id}"):
                out.write(frame)
            out.release()

        if save_temp:
            with open(pkl_path, "w") as fp:
                json.dump(collector, fp)

        collector_all[cam_id] = collector

    return collector_all


def filter_smpl_meta_to_instance_frames(scene_dir: str, smpl_meta: Dict[int, Dict]):
    """Keep SMPL frames only where ClassLab instances have matching bbox frames."""
    instances_path = os.path.join(scene_dir, "instances", "instances_info.json")
    with open(instances_path, "r") as fp:
        instances_info = json.load(fp)

    filtered = {}
    for instance_id, instance_smpl in smpl_meta.items():
        instance_info = instances_info.get(str(instance_id))
        if instance_info is None:
            logger.warning(
                "Dropping SMPL track %s because it has no ClassLab instance.",
                instance_id,
            )
            continue

        valid_mask = instance_smpl["valid_mask"].bool()
        bbox_frame_mask = torch.zeros_like(valid_mask)
        frame_ids = instance_info["frame_annotations"]["frame_idx"]
        for frame_idx in frame_ids:
            if 0 <= frame_idx < len(bbox_frame_mask):
                bbox_frame_mask[frame_idx] = True

        aligned_mask = valid_mask & bbox_frame_mask
        dropped = int(valid_mask.sum().item() - aligned_mask.sum().item())
        if dropped > 0:
            logger.warning(
                "ClassLab SMPL track %s: dropped %d frames without bbox annotations.",
                instance_id,
                dropped,
            )
        if not aligned_mask.any():
            logger.warning(
                "Dropping SMPL track %s because no valid frames remain after bbox alignment.",
                instance_id,
            )
            continue

        instance_smpl["valid_mask"] = aligned_mask
        if "matched_mask" in instance_smpl:
            instance_smpl["matched_mask"] = instance_smpl["matched_mask"] & aligned_mask
        if "must_appear_mask" in instance_smpl:
            instance_smpl["must_appear_mask"] = (
                instance_smpl["must_appear_mask"] & aligned_mask
            )
        if "selected_cam_idx" in instance_smpl:
            instance_smpl["selected_cam_idx"][~aligned_mask] = -1
        filtered[instance_id] = instance_smpl

    return filtered
