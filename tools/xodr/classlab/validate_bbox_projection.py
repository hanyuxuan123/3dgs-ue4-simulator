import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.classlab.classlab_preprocess import DEFAULT_CAMERA_TOPICS


BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project processed ClassLab 3D instance boxes onto camera images."
    )
    parser.add_argument(
        "--processed_scene",
        required=True,
        help="Processed scene directory containing images, intrinsics, extrinsics, instances.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Debug output directory. Defaults to <processed_scene>/debug_bbox.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=None,
        help="Inclusive processed frame start. Defaults to the first available frame.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Inclusive processed frame end. Defaults to the last available frame.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5],
        help="Camera ids to render.",
    )
    parser.add_argument(
        "--max_boxes_per_image",
        type=int,
        default=-1,
        help="Limit drawn boxes per image. Use -1 to draw all visible boxes.",
    )
    parser.add_argument(
        "--near_clip",
        type=float,
        default=0.1,
        help="Minimum positive camera z depth used for projection.",
    )
    parser.add_argument(
        "--apply_distortion",
        action="store_true",
        help="Apply the saved k1,k2,p1,p2,k3 distortion during projection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    from PIL import Image, ImageDraw

    processed_scene = Path(args.processed_scene)
    output_dir = Path(args.output_dir) if args.output_dir else (
        processed_scene / "debug_bbox"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    instances_info, frame_instances = _load_instances(processed_scene)
    annotations = _build_frame_annotation_lookup(instances_info)
    frame_indices = _select_frame_indices(
        frame_instances,
        args.start_frame,
        args.end_frame,
    )
    camera_topics = _camera_topics_for_labels(args.cameras)

    total_images = 0
    total_drawn = 0
    for cam_id in args.cameras:
        cam_output_dir = output_dir / f"cam_{cam_id}"
        cam_output_dir.mkdir(parents=True, exist_ok=True)
        intrinsic, distortion = _load_intrinsic(processed_scene, cam_id, np)
        for frame_idx in frame_indices:
            image_path = processed_scene / "images" / f"{frame_idx:03d}_{cam_id}.jpg"
            extrinsic_path = (
                processed_scene / "extrinsics" / f"{frame_idx:03d}_{cam_id}.txt"
            )
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")
            if not extrinsic_path.exists():
                raise FileNotFoundError(f"Missing extrinsic: {extrinsic_path}")

            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            camera_to_world = np.loadtxt(extrinsic_path)
            world_to_camera = np.linalg.inv(camera_to_world)
            visible_count = _draw_frame_boxes(
                draw,
                image.size,
                frame_idx,
                frame_instances,
                annotations,
                world_to_camera,
                intrinsic,
                distortion,
                args.apply_distortion,
                args.near_clip,
                args.max_boxes_per_image,
                np,
            )
            draw.text((8, 8), f"frame {frame_idx:03d} cam {cam_id}", fill=(255, 255, 0))
            draw.text(
                (8, 24),
                f"topic {camera_topics.get(cam_id, 'unknown')}",
                fill=(255, 255, 0),
            )
            draw.text((8, 40), f"boxes {visible_count}", fill=(255, 255, 0))
            image.save(cam_output_dir / f"{frame_idx:03d}.jpg", quality=95)
            total_images += 1
            total_drawn += visible_count

    print(f"debug bbox output: {output_dir}")
    print(f"rendered images: {total_images}")
    print(f"drawn boxes: {total_drawn}")


def _load_instances(processed_scene: Path):
    instances_dir = processed_scene / "instances"
    with open(instances_dir / "instances_info.json", "r", encoding="utf-8") as fp:
        instances_info = json.load(fp)
    with open(instances_dir / "frame_instances.json", "r", encoding="utf-8") as fp:
        frame_instances = json.load(fp)
    return instances_info, frame_instances


def _camera_topics_for_labels(cameras: Sequence[int]) -> Dict[int, str]:
    labels = {}
    for cam_id in cameras:
        if 0 <= cam_id < len(DEFAULT_CAMERA_TOPICS):
            labels[cam_id] = DEFAULT_CAMERA_TOPICS[cam_id]
        else:
            labels[cam_id] = "unknown"
    return labels


def _build_frame_annotation_lookup(instances_info: Dict) -> Dict[int, Dict[int, Dict]]:
    lookup = {}
    for instance_id, instance_info in instances_info.items():
        simple_id = int(instance_id)
        frame_annotations = instance_info["frame_annotations"]
        for frame_idx, obj_to_world, box_size in zip(
            frame_annotations["frame_idx"],
            frame_annotations["obj_to_world"],
            frame_annotations["box_size"],
        ):
            lookup.setdefault(int(frame_idx), {})[simple_id] = {
                "obj_to_world": obj_to_world,
                "box_size": box_size,
                "class_name": instance_info["class_name"],
                "id": instance_info["id"],
            }
    return lookup


def _select_frame_indices(
    frame_instances: Dict,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> List[int]:
    available = sorted(int(frame_idx) for frame_idx in frame_instances)
    if not available:
        raise ValueError("frame_instances.json does not contain any frames")
    start = available[0] if start_frame is None else start_frame
    end = available[-1] if end_frame is None else end_frame
    return [frame_idx for frame_idx in available if start <= frame_idx <= end]


def _load_intrinsic(processed_scene: Path, cam_id: int, np):
    values = np.loadtxt(processed_scene / "intrinsics" / f"{cam_id}.txt").reshape(-1)
    if values.shape[0] != 9:
        raise ValueError(f"Camera {cam_id} intrinsic must have 9 values")
    intrinsic = np.asarray(
        [
            [values[0], 0.0, values[2]],
            [0.0, values[1], values[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = values[4:9].astype(np.float64)
    return intrinsic, distortion


def _draw_frame_boxes(
    draw,
    image_size: Tuple[int, int],
    frame_idx: int,
    frame_instances: Dict,
    annotations: Dict[int, Dict[int, Dict]],
    world_to_camera,
    intrinsic,
    distortion,
    apply_distortion: bool,
    near_clip: float,
    max_boxes_per_image: int,
    np,
) -> int:
    width, height = image_size
    visible_count = 0
    frame_annotation = annotations.get(frame_idx, {})
    for instance_id in frame_instances.get(str(frame_idx), []):
        annotation = frame_annotation.get(int(instance_id))
        if annotation is None:
            continue
        corners_world = _box_corners_world(
            annotation["obj_to_world"],
            annotation["box_size"],
            np,
        )
        projected = _project_points(
            corners_world,
            world_to_camera,
            intrinsic,
            distortion,
            apply_distortion,
            near_clip,
            np,
        )
        if not _has_visible_point(projected, width, height):
            continue
        color = _instance_color(int(instance_id))
        _draw_projected_box(draw, projected, color)
        label_xy = _label_position(projected)
        if label_xy is not None:
            draw.text(
                label_xy,
                f"{instance_id}:{_short_class(annotation['class_name'])}",
                fill=color,
            )
        visible_count += 1
        if 0 <= max_boxes_per_image <= visible_count:
            break
    return visible_count


def _box_corners_world(obj_to_world, box_size: Sequence[float], np):
    length, width, height = [float(value) for value in box_size]
    local = np.asarray(
        [
            [-length / 2.0, -width / 2.0, -height / 2.0, 1.0],
            [length / 2.0, -width / 2.0, -height / 2.0, 1.0],
            [length / 2.0, width / 2.0, -height / 2.0, 1.0],
            [-length / 2.0, width / 2.0, -height / 2.0, 1.0],
            [-length / 2.0, -width / 2.0, height / 2.0, 1.0],
            [length / 2.0, -width / 2.0, height / 2.0, 1.0],
            [length / 2.0, width / 2.0, height / 2.0, 1.0],
            [-length / 2.0, width / 2.0, height / 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = np.asarray(obj_to_world, dtype=np.float64).reshape(4, 4)
    return (transform @ local.T).T[:, :3]


def _project_points(
    points_world,
    world_to_camera,
    intrinsic,
    distortion,
    apply_distortion: bool,
    near_clip: float,
    np,
):
    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_camera = (world_to_camera @ points_h.T).T[:, :3]
    output = []
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    k1, k2, p1, p2, k3 = distortion
    for point in points_camera:
        x, y, z = point
        if z <= near_clip:
            output.append(None)
            continue
        xn, yn = x / z, y / z
        if apply_distortion:
            r2 = xn * xn + yn * yn
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            x_dist = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
            y_dist = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
            xn, yn = x_dist, y_dist
        output.append((fx * xn + cx, fy * yn + cy))
    return output


def _has_visible_point(projected: Sequence[Optional[Tuple[float, float]]], width: int, height: int):
    for point in projected:
        if point is None:
            continue
        x, y = point
        if 0 <= x < width and 0 <= y < height:
            return True
    return False


def _draw_projected_box(draw, projected, color):
    for start_idx, end_idx in BBOX_EDGES:
        start = projected[start_idx]
        end = projected[end_idx]
        if start is None or end is None:
            continue
        draw.line([start, end], fill=color, width=3)
    for point in projected:
        if point is None:
            continue
        x, y = point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def _label_position(projected) -> Optional[Tuple[float, float]]:
    valid_points = [point for point in projected if point is not None]
    if not valid_points:
        return None
    x = min(point[0] for point in valid_points)
    y = min(point[1] for point in valid_points)
    return x, y


def _instance_color(instance_id: int) -> Tuple[int, int, int]:
    return (
        50 + (37 * instance_id) % 206,
        50 + (91 * instance_id) % 206,
        50 + (151 * instance_id) % 206,
    )


def _short_class(class_name: str) -> str:
    if class_name.startswith("human.pedestrian"):
        return "ped"
    if class_name == "vehicle.bicycle":
        return "bike"
    if class_name.startswith("vehicle."):
        return class_name.split(".")[-1]
    return class_name.split(".")[-1]


if __name__ == "__main__":
    main()
