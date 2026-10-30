import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def load_intrinsics(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = values[:4]
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def read_lidar_bin(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Invalid lidar bin: {path}")
    return points.reshape(-1, 4)


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_xyz.astype(np.float64), ones], axis=1)
    return (points_h @ transform.T)[:, :3]


def write_ply(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    points_xyz = points_xyz.astype(np.float32)
    colors_rgb = np.clip(colors_rgb, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points_xyz, colors_rgb):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def discover_frames(scene_dir: Path) -> List[int]:
    lidar_dir = scene_dir / "lidar"
    frame_indices = []

    for path in lidar_dir.glob("*.bin"):
        try:
            frame_indices.append(int(path.stem))
        except ValueError:
            pass

    frame_indices = sorted(frame_indices)
    if not frame_indices:
        raise ValueError(f"No lidar frames found in {lidar_dir}")

    return frame_indices


def build_frame_boxes(scene_dir: Path) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
    info_path = scene_dir / "instances" / "instances_info.json"
    frame_boxes: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

    if not info_path.exists():
        return frame_boxes

    with open(info_path, "r", encoding="utf-8") as f:
        instances_info = json.load(f)

    for instance in instances_info.values():
        ann = instance["frame_annotations"]

        frame_indices = ann["frame_idx"]
        obj_to_world_list = ann["obj_to_world"]
        box_size_list = ann["box_size"]

        for frame_idx, obj_to_world, box_size in zip(
            frame_indices,
            obj_to_world_list,
            box_size_list,
        ):
            frame_idx = int(frame_idx)
            obj_to_world = np.asarray(obj_to_world, dtype=np.float64).reshape(4, 4)
            box_size = np.asarray(box_size, dtype=np.float64).reshape(3)
            frame_boxes.setdefault(frame_idx, []).append((obj_to_world, box_size))

    return frame_boxes


def remove_points_inside_boxes_world(
    points_world: np.ndarray,
    boxes: List[Tuple[np.ndarray, np.ndarray]],
    bbox_margin: float,
) -> np.ndarray:
    keep_mask = np.ones((points_world.shape[0],), dtype=bool)

    if not boxes:
        return keep_mask

    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_world.astype(np.float64), ones], axis=1)

    for obj_to_world, box_size in boxes:
        world_to_obj = np.linalg.inv(obj_to_world)
        points_obj = points_h @ world_to_obj.T
        local_xyz = points_obj[:, :3]

        length, width, height = box_size.tolist()

        half_size = np.array(
            [
                length * 0.5 + bbox_margin,
                width * 0.5 + bbox_margin,
                height * 0.5 + bbox_margin,
            ],
            dtype=np.float64,
        )

        inside = (
            (np.abs(local_xyz[:, 0]) <= half_size[0])
            & (np.abs(local_xyz[:, 1]) <= half_size[1])
            & (np.abs(local_xyz[:, 2]) <= half_size[2])
        )

        keep_mask &= ~inside

    return keep_mask


def color_lidar_by_6cams_nearest_axis(
    lidar_xyz: np.ndarray,
    lidar_to_world: np.ndarray,
    scene_dir: Path,
    frame_idx: int,
    skip_uncolored: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        colors_rgb: [N, 3]
        valid_mask: [N]
    """

    image_dir = scene_dir / "images"
    intrinsics_dir = scene_dir / "intrinsics"
    extrinsics_dir = scene_dir / "extrinsics"

    n = lidar_xyz.shape[0]

    colors_rgb = np.zeros((n, 3), dtype=np.uint8)
    valid_mask = np.zeros((n,), dtype=bool)
    best_score = np.full((n,), np.inf, dtype=np.float64)

    points_world = transform_points(lidar_xyz, lidar_to_world)

    for cam_idx in range(6):
        image_path = image_dir / f"{frame_idx:03d}_{cam_idx}.jpg"
        intrinsic_path = intrinsics_dir / f"{cam_idx}.txt"
        extrinsic_path = extrinsics_dir / f"{frame_idx:03d}_{cam_idx}.txt"

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        k = load_intrinsics(intrinsic_path)
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]

        camera_to_world = load_matrix(extrinsic_path)
        world_to_camera = np.linalg.inv(camera_to_world)

        points_cam = transform_points(points_world, world_to_camera)

        z = points_cam[:, 2]
        valid_depth = z > 1e-6

        u = fx * points_cam[:, 0] / np.maximum(z, 1e-6) + cx
        v = fy * points_cam[:, 1] / np.maximum(z, 1e-6) + cy

        valid_image = (
            valid_depth
            & (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
        )

        if not np.any(valid_image):
            continue

        # nearest optical axis:
        # 使用归一化成像平面半径作为分数，越小越靠近光轴。
        score = ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2

        update_mask = valid_image & (score < best_score)

        if not np.any(update_mask):
            continue

        u_int = np.rint(u[update_mask]).astype(np.int64)
        v_int = np.rint(v[update_mask]).astype(np.int64)

        u_int = np.clip(u_int, 0, w - 1)
        v_int = np.clip(v_int, 0, h - 1)

        colors_rgb[update_mask] = image_rgb[v_int, u_int]
        valid_mask[update_mask] = True
        best_score[update_mask] = score[update_mask]

    if not skip_uncolored:
        colors_rgb[~valid_mask] = np.array([128, 128, 128], dtype=np.uint8)
        valid_mask[:] = True

    return colors_rgb, valid_mask


def voxel_downsample_colored(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0 or points_xyz.shape[0] == 0:
        return points_xyz, colors_rgb

    voxel_idx = np.floor(points_xyz / voxel_size).astype(np.int64)

    _, inverse, counts = np.unique(
        voxel_idx,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    num_voxels = counts.shape[0]

    points_sum = np.zeros((num_voxels, 3), dtype=np.float64)
    colors_sum = np.zeros((num_voxels, 3), dtype=np.float64)

    np.add.at(points_sum, inverse, points_xyz)
    np.add.at(colors_sum, inverse, colors_rgb.astype(np.float64))

    points_ds = points_sum / counts[:, None]
    colors_ds = colors_sum / counts[:, None]

    return points_ds.astype(np.float32), np.clip(colors_ds, 0, 255).astype(np.uint8)


def process_scene(args):
    scene_dir = Path(args.scene_dir).resolve()
    output_path = Path(args.output_path).resolve()

    frame_indices = discover_frames(scene_dir)
    frame_boxes = build_frame_boxes(scene_dir)

    merged_points = []
    merged_colors = []

    for frame_idx in frame_indices:
        lidar_path = scene_dir / "lidar" / f"{frame_idx:03d}.bin"
        lidar_pose_path = scene_dir / "lidar_pose" / f"{frame_idx:03d}.txt"

        lidar_xyzi = read_lidar_bin(lidar_path)
        lidar_xyz = lidar_xyzi[:, :3].astype(np.float64)

        lidar_to_world = load_matrix(lidar_pose_path)
        lidar_world = transform_points(lidar_xyz, lidar_to_world)

        colors_rgb, color_mask = color_lidar_by_6cams_nearest_axis(
            lidar_xyz=lidar_xyz,
            lidar_to_world=lidar_to_world,
            scene_dir=scene_dir,
            frame_idx=frame_idx,
            skip_uncolored=args.skip_uncolored,
        )

        static_mask = remove_points_inside_boxes_world(
            points_world=lidar_world,
            boxes=frame_boxes.get(frame_idx, []),
            bbox_margin=args.bbox_margin,
        )

        final_mask = static_mask & color_mask

        merged_points.append(lidar_world[final_mask].astype(np.float32))
        merged_colors.append(colors_rgb[final_mask].astype(np.uint8))

        print(
            f"[{frame_idx:03d}] "
            f"raw={lidar_xyz.shape[0]} "
            f"colored={int(color_mask.sum())} "
            f"static={int(static_mask.sum())} "
            f"kept={int(final_mask.sum())} "
            f"boxes={len(frame_boxes.get(frame_idx, []))}"
        )

    merged_points = np.concatenate(merged_points, axis=0)
    merged_colors = np.concatenate(merged_colors, axis=0)

    if args.voxel_size > 0:
        merged_points, merged_colors = voxel_downsample_colored(
            merged_points,
            merged_colors,
            args.voxel_size,
        )

    write_ply(output_path, merged_points, merged_colors)

    print(f"Saved: {output_path}")
    print(f"Final points: {merged_points.shape[0]}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--bbox_margin",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--skip_uncolored",
        action="store_true",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    process_scene(parse_args())