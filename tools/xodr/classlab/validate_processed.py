import argparse
import html
import sys
from pathlib import Path
from typing import Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.classlab.classlab_preprocess import (
    DEFAULT_CAMERA_TOPICS,
    _compute_undistorted_intrinsic_matrix,
    _find_closest_index,
    _load_classlab_calibration,
    _normalize_topic,
    _read_image_size,
    _read_mapping_poses,
    _read_pcd_xyzi,
    _read_sync_rows,
    _transform_from_xyz_rpy,
)


CAMERA_LABELS = [
    "front",
    "front_left",
    "front_right",
    "back_left",
    "back_right",
    "back",
]
COLORS = [
    "#111111",
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate ClassLab NuScenes-style processed output and plot lidar/camera "
            "trajectories."
        )
    )
    parser.add_argument(
        "--processed_scene",
        required=True,
        help="Processed scene directory, e.g. data/classlab/processed/2025-07-17-17-33-26",
    )
    parser.add_argument(
        "--raw_sequence_dir",
        default=None,
        help="Raw ClassLab sequence directory. Enables raw calibration and PCD checks.",
    )
    parser.add_argument(
        "--source_start_frame",
        type=int,
        default=None,
        help="Original lidar_frame corresponding to processed frame 000.",
    )
    parser.add_argument(
        "--calib_file",
        default=None,
        help="Calibration yaml path. Defaults to <raw_sequence_dir>/car__2024-12-12.yaml.",
    )
    parser.add_argument(
        "--camera_topics",
        nargs="+",
        default=None,
        help="Camera topics for output ids 0..5, matching preprocessing.",
    )
    parser.add_argument(
        "--plot_path",
        default=None,
        help="Output SVG path. Defaults to <processed_scene>/sensor_trajectories.svg.",
    )
    parser.add_argument(
        "--matrix_atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for matrix/intrinsic comparisons.",
    )
    parser.add_argument(
        "--pcd_atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for raw PCD to processed bin comparisons.",
    )
    parser.add_argument(
        "--max_pcd_checks",
        type=int,
        default=5,
        help="Number of processed frames to compare against raw PCD. Use -1 for all, 0 to skip.",
    )
    parser.add_argument(
        "--undistorted_images",
        action="store_true",
        help="Validate intrinsics produced with ClassLab --undistort_images.",
    )
    parser.add_argument(
        "--undistort_alpha",
        type=float,
        default=0.0,
        help="OpenCV getOptimalNewCameraMatrix alpha used during preprocessing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np

    processed_scene = Path(args.processed_scene)
    if not processed_scene.is_dir():
        raise FileNotFoundError(f"Processed scene does not exist: {processed_scene}")

    camera_topics = [
        _normalize_topic(topic) for topic in (args.camera_topics or DEFAULT_CAMERA_TOPICS)
    ]
    if len(camera_topics) != 6:
        raise ValueError("--camera_topics must provide exactly 6 topics")

    frame_indices = _discover_frame_indices(processed_scene)
    failures = []

    lidar_poses = _load_lidar_poses(processed_scene, frame_indices, np, failures)
    camera_poses = _load_camera_poses(processed_scene, frame_indices, np, failures)

    _validate_lidar_bins(processed_scene, frame_indices, np, failures)

    if args.raw_sequence_dir:
        raw_sequence_dir = Path(args.raw_sequence_dir)
        calib_file = Path(args.calib_file) if args.calib_file else (
            raw_sequence_dir / "car__2024-12-12.yaml"
        )
        calibration = _load_classlab_calibration(str(calib_file))
        _validate_intrinsics(
            processed_scene,
            calibration,
            camera_topics,
            np,
            args.matrix_atol,
            args.undistorted_images,
            args.undistort_alpha,
            failures,
        )
        _validate_extrinsics(
            calibration,
            camera_topics,
            lidar_poses,
            camera_poses,
            np,
            args.matrix_atol,
            failures,
        )
        if args.source_start_frame is None:
            print("skip raw lidar pose / PCD checks: --source_start_frame not provided")
        else:
            _validate_mapping_lidar_poses(
                raw_sequence_dir,
                args.source_start_frame,
                frame_indices,
                lidar_poses,
                calibration,
                np,
                args.matrix_atol,
                failures,
            )
            if args.max_pcd_checks != 0:
                _validate_raw_pcd_conversion(
                    raw_sequence_dir,
                    args.source_start_frame,
                    frame_indices,
                    processed_scene,
                    np,
                    args.max_pcd_checks,
                    args.pcd_atol,
                    failures,
                )
    else:
        print("skip raw calibration / PCD checks: --raw_sequence_dir not provided")

    _print_offset_summary(lidar_poses, camera_poses, np)

    plot_path = Path(args.plot_path) if args.plot_path else (
        processed_scene / "sensor_trajectories.svg"
    )
    _write_svg_plot(plot_path, lidar_poses, camera_poses)
    print(f"trajectory plot: {plot_path}")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)

    print("\nPASSED")


def _discover_frame_indices(processed_scene: Path) -> List[int]:
    pose_dir = processed_scene / "lidar_pose"
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Missing lidar_pose directory: {pose_dir}")
    frame_indices = sorted(int(path.stem) for path in pose_dir.glob("*.txt"))
    if not frame_indices:
        raise ValueError(f"No lidar pose files found in {pose_dir}")
    return frame_indices


def _load_lidar_poses(processed_scene: Path, frame_indices: Sequence[int], np, failures):
    lidar_poses = {}
    for frame_idx in frame_indices:
        path = processed_scene / "lidar_pose" / f"{frame_idx:03d}.txt"
        matrix = _load_matrix(path, np, failures)
        if matrix is not None:
            lidar_poses[frame_idx] = matrix
    print(f"lidar poses: {len(lidar_poses)}/{len(frame_indices)}")
    return lidar_poses


def _load_camera_poses(processed_scene: Path, frame_indices: Sequence[int], np, failures):
    camera_poses = {cam_idx: {} for cam_idx in range(6)}
    for frame_idx in frame_indices:
        for cam_idx in range(6):
            path = processed_scene / "extrinsics" / f"{frame_idx:03d}_{cam_idx}.txt"
            matrix = _load_matrix(path, np, failures)
            if matrix is not None:
                camera_poses[cam_idx][frame_idx] = matrix
    counts = ", ".join(
        f"{cam_idx}:{len(camera_poses[cam_idx])}" for cam_idx in range(6)
    )
    print(f"camera poses: {counts}")
    return camera_poses


def _load_matrix(path: Path, np, failures):
    if not path.exists():
        failures.append(f"missing matrix file: {path}")
        return None
    matrix = np.loadtxt(path)
    if matrix.shape != (4, 4):
        failures.append(f"{path} has shape {matrix.shape}, expected (4, 4)")
        return None
    if not np.all(np.isfinite(matrix)):
        failures.append(f"{path} contains non-finite values")
        return None
    return matrix.astype(np.float64, copy=False)


def _validate_lidar_bins(processed_scene: Path, frame_indices, np, failures):
    checked = 0
    for frame_idx in frame_indices:
        path = processed_scene / "lidar" / f"{frame_idx:03d}.bin"
        if not path.exists():
            failures.append(f"missing lidar bin: {path}")
            continue
        values = np.fromfile(path, dtype=np.float32)
        if values.size % 4 != 0:
            failures.append(f"{path} has {values.size} float32 values, not divisible by 4")
            continue
        points = values.reshape(-1, 4)
        if points.shape[0] == 0:
            failures.append(f"{path} has zero points")
            continue
        if not np.isfinite(points[:, :3]).all():
            failures.append(f"{path} contains non-finite xyz values")
            continue
        checked += 1
    print(f"lidar bins: {checked}/{len(frame_indices)} valid xyzi files")


def _validate_intrinsics(
    processed_scene: Path,
    calibration: Dict,
    camera_topics: Sequence[str],
    np,
    atol: float,
    undistorted_images: bool,
    undistort_alpha: float,
    failures,
):
    max_error = 0.0
    for cam_idx, topic in enumerate(camera_topics):
        path = processed_scene / "intrinsics" / f"{cam_idx}.txt"
        if not path.exists():
            failures.append(f"missing intrinsic file: {path}")
            continue
        actual = np.loadtxt(path).reshape(-1)
        if actual.shape != (9,):
            failures.append(f"{path} has shape {actual.shape}, expected 9 values")
            continue
        camera = calibration["cameras"].get(topic)
        if camera is None:
            failures.append(f"calibration has no camera topic {topic}")
            continue
        image_size = None
        if undistorted_images:
            image_size = _first_processed_image_size(processed_scene, cam_idx)
        expected = _expected_intrinsic_vector(
            camera,
            np,
            undistorted_images,
            image_size,
            undistort_alpha,
        )
        error = float(np.max(np.abs(actual - expected)))
        max_error = max(max_error, error)
        if error > atol:
            failures.append(
                f"intrinsics cam {cam_idx} topic {topic} max error {error:.3e}"
            )
    print(f"intrinsics check max error: {max_error:.3e}")


def _first_processed_image_size(processed_scene: Path, cam_idx: int):
    image_paths = sorted((processed_scene / "images").glob(f"*_{cam_idx}.jpg"))
    if not image_paths:
        raise FileNotFoundError(
            f"No processed images found for camera {cam_idx}; "
            "undistorted intrinsic validation needs image size"
        )
    return _read_image_size(str(image_paths[0]))


def _expected_intrinsic_vector(
    camera: Dict,
    np,
    undistorted_images: bool,
    image_size,
    undistort_alpha: float,
):
    intrinsics = camera["intrinsics"]
    distortion = camera["distortion"]
    if undistorted_images:
        new_intrinsics = _compute_undistorted_intrinsic_matrix(
            intrinsics,
            distortion,
            image_size,
            undistort_alpha,
        )
        return np.asarray(
            [
                new_intrinsics[0, 0],
                new_intrinsics[1, 1],
                new_intrinsics[0, 2],
                new_intrinsics[1, 2],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    # NuScenesPixelSource reads fx, fy, cx, cy, k1, k2, p1, p2, k3.
    return np.asarray(
        [
            intrinsics[0],
            intrinsics[4],
            intrinsics[2],
            intrinsics[5],
            *distortion[:5],
        ],
        dtype=np.float64,
    )


def _validate_extrinsics(
    calibration: Dict,
    camera_topics: Sequence[str],
    lidar_poses: Dict[int, object],
    camera_poses: Dict[int, Dict[int, object]],
    np,
    atol: float,
    failures,
):
    lidar_to_vehicle = np.asarray(calibration["lidar_to_vehicle"], dtype=np.float64)
    vehicle_to_lidar = np.linalg.inv(lidar_to_vehicle)
    max_error = 0.0
    checked = 0
    for frame_idx, lidar_to_world in lidar_poses.items():
        for cam_idx, topic in enumerate(camera_topics):
            actual = camera_poses[cam_idx].get(frame_idx)
            if actual is None:
                failures.append(f"missing extrinsic frame {frame_idx:03d} camera {cam_idx}")
                continue
            camera = calibration["cameras"].get(topic)
            if camera is None:
                failures.append(f"calibration has no camera topic {topic}")
                continue
            camera_to_vehicle = np.asarray(
                camera["camera_to_vehicle"], dtype=np.float64
            )
            expected = (
                lidar_to_world
                @ vehicle_to_lidar
                @ camera_to_vehicle
            )
            error = float(np.max(np.abs(actual - expected)))
            max_error = max(max_error, error)
            checked += 1
            if error > atol:
                failures.append(
                    f"extrinsic frame {frame_idx:03d} cam {cam_idx} "
                    f"topic {topic} max error {error:.3e}"
                )
    print(f"extrinsics check: {checked} matrices, max error {max_error:.3e}")


def _validate_mapping_lidar_poses(
    raw_sequence_dir: Path,
    source_start_frame: int,
    frame_indices: Sequence[int],
    lidar_poses: Dict[int, object],
    calibration: Dict,
    np,
    atol: float,
    failures,
):
    mapping_path = raw_sequence_dir / "result" / "mapping" / "mapping_pose.txt"
    poses = _read_mapping_poses(str(mapping_path))
    lidar_to_vehicle = np.asarray(calibration["lidar_to_vehicle"], dtype=np.float64)
    max_error = 0.0
    checked = 0
    for frame_idx in frame_indices:
        source_lidar_frame = source_start_frame + frame_idx
        pose = poses.get(source_lidar_frame)
        if pose is None:
            failures.append(f"mapping pose missing lidar_frame {source_lidar_frame}")
            continue
        actual = lidar_poses.get(frame_idx)
        if actual is None:
            continue
        vehicle_to_world = np.asarray(
            _transform_from_xyz_rpy(
                pose["x"],
                pose["y"],
                pose["z"],
                pose["roll"],
                pose["pitch"],
                pose["yaw"],
            ),
            dtype=np.float64,
        )
        expected = vehicle_to_world @ lidar_to_vehicle
        error = float(np.max(np.abs(actual - expected)))
        max_error = max(max_error, error)
        checked += 1
        if error > atol:
            failures.append(
                f"lidar_pose frame {frame_idx:03d} source lidar_frame "
                f"{source_lidar_frame} max error {error:.3e}"
            )
    print(
        "mapping vehicle pose -> lidar pose check: "
        f"{checked} matrices, max error {max_error:.3e}"
    )


def _validate_raw_pcd_conversion(
    raw_sequence_dir: Path,
    source_start_frame: int,
    frame_indices: Sequence[int],
    processed_scene: Path,
    np,
    max_pcd_checks: int,
    atol: float,
    failures,
):
    mapping_path = raw_sequence_dir / "result" / "mapping" / "mapping_pose.txt"
    sync_path = raw_sequence_dir / "raw" / "sync_time_list.txt"
    poses = _read_mapping_poses(str(mapping_path))
    sync_rows = _read_sync_rows(str(sync_path))
    sync_timestamps = [row["middle"] for row in sync_rows]

    check_frames = list(frame_indices)
    if max_pcd_checks > 0:
        check_frames = check_frames[:max_pcd_checks]

    max_error = 0.0
    checked = 0
    for frame_idx in check_frames:
        source_lidar_frame = source_start_frame + frame_idx
        pose = poses.get(source_lidar_frame)
        if pose is None:
            failures.append(f"mapping pose missing lidar_frame {source_lidar_frame}")
            continue
        lidar_ns = int(round(pose["timestamp_sec"] * 1_000_000_000))
        sync_idx = _find_closest_index(sync_timestamps, lidar_ns)
        pcd_timestamp = sync_rows[sync_idx]["middle"]
        pcd_path = raw_sequence_dir / "raw" / "pcd" / "middle" / f"{pcd_timestamp}.pcd"
        bin_path = processed_scene / "lidar" / f"{frame_idx:03d}.bin"
        raw_points = _read_pcd_xyzi(str(pcd_path))
        bin_points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        if raw_points.shape != bin_points.shape:
            failures.append(
                f"PCD/bin shape mismatch frame {frame_idx:03d}: "
                f"{raw_points.shape} vs {bin_points.shape}"
            )
            continue
        error = float(np.max(np.abs(raw_points - bin_points))) if raw_points.size else 0.0
        max_error = max(max_error, error)
        checked += 1
        if error > atol:
            failures.append(
                f"PCD/bin max error frame {frame_idx:03d}: {error:.3e}"
            )
    print(f"raw PCD -> bin check: {checked} frames, max error {max_error:.3e}")


def _print_offset_summary(lidar_poses, camera_poses, np):
    if not lidar_poses:
        return
    lidar_positions = {
        frame_idx: pose[:3, 3] for frame_idx, pose in lidar_poses.items()
    }
    print("camera-lidar distance summary:")
    for cam_idx in range(6):
        distances = []
        for frame_idx, camera_pose in camera_poses[cam_idx].items():
            lidar_position = lidar_positions.get(frame_idx)
            if lidar_position is None:
                continue
            distances.append(float(np.linalg.norm(camera_pose[:3, 3] - lidar_position)))
        if distances:
            values = np.asarray(distances, dtype=np.float64)
            print(
                f"  cam {cam_idx} {CAMERA_LABELS[cam_idx]}: "
                f"mean={values.mean():.4f}m std={values.std():.6f}m"
            )


def _write_svg_plot(plot_path: Path, lidar_poses, camera_poses):
    trajectories = [("lidar", [pose[:3, 3] for _, pose in sorted(lidar_poses.items())])]
    for cam_idx in range(6):
        camera_traj = [
            pose[:3, 3] for _, pose in sorted(camera_poses[cam_idx].items())
        ]
        trajectories.append((f"cam{cam_idx} {CAMERA_LABELS[cam_idx]}", camera_traj))

    xy_points = [
        (float(point[0]), float(point[1]))
        for _, trajectory in trajectories
        for point in trajectory
    ]
    if not xy_points:
        raise ValueError("No trajectory points available for plotting")

    min_x = min(point[0] for point in xy_points)
    max_x = max(point[0] for point in xy_points)
    min_y = min(point[1] for point in xy_points)
    max_y = max(point[1] for point in xy_points)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    pad_x = span_x * 0.08
    pad_y = span_y * 0.08
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    width, height = 1200, 900
    left, top, right, bottom = 70, 40, 280, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    def project(point):
        x = (float(point[0]) - min_x) / (max_x - min_x)
        y = (float(point[1]) - min_y) / (max_y - min_y)
        return left + x * plot_w, top + (1.0 - y) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
            'fill="#fafafa" stroke="#d0d0d0"/>'
        ),
        (
            f'<text x="{left}" y="24" font-family="monospace" font-size="16">'
            "ClassLab sensor trajectories, top-down XY"
            "</text>"
        ),
    ]

    for idx, (name, trajectory) in enumerate(trajectories):
        if not trajectory:
            continue
        color = COLORS[idx % len(COLORS)]
        points = " ".join(
            f"{svg_x:.2f},{svg_y:.2f}" for svg_x, svg_y in map(project, trajectory)
        )
        stroke_width = 3 if idx == 0 else 2
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}" stroke-linejoin="round" '
            'stroke-linecap="round"/>'
        )
        start_x, start_y = project(trajectory[0])
        end_x, end_y = project(trajectory[-1])
        parts.append(f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="4" fill="{color}"/>')
        parts.append(
            f'<rect x="{end_x - 4:.2f}" y="{end_y - 4:.2f}" width="8" '
            f'height="8" fill="{color}"/>'
        )
        legend_y = top + 24 + idx * 26
        legend_x = width - right + 40
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 34}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 44}" y="{legend_y + 5}" '
            'font-family="monospace" font-size="14">'
            f"{html.escape(name)}</text>"
        )

    parts.extend(
        [
            (
                f'<text x="{left}" y="{height - 26}" font-family="monospace" '
                f'font-size="12">x: {min_x:.3f} .. {max_x:.3f} m</text>'
            ),
            (
                f'<text x="{left + 260}" y="{height - 26}" font-family="monospace" '
                f'font-size="12">y: {min_y:.3f} .. {max_y:.3f} m</text>'
            ),
            (
                f'<text x="{width - right + 40}" y="{height - 26}" '
                'font-family="monospace" font-size="12">circle=start, square=end</text>'
            ),
            "</svg>",
        ]
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
