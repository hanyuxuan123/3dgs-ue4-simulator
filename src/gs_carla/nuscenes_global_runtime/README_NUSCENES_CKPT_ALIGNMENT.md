# nuScenes checkpoint 与 CARLA/3DGS 对齐说明

> 运行命令见 [nuScenes global runtime](README.md)；项目流程见 [仓库 README](../../../README.md)。

本文说明 `run_nuscenes_global_scene.py` 这条命令如何输入 nuScenes 训练得到的 checkpoint，如何把 nuScenes global、OpenDRIVE/CARLA 和 3DGS 渲染坐标对齐，以及如何调用生成基于 nuScenes 的 CARLA + 3DGS 联合渲染。

对应命令：

```bash
python -m gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir outputs/nuscenes_global_runtime/003_global_control \
  --start-frame 0 \
  --end-frame 48 \
  --cameras 0 \
  --pygame \
  --export-background \
  --motion-mode control
```

## 总体数据流

`run_nuscenes_global_scene.py` 是一个 wrapper，实际分两步：

1. 调 `prepare_nuscenes_global_scene.py` 准备运行包。
   - 读取 `processed-scene/lidar_pose/*.txt`。
   - 写出 `mapping_pose.txt`，保留 nuScenes/global 原始 ego/lidar pose。
   - 写出 `core_pose_global.csv`，转换成 CARLA world 里可 replay/control 的 hero 轨迹。
   - 如传入 `--export-background`，从 checkpoint 导出 `background/background_gaussians.pth` 和 `background/sky_envlight.pth`。
   - 写出 `scene_package.json`，把 XODR、processed scene、mapping pose、core pose、background GS 串起来。

2. 调 `gs_carla.scene_runtime_external_control` 启动 runtime。
   - CARLA 从 `--xodr` 加载 OpenDRIVE world。
   - Runtime spawn 一个 `role_name=hero` 的车辆。
   - 根据 `--motion-mode` 决定 hero 是按轨迹 replay、由外部控制，还是 path-follow。
   - 每个 render step 用当前 hero/reference pose 组合 processed camera 外参，调用 `render_background_gsplat.py::render()` 渲染 3DGS。

## checkpoint 输入如何处理

`--checkpoint` 不直接给 runtime 使用，而是在 prepare 阶段传给：

```bash
gs_carla.export_drivestudio_background
```

该脚本从 checkpoint 中读取：

- `ckpt["models"]["Background"]`
  - `_means` -> `means`
  - `_scales` -> `scales_log`
  - `_quats` -> `quats_raw`
  - `_features_dc` -> `features_dc`
  - `_features_rest` -> `features_rest`
  - `_opacities` -> `opacities_logit`
- `ckpt["models"]["Sky"]["base"]` -> `sky_envlight.pth`

也就是说，当前 nuScenes global runtime 只导出背景 Gaussian 和天空 cubemap。动态物体 Gaussian/RigidNodes 不是这条命令的必需输入。

导出的文件位置：

```text
<output-dir>/background/background_gaussians.pth
<output-dir>/background/sky_envlight.pth
<output-dir>/background/metadata.json
```

注意：`export_drivestudio_background.py` 是通用导出脚本，它的 `metadata.json` 里可能仍写 `DriveStudio local world`。但这个 nuScenes global wrapper 在 `scene_package.json` 中设置：

```json
"gaussian_coordinate_frame": "mapping-absolute"
```

因此 runtime 会把 checkpoint 里的 Gaussian 坐标直接当成 nuScenes/global mapping 坐标使用，不再做旧版 `inv(extrinsics/000_0.txt)` 的 DriveStudio local 原点变换。

## 坐标对齐的核心假设

这套 nuScenes global runtime 的核心假设是：

```text
XODR global XY
processed_scene/lidar_pose/*.txt
processed_scene/extrinsics/*.txt
checkpoint Background Gaussian means
```

都已经处在同一套 nuScenes/global mapping 坐标里。

也就是说，对 3DGS 渲染来说：

```text
T_3dgs_from_abs = I
```

代码位置在 `scene_runtime_external_control.py`：

```python
if args.gaussian_coordinate_frame == "mapping-absolute":
    t_3dgs_from_abs = np.eye(4, dtype=np.float32)
else:
    t_3dgs_from_abs = np.linalg.inv(t_abs_cam_origin)
```

旧版 ClassLab/DriveStudio local 流程会把 absolute mapping world 转到某一帧 camera0 local world；这里不这样做。

## nuScenes/global 到 CARLA 的转换

CARLA 导入 OpenDRIVE 后，地图 Y 轴相对原始 OpenDRIVE/global frame 会翻转。因此 prepare 阶段写 `core_pose_global.csv` 时做了一个固定转换：

```text
x_carla   =  x_global
y_carla   = -y_global
z_carla   =  z_global
yaw_carla = -yaw_global
```

代码在 `prepare_nuscenes_global_scene.py::write_core_pose()`：

```python
carla_x = float(x)
carla_y = -float(y)
carla_yaw = -float(yaw)
```

默认 `--core-yaw-source trajectory` 时，yaw 不直接使用 lidar pose 里的 yaw，而是用相邻 ego 位置的轨迹方向重新估计。这样 CARLA hero 的车头方向更贴合道路运动方向。

`mapping_pose.txt` 不做这个 Y 翻转，它保留原始 global pose：

```text
lidar_frame x y z roll pitch yaw
```

所以这里有两套 pose 文件：

- `mapping_pose.txt`：给 3DGS/reference/camera 外参使用，保持 nuScenes/global。
- `core_pose_global.csv`：给 CARLA hero 使用，已经转成 CARLA world。

## CARLA actor pose 回到 3DGS/global

当 `--render-pose-source actor` 时，runtime 用 CARLA hero 的实时 pose 驱动 3DGS camera。因为 `scene_package.json` 设置了：

```json
"carla_pose_frame": "opendrive-global"
```

所以 runtime 会把 CARLA actor pose 转回原始 global frame：

```text
x_global   =  x_carla
y_global   = -y_carla
yaw_global = -yaw_carla
```

代码在 `scene_runtime_external_control.py::carla_transform_to_opendrive_global_matrix()`。

最终 actor 渲染链为：

```text
T_abs_vehicle =
  carla_transform_to_opendrive_global_matrix(hero_transform)

T_cam_3dgs =
  T_3dgs_from_abs
  @ T_abs_vehicle
  @ T_camera_from_vehicle
```

在 nuScenes global 模式下 `T_3dgs_from_abs = I`，所以实际是：

```text
T_cam_3dgs = T_abs_vehicle @ T_camera_from_vehicle
```

其中 `T_camera_from_vehicle` 来自 processed scene：

```text
T_camera_from_vehicle =
  inv(T_abs_vehicle_origin) @ T_abs_camera_origin
```

代码读取的是：

```text
processed_scene/extrinsics/<processed_origin_frame>_<camera>.txt
```

默认 `processed_origin_frame=0`，camera 由 `--camera` 或 `--cameras` 指定。

## reference pose 与 actor pose 的区别

`--render-pose-source actor`：

- 用 CARLA hero 当前真实 pose 渲染。
- 适合 `--motion-mode control`，因为 throttle/steer/brake 会改变 actor pose。
- 3DGS 内容帧 `source_frame` 只负责时间相关内容，例如动态 Gaussian 或日志匹配。

`--render-pose-source reference`：

- 不使用 CARLA hero 当前 pose。
- 直接用 `mapping_pose.txt` 中的 recorded/global ego pose 渲染。
- 适合先检查 checkpoint、processed extrinsics、相机内参、背景 GS 是否正确对齐。

reference 渲染链为：

```text
T_abs_vehicle = mapping_pose_matrix(mapping_pose, render_frame)
T_cam_3dgs    = T_3dgs_from_abs @ T_abs_vehicle @ T_camera_from_vehicle
```

nuScenes global 下同样因为 `T_3dgs_from_abs = I`，所以不会换到 camera0 local 坐标。

## control 模式如何运行

你的命令使用：

```bash
--motion-mode control
```

wrapper 会给 runtime 增加：

```bash
--wait-for-control-client
--no-exit-on-path-end
--source-frame-mode nearest
```

含义：

- Runtime 启动 CARLA/3DGS 后，在 `127.0.0.1:29001` 开 TCP JSON-lines control server。
- 每个控制 step 先发 observation 给 client，再等待 client 返回控制量。
- 控制量格式：

```json
{"throttle": 0.25, "steer": 0.0, "brake": 0.0}
```

可以用仓库里的测试 client：

```bash
python -m gs_carla.nuscenes_global_runtime.send_control \
  --throttle 0.25 \
  --steer 0.0 \
  --brake 0.0
```

如果要手动连续输入：

```bash
python -m gs_carla.nuscenes_global_runtime.send_control --interactive
```

## 生成的关键输出

以你的 `--output-dir outputs/nuscenes_global_runtime/003_global_control` 为例：

```text
scene_package.json
mapping_pose.txt
core_pose_global.csv
background/background_gaussians.pth
background/sky_envlight.pth
runtime_pose.csv
runtime_path.json
runtime_frames/live_XXXX_cam0.png
```

其中：

- `scene_package.json` 是 runtime 的总入口配置。
- `mapping_pose.txt` 是 nuScenes/global ego pose。
- `core_pose_global.csv` 是 CARLA hero 轨迹。
- `runtime_pose.csv` 记录每个 control step 的 CARLA actor 状态、速度、控制量、nearest frame 等。
- `runtime_path.json` 记录每次渲染用到的 `camera_to_3dgs_world` 矩阵。
- `runtime_frames/*.png` 是 CARLA actor/reference pose 下的 3DGS 渲染结果。

## 最小调试顺序

建议按以下顺序确认对齐问题：

1. 先用 `--render-pose-source reference --motion-mode kinematic` 检查 checkpoint + processed scene 本身是否能复现 recorded camera 视角。
2. 再用默认 actor 渲染和 `--motion-mode kinematic` 检查 CARLA XODR 的 Y 翻转、hero 轨迹、camera mount 是否一致。
3. 最后切到 `--motion-mode control`，检查外部控制后的 actor pose 是否还能和 3DGS global background 保持一致。

如果 reference 对齐但 actor 不对，问题通常在 CARLA/OpenDRIVE Y 翻转、hero yaw、或者 actor camera mount。
如果 reference 本身不对，问题通常在 checkpoint 是否真的用 global frame 训练、processed `extrinsics/*.txt` 是否与 checkpoint 坐标一致，或 camera intrinsics/downscale 是否匹配。
