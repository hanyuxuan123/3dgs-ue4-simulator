# 3DGS-CARLA 仿真器 Bridge 说明

这份文档整理当前 ClassLab 3DGS-CARLA simulator bridge 的构建思路、数据流、主要代码、参数含义，以及如何把相关代码整理成一个独立可复用的数据包。

核心目标是：把真实采集数据里的道路、相机、3DGS 背景、动态交通实例和 ego 轨迹统一到 CARLA 里，让 CARLA 负责物理仿真和 actor 状态，让 DriveStudio/3DGS 负责相机画面渲染，从而形成一个可控制、可输出轨迹和标注的闭环仿真器。

## 总体思路

整个系统围绕一个 scene package JSON 工作：

```text
classlab_3dgs_carla_scene_package.json
```

这个 package 记录所有稳定的数据路径和坐标锚点，例如：

- OpenDRIVE `.xodr` 地图
- DriveStudio background Gaussians
- DriveStudio sky envlight
- processed camera intrinsics/extrinsics
- mapping_pose
- ego core pose CSV
- instances_info 动态交通实例
- sequence origin / processed origin / instance origin 等关键 frame

运行仿真时，runtime 只需要读这个 package，再通过命令行覆盖实验窗口、控制参数、输出路径和渲染质量参数。

## 仿真器由三层组成

### 1. CARLA 世界

CARLA 负责：

- 从 `.xodr` 加载局部道路地图
- spawn ego vehicle
- spawn/update traffic actors
- 同步物理仿真
- collision sensor
- spectator/debug bbox 可视化

相关脚本主要在：

```text
src/gs_carla/scene_runtime_external_control.py
src/gs_carla/load_xodr_town_and_dump_poses.py
src/gs_carla/load_xodr_town_with_instances.py
```

### 2. 3DGS 视觉世界

3DGS 负责：

- 读取 DriveStudio 导出的 `background_gaussians.pth`
- 读取 `sky_envlight.pth`
- 根据 CARLA ego 当前 pose 构造 `camera_to_3dgs_world`
- 调用 `gsplat` 渲染 camera image

相关脚本：

```text
src/gs_carla/export_drivestudio_background.py
src/gs_carla/render_background_gsplat.py
src/gs_carla/render_aligned_mapping_or_carla_path_gsplat.py
```

### 3. 控制与标注桥接

控制和输出分成两个进程：

```text
scene_runtime_external_control.py 负责 CARLA + 3DGS + traffic + outputs
pose_control_client.py            负责连接 runtime 并发送 throttle/steer/brake
```

两者通过 JSON-lines TCP 通信。runtime 每个 step 发 observation，client 返回控制量。

## 坐标系统原则

当前 pipeline 的关键原则是：

```text
mapping_pose 是所有模块共享的坐标族
```

也就是说：

- BEV 局部 road map 用 `mapping_pose.txt` 构建
- OpenDRIVE `.xodr` 从这个 BEV map 导出
- ego core pose 从同一组 mapping pose/ENU pose 得到
- processed scene 的 camera extrinsics 和 3DGS local world 通过 processed origin 对齐
- traffic instances 通过 `instance_origin_sequence_frame` 和 `instance_transform_mode` 转到 CARLA

这样 CARLA 地图、ego 轨迹、traffic actor、3DGS 背景才能落到同一个局部场景里。

## 完整构建流程

```text
原始 ClassLab sequence
  ├─ result/lane.csv
  ├─ result/mapping/mapping_pose.txt
  ├─ DriveStudio processed scene
  ├─ DriveStudio checkpoint
  └─ instances/instances_info.json

步骤 1: lane.csv + mapping_pose 生成 BEV semantic road map
步骤 2: BEV road fills 导出 OpenDRIVE .xodr
步骤 3: DriveStudio checkpoint 导出 background_gaussians.pth / sky_envlight.pth
步骤 4: make_3dgs_carla_scene_package.py 生成 scene package JSON
步骤 5: scene_runtime_external_control.py 启动 CARLA + 3DGS runtime
步骤 6: pose_control_client.py 发送外部控制
步骤 7: 可选 traffic bbox projection / live traffic 工具
```

## 步骤 1: 构建 BEV 局部语义地图

脚本：

```text
PythonAPI/examples/drivestudio_clear/datasets/classlab/track_classlab_roadlanes_bev_semantic_mapping_pose.py
```

命令：

```bash
python PythonAPI/examples/drivestudio_clear/datasets/classlab/track_classlab_roadlanes_bev_semantic_mapping_pose.py \
  --sequence-dir /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26 \
  --lane-csv /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26/result/lane.csv \
  --pose-file /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26/result/mapping/mapping_pose.txt \
  --output-dir PythonAPI/examples/drivestudio_clear/data/classlab/roadlane_tracking/frames_250_450_mapping_pose_semantic \
  --start-frame 250 \
  --end-frame 450 \
  --sequence-origin-frame 0
```

主要作用：

- 从 `lane.csv` 读取每帧 lane observation
- 从 `mapping_pose.txt` 读取 ego pose
- 以 `--sequence-origin-frame` 指定的 mapping frame 作为局部坐标原点
- 将多帧 lane observation 累积成 BEV road/lane 结构
- 输出 road lanes、road fills、connectors 和 corrected pose CSV

关键参数：

| 参数 | 含义 |
| --- | --- |
| `--sequence-dir` | 原始 sequence 目录 |
| `--lane-csv` | lane 检测结果 |
| `--pose-file` | mapping pose 文件 |
| `--output-dir` | BEV 语义地图输出目录 |
| `--start-frame`, `--end-frame` | 用于建图的帧范围 |
| `--sequence-origin-frame` | 局部坐标原点 frame |

这里使用 mapping pose 作为固定轨迹，不做 lane matching pose correction。这样 BEV map 和后面的 DriveStudio/3DGS processed pose 保持在同一个 pose 体系里。

## 步骤 2: BEV 结果生成 OpenDRIVE

脚本：

```text
PythonAPI/examples/drivestudio_clear/datasets/classlab/classlab_mapping_pose_semantic_to_xodr.py
```

命令：

```bash
python PythonAPI/examples/drivestudio_clear/datasets/classlab/classlab_mapping_pose_semantic_to_xodr.py \
  --sequence-dir /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26 \
  --lane-csv /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26/result/lane.csv \
  --pose-file /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26/result/mapping/mapping_pose.txt \
  --end-frame 450 \
  --sequence-origin-frame 0 \
  --output PythonAPI/examples/drivestudio_clear/data/classlab/roadlane_tracking/frames_250_450_mapping_pose_semantic_4lane_correct/classlab_mapping_pose_semantic_4lane.xodr
```

主要作用：

- 读取 BEV semantic road fills
- 选择主道路方向
- 将中心线、lane width、lane section 写成 OpenDRIVE
- 输出 CARLA 可加载的 `.xodr`

关键参数：

| 参数 | 含义 |
| --- | --- |
| `--sequence-dir` | 从原始 sequence 重新构建 road network |
| `--road-network` | 直接使用已有 road network JSON，和 `--sequence-dir` 二选一 |
| `--lane-csv`, `--pose-file` | 构建 road network 时的输入 |
| `--sequence-origin-frame` | 必须和 BEV 建图、scene package 保持一致 |
| `--layout` | 默认 `separate-roads`，两个方向分成两条 road |
| `--lanes-per-road` | 每条 road 生成几条同向 lane |
| `--output` | 输出 `.xodr` |

生成的 `.xodr` 是 CARLA 中真实加载的局部地图。它不是图片，而是 CARLA 的道路几何和 waypoint 来源。

## 步骤 3: 导出 3DGS 背景和天空

核心脚本：

```text
src/gs_carla/export_drivestudio_background.py
src/gs_carla/render_background_gsplat.py
```

需要得到的运行资产：

```text
background_gaussians.pth
sky_envlight.pth
processed scene intrinsics/extrinsics
```

`render_background_gsplat.py` 使用 processed scene 的相机内参，并根据 runtime 给出的 `camera_to_3dgs_world` 渲染图像。

runtime 中的 pose 链路大致是：

```text
CARLA ego transform
  -> sequence-local vehicle transform
  -> mapping absolute transform
  -> 3DGS local/world transform
  -> camera_to_3dgs_world
```

关键渲染参数：

| 参数 | 含义 |
| --- | --- |
| `--max-gaussians` | 最多参与渲染的 Gaussian 数量 |
| `--opacity-threshold` | 过滤低 opacity Gaussian |
| `--crop-radius` | 以相机/车辆附近裁剪 Gaussian |
| `--near`, `--far` | 相机裁剪范围 |
| `--downscale` | 图像降采样比例，例如 3 表示 1920x1080 变 640x360 |
| `--sh-degree` | 可选 SH degree |

## 步骤 4: 生成 scene package

脚本：

```text
src/gs_carla/make_3dgs_carla_scene_package.py
```

命令：

```bash
python -m gs_carla.make_3dgs_carla_scene_package \
  --output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/classlab_3dgs_carla_scene_package.json \
  --name classlab_2025_07_17_3dgs_carla \
  --description "ClassLab OpenDRIVE + DriveStudio 3DGS background + replayed instances + corrected ENU core pose window" \
  --xodr PythonAPI/examples/drivestudio_clear/data/classlab/roadlane_tracking/frames_250_450_mapping_pose_semantic_4lane_correct/classlab_mapping_pose_semantic_4lane.xodr \
  --instances-info PythonAPI/examples/drivestudio_clear/work_dirs/000_6cams_omnire_ext/instances/instances_info.json \
  --instance-origin-sequence-frame 300 \
  --instance-transform-mode mapping-absolute \
  --background PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/background_gaussians.pth \
  --sky PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/sky_envlight.pth \
  --processed-scene PythonAPI/examples/drivestudio_clear/data/classlab/processed/2025-07-17-17-33-26 \
  --mapping-pose /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26/result/mapping/mapping_pose.txt \
  --camera 0 \
  --processed-origin-frame 0 \
  --processed-origin-mapping-frame 300 \
  --sequence-origin-frame 0 \
  --core-pose-csv PythonAPI/examples/drivestudio_clear/data/classlab/roadlane_tracking/frames_250_450_mapping_pose_semantic_4lane_correct/corrected_pose_enu.csv \
  --core-start-frame 315 \
  --core-end-frame 390 \
  --core-pose-coordinate-frame enu \
  --sequence-origin-pose-file /media/classlab/2E0CECBF0CEC82E7/2025-07-17-17-33-26/result/mapping/mapping_pose.txt
```

package 里的字段含义：

| 字段 | 含义 |
| --- | --- |
| `assets.xodr` | CARLA OpenDRIVE 地图 |
| `assets.instances_info` | DriveStudio 动态实例轨迹 |
| `assets.background` | background Gaussian 文件 |
| `assets.sky` | sky envlight 文件 |
| `assets.processed_scene` | processed camera 数据 |
| `assets.mapping_pose` | 坐标对齐用 mapping pose |
| `assets.core_pose_csv` | ego 路径 |
| `assets.sequence_origin_pose_file` | ENU 到 sequence-local 转换时用的 pose 文件 |
| `frames.instance_origin_sequence_frame` | instance frame 0 对应的 sequence frame |
| `frames.processed_origin_frame` | processed scene 的 origin camera frame |
| `frames.processed_origin_mapping_frame` | processed origin 对应的 mapping frame |
| `frames.sequence_origin_frame` | sequence-local 坐标原点 |
| `frames.core_start_frame`, `frames.core_end_frame` | 默认 ego 路径窗口 |
| `values.camera` | 使用哪个 processed camera |
| `values.instance_transform_mode` | instance 矩阵如何转到 sequence/CARLA |
| `values.core_pose_coordinate_frame` | core pose CSV 的坐标格式 |

## 步骤 5: 启动仿真 runtime

脚本：

```text
src/gs_carla/scene_runtime_external_control.py
```

这个脚本是当前 simulator bridge 的主 runtime。它负责：

- 读取 scene package
- 加载 `.xodr` 到 CARLA
- spawn ego vehicle
- 从 `instances_info` spawn/update traffic actors
- 启动 control server，等待外部控制器连接
- 用 live ego pose 渲染 3DGS 图像
- 输出 pose/path/traffic/collision/bbox

推荐命令：

```bash
python -m gs_carla.scene_runtime_external_control \
  --scene-package PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/classlab_3dgs_carla_scene_package.json \
  --control-host 127.0.0.1 \
  --control-port 29001 \
  --core-start-frame 315 \
  --core-end-frame 360 \
  --render-pose-source actor \
  --follow-speed-kmh 3 \
  --follow-start-z-offset 0.35 \
  --follow-settle-ticks 30 \
  --control-dt 0.05 \
  --spectator-follow \
  --pygame \
  --display-fps 0 \
  --pose-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/external_control_pose.csv \
  --instance-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/external_control_instances.csv \
  --traffic-bbox-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/external_control_traffic_bboxes.json \
  --collision-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/external_control_collisions.csv \
  --draw-traffic-bboxes \
  --path-json-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/external_control_path.json \
  --max-gaussians 5000000 \
  --opacity-threshold 0.02 \
  --crop-radius 160 \
  --far 220 \
  --downscale 3
```

如果不写 `--core-end-frame`，runtime 会使用 package 里的 `frames.core_end_frame`。你现在 package 里是 390，所以如果 controller 写 360，而 runtime 不写 end frame，就会出现 controller 到 360 停、runtime 继续到 package 默认终点的情况。为了实验清晰，建议 runtime 和 client 都显式写同一个 end frame。

runtime 参数分组：

| 分组 | 参数 |
| --- | --- |
| package | `--scene-package` |
| CARLA | `--host`, `--port`, `--timeout`, `--vertex-distance`, `--max-road-length`, `--wall-height`, `--additional-width`, `--no-smooth-junctions`, `--no-mesh-visibility` |
| 外部控制服务 | `--control-host`, `--control-port` |
| ego 路径窗口 | `--core-start-frame`, `--core-end-frame`, `--exit-on-path-end` |
| traffic replay | `--no-live-instances`, `--instance-class-prefix`, `--max-instance-actors`, `--instance-z-offset`, `--no-instance-map-z`, `--instance-local-offset-x`, `--instance-local-offset-y`, `--instance-local-yaw-offset-deg`, `--hidden-actor-z` |
| ego 初始放置 | `--follow-start-z-offset`, `--follow-settle-ticks`, `--no-follow-map-z` |
| render pose | `--render-pose-source`, `--vehicle-zrp-source`, `--carla-z-mode` |
| 3DGS 渲染 | `--max-gaussians`, `--opacity-threshold`, `--crop-radius`, `--near`, `--far`, `--downscale`, `--sh-degree`, `--device` |
| 显示和输出 | `--pygame`, `--display-fps`, `--output-dir`, `--video-output`, `--video-fps`, `--no-save-frames`, `--pose-output`, `--path-json-output` |
| traffic/collision 输出 | `--instance-output`, `--traffic-bbox-output`, `--draw-traffic-bboxes`, `--traffic-bbox-life-time`, `--collision-output` |

注意：

- `--pygame` 显示的是 3DGS 渲染图像。
- CARLA traffic actors 和 debug bbox 是在 CARLA world/spectator/debug 里可见，不一定直接合成到 3DGS 图像像素里。
- 如果 package 有 `instances_info` 且没有加 `--no-live-instances`，runtime 会自动进行 traffic update。

## 步骤 6: 启动外部控制器

脚本：

```text
src/gs_carla/pose_control_client.py
```

命令：

```bash
python -m gs_carla.pose_control_client \
  --scene-package PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/classlab_3dgs_carla_scene_package.json \
  --control-host 127.0.0.1 \
  --control-port 29001 \
  --core-start-frame 315 \
  --core-end-frame 360 \
  --control-mode path \
  --follow-speed-kmh 3 \
  --follow-lookahead-distance 10.0 \
  --control-dt 0.05 \
  --steer-gain 0.35 \
  --max-steer 0.30
```

主要逻辑：

- 连接 runtime 的 TCP control server
- 接收 observation：
  - `step`
  - `source_frame`
  - `nearest_index`
  - `nearest_dist`
  - `pose`
  - `speed_mps`
  - `path_finished`
- 根据 core pose path 计算 steering 和 speed PID
- 返回：
  - `throttle`
  - `steer`
  - `brake`
- 到终点时发刹车并断开

关键参数：

| 参数 | 含义 |
| --- | --- |
| `--control-mode` | `path`, `steer-bias`, `constant-steer` |
| `--follow-speed-kmh` | 目标速度 |
| `--follow-lookahead-distance` | 路径前视距离 |
| `--steer-gain`, `--max-steer` | 横向控制参数 |
| `--speed-kp`, `--speed-ki`, `--speed-kd` | 纵向速度 PID |
| `--stop-at-path-end` | 到终点是否停止 |

## 可选: traffic bbox 工具

脚本：

```text
src/gs_carla/traffic_bbox_projection_from_package.py
```

这个脚本有两种模式。

### 离线 2D bbox 投影

读取 runtime 输出的 `external_control_path.json`，按照实际渲染相机轨迹投影 traffic 2D bbox。

```bash
python -m gs_carla.traffic_bbox_projection_from_package \
  --scene-package PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/classlab_3dgs_carla_scene_package.json \
  --output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/traffic_bbox_projection_runtime_315_360.json \
  --frame-start 315 \
  --frame-end 360 \
  --camera-path-json PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/external_control_path.json \
  --control-dt 0.05 \
  --downscale 3
```

### CARLA live traffic replay

让第三个脚本连接正在运行的 CARLA world，单独负责 traffic actors 的 replay 和 bbox 输出。

```bash
python -m gs_carla.traffic_bbox_projection_from_package \
  --scene-package PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/classlab_3dgs_carla_scene_package.json \
  --output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/live_traffic_bboxes_315_360.json \
  --instance-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/live_traffic_instances_315_360.csv \
  --frame-start 315 \
  --frame-end 360 \
  --live-carla \
  --draw-traffic-bboxes \
  --control-dt 0.05
```

如果用第三个脚本负责 live traffic，那么 runtime 要加：

```bash
--no-live-instances
```

否则 runtime 和第三个脚本会各自 spawn 一套 traffic actors，造成重复车辆。

如果 runtime 已经在同步 tick CARLA，不要给第三个脚本加 `--tick-world`。`--tick-world` 只适合第三个脚本单独测试 traffic replay。

## 输出文件和时间对齐

| 输出 | 生成脚本 | 内容 |
| --- | --- | --- |
| `external_control_pose.csv` | runtime | ego 每个 step 的 pose、速度、控制量 |
| `external_control_path.json` | runtime | 每个 render frame 的 `camera_to_3dgs_world` |
| `external_control_instances.csv` | runtime | traffic actor 每帧状态 |
| `external_control_traffic_bboxes.json` | runtime | CARLA 3D bbox |
| `external_control_collisions.csv` | runtime | ego collision sensor 事件 |
| `traffic_bbox_projection_runtime_*.json` | bbox 工具 | 与 runtime path 对齐的 2D bbox |

时间字段：

| 字段 | 含义 |
| --- | --- |
| `step` | CARLA/control 同步步 |
| `elapsed_time` | `step * control_dt` |
| `frame` / `sequence_frame` | 数据集/mapping frame |
| `render_index` | 3DGS 渲染输出帧序号 |

低速跟随时，多个 `step` 可能对应同一个 `sequence_frame`，这是正常的。`step` 是仿真时间，`sequence_frame` 是当前最近路径点对应的数据帧。

## Tools 目录里的代码对应关系

当前 `src/gs_carla` 里和 bridge 直接相关的脚本关系如下：

| 文件 | 角色 | 被谁使用 |
| --- | --- | --- |
| `make_3dgs_carla_scene_package.py` | 生成 scene package JSON | 手动运行 |
| `scene_runtime_external_control.py` | 当前主 runtime，CARLA + 3DGS + traffic + external control | 手动运行 |
| `pose_control_client.py` | 外部 path controller | 手动运行，连接 runtime |
| `traffic_bbox_projection_from_package.py` | 离线 2D bbox / 可选 live traffic replay | 手动运行 |
| `load_xodr_town_and_dump_poses.py` | CARLA map、ego pose、path control 基础函数 | runtime、client |
| `load_xodr_town_with_instances.py` | instance track 加载、spawn、update、CSV 输出 | runtime、traffic 工具 |
| `render_background_gsplat.py` | 3DGS rasterization 和 sky 混合 | runtime |
| `render_aligned_mapping_or_carla_path_gsplat.py` | mapping pose 读取、pose matrix 工具 | runtime、bbox 工具 |
| `export_drivestudio_background.py` | 从 DriveStudio checkpoint 导出 background/sky | 预处理 |
| `traffic_bbox_carla_loader.py` | CARLA bbox helper/legacy live loader | 可选/历史模块 |
| `carla_xodr_live_3dgs_bridge_with_instances.py` | 旧的一体化 runtime，内置 controller + instances | 参考和对照 |
| `carla_3dgs_scene_runtime.py` | 更模块化 runtime 版本 | 可选/参考 |
| `pose_control_signal.py` | 控制信号辅助模块 | 模块化 runtime |

也就是说，现在推荐使用的主路径是：

```text
make_3dgs_carla_scene_package.py
  -> scene_runtime_external_control.py
      -> load_xodr_town_and_dump_poses.py
      -> load_xodr_town_with_instances.py
      -> render_background_gsplat.py
      -> render_aligned_mapping_or_carla_path_gsplat.py
  -> pose_control_client.py
  -> traffic_bbox_projection_from_package.py  可选
```

旧脚本 `carla_xodr_live_3dgs_bridge_with_instances.py` 仍然保留，因为它是最早把 CARLA、3DGS、instances 放在一个进程里跑通的版本。现在的新流程把控制拆成 external client，更方便替换 controller。

## 独立代码包

我已经添加了代码包清单：

```text
src/gs_carla/bridge_bundle_manifest.json
```

以及打包脚本：

```text
src/gs_carla/bundle_bridge_package.py
```

打包命令：

```bash
python -m gs_carla.bundle_bridge_package \
  --output-dir /tmp/classlab_3dgs_carla_bridge_bundle
```

如果要连示例数据路径里的文件也尝试复制：

```bash
python -m gs_carla.bundle_bridge_package \
  --output-dir /tmp/classlab_3dgs_carla_bridge_bundle \
  --include-data-examples
```

这个 bundle 会复制：

- `src/gs_carla` 下的 runtime、controller、renderer、traffic、package 工具
- `PythonAPI/examples/drivestudio_clear/datasets/classlab` 下的 BEV 建图和 XODR 生成工具
- 本 README
- bundle manifest

它不是完整 Python 环境，也不会自动打包 CARLA、PyTorch、gsplat 或大体积数据。后续使用时需要：

- 有 CARLA PythonAPI 可 import
- 有 PyTorch 和 `gsplat`
- 有 NumPy、PIL 等依赖
- scene package 里的数据路径有效，或者重新生成 package JSON

## 最小运行检查表

1. 启动 CARLA server。
2. 进入包含 CARLA PythonAPI、PyTorch、gsplat 的 Python 环境。
3. 确认 `classlab_3dgs_carla_scene_package.json` 里的路径存在。
4. 运行 `scene_runtime_external_control.py`。
5. runtime 打印 control server listening 后，运行 `pose_control_client.py`。
6. 检查：
   - `external_control_pose.csv`
   - `external_control_path.json`
   - `external_control_instances.csv`
   - `external_control_traffic_bboxes.json`
   - `external_control_collisions.csv`

## 常见注意点

- runtime 和 controller 的 `--core-start-frame / --core-end-frame` 最好一致。
- 如果 runtime 不写 `--core-end-frame`，会使用 package 默认值。
- `--pygame` 是 3DGS 图像窗口，不是 CARLA RGB camera。
- traffic actor 在 CARLA world 里更新；3DGS 画面目前主要渲染 background/sky，不一定把 CARLA traffic 直接合成到图像里。
- 如果第三个脚本 live replay traffic，runtime 要加 `--no-live-instances`。
- 同步 CARLA world 只能由一个主进程 tick；不要让多个进程同时 `world.tick()`。
