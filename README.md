# DriveStudio GS 到 CARLA 背景渲染 Demo

本文件夹包含一个最小化的桥接实验，用于在不依赖 NuRec 的情况下使用 DriveStudio checkpoint。当前阶段只关注背景：从 DriveStudio checkpoint 中导出 `Background` 高斯和 `Sky` 环境贴图，先在 processed camera 上验证坐标，再通过 CARLA bridge 读取 hero pose 做背景渲染。Rigid nodes 和动态阴影暂时不进入第一版闭环。

## 当前目标

第一阶段用于检查 DriveStudio 的背景 Gaussian 世界坐标是否与已处理的相机坐标系统对齐，然后把 CARLA hero/camera pose 转到 DriveStudio local world。当前 `gussian-sim` conda 环境已经有 `torch 2.4.0+cu118` 和 `gsplat 1.5.3`，但运行真实渲染时仍要求当前会话能访问 CUDA GPU。

## 文件说明

### `export_drivestudio_background.py`

加载 DriveStudio 的 `checkpoint_final.pth`，只提取 `models["Background"]`，并将 Gaussian tensor 保存为 `background_gaussians.pth`。该脚本会保留原始训练参数，例如 log-scales、raw quaternions、SH features 和 opacity logits。同时会写出 `metadata.json`，并可选导出一个降采样后的预览 PLY 文件。

### `render_background_point_demo.py`

使用与 DriveStudio 相同的 local-world 对齐方式，将导出的背景 Gaussian 中心点投影到一帧已处理的相机图像中。这个脚本只是一个坐标调试渲染器，并不是真正的 Gaussian splatting。它会输出点投影图像，并可选生成与原始相机图像叠加的 overlay，用于检查坐标轴、尺度和相机朝向是否正确。

### `render_background_gsplat.py`

使用 `gsplat.rendering.rasterization` 对导出的 DriveStudio background gaussians 做真实 3DGS 渲染。脚本会激活 log-scale、opacity logit、raw quaternion 和 SH degree 3 features，并支持 processed scene pose 或外部 JSON camera pose。它也能读取 `sky_envlight.pth`，按像素 viewdir 从 DriveStudio `EnvLight` cubemap 采样天空，再用 `rgb_gaussians + rgb_sky * (1-alpha)` 混合。

### `carla_background_gs_bridge.py`

连接 CARLA server，必要时从 `.xodr` 生成 CARLA world，查找 `role_name=hero` 的车辆，读取车辆 transform 和相机外参，然后通过 `T_drivestudio_from_carla` 转成 DriveStudio camera-to-world。最后调用 `render_background_gsplat.py` 中的 renderer 输出一帧背景 GS 图像。

### `coordinate_calibration_identity.json`

默认的坐标标定占位文件，只包含 identity `T_drivestudio_from_carla`。它不能代表真实对齐结果，只用于跑通接口。后续需要根据道路几何、ego 轨迹或人工选点估计 CARLA world 到 DriveStudio local world 的刚体变换，并替换这个矩阵。

## 已生成的 Demo 输出

当前 demo 输出写入到了：

```text
PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/
```

已经生成的文件包括：

```text
background_gaussians.pth
sky_envlight.pth
metadata.json
background_preview_500k.ply
frame000_cam0_points.png
frame000_cam0_overlay.png
```

在可访问 CUDA GPU 的 session 中运行真实 gsplat 命令后，会生成：

```text
frame000_cam0_gsplat.png
frame000_cam0_gsplat_overlay.png
```

## 使用过的命令

导出背景 Gaussians：

```bash
 python Tools/gs_carla/export_drivestudio_background.py \
  --checkpoint PythonAPI/examples/drivestudio_clear/work_dirs/000_6cams_omnire_ext/checkpoint_final.pth \
  --config PythonAPI/examples/drivestudio_clear/work_dirs/000_6cams_omnire_ext/config.yaml \
  --processed-scene PythonAPI/examples/drivestudio_clear/data/classlab/processed/2025-07-17-17-33-26 \
  --output-dir PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo \
  --ply PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/background_preview_500k.ply \
  --ply-max-points 500000
```

渲染粗略点投影 overlay：

```bash
  /home/classlab/anaconda3/envs/gussian-sim/bin/python \
  Tools/gs_carla/render_background_point_demo.py \
  --background PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/background_gaussians.pth \
  --processed-scene PythonAPI/examples/drivestudio_clear/data/classlab/processed/2025-07-17-17-33-26 \
  --frame 0 \
  --camera 0 \
  --output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/frame000_cam0_points.png \
  --overlay-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/frame000_cam0_overlay.png \
  --max-points 1000000 \
  --opacity-threshold 0.02 \
  --far 200 \
  --radius 1 \
  --downscale 2
```

真实 gsplat 背景渲染：

```bash
/home/classlab/anaconda3/envs/gussian-sim/bin/python \
  Tools/gs_carla/render_background_gsplat.py \
  --background PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/background_gaussians.pth \
  --sky PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/sky_envlight.pth \
  --processed-scene PythonAPI/examples/drivestudio_clear/data/classlab/processed/2025-07-17-17-33-26 \
  --frame 0 \
  --camera 0 \
  --output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/frame000_cam0_gsplat.png \
  --overlay-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/frame000_cam0_gsplat_overlay.png \
  --sky-output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/frame000_cam0_sky.png \
  --max-gaussians 10000000 \
  --opacity-threshold 0.02 \
  --crop-radius 160 \
  --far 220 \
  --downscale 2
```

当前 shell 里如果 `torch.cuda.is_available()` 为 `False`，该命令会直接报错。真实出图需要在可访问 GPU 的 session 中运行。

CARLA hero pose 到 GS background 的桥接渲染：

```bash
/home/classlab/anaconda3/envs/gussian-sim/bin/python \
  Tools/gs_carla/carla_background_gs_bridge.py \
  --host 127.0.0.1 \
  --port 2000 \
  --xodr singapore_onenorth.xodr \
  --background PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/background_gaussians.pth \
  --sky PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/sky_envlight.pth \
  --calibration Tools/gs_carla/coordinate_calibration_identity.json \
  --output PythonAPI/examples/drivestudio_clear/work_dirs/background_only_demo/carla_hero_background.png \
  --width 960 \
  --height 540 \
  --fx 900 \
  --fy 900 \
  --max-gaussians 800000 \
  --crop-radius 160
```

这个命令要求 CARLA server 已经在 `--host/--port` 运行，并且 world 里已经有一个 `role_name=hero` 的 vehicle。`coordinate_calibration_identity.json` 只是占位；真实对齐前，画面位置大概率不正确。

## 坐标说明

DriveStudio 使用以第一帧前视相机为基准对齐的 local world。对于当前这个 ClassLab / NuScenes 风格的数据加载器，相机位姿和实例位姿会按照下面方式变换：

```python
local_pose = inv(extrinsics/000_0.txt) @ raw_pose
```

导出的背景 Gaussians 已经位于这个 DriveStudio local world 中。点投影 demo 在投影前也使用相同的变换方式。

CARLA bridge 使用下面的 pose 链：

```python
T_drivestudio_from_camera_opencv =
    T_drivestudio_from_carla
    @ T_carla_from_hero
    @ T_hero_from_camera_sensor
    @ T_carla_sensor_from_opencv_camera
```

其中 `T_carla_sensor_from_opencv_camera` 把 OpenCV camera 坐标约定转换到 CARLA sensor local 坐标：OpenCV `x=right, y=down, z=forward`，CARLA sensor local `x=forward, y=right, z=up`。真正需要标定的是 `T_drivestudio_from_carla`。

## Sky 说明

当前 checkpoint 的 `Sky` 不是 Gaussian，也不是普通 equirectangular 图片，而是 DriveStudio `EnvLight` cubemap：

```text
models["Sky"]["base"]: [6, 1024, 1024, 3]
```

导出脚本会把它保存成 `sky_envlight.pth`。渲染脚本会根据相机内参和 `camera_to_world` 生成每个像素的 world-space ray direction，然后近似复现 DriveStudio 中的方向变换：

```python
viewdir_opengl = viewdir_world @ [[1,0,0], [0,0,1], [0,-1,0]].T
```

随后按 cubemap face 做双线性采样。这里没有依赖 `nvdiffrast`，所以 face orientation 是按 OpenGL cubemap 约定近似实现的；如果天空方向看起来左右/上下不对，优先调试 cubemap face 映射，而不是 GS 坐标。

## 当前验证结果

点投影结果不是空的，并且覆盖了约 81% 的降采样前视相机像素。overlay 图像在道路、树木和建筑物区域上具有较好的视觉对齐效果，因此可以初步认为 DriveStudio 背景坐标和已处理相机位姿大致一致。

`gussian-sim` 环境中已经可以 import `gsplat`，脚本语法检查也通过。但当前执行会话没有 CUDA runtime，`torch.cuda.is_available()` 为 `False`，因此真实 gsplat rasterization 无法在这里实际出图。换到能访问 GPU 的 shell 后，可以直接运行上面的真实 gsplat 命令。

## 下一步工作

1. 在可访问 GPU 的 session 中运行 `render_background_gsplat.py`，生成 `frame000_cam0_gsplat.png`。
2. 调整 `max-gaussians`、`crop-radius`、`opacity-threshold`，找到可接受的显存和画质折中。
3. 用 CARLA 的 OpenDRIVE world 和 processed ego 轨迹估计 `T_drivestudio_from_carla`。
4. 跑 `carla_background_gs_bridge.py`，确认 CARLA hero camera 与 background GS 的道路方向和尺度一致。
5. 背景稳定后，再加入 RigidNodes，并用 CARLA actor pose 覆盖 DriveStudio replay pose。
