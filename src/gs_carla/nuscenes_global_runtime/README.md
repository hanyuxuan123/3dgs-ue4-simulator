# nuScenes 全局坐标场景的 CARLA + 3DGS 运行

本目录负责把全局坐标的 nuScenes XODR、DriveStudio checkpoint 和 processed scene 转为场景包，并启动 CARLA + 3DGS 联合仿真。项目总入口见[仓库 README](../../../README.md)；坐标变换和 checkpoint 假设见[对齐说明](README_NUSCENES_CKPT_ALIGNMENT.md)。

## 运行过程

`run_nuscenes_global_scene.py` 先调用 `prepare_nuscenes_global_scene.py`，生成 `scene_package.json`、`core_pose_global.csv`、`mapping_pose.txt` 及可选的背景和天空资产；随后调用 `scene_runtime_external_control.py`，将 XODR 加载到 CARLA、生成 hero 车辆并渲染 3DGS 相机。processed scene 的 `lidar_pose/*.txt` 和 checkpoint 使用同一全局坐标系。

`--motion-mode control` 使用 CARLA 物理运动和外部 `throttle/steer/brake`，会等待控制客户端连接。此模式设置 `--source-frame-mode nearest`：运行时用当前 CARLA hero 位姿匹配 `core_pose_global.csv` 中最近的帧，以该帧选择 3DGS 动态内容。控制模式下车辆可以偏离记录的轨迹速度，因此匹配的是当前位置，而不是固定播放时间。

## 准备场景

下面的本机路径是 mini/003 示例；使用其他数据时需替换四个输入路径。只准备资产、不启动 CARLA 时运行：

```bash
python -m gs_carla.nuscenes_global_runtime.prepare_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir outputs/nuscenes_global_runtime/003_global_control \
  --start-frame 0 --end-frame 48 --export-background
```

准备结果主要包括 `mapping_pose.txt`、`core_pose_global.csv`、`scene_package.json`、`background/background_gaussians.pth` 和 `background/sky_envlight.pth`。`--export-background` 用于导出背景资产；重新强制导出时可用 `--force-export-background`。

## 启动仿真并手动控制

先启动 CARLA server，默认地址为 `127.0.0.1:2000`。在包含 DriveStudio/GS 依赖的环境中，从仓库根目录运行用户提供的 control 命令：

```bash
python -m gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir outputs/nuscenes_global_runtime/003_global_control \
  --start-frame 0 --end-frame 48 --cameras 0 --pygame \
  --export-background --motion-mode control
```

上述命令会停在等待外部控制客户端的状态。在第二个终端运行以下命令，每次输入 `油门 转向 制动` 三个数：

```bash
python -m gs_carla.nuscenes_global_runtime.send_control --interactive
```

如需固定控制输入，可运行 `python -m gs_carla.nuscenes_global_runtime.send_control --throttle 0.25 --steer 0.0 --brake 0.0`。控制口一次只连接一个客户端。

输出目录中的 `scene_package.json`、`core_pose_global.csv` 属于准备阶段；`runtime_pose.csv`、`runtime_path.json` 和 `runtime_frames/` 属于仿真阶段。`runtime_pose.csv` 记录每步的车辆位姿和控制量，可用于查看 control 的实际执行情况。若想只按参考轨迹检查渲染，可省略 `--motion-mode control`，默认使用 `kinematic`；`--render-pose-source reference` 可检查原始相机姿态与 GS 资产的对齐。
