# UniAD 接入 CARLA–3DGS 闭环仿真

> 总入口：[仓库 README](../README.md)。runtime 的准备和单相机 control 验证见 [nuScenes global runtime](../src/gs_carla/nuscenes_global_runtime/README.md)；Benchmark actor 生成见 [Bench2Drive-Real](../tools/bench2drive_real/README.md)；计分见 [闭环评测](EVALUATION.md)。

这个目录把 `Bench2DriveZoo/team_code/uniad_b2d_agent.py` 接到现有`scene_runtime_external_control.py` 的 JSON-line 控制口。固定的`send_control.py` 不再参与闭环：3DGS 六相机图像与 CARLA 运动学进入UniAD，UniAD 的 `throttle/steer/brake` 返回 CARLA；下一 tick 的 CARLA位姿继续驱动 3DGS 相机。

```text
CARLA hero pose ──> 3DGS 六相机渲染 ──> UniAD ──> VehicleControl
      ^                                             │
      └──────────── CARLA physics tick <────────────┘
```

## 文件

- `run_uniad_agent.py`：连接控制 socket、组装六相机/GPS/IMU/速度输入并执行推理。
- `team_code/uniad_gs_agent.py`：复用原始 `UniadAgent` 模型和 PID，仅替换 Leaderboard 的传感器所有权与路线初始化。
- `camera_map_nuscenes.json`：当前 nuScenes/DriveStudio 六相机编号到 UniAD 相机名的映射。

## 代码结构与每一层职责

```text
UniAD_GS_CARLA/
├── run_uniad_tiny.sh               # Tiny config/checkpoint 的固定启动入口
├── run_uniad_agent.py              # JSON-line client、模型生命周期、control 安全裁剪
├── score_episode.py                # episode facts -> 完成率/碰撞惩罚/驾驶分数
├── aggregate_scores.py             # 多 route 的均值、成功率与碰撞汇总
├── EVALUATION.md                   # 碰撞退出、终点成功、超时与计分说明
├── camera_map_nuscenes.json        # 3DGS camera id -> UniAD sensor name
├── checkpoints/                    # 本地权重（不应提交到 Git）
└── team_code/
    └── uniad_gs_agent.py           # observation -> Leaderboard input_data 适配
```

职责边界如下：

```text
scene_runtime_external_control.py
  拥有 CARLA world/hero、同步 tick、3DGS renderer 和 control socket
        │ observation：六图路径 + 标定 + GPS/IMU/speed 等价量
        ▼
run_uniad_agent.py
  加载 checkpoint、连接 socket、执行一次 run_step、裁剪 control
        ▼
team_code/uniad_gs_agent.py
  复用 Bench2DriveZoo UniadAgent 与 PID，仅替换传感器/route 输入
        │ throttle / steer / brake
        └──────────────────────────────> runtime -> CARLA physics -> 下一帧 GS
```

所以输出 control 使用的运动学确实来自 CARLA：`pose/speed/acceleration/angular_velocity`由 hero actor 每 tick 读取；视觉来自当前 CARLA 位姿驱动的 3DGS 六相机渲染。

运行时 observation 新增了 `sensor_packet`、加速度、角速度和 compass。图像通过本机绝对路径传递，避免每 tick 在 JSON 中编码六张大图。第一 tick尚无已渲染图像，因此 agent 返回全制动；从第二 tick 起进入正常推理。

## 1. 环境

使用能运行 Bench2DriveZoo UniAD 的环境，确保 CUDA、PyTorch、项目内 MMCV扩展和 CARLA Python API 都可导入。UniAD 原代码固定调用 CUDA，不能用 CPU完成模型推理。准备官方/项目对应的 B2D checkpoint，例如：

```text
Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py
/path/to/uniad_base_b2d.pth
```

所有命令建议从仓库根目录执行。

本仓库 `Bench2DriveZoo/requirements.txt` 固定了 `numpy==1.20.0` 和`numba==0.48.0`。如果用户目录的 `~/.local` 中有较新的 NumPy 覆盖 conda环境，会在导入 numba 时失败；应先把 Bench2DriveZoo 环境依赖补齐，再使用`PYTHONNOUSERSITE=1` 启动，不能靠给旧 numba 临时补 `np.long` 来解决二进制兼容问题。

## 2. 启动 CARLA–3DGS runtime

沿用原来的场景准备命令，但 control 模式必须增加六相机、逐 tick 渲染、原始 1600x900 分辨率和 observation 发布参数：

```bash
python -m gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene \
  --xodr /path/to/map.xodr \
  --checkpoint /path/to/3dgs_checkpoint.ckpt \
  --processed-scene /path/to/processed_scene \
  --output-dir /path/to/work_dir \
  --motion-mode control \
  --render-pose-source actor \
  --cameras 0 1 2 3 4 5 \
  --render-every-n 1 \
  --downscale 1 \
  --publish-rendered-observations \
  --control-port 29001 \
  --no-save-frames
```

使用 `--no-save-frames` 时，runtime output 目录只保留六个`agent_latest_cam*.png` 并逐 tick 覆盖，不会无限增长；去掉该参数则会按原来的`live_*.png` 规则保留全部渲染帧以便复现实验。

如果场景包已经准备好，也可以直接运行底层 runtime：

```bash
python -m gs_carla.scene_runtime_external_control \
  --scene-package /path/to/work_dir/scene_package.json \
  --wait-for-control-client \
  --cameras 0 1 2 3 4 5 \
  --render-every-n 1 \
  --downscale 1 \
  --render-pose-source actor \
  --source-frame-mode nearest \
  --output-dir /path/to/work_dir/runtime_frames \
  --no-save-frames \
  --publish-rendered-observations
```

## 3. 启动 UniAD 控制端

在第二个终端执行：

```bash
CUDA_VISIBLE_DEVICES=0 python UniAD_GS_CARLA/run_uniad_agent.py \
  --config Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py \
  --checkpoint /path/to/uniad_base_b2d.pth \
  --scene-package /path/to/work_dir/scene_package.json \
  --camera-map UniAD_GS_CARLA/camera_map_nuscenes.json \
  --control-port 29001 \
  --log /path/to/work_dir/uniad_control.jsonl
```

若 `carla` 或 `agents.navigation` 不能导入，再传：

```bash
  --carla-python-api /path/to/CARLA
```

该参数既接受 CARLA 根目录，也接受其 `PythonAPI` 目录。

## UniAD-Tiny B2D

Tiny 必须使用以下配对（以仓库现有`Bench2Drive/leaderboard/scripts/run_evaluation_multi_uniad_tiny.sh` 为准）：

```text
config:     Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py
checkpoint: uniad_tiny_b2d.pth
```

注意：`Bench2DriveZoo/README.md` 结果表中 UniAD-Tiny/UniAD-Base 的 config链接写反了；Tiny 不应搭配 `base_e2e_b2d.py`。

推荐把权重放在：

```bash
mkdir -p UniAD_GS_CARLA/checkpoints
curl -L --fail --continue-at - \
  -o UniAD_GS_CARLA/checkpoints/uniad_tiny_b2d.pth \
  https://huggingface.co/rethinklab/Bench2DriveZoo/resolve/main/uniad_tiny_b2d.pth
```

然后用 Tiny 专用包装脚本启动；它会自动选择正确 config/checkpoint：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  bash UniAD_GS_CARLA/run_uniad_tiny.sh \
  --scene-package /path/to/work_dir/scene_package.json \
  --camera-map UniAD_GS_CARLA/camera_map_nuscenes.json \
  --carla-python-api /path/to/CARLA \
  --control-port 29001 \
  --log /path/to/work_dir/uniad_tiny_control.jsonl
```

权重放在其他位置时，无需复制：

```bash
UNIAD_TINY_CHECKPOINT=/absolute/path/uniad_tiny_b2d.pth \
  bash UniAD_GS_CARLA/run_uniad_tiny.sh --scene-package /path/to/scene_package.json
```

## 数据与坐标约定

- 默认相机顺序为 `0=front, 1=front-left, 2=front-right, 3=back-left,
  4=back-right, 5=back`；不同数据集必须修改 JSON 映射。
- route 来自场景包的 `assets.core_pose_csv`。当前适配器要求
  `values.core_pose_coordinate_frame` 为 `carla`，避免把 ENU/local 坐标误当成  CARLA 世界坐标。
- CARLA XY 被无损地编码成以 `(lat_ref, lon_ref)=(0,0)` 为参考的合成 GPS，
  只是为了复用 UniAD 原始 `RoutePlanner`，不是实际经纬度。
- compass 按 CARLA IMU 语义生成：北为 0、东为 `pi/2`；角速度从 CARLA 的
  degree/s 转为 UniAD IMU 所需的 rad/s。
- 模型仍使用发布版 B2D checkpoint 内约定的 `lidar2cam/lidar2img`。因此 3DGS
  六个虚拟相机应尽量匹配 B2D 的机位和 1600x900 成像；仅把任意六张图映射进来  可以跑通代码，但几何不匹配会明显降低控制质量。
- 当前示例 `nuscenes/mini/003/images` 实际只有 `cam_0` 的 201 张训练图。虽然
  runtime 能按外参渲染 camera 0..5，但实测 camera 3/4 仅部分覆盖、camera 5   的 alpha 为 0（纯灰背景）。这足以验证接口闭环，不代表有效的六视角感知；  正式实验应换用六相机共同训练的 360° 3DGS checkpoint，或另行设计缺失视角回退。

## 安全行为与检查

- 没有图像、图像缺失、非有限 control 时应停车；首帧明确全制动。
- runtime 与 agent 是同步闭环，UniAD 推理期间 CARLA 不会偷偷前进。
- `uniad_control.jsonl` 同时记录 observation/control，可检查每一步是否真的来自
  `state=inference`，而不是 warmup 制动。
- 首次建议加 `--max-steps 20` 做短测，并观察输出 control、六张图方向和 CARLA
  hero 运动是否一致。

## 闭环评测

需要“碰撞即退出、到终点即成功、超时失败并自动结算”时，runtime 增加`--evaluation-mode`，UniAD 增加 `--episode-result` 和 `--score-output`。完整命令、计分公式、输出文件和 3DGS 覆盖范围要求见 [EVALUATION.md](EVALUATION.md)。

当前实现是可复现的轻量 Route Completion × Collision Penalty 评测，不是官方Bench2Drive 全 criteria 分数；正式对比还要加入红灯、车道侵入、阻塞和各 scenario完成条件。

## 换成 TransFuser++ 或其他 agent

仿真端不需要改成某个模型专用版本。保留 JSON-line 协议，另建例如`TransFuserPP_GS_CARLA/`，实现与 `run_uniad_agent.py` 相同的四个边界：

1. 接收一行 observation，并读取 `sensor_packet.images/cameras`。
2. 把相机、CARLA 运动学、route command 转成该模型训练时的尺寸、归一化和坐标系。
3. 加载模型并执行一次推理；如果输出 waypoints/trajectory，再调用该方法配套 PID/LQR。
4. 向 socket 回一行 `{"throttle": ..., "steer": ..., "brake": ...}`。

需要特别注意，TransFuser 系方法通常依赖 RGB + LiDAR、目标点/导航命令和指定的相机内外参。当前协议已经有六路 RGB、标定与运动学，但没有 LiDAR 点云。应选择真正的camera-only checkpoint，或在 runtime 增加同步 CARLA LiDAR/GS depth 发布；不能用空点云冒充训练输入。任何替换模型都必须保持 checkpoint、config、sensor suite 和训练时约定完全配对。
