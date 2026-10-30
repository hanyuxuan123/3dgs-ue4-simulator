# PDM-Lite × CARLA–3DGS 专家数据采集桥

> 总入口：[仓库 README](../README.md)。先按 [nuScenes global runtime](../src/gs_carla/nuscenes_global_runtime/README.md) 准备场景与控制口；这里说明如何接专家后端并保存训练数据。

这个目录负责一件事：让规划专家接管 control socket，在闭环 CARLA 动力学中驾驶，同时把每次专家决策对应的六路 3DGS 图像、自车状态、路线、交通参与者真值和`throttle/steer/brake` 保存成严格对齐的训练样本。

PDM-Lite 的准确来源是 OpenDriveLab/DriveLM 仓库的 `DriveLM-CARLA` 分支：[PDM-Lite 官方代码](https://github.com/OpenDriveLab/DriveLM/tree/DriveLM-CARLA/pdm_lite)。它是 CARLA Leaderboard 2.0 的 privileged rule-based expert，不是 NAVSIM 的PDM-Closed/PDM score。官方也发布了[PDM-Lite CARLA LB2 数据集](https://huggingface.co/datasets/autonomousvision/PDM_Lite_Carla_LB2)。本仓库当前没有复制这套源码，因此这里提供稳定桥接接口，实际 PDM-Lite 后端通过`module:factory` 注入。

## 数据流

```text
CARLA ego/traffic 真值 ─┐
core route ─────────────┼─> PDM-Lite backend ─> throttle/steer/brake ─> CARLA
3DGS 六相机图像 ────────┘                                  │
       │                                                    └─> 下一 tick 位姿
       └─> copy before control reply ─> 专家数据集
```

`scene_runtime_external_control.py` 是唯一仿真 owner。采集器是唯一 control client，它在同一进程中调用专家并记录数据。不能同时再启动 UniAD，因为端口 `29001` 一次只允许一个控制客户端。

## 目录与接口

```text
PDMLite_GS_CARLA/
├── collect_expert_data.py          # socket 闭环、图像固化、JSONL 标注
├── camera_map_nuscenes.json        # camera id 到标准六相机名
├── pdmlite.example.json            # 后端配置模板
└── team_code/
    └── pdmlite_adapter.py          # 稳定输入协议 -> 用户的 PDM-Lite 后端
```

后端 factory 的签名为：

```python
def create_backend(*, context, checkpoint, **kwargs):
    return Backend(...)

class Backend:
    def run_step(self, expert_input):
        # expert_input 包含 ego_state、route、tracked_objects、六相机路径/标定
        return {
            "control": {"throttle": 0.2, "steer": 0.01, "brake": 0.0},
            "annotations": {"trajectory": [[0.0, 0.0], [1.0, 0.0]]},
        }
```

如果 PDM-Lite 输出的是未来轨迹而不是车辆控制，需要在这个 backend 内使用其原生LQR/PID/车辆模型转换成 control；不要在通用采集器里猜轨迹坐标系。

## 启动

先按官方说明准备 PDM-Lite。建议克隆到本仓库同级而不是复制进本目录，并固定 commit：

```bash
cd /home/classlab/TPAMI
git clone --branch DriveLM-CARLA https://github.com/OpenDriveLab/DriveLM.git DriveLM-CARLA
cd DriveLM-CARLA/pdm_lite
conda env create -f environment.yml
```

官方实现面向 CARLA 0.9.14 + Leaderboard 2.0，并通过 `CarlaDataProvider` 访问 HD map、交通灯、stop sign 和所有 actors。当前项目的 CARLA 版本、XODR 注入和同步 tick owner不同，不能直接把 `run_pdm_lite_local.sh` 与本 runtime 同时启动。需要在用户自己的`your_pdmlite_package.carla_backend:create_backend` 中复用 PDM-Lite planner：连接现有 CARLA server 但不调用 `world.tick()`，把 scene package route 注册给 planner，并让 `run_step` 返回 control。`expert_state` 可提供 route 和本项目生成的 scenario actor boxes；交通灯、路标和其他 CARLA actors 应由 backend 从同一个 CARLA world读取。完成后再把该 entrypoint 写入 `pdmlite.json`。

终端 1 使用 `gussian-sim` 启动六相机 runtime。和普通 UniAD 闭环相比，多加`--publish-expert-state`：

```bash
conda activate gussian-sim
cd /home/classlab/TPAMI/3dgs-ue4-simulator-2

python -m gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir outputs/nuscenes_global_runtime/003_global_pdmlite \
  --start-frame 0 --end-frame 48 \
  --cameras 0 1 2 3 4 5 \
  --render-every-n 1 --downscale 1 \
  --render-pose-source actor \
  --motion-mode control --control-port 29001 \
  --publish-rendered-observations --publish-expert-state \
  --no-save-frames
```

终端 2 使用安装了 PDM-Lite 的环境。先复制并修改配置：

```bash
cp PDMLite_GS_CARLA/pdmlite.example.json /path/to/pdmlite.json

python PDMLite_GS_CARLA/collect_expert_data.py \
  --scene-package outputs/nuscenes_global_runtime/003_global_pdmlite/scene_package.json \
  --expert-config /path/to/pdmlite.json \
  --camera-map PDMLite_GS_CARLA/camera_map_nuscenes.json \
  --output-dir /path/to/expert_dataset/episode_000001 \
  --sample-every 5 \
  --control-port 29001
```

上例 runtime 的 `control-dt=0.05` 即 20 Hz；`--sample-every 5` 保持专家以 20 Hz闭环控制，但以 4 Hz 保存训练样本，与官方 PDM-Lite 数据集的采样率一致。需要稠密轨迹监督时改为 1。

## 输出格式

```text
episode_000001/
├── metadata.json
├── frames.jsonl
└── images/
    ├── CAM_FRONT/000000.png
    ├── CAM_FRONT_LEFT/000000.png
    └── ... 六个视角
```

每行标注包含图像相对路径、内外参、自车运动学、前方 route window、交通目标 3D box、专家 control 和可选轨迹/代价等 annotations。runtime 的 `agent_latest_cam*.png`会覆盖，因此采集器必须在回复 control 前复制图像；当前实现已经这样做，不能改成hard link。

## 生成可训练专家数据时的要求

- 先验证六相机 3DGS checkpoint 确实覆盖 360°。当前 mini/003 只有 cam0 训练图，
  cam5 基本是灰图，只适合接口测试，不能作为可靠专家视觉数据。
- 专家可以读取 route/交通真值；被训练的 student 只能读取计划部署时存在的传感器，
  否则会产生 privileged-information leakage。
- 固定 CARLA/GS/PDM 版本、随机种子、地图、天气、场景包 hash 和 checkpoint hash。
- 划分 train/val/test 时按 scene/route 划分，不能随机拆同一段连续帧。
- `render-every-n` 应为 1。若不是 1，采集器会继续闭环，但相同 render index 只保存一次。
- 专家失败、碰撞、脱离路线和紧急制动应保留状态标签；不要静默混入正常 imitation 样本。
- 当前 `expert_state.traffic_objects` 只覆盖由 scene package/instances 生成的 live
  actors，不等同于完整 CARLA world。若启用 Traffic Manager，后端必须直接读取其 actors。
