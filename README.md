# CARLA + 3DGS 场景、控制与评测

本仓库把 OpenDRIVE 地图、DriveStudio 3DGS、CARLA 车辆动力学和外部控制器接成同一个仿真流程。**v0.2.0 的方案是把 nuScenes 与自采 ClassLab 数据接入统一的 GS + CARLA 仿真链路**：两种数据分别构建各自的 XODR、位姿与 3DGS 场景包，再共用场景运行时、车辆动力学、渲染和控制接口。这里的“合在一起”指统一代码与运行接口；跨数据集训练、资产融合或把两套坐标直接拼成一个场景不属于当前实现。本文是总入口；下面的 `/path/to/...` 均需换成本机的 XODR、checkpoint、config 和 processed scene。

## 阅读入口

- nuScenes global 场景准备、GS + CARLA 和手动控制：[nuScenes runtime](src/gs_carla/nuscenes_global_runtime/README.md)；checkpoint 与坐标假设见[对齐说明](src/gs_carla/nuscenes_global_runtime/README_NUSCENES_CKPT_ALIGNMENT.md)。
- 从地图、轨迹和 3DGS 资产构建一般 scene package：[仿真器 bridge 全流程](docs/SIMULATOR_BRIDGE_README.md)。
- Bench2Drive XML 场景迁移到当前路线并生成 actor：[Bench2Drive-Real 场景构建](tools/bench2drive_real/README.md)。
- UniAD 六相机闭环：[UniAD 接入](UniAD_GS_CARLA/README_CN.md)；终止条件和分数见[闭环评测](UniAD_GS_CARLA/EVALUATION.md)。
- PDM-Lite 专家接口和训练数据采集：[PDM-Lite 采集桥](PDMLite_GS_CARLA/README.md)。
- 环境与本机资产：[部署说明](docs/GITHUB_DEPLOY.md)；环境检查为 `scripts/check_environment.py`。

`Bench2Drive/` 和 `Bench2DriveZoo/` 是外部上游项目，本目录不复制其源码和权重。这里的 Bench2Drive-Real 场景迁移不等同于原版 leaderboard 评测。PDM-Lite planner 后端通过 `module:factory` 接入，本目录提供控制和数据采集接口。

## 数据和运行链路

```text
nuScenes map ──> nuScenes XODR ──> 全局 lidar_pose + DriveStudio checkpoint ─┐
                                                                         ├─> scene_package.json
ClassLab 道路语义/轨迹 ──> ClassLab XODR ──> mapping_pose + 3DGS 背景/天空 ──┘
                                      │
                                      ├─> Bench2Drive 场景 / Traffic Manager 轨迹
                                      └─> CARLA world + hero + traffic actor + 同步 tick
                                             │
                                             └─> 按 CARLA 位姿渲染 3DGS 相机
                                                     │
                                                     └─> control socket 29001
                                                          ↔ 手动控制 / UniAD / PDM-Lite
```

`--motion-mode control` 或底层 runtime 的 `--wait-for-control-client` 会等待外部客户端：客户端接收 observation，返回 `throttle/steer/brake`，CARLA 前进一步，再按新位姿渲染下一帧。一个 runtime 同时只连接一个控制客户端。单视角调试用 `--cameras 0`；UniAD 和专家采集需六相机、逐 tick 渲染及 observation 发布。

## 安装与统一入口

在具备 CARLA PythonAPI、DriveStudio、PyTorch、gsplat 和 CUDA 的环境中，从本目录执行：

```bash
cd /path/to/3dgs-ue4-simulator
python -m pip install -e .
python scripts/check_environment.py
python scripts/run_scene.py --help
```

`requirements-gpu-cu118.txt` 是原项目的 CUDA 11.8 依赖参考。运行前还需启动 CARLA server，默认地址 `127.0.0.1:2000`。总入口会将本目录的 `src/` 加入子进程 `PYTHONPATH`；其他上游依赖仍需安装。

```bash
python scripts/run_scene.py --dataset {nuscenes,classlab} --stage {xodr,export-background,prepare,run,runtime,traffic,control} -- <原工具参数>
```

| 阶段 | `nuscenes` 实际调用 | `classlab` 实际调用 |
| --- | --- | --- |
| `xodr` | `tools/xodr/nuscenes_to_xodr.py` | `tools/xodr/classlab/classlab_mapping_pose_semantic_to_xodr_with_connectors.py` |
| `export-background` | `gs_carla.export_drivestudio_background` | 同左 |
| `prepare` | `gs_carla.nuscenes_global_runtime.prepare_nuscenes_global_scene` | `gs_carla.make_3dgs_carla_scene_package` |
| `run` | nuScenes 准备器 + `scene_runtime_external_control` | 读取已准备的 scene package，调用 `scene_runtime_external_control` |
| `runtime` | 直接读取已准备的 scene package | 同左 |
| `traffic` | `gs_carla.generate_tm_insert_package` | 同左 |
| `control` | `gs_carla.nuscenes_global_runtime.send_control` | 同左 |

`--` 后的参数原样交给表中工具。查看实际参数可运行 `python scripts/run_scene.py --dataset nuscenes --stage run -- --help`；用 `--dry-run` 只打印映射后的命令。

## 路线 A：nuScenes 全局坐标

已有全局 XODR 可跳过第一步。nuScenes 地图转换器会近似补全 OpenDRIVE 道路结构；生成后须在 CARLA 检查车道连通和高度：

```bash
python scripts/run_scene.py --dataset nuscenes --stage xodr -- \
  --dataroot /path/to/nuscenes --map-name boston-seaport \
  --output assets/local/maps/boston-seaport.xodr
```

XODR、`processed_scene/lidar_pose/*.txt` 和 DriveStudio checkpoint 必须使用同一个全局 XY 坐标。仅准备场景包与背景时运行：

```bash
python scripts/run_scene.py --dataset nuscenes --stage prepare -- \
  --xodr assets/local/maps/boston-seaport.xodr \
  --checkpoint /path/to/checkpoint_final.pth --config /path/to/config.yaml \
  --processed-scene /path/to/processed/003 \
  --output-dir outputs/nuscenes_003 --start-frame 0 --end-frame 48 \
  --export-background
```

输出 `scene_package.json`、`mapping_pose.txt`、`core_pose_global.csv`、`background/background_gaussians.pth` 和 `background/sky_envlight.pth`。一键准备并启动单视角 control 模式：

```bash
python scripts/run_scene.py --dataset nuscenes --stage run -- \
  --xodr assets/local/maps/boston-seaport.xodr \
  --checkpoint /path/to/checkpoint_final.pth --config /path/to/config.yaml \
  --processed-scene /path/to/processed/003 \
  --output-dir outputs/nuscenes_003 --start-frame 0 --end-frame 48 \
  --cameras 0 --pygame --motion-mode control --export-background
```

背景已导出时可省略 `--export-background`。`run` 先准备场景，再启动 runtime；输出目录中的 `runtime_pose.csv`、`runtime_path.json` 和 `runtime_frames/` 来自仿真阶段。要按参考轨迹检查渲染，可改 `--motion-mode kinematic`。nuScenes 控制模式按当前 CARLA 位姿匹配最近的参考帧，选择对应动态内容。

## 路线 B：自采 ClassLab 场景

先从道路语义和轨迹生成 XODR。原始采集数据预处理需 DriveStudio ClassLab 数据工具；已有 `road_network_info_local.json` 时可直接执行：

```bash
python scripts/run_scene.py --dataset classlab --stage xodr -- \
  --road-network /path/to/road_network_info_local.json \
  --output assets/local/maps/classlab.xodr
```

导出 checkpoint 背景和天空，再用通用场景包工具连接 XODR、mapping pose、core pose、processed scene 和可选交通实例：

```bash
python scripts/run_scene.py --dataset classlab --stage export-background -- \
  --checkpoint /path/to/checkpoint_final.pth \
  --processed-scene /path/to/processed/classlab_sequence \
  --output-dir assets/local/classlab_background

python scripts/run_scene.py --dataset classlab --stage prepare -- \
  --output assets/local/classlab_scene.local.json \
  --xodr assets/local/maps/classlab.xodr \
  --background assets/local/classlab_background/background_gaussians.pth \
  --sky assets/local/classlab_background/sky_envlight.pth \
  --processed-scene /path/to/processed/classlab_sequence \
  --mapping-pose /path/to/mapping_pose.txt \
  --core-pose-csv /path/to/core_pose.csv \
  --processed-origin-mapping-frame 300 --sequence-origin-frame 0 \
  --instance-origin-sequence-frame 300 \
  --instance-transform-mode mapping-absolute
```

帧号 `300` 仅表示参数位置，必须按实际数据修改。已有 `instances_info.json` 时在 `prepare` 命令中加 `--instances-info /path/to/instances_info.json`。`--core-pose-coordinate-frame` 必须与 CSV 坐标系一致。标定和坐标字段见[仿真器 bridge 全流程](docs/SIMULATOR_BRIDGE_README.md)。

启动 ClassLab 场景 control 模式：

```bash
python scripts/run_scene.py --dataset classlab --stage run -- \
  --scene-package assets/local/classlab_scene.local.json \
  --host 127.0.0.1 --port 2000 --control-port 29001 \
  --cameras 0 --pygame --wait-for-control-client \
  --output-dir outputs/classlab_frames \
  --pose-output outputs/classlab_pose.csv \
  --path-json-output outputs/classlab_path.json
```

两条路线已有 scene package 时都可用 `--stage runtime -- --scene-package ...` 直接运行同一个底层 runtime。ClassLab 的 `run` 本身就是这种直接运行方式；nuScenes 的 `run` 会额外执行场景准备。

## 最小 control 示例

先启动上面任一 `--stage run` 命令，另开终端交互输入油门、转向、制动：

```bash
python scripts/run_scene.py --dataset nuscenes --stage control -- --interactive
```

控制 ClassLab 时将 `--dataset` 改为 `classlab`。固定输入可用 `-- --throttle 0.25 --steer 0 --brake 0`。统一入口只选择并启动工具；control 模式不会自动产生 UniAD 决策或 PDM-Lite 标注。

## 交通流与 Bench2Drive 场景

`tools/bench2drive_real/` 可把 XML 场景语义投影到当前 XODR 路线并生成 `instances/instances_info.json`。在 nuScenes 的 `run` 命令中加 `--instances-info ... --instance-transform-mode mapping-absolute`，或在 ClassLab 的 `prepare` 中加 `--instances-info ...`。原 Town 坐标不能直接当作 nuScenes 或 ClassLab 坐标。完整三步命令与支持的场景类型见[场景构建说明](tools/bench2drive_real/README.md)。

`--stage traffic` 调用 `gs_carla.generate_tm_insert_package`，需要当前 XODR、mapping pose 和 processed scene 等输入。运行 `python scripts/run_scene.py --dataset classlab --stage traffic -- --help` 查看参数。生成轨迹需按场景包的实例坐标模式转换，再以 `instances_info.json` 接入。hero 与交通 actor 由同一 CARLA world 的同步 tick 驱动。

## UniAD 与 PDM-Lite

UniAD 六相机闭环：runtime 加 `--cameras 0 1 2 3 4 5 --render-every-n 1 --downscale 1 --publish-rendered-observations`；另一终端运行 `UniAD_GS_CARLA/run_uniad_tiny.sh --scene-package <本次场景包>`，并设置 `BENCH2DRIVE_ZOO_ROOT`、`BENCH2DRIVE_ROOT`、`UNIAD_TINY_CHECKPOINT`。评测增加 `--evaluation-mode`。UniAD 适配器目前要求 `values.core_pose_coordinate_frame="carla"`；ClassLab 路线需先导出 CARLA 世界坐标的 core pose CSV。详见[UniAD 接入](UniAD_GS_CARLA/README_CN.md)和[评测](UniAD_GS_CARLA/EVALUATION.md)。

PDM-Lite 专家采集：runtime 还需 `--publish-expert-state`；另一终端运行 `PDMLite_GS_CARLA/collect_expert_data.py --scene-package <本次场景包> --expert-config <配置> --output-dir <数据集目录>`。planner 后端需按[采集桥说明](PDMLite_GS_CARLA/README.md)实现；输出包括 `metadata.json`、`frames.jsonl` 和六相机图像。两个控制器不能同时连接同一个控制口。

## 代码与验证

| 目录 | 作用 |
| --- | --- |
| `scripts/run_scene.py` | 单一入口，按数据集和阶段映射到现有 Python 工具 |
| `src/gs_carla/` | scene package、GS 渲染、CARLA runtime、控制 socket 和交通工具 |
| `tools/xodr/` | nuScenes 与 ClassLab 各自的 XODR 构建方法 |
| `UniAD_GS_CARLA/`、`PDMLite_GS_CARLA/` | 控制器与专家采集客户端 |

```bash
python -m compileall -q src scripts tools UniAD_GS_CARLA PDMLite_GS_CARLA
PYTHONPATH=src python -m unittest discover -s tests -v
```

测试覆盖双路线入口映射、ClassLab XODR 和两种 scene package。完整闭环仍需 CARLA server、CUDA 可用的 PyTorch、3DGS checkpoint 和实际数据，才能实测渲染、交通 actor 位姿与控制效果。
