# 3DGS-CARLA Simulator Bridge

这个仓库用于把 DriveStudio/3D Gaussian Splatting 场景接入 CARLA：CARLA 负责道路、车辆、traffic actor 和物理仿真，DriveStudio/3DGS 负责根据 CARLA pose 渲染相机画面。当前代码已经整理成 Python package，源码在 `src/gs_carla/`。

更详细的说明：

- GitHub 发布、环境配置、外部资产放置、部署步骤：见 [docs/GITHUB_DEPLOY.md](docs/GITHUB_DEPLOY.md)
- 完整 CARLA + 3DGS + traffic + external control 数据流：见 [docs/SIMULATOR_BRIDGE_README.md](docs/SIMULATOR_BRIDGE_README.md)

## 仓库结构

```text
3dgs-ue4-simulator/
  README.md
  pyproject.toml
  requirements.txt
  requirements-gpu-cu118.txt
  environment.yml
  .env.example
  coordinate_calibration_identity.json
  docs/
    GITHUB_DEPLOY.md
    SIMULATOR_BRIDGE_README.md
  examples/
    scene_package.example.json
  scripts/
    check_environment.py
  src/gs_carla/
    *.py
  assets/local/      # 本机数据，不上传 GitHub
  outputs/           # 运行输出，不上传 GitHub
```

## 不要上传到 GitHub 的内容

仓库只应该提交代码、文档、依赖清单、小型 JSON 模板和默认标定文件。下面这些内容不要提交：

- 原始采集数据：`data/`、`datasets/`
- DriveStudio checkpoint：`*.pth`、`*.pt`、`*.ckpt`
- 导出的 3DGS 资产：`background_gaussians.pth`、`sky_envlight.pth`、`*.ply`
- CARLA/实验输出：`work_dirs/`、`outputs/`、`results/`、`runs/`
- 图片和视频：`*.png`、`*.jpg`、`*.mp4`、`*.avi`
- 本机路径配置：`.env`、`*.local.json`

这些规则已经写入 `.gitignore`。如果大文件已经被 Git 跟踪，需要用 `git rm --cached path/to/file` 从索引移除，本机文件不会被删除。

## 环境安装

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate 3dgs-carla
pip install -e .
```

如果不用 conda：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gpu-cu118.txt
pip install -e .
```

还需要单独准备 CARLA PythonAPI。复制 `.env.example` 为 `.env` 后按本机路径修改，或直接在 shell 里设置：

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
export PYTHONPATH="$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg"
```

检查环境：

```bash
python scripts/check_environment.py
```

`torch.cuda.is_available()` 必须为 `True`，否则 `gsplat` 真实渲染不能运行。

## 本机资产处理

建议把所有数据放在被 `.gitignore` 排除的 `assets/local/` 下：

```text
assets/local/
  maps/
    classlab_mapping_pose_semantic_4lane.xodr
  3dgs/
    background_gaussians.pth
    sky_envlight.pth
  processed/
    2025-07-17-17-33-26/
  raw/
    2025-07-17-17-33-26/
  instances/
    instances_info.json
  roadlane_tracking/
    corrected_pose_enu.csv
```

然后复制示例 package：

```bash
cp examples/scene_package.example.json assets/local/scene_package.local.json
```

修改 `assets/local/scene_package.local.json` 里的路径，使它指向本机 `.xodr`、processed scene、mapping pose、instances、background 和 sky 文件。`*.local.json` 不会上传 GitHub。

## 部署运行

1. 启动 CARLA server：

```bash
/path/to/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen
```

2. 生成 scene package，也可以直接使用已经改好的 `assets/local/scene_package.local.json`：

```bash
python -m gs_carla.make_3dgs_carla_scene_package \
  --output assets/local/scene_package.local.json \
  --name classlab_3dgs_carla \
  --description "Local CARLA + DriveStudio 3DGS scene" \
  --xodr assets/local/maps/classlab_mapping_pose_semantic_4lane.xodr \
  --instances-info assets/local/instances/instances_info.json \
  --instance-origin-sequence-frame 300 \
  --instance-transform-mode mapping-absolute \
  --background assets/local/3dgs/background_gaussians.pth \
  --sky assets/local/3dgs/sky_envlight.pth \
  --processed-scene assets/local/processed/2025-07-17-17-33-26 \
  --mapping-pose assets/local/raw/2025-07-17-17-33-26/result/mapping/mapping_pose.txt \
  --camera 0 \
  --sequence-origin-frame 0 \
  --processed-origin-frame 0 \
  --core-pose-csv assets/local/roadlane_tracking/corrected_pose_enu.csv
```

3. 启动主 runtime：

```bash
python -m gs_carla.carla_3dgs_scene_runtime \
  --scene-package assets/local/scene_package.local.json \
  --host 127.0.0.1 \
  --port 2000 \
  --start-frame 315 \
  --end-frame 360 \
  --output-dir outputs/runtime_demo \
  --pygame \
  --downscale 3 \
  --max-gaussians 800000 \
  --crop-radius 160
```

4. 另开一个终端发送控制：

```bash
python -m gs_carla.pose_control_client \
  --host 127.0.0.1 \
  --port 25000
```

所有工具都可以用 `python -m gs_carla.<module_name> --help` 查看参数。

## 常用模块

- `gs_carla.export_drivestudio_background`：从 DriveStudio checkpoint 导出 `background_gaussians.pth` 和 `sky_envlight.pth`
- `gs_carla.render_background_gsplat`：用 `gsplat` 渲染 3DGS 背景和天空
- `gs_carla.make_3dgs_carla_scene_package`：生成统一 scene package JSON
- `gs_carla.carla_3dgs_scene_runtime`：主仿真 runtime
- `gs_carla.scene_runtime_external_control`：CARLA + 3DGS + traffic + external control runtime
- `gs_carla.pose_control_client`：外部控制客户端
- `gs_carla.traffic_bbox_projection_from_package`：traffic bbox 输出和投影工具

## 发布前检查

```bash
git status --short
git check-ignore -v assets/local/3dgs/background_gaussians.pth
python scripts/check_environment.py
python -m compileall -q src scripts
```

确认 `git status` 中没有数据、权重、图片、视频、`.env` 或 `*.local.json`。
