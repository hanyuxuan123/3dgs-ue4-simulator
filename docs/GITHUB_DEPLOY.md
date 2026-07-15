# GitHub 发布与部署说明

这份说明用于把当前 `3dgs-ue4-simulator` 文件夹整理成一个可公开或私有发布的 GitHub 仓库。仓库只提交代码、文档、小型 JSON 示例和配置模板；原始数据、DriveStudio checkpoint、导出的 3DGS 资产、CARLA 生成缓存和实验输出不要提交。

## 建议仓库结构

当前仓库已经采用 `src` package 布局。源码在 `src/gs_carla/`，命令行入口统一使用 `python -m gs_carla.<module>`。

```text
3dgs-ue4-simulator/
  README.md
  docs/
    GITHUB_DEPLOY.md
    SIMULATOR_BRIDGE_README.md
  examples/
    scene_package.example.json
  scripts/
    check_environment.py
  src/gs_carla/
    *.py
  assets/local/          # 本机数据占位，不提交
  outputs/               # 输出占位，不提交
```

## 应提交到 GitHub 的内容

- `src/gs_carla/*.py` 源码脚本
- `pyproject.toml`
- `README.md`、`docs/*.md`
- `.gitignore`
- `requirements.txt`
- `requirements-gpu-cu118.txt`
- `environment.yml`
- `.env.example`
- `coordinate_calibration_identity.json`
- `examples/*.json`

## 不应提交的内容

- 原始采集数据：`data/`、`datasets/`
- DriveStudio checkpoint：`*.pth`、`*.pt`、`*.ckpt`
- 导出的 3DGS 资产：`background_gaussians.pth`、`sky_envlight.pth`、`*.ply`
- 运行输出：`work_dirs/`、`outputs/`、`results/`、`runs/`
- 图片和视频输出：`*.png`、`*.jpg`、`*.mp4` 等
- 本机路径配置：`.env`、`*.local.json`
- Python 缓存和 IDE 文件：`__pycache__/`、`.vscode/`、`.idea/`

大文件如果确实需要版本管理，建议使用 Git LFS 或外部对象存储，并在 README 里写清下载链接、校验和与放置路径。

## 环境准备

系统侧要求：

- Ubuntu/Linux 环境
- NVIDIA GPU 和可用 CUDA driver
- CARLA server，建议使用和项目匹配的 CARLA 0.9.x 版本
- DriveStudio 已训练好的 checkpoint，或已经导出的 `background_gaussians.pth` 与 `sky_envlight.pth`
- 已处理的 DriveStudio processed scene，包括相机内外参

Python 环境：

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

CARLA PythonAPI 不通过 pip 安装到本仓库。设置方式二选一：

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.15
export PYTHONPATH="$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg"
```

或把本仓库放在 CARLA 工程的工具目录中运行，使脚本能找到 `PythonAPI/carla/dist`。

检查环境：

```bash
python scripts/check_environment.py
```

其中 `torch.cuda.is_available()` 必须是 `True`，否则 `gsplat` 真实渲染不能运行。

## 本机资产放置

建议把本机数据放到被 `.gitignore` 排除的目录，例如：

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
```

然后复制 `examples/scene_package.example.json`，改成本机路径，例如 `assets/local/scene_package.local.json`。这个 local JSON 不要提交。

## 部署与运行

1. 启动 CARLA server：

```bash
/path/to/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen
```

2. 生成或准备 scene package：

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

## GitHub 发布前检查

```bash
git status --short
git check-ignore -v assets/local/3dgs/background_gaussians.pth
python scripts/check_environment.py
```

确认 `git status` 里没有数据、权重、图片、视频或本机 `.env`。如果有大文件已经被 Git 跟踪，需要先从索引移除：

```bash
git rm --cached path/to/large_file
```

这个命令只取消 Git 跟踪，不删除本机文件。
