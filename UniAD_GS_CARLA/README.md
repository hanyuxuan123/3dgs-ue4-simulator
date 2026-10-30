# UniAD × CARLA–3DGS

> 项目总入口：[仓库 README](../README.md)。

本目录将 Bench2DriveZoo UniAD 接入 CARLA 动力学与 3DGS 六相机渲染的同步闭环。完整启动命令、环境、坐标约定、目录结构和替换其他 agent 的方法见[README_CN.md](README_CN.md)。碰撞退出、终点成功、超时和自动计分见 [EVALUATION.md](EVALUATION.md)。

核心链路为：

```text
CARLA 运动学 + 3DGS 六相机
       -> UniAD 适配器、模型与 PID
       -> 油门、转向、制动
       -> CARLA 物理更新
       -> 新位姿驱动下一帧 3DGS 渲染
```

Tiny 推荐入口：

```bash
conda activate b2d_zoo
cd /home/classlab/TPAMI/3dgs-ue4-simulator-2

CUDA_VISIBLE_DEVICES=0 bash UniAD_GS_CARLA/run_uniad_tiny.sh \
  --scene-package outputs/nuscenes_global_runtime/003_global_control_b2dr/scene_package.json \
  --camera-map UniAD_GS_CARLA/camera_map_nuscenes.json \
  --carla-python-api /home/classlab/TPAMI/CARLA_Latest \
  --control-port 29001 \
  --log outputs/nuscenes_global_runtime/003_global_control_b2dr/uniad_tiny_control.jsonl
```

`run_uniad_tiny.sh` 自动隔离用户 site-packages，并使用`UniAD_GS_CARLA/checkpoints/uniad_tiny_b2d.pth`。runtime 必须以六相机、`--publish-rendered-observations` 和相同 control port 启动。
