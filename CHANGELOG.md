# 更新记录

## 0.2.0 — nuScenes + 自采数据的 GS + CARLA 统一方案

- 将 nuScenes 全局坐标场景与自采 ClassLab 场景接入同一套场景包、CARLA 运行时、3DGS 渲染和控制接口。
- 新增 `scripts/run_scene.py`，按数据集与阶段调用地图构建、背景导出、场景准备、运行、交通和控制工具。
- 增加 nuScenes 场景准备、ClassLab OpenDRIVE 构建、交通实例、UniAD 闭环和 PDM-Lite 专家数据采集接口。
- 两类数据保留独立的坐标对齐与资产准备流程；当前版本不直接混合原始数据或训练联合模型。
