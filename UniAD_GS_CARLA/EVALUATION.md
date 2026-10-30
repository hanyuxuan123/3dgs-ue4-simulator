# UniAD × CARLA–3DGS 闭环评测

> 项目总入口：[仓库 README](../README.md)；UniAD 接口、环境和启动方式见 [接入说明](README_CN.md)。

## 评测闭环

```text
runtime 发 observation
  -> UniAD 返回 throttle/steer/brake
  -> CARLA physics tick
  -> 更新 route progress / collision sensor
  -> 未结束：用新位姿渲染下一帧 3DGS
  -> 已结束：刹车、关闭 socket、写 episode_result.json
  -> UniAD 收到 EOF，写 uniad_score.json
```

评测模式有四种主要终止原因：

- `route_completed`：到达所选 core route 末端，成功。
- `collision`：碰撞冲量达到阈值，立即终止，失败。
- `timeout`：超过 episode 时间仍未到达终点，失败。
- `runtime_stopped/interrupted`：窗口关闭、人工中断或其他非正常结束，失败。

runtime 退出前会给 hero 施加全制动，然后销毁 actor。因此到终点或撞车后不会继续向前运动。control socket 关闭后 UniAD 推理循环也会自然退出。

## 启动命令

下面使用本机已存在的 `mini/003 + ParkingCutIn + UniAD-Tiny`。每次实验更换`EVAL_DIR`，不要让新旧结果共用目录。

终端 0 启动 CARLA server：

```bash
cd /home/classlab/TPAMI/CARLA_Latest
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

终端 1 使用 `gussian-sim` 启动 3DGS/runtime：

```bash
conda activate gussian-sim
cd /home/classlab/TPAMI/3dgs-ue4-simulator-2

EVAL_DIR=outputs/nuscenes_global_runtime/003_uniad_tiny_eval_001

python -m gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir "$EVAL_DIR" \
  --export-background \
  --start-frame 0 --end-frame 48 \
  --cameras 0 1 2 3 4 5 \
  --render-every-n 1 --downscale 1 \
  --render-pose-source actor \
  --motion-mode control --control-port 29001 \
  --publish-rendered-observations --no-save-frames \
  --instances-info outputs/bench2drive_real/003_global_control/generated_parkingcutin/instances/instances_info.json \
  --instance-transform-mode mapping-absolute \
  --no-render-dynamic-gaussians \
  --evaluation-mode \
  --evaluation-timeout-seconds 120 \
  --finish-distance 3.0 \
  --collision-impulse-threshold 0.0
```

UniAD 终端使用 `b2d_zoo`，增加结果路径。agent 在 runtime 结束后会自动评分：

```bash
conda activate b2d_zoo
cd /home/classlab/TPAMI/3dgs-ue4-simulator-2

EVAL_DIR=outputs/nuscenes_global_runtime/003_uniad_tiny_eval_001

CUDA_VISIBLE_DEVICES=0 bash UniAD_GS_CARLA/run_uniad_tiny.sh \
  --scene-package "$EVAL_DIR/scene_package.json" \
  --camera-map UniAD_GS_CARLA/camera_map_nuscenes.json \
  --carla-python-api /home/classlab/TPAMI/CARLA_Latest \
  --control-port 29001 \
  --log "$EVAL_DIR/uniad_tiny_control.jsonl" \
  --episode-result "$EVAL_DIR/episode_result.json" \
  --score-output "$EVAL_DIR/uniad_score.json"
```

启动顺序为终端 0、终端 1、终端 2。终端 1 出现`[control-server] listening on 127.0.0.1:29001` 后启动终端 2。

也可以单独重新计算分数：

```bash
python UniAD_GS_CARLA/score_episode.py \
  --episode-result "$EVAL_DIR/episode_result.json" \
  --output "$EVAL_DIR/uniad_score.json"
```

查看结果：

```bash
python -m json.tool "$EVAL_DIR/episode_result.json"
python -m json.tool "$EVAL_DIR/uniad_score.json"
tail -n 3 "$EVAL_DIR/uniad_tiny_control.jsonl"
```

脚本在失败 episode 时返回退出码 2，便于批量任务识别失败；结果 JSON 仍会正常写出。

多条路线完成后可以汇总：

```bash
python UniAD_GS_CARLA/aggregate_scores.py \
  work_dirs/route_*/uniad_score.json \
  --output work_dirs/uniad_summary.json
```

汇总结果包含平均 Driving Score、平均 Route Completion、Success Rate 和碰撞总数。

## 分数定义

当前是透明的轻量评测协议，不冒充官方 Bench2Drive evaluator：

```text
Route Completion = 沿参考 core route 已完成距离 / route 总距离 × 100
Infraction Penalty = 无碰撞为 1；发生碰撞默认 0
Driving Score = Route Completion × Infraction Penalty
Success = 到达 route 终点，并且没有达到阈值的碰撞
```

例如走到 60% 撞车：episode 立即结束，`Route Completion=60`、默认`Driving Score=0`、`Success=false`。需要保留部分碰撞得分时，可向 agent 或`score_episode.py` 传 `--collision-penalty 0.5`。

官方 Bench2Drive 还统计红灯、stop sign、车道侵入、路线偏离、阻塞、scenario完成情况等。若需要论文中可直接对比的官方 Driving Score，必须继续接入这些 criteria及其官方 penalty；当前 JSON schema 可以继续增加对应事实，但不能仅凭 collision声称是官方 Bench2Drive 分数。

## 3DGS 覆盖区域和终点

评测轨迹由 scene package 的 `core_pose_csv` 以及 `--start-frame/--end-frame` 定义。终点是该 route segment 的最后一个 pose，不是系统自动从 checkpoint 推断出的“3DGS边界”。因此：

1. 先离线检查所选 route 全程的六相机 render/alpha。
2. 把 start/end 限制在 3DGS 有效重建区域内。
3. 只有该范围内六相机质量达到要求，这条 episode 才是有效视觉评测。
4. 车辆驶出 GS 覆盖但仍在 route 上时，目前不会自动判失败；应在正式 benchmark
   中加入 coverage mask/alpha threshold criterion，或将 route 截短。

当前 mini/003 只有 cam0 训练图，cam5 几乎无覆盖，所以它可以测试终止与计分代码，但不适合报告 UniAD 的正式闭环性能。

## 输出文件

- `episode_result.json`：runtime 的客观事实、终止原因、进度、碰撞和时间。
- `runtime_pose.csv`：每 tick 位姿、control、最近 route index 和碰撞累计数。
- `runtime_collisions.csv`：碰撞 actor、冲量和发生 step；无碰撞时文件可能不存在。
- `uniad_tiny_control.jsonl`：UniAD 每次 observation/control 记录。
- `uniad_score.json`：轻量 Route Completion、Penalty、Driving Score、Success。
- `uniad_summary.json`：多 episode 聚合结果。
