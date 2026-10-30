# Bench2Drive-Real 场景 actor 构建

> 总入口：[仓库 README](../../README.md)。运行生成的 actor 见 [nuScenes global runtime](../../src/gs_carla/nuscenes_global_runtime/README.md)；UniAD 闭环评测见 [评测说明](../../UniAD_GS_CARLA/EVALUATION.md)。

这个文件夹实现一个轻量版 Bench2Drive-Real actor benchmark driver，用来把 Bench2Drive 的场景语义迁移到当前 nuScenes global XODR + CARLA dynamics + 3DGS actor 渲染流程中。

当前目标不是完整复刻 Bench2Drive scenario_runner，而是先建立一条可验证的数据链：

```text
Bench2Drive XML
  -> scenario templates
  -> nuScenes/CARLA route anchors
  -> CARLA actor trajectories
  -> 3DGS actor obj_to_world trajectories
  -> patched scene_package.json
```

第一阶段支持 8 类高价值场景：

```text
ParkedObstacle
ConstructionObstacle
Accident
HardBreakRoute
HighwayCutIn
ParkingCutIn
PedestrianCrossing
DynamicObjectCrossing
```

## Bench2Drive 220 XML 是什么

`Bench2Drive/leaderboard/data/bench2drive220.xml` 是 Bench2Drive 的 220 条验证 route 配置文件。它不是一个已经展开好的“所有 actor 逐帧轨迹文件”，而是 CARLA leaderboard/scenario_runner 使用的 route + scenario 配置。

它的基本结构是：

```xml
<routes>
  <route id="1711" town="Town12">
    <waypoints>
      <position x="..." y="..." z="..." />
      ...
    </waypoints>
    <scenarios>
      <scenario name="ParkingCutIn_1" type="ParkingCutIn">
        <direction value="left" />
        <trigger_point x="..." y="..." z="..." yaw="..." />
      </scenario>
    </scenarios>
    <weathers>
      <weather route_percentage="0" ... />
      <weather route_percentage="100" ... />
    </weathers>
  </route>
</routes>
```

本地统计结果：

```text
routes: 220
scenario types: 44
per type: 5 routes
towns: Town01/Town02/Town03/Town04/Town05/Town06/Town07/Town10HD/Town11/Town12/Town13/Town15
```

这些坐标是 Bench2Drive 对应 CARLA towns 的坐标，不是 nuScenes boston-seaport XODR 的坐标。因此本项目不能直接复制 `x/y/z/yaw`，而是抽取 scenario 语义和局部几何关系，再迁移到 nuScenes global route。

## Bench2Drive 原始调用关系

原始 Bench2Drive/leaderboard 大致按下面的链路调用 CARLA：

```text
leaderboard evaluator
  -> 启动/连接 CARLA server
  -> RouteParser.parse_routes_file(bench2drive220.xml)
  -> RouteScenario(world, route_config)
  -> interpolate_trajectory(route_config.keypoints)
  -> spawn hero at first interpolated route transform
  -> filter scenarios whose trigger_point lies on route
  -> instantiate scenario class from srunner/scenarios/*.py
  -> py_trees behavior tree controls scenario actors
```

代码位置：

```text
Bench2Drive/leaderboard/leaderboard/utils/route_parser.py
Bench2Drive/scenario_runner/srunner/scenarios/route_scenario.py
Bench2Drive/scenario_runner/srunner/scenarios/*.py
```

`RouteParser.parse_routes_file()` 做三件事：

- 把 `<route town="...">` 读成一个 `RouteScenarioConfiguration`。
- 把 `<waypoints>/<position>` 读成 `route_config.keypoints`。
- 把 `<scenarios>/<scenario>` 读成 `ScenarioConfiguration`，其中 `trigger_point` 是触发位置，其他 XML 子节点是 scenario 参数。

`RouteScenario` 做四件事：

- 用 `interpolate_trajectory(config.keypoints)` 把稀疏 waypoints 插值成 CARLA 可跟踪的稠密 route。
- 在 route 第一个点 spawn ego/hero。
- 用 `RouteParser.is_scenario_at_route()` 检查 trigger 是否接近 route。原代码阈值约为位置 2 m、yaw 10 deg。
- 根据 `scenario.type` 动态加载 `srunner/scenarios/*.py` 里的同名 scenario class，生成对应 traffic actor 和 behavior tree。

Bench2Drive 还提供了 `leaderboard/scripts/scenario_creator.py` 这类交互式工具。它通过 CARLA spectator 当前所在位置调用：

```text
tmap.get_waypoint(spectator.get_location())
```

然后把最近 waypoint 的 `x/y/z/yaw` 保存成 XML 中的 `trigger_point` 或其他 scenario 参数。这说明 XML 中很多位置点本质上是 CARLA map waypoint 上的人工/半自动标注点。

## 路线点的作用

`waypoints` 是 ego route 的稀疏关键点，不是交通参与者轨迹。

它们的作用是：

- 定义 ego 从哪里开到哪里。
- 被 CARLA `interpolate_trajectory()` 插值成稠密 route。
- 给 agent 提供 global plan。
- 给 route completion / in-route / off-route 等评价指标提供参考路径。
- 给 scenario trigger 判断提供空间参照。

`trigger_point` 则是某个 scenario 在 route 上开始生效的位置。比如 ego 接近 `trigger_point` 后，`ParkingCutIn` 才会生成/激活旁边车辆 cut-in 行为。

因此，对 Bench2Drive-Real 来说：

```text
Bench2Drive waypoints -> 不直接复用坐标
Bench2Drive trigger_point -> 转成 trigger_route_fraction / scenario template
nuScenes core_pose_global.csv -> 作为新的 ego route
nuScenes route anchor -> 替代原 TownXX trigger_point
```

## 本项目如何构建 anchor

本项目的 anchor 是“把 Bench2Drive 场景放到 nuScenes route 上的位置”。它不是从 Bench2Drive XML 直接复制出来的，而是由 `nuscenes_route_anchor_finder.py` 在当前 3DGS/CARLA route 上重新生成。

输入：

```text
scene_package.json
  -> assets.core_pose_csv
  -> assets.xodr

templates.json
  -> scenario_type
  -> trigger_route_fraction
  -> required_topology
```

处理流程：

```text
core_pose_global.csv
  -> 读取 CARLA pose
  -> 转回 nuScenes/global pose
  -> 计算 route_s 累计距离
  -> 可选读取 CARLA XODR waypoint/lane/junction 信息
  -> 按 scenario_type 评分
  -> 输出 anchors.json
```

评分目前是轻量 heuristic：

- `ParkedObstacle / ConstructionObstacle / Accident / HardBreakRoute` 偏好直路、非路口。
- `HighwayCutIn / ParkingCutIn` 偏好直路和存在相邻车道的位置。
- `PedestrianCrossing / DynamicObjectCrossing` 偏好前视相机可见、靠近 ego route 的位置。
- 默认 `--target-mode spread` 会把 8 类场景分散到当前 3DGS route 区间，避免短片段内所有 trigger 堆在开头。
- `--target-mode template` 会使用 Bench2Drive XML 中原始 trigger 在 source route 上的相对进度。

`anchors.json` 中每个 anchor 包含：

```text
scenario_type
frame
route_s
global_pose
carla_pose
map_features
score
reasons
```

后续 `scenario_instance_generator.py` 会把每个 anchor 展开成 actor schedule：

```text
anchor + scenario template
  -> actor initial pose
  -> actor per-frame pose
  -> CARLA trajectory
  -> GS obj_to_world trajectory
```

## 设计原则

CARLA actor 和 3DGS actor 使用同一个轨迹源。

- `actor_trajectories_carla.json`：CARLA world 坐标，给 live CARLA actor replay/校验使用。
- `instances/instances_info.json`：mapping/global absolute `obj_to_world`，给现有 3DGS actor 替换/渲染链使用。
- `bench2drive_real_scenarios.json`：场景实例、anchor、trigger、actor id 的总索引。

这样 CARLA 可以准确输出/校验 actor 轨迹，3DGS actor 可以按同一条轨迹移动。当前 generator 先生成确定性 kinematic schedule；如果后续要使用 CARLA Traffic Manager 或自定义 controller，只需要把 live replay 后记录的实际 pose 再转回 `instances_info.json`。

## 文件说明

```text
common.py
  坐标转换、JSON IO、pose matrix、scene_package patch 公共函数。

bench2drive_xml_parser.py
  解析 Bench2Drive route XML，抽取 portable scenario templates。

nuscenes_route_anchor_finder.py
  沿 nuScenes global runtime 生成的 core_pose_global.csv 找 anchor。
  如果当前 Python 能 import carla，会读取 XODR lane/junction 信息辅助打分。

scenario_instance_generator.py
  把 template + anchor 生成 actor trajectories、instances_info 和 patched scene_package。

traffic_scenario_runtime.py
  连接 live CARLA world，按 actor_trajectories_carla.json replay actors，并记录实际执行 pose。

carla_actual_to_instances.py
  把 live CARLA replay 记录的 actual pose CSV 转回 mapping/global instances_info.json。
```

## 调用流程

以下命令假设你已经用 `prepare_nuscenes_global_scene.py` 生成了：

```text
outputs/nuscenes_global_runtime/003_global_control/scene_package.json
outputs/nuscenes_global_runtime/003_global_control/core_pose_global.csv
```

## 与 nuScenes global runtime 的关系

`nuscenes_global_runtime` 是唯一的地图、ego route 和最终联合运行入口；`bench2drive_real` 只是它的离线场景生成层。两者不是两套 runtime，也不需要先启动一次 control runtime 再重启一次。

推荐顺序如下：

```text
1. prepare_nuscenes_global_scene.py：离线生成 scene_package.json 和 core_pose_global.csv
2. bench2drive_real：从 XML 抽模板，在 core pose route 上生成 instances_info.json
3. run_nuscenes_global_scene.py --instances-info：唯一一次启动 CARLA + 3DGS + 外部控制
```

第一步必须用 prepare-only 命令，而不是下面的 control 命令：后者会启动 CARLA 并等待控制客户端，不能作为后续 actor 生成的前置步骤。

```bash
python -m gs_carla.nuscenes_global_runtime.prepare_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir outputs/nuscenes_global_runtime/003_global_control \
  --start-frame 0 \
  --end-frame 48 \
  --export-background
```

这一步生成：

```text
scene_package.json
core_pose_global.csv
mapping_pose.txt
background/*
```

其中 `core_pose_global.csv` 是 Bench2Drive-Real 的新 ego route，包含 frame 0 到 48 的ego pose：

```text
frame,x,y,z,yaw
0,...
1,...
...
48,...
```

因此可以这样理解：

```text
core_pose_global.csv 第 0 帧     -> 新 route 起点
core_pose_global.csv 第 48 帧    -> 新 route 终点
中间某一帧，例如第 21 帧         -> scenario trigger anchor
该帧的 yaw 或邻近路线方向         -> trigger 方向
```

这里不需要把 Bench2Drive XML 中 Town12/Town13 的原始坐标搬过来。那些坐标属于 Bench2Drive 自己的 CARLA town，不能直接用于 boston-seaport 的 XODR。正确做法是：

```text
Bench2Drive XML
  -> 只抽取 scenario type、参数、trigger 相对 route 进度

当前 nuScenes global route
  -> 使用 core_pose_global.csv 的第 0 到 48 帧作为新 route
  -> 在这条 route 中间选 anchor/trigger
  -> 根据 anchor 生成 actor 轨迹
```

XML 本身不会跟着仿真移动。它只是 template 来源。真正会随仿真时间移动的是生成后的：

```text
generated/instances/instances_info.json
generated/actor_trajectories_carla.json
```

最终由 `run_nuscenes_global_scene.py --instances-info` 写入本次运行的`scene_package.json`；不要把 generator 输出的 `scene_package_b2dr*.json` 当作另一条runtime 入口。底层 runtime 按当前 `source_frame` 更新 actor。也就是说，XML 不在线参与移动；移动的是生成出来的 actor timeline。

### 1. 抽取 Bench2Drive templates

```bash
python tools/bench2drive_real/bench2drive_xml_parser.py \
  --xml Bench2Drive/leaderboard/data/bench2drive220.xml \
  --output outputs/bench2drive_real/003_global_control/templates.json
```

输出：

```text
templates.json
```

它保留 scenario type、trigger 在原 route 上的相对进度、参数字段和拓扑需求，不使用 Bench2Drive Town12/Town13 的绝对坐标。

### 2. 在 nuScenes XODR route 上找 anchor

```bash
python tools/bench2drive_real/nuscenes_route_anchor_finder.py \
  --scene-package outputs/nuscenes_global_runtime/003_global_control/scene_package.json \
  --templates outputs/bench2drive_real/003_global_control/templates.json \
  --output outputs/bench2drive_real/003_global_control/anchors.json
```

输出：

```text
anchors.json
```

`anchors.json` 中每个 anchor 包含：

```text
scenario_type
frame
route_s
global_pose
carla_pose
map_features
score
```

如果 CARLA PythonAPI 不可用，脚本会退化为 route-only anchor matching。

默认 `--target-mode spread` 会把 8 类场景分布到当前 3DGS route 区间内，避免短片段里所有 trigger 都挤在开头。若要使用 Bench2Drive XML 里原始 route trigger fraction，可以改成：

```bash
--target-mode template
```

如果只想先验证一个场景，例如 `ParkingCutIn`，可以只生成一个 anchor：

```bash
python tools/bench2drive_real/nuscenes_route_anchor_finder.py \
  --scene-package outputs/nuscenes_global_runtime/003_global_control/scene_package.json \
  --templates outputs/bench2drive_real/003_global_control/templates.json \
  --output outputs/bench2drive_real/003_global_control/anchors_parkingcutin.json \
  --types ParkingCutIn \
  --target-mode spread
```

对于 frame 0 到 48 这种短片段，`--target-mode spread` 加单个 type 时通常会选路线中点附近作为 trigger。也就是：

```text
trigger_frame ~= 中间帧
trigger_route_fraction ~= 0.5
```

如果当前 Python 环境能 import `carla`，脚本会读取 XODR，调用 `carla.Map(...).get_waypoint()` 检查 lane、junction、相邻车道等信息。
如果不能 import `carla`，会退化为只根据 `core_pose_global.csv` 的 route 几何选点。这个模式可以先跑通数据链，但 lane 判断只是保守默认值。

### 3. 生成 actor benchmark 轨迹

```bash
python tools/bench2drive_real/scenario_instance_generator.py \
  --scene-package outputs/nuscenes_global_runtime/003_global_control/scene_package.json \
  --templates outputs/bench2drive_real/003_global_control/templates.json \
  --anchors outputs/bench2drive_real/003_global_control/anchors.json \
  --output-dir outputs/bench2drive_real/003_global_control/generated \
  --output-scene-package outputs/bench2drive_real/003_global_control/scene_package_b2dr.json
```

输出：

```text
generated/instances/instances_info.json
generated/instances/frame_instances.json
generated/actor_trajectories_carla.json
generated/bench2drive_real_scenarios.json
scene_package_b2dr.json (generator 的调试产物；正常运行不直接使用)
```

`scene_package_b2dr.json` 会在原始 scene package 基础上增加：

```json
"assets": {
  "instances_info": ".../generated/instances/instances_info.json"
},
"values": {
  "instance_transform_mode": "mapping-absolute"
}
```

单场景 `ParkingCutIn` 的最小验证命令：

```bash
python tools/bench2drive_real/scenario_instance_generator.py \
  --scene-package outputs/nuscenes_global_runtime/003_global_control/scene_package.json \
  --templates outputs/bench2drive_real/003_global_control/templates.json \
  --anchors outputs/bench2drive_real/003_global_control/anchors_parkingcutin.json \
  --output-dir outputs/bench2drive_real/003_global_control/generated_parkingcutin \
  --output-scene-package outputs/bench2drive_real/003_global_control/scene_package_b2dr_parkingcutin.json \
  --types ParkingCutIn \
  --max-scenarios 1
```

生成结果中最关键的是：

```text
bench2drive_real_scenarios.json
  记录 trigger_frame、anchor pose、actor ids

actor_trajectories_carla.json
  CARLA 坐标下的 actor 每帧 pose，可用于 live replay/校验

instances/instances_info.json
  mapping/global absolute 的 3DGS actor obj_to_world，每帧渲染时使用

scene_package_b2dr_parkingcutin.json
  generator 的 patched package，可用于直接调试底层 runtime；正常联合运行只传
  instances_info.json 给 wrapper
```

当前 generator 的 `ParkingCutIn` 是一个确定性 kinematic 版本：车辆从 ego 侧前方/旁边开始，随帧数逐渐横向并入 ego route。它不是原始 Bench2Drive `ParkingCutIn` 的 py_trees/TrafficManager 完整复刻，但可以用于验证“trigger 位置、actor 轨迹、CARLA/3DGS 坐标转换、渲染结果是否一致”。

## 接入现有 CARLA + 3DGS runtime

最终运行时，wrapper 会把 `instances_info.json` 写入本次的 `scene_package.json`，由`scene_runtime_external_control.py` 按 `source_frame` 更新 actor，并输出bbox/pose/collision 日志。

核心坐标约定：

```text
instances_info obj_to_world:
  nuScenes/global mapping absolute

scene_package values.instance_transform_mode:
  mapping-absolute

CARLA replay trajectory:
  x_carla = x_global
  y_carla = -y_global
  yaw_carla = -yaw_global
```

对于你们已经在做的 3DGS actor 替换链，直接消费：

```text
generated/instances/instances_info.json
```

每个 actor 的 `frame_annotations.obj_to_world` 就是 GS actor 每帧应该使用的 global pose。

要把生成的场景放回现有 runtime 里跑，推荐直接在 `run_nuscenes_global_scene.py` 命令上追加 `--instances-info`。wrapper 会先 prepare 原始 `scene_package.json`，然后把 `assets.instances_info` 和 `values.instance_transform_mode` 写进去，再启动 `scene_runtime_external_control.py`。

### 4. 启动含 Bench2Drive-Real actor 的联合仿真

生成 actor 后，下面是唯一需要启动的 online runtime。以 `ParkingCutIn` 为例：

```text
--instances-info .../generated_parkingcutin/instances/instances_info.json
--instance-transform-mode mapping-absolute
```

完整命令：

```bash
python -m gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir outputs/nuscenes_global_runtime/003_global_control_b2dr \
  --start-frame 0 \
  --end-frame 48 \
  --cameras 0 \
  --pygame \
  --export-background \
  --motion-mode control \
  --instances-info outputs/bench2drive_real/003_global_control/generated_parkingcutin/instances/instances_info.json \
  --instance-transform-mode mapping-absolute \
  --no-render-dynamic-gaussians
```

这条命令做的事情是：

```text
1. prepare_nuscenes_global_scene.py
   -> 生成 scene_package.json、core_pose_global.csv、background/*

2. run_nuscenes_global_scene.py
   -> 把 generated_parkingcutin/instances/instances_info.json 写入 scene_package.json

3. scene_runtime_external_control.py
   -> 加载 XODR/CARLA world
   -> 加载 3DGS background/sky
   -> spawn hero
   -> 读取 Bench2Drive-Real extra instances
   -> 等待外部 throttle/steer/brake 控制
   -> 用 nearest source_frame 同步 ego 进度、3DGS 渲染和 extra actor
```

仍然从另一个 terminal 发送控制：

```bash
python -m gs_carla.nuscenes_global_runtime.send_control \
  --throttle 0.25 \
  --steer 0.0 \
  --brake 0.0
```

如果要手动持续输入控制：

```bash
python -m gs_carla.nuscenes_global_runtime.send_control --interactive
```

### 运行时参数

主要输入参数：

```text
--xodr
  当前 CARLA OpenDRIVE 地图。这里是 boston-seaport 的 nuScenes global XODR。

--checkpoint
  DriveStudio/3DGS checkpoint。prepare 阶段会从这里导出 background gaussians。

--config
  3DGS/DriveStudio 配置文件，用来解释 checkpoint、网络结构和数据配置。

--processed-scene
  nuScenes processed scene 路径。这里读取 lidar_pose、camera extrinsics/intrinsics 等原始帧信息。

--output-dir
  本次 runtime 工作目录。会写入 scene_package.json、core_pose_global.csv、runtime_pose.csv、runtime_path.json、runtime_frames/ 等。

--start-frame / --end-frame
  选择当前 route 的帧范围。这里 0 到 48 会形成一条短 ego route。

--cameras
  要渲染的 camera id。`--cameras 0` 只渲染 cam0。

--pygame
  打开 pygame 实时显示窗口。

--export-background
  如果 output-dir/background 下没有 3DGS background，就从 checkpoint 导出。

--motion-mode control
  使用外部控制模式。runtime spawn hero 后等待 send_control.py 发 throttle/steer/brake。
```

Bench2Drive-Real 相关参数：

```text
--instances-info
  额外 actor 的 instances_info.json。它来自 scenario_instance_generator.py。
  里面存的是每个 actor 在每个 frame 的 obj_to_world 4x4 矩阵。

--instance-transform-mode mapping-absolute
  告诉 runtime：instances_info 里的 obj_to_world 是 nuScenes/global mapping absolute 坐标。

--no-render-dynamic-gaussians
  只渲染 3DGS background/sky，不 splat checkpoint 中的动态 RigidNodes。
  CARLA actor 的生成、运动学更新、bbox 和碰撞记录保持开启，适合先验证 actor 轨迹。
```

control 模式下 wrapper 隐式传给底层 runtime 的关键参数：

```text
--source-frame-mode nearest
  用 live CARLA ego 当前 x/y 找 core_pose_global.csv 中最近的帧。
  这个 source_frame 同时驱动 3DGS 渲染帧和 Bench2Drive-Real extra actor 帧。

--wait-for-control-client
  等待外部控制客户端连接。

--no-exit-on-path-end
  到达参考 route 末端后不立刻退出，方便持续调试。
```

wrapper 在 `--motion-mode control` 下会使用：

```text
--source-frame-mode nearest
--wait-for-control-client
--no-exit-on-path-end
```

其中 `nearest` 是关键：runtime 会把 live CARLA ego 当前 `x/y` 匹配到 `core_pose_global.csv` 中最近的 frame，然后用这个 frame 更新 Bench2Drive-Real actor。因此 extra instance 不是按固定墙钟时间硬播，而是跟随 ego 在 route 上的空间进度。

如果需要更底层地调试，也可以直接调用 `scene_runtime_external_control.py` 并传入 patched scene package：

```bash
python -m gs_carla.scene_runtime_external_control \
  --scene-package outputs/bench2drive_real/003_global_control/scene_package_b2dr_parkingcutin.json \
  --host 127.0.0.1 \
  --port 2000 \
  --control-port 29001 \
  --core-start-frame 0 \
  --core-end-frame 48 \
  --source-frame-mode nearest \
  --source-frame-fps 10.0 \
  --follow-speed-kmh 12.0 \
  --control-dt 0.05 \
  --render-every-n 1 \
  --cameras 0 \
  --pygame \
  --wait-for-control-client \
  --no-exit-on-path-end \
  --output-dir outputs/bench2drive_real/003_global_control/runtime_frames_b2dr \
  --pose-output outputs/bench2drive_real/003_global_control/runtime_pose_b2dr.csv \
  --path-json-output outputs/bench2drive_real/003_global_control/runtime_path_b2dr.json
```

`--source-frame-mode nearest` 的含义是：用 live CARLA ego 当前所在位置去找 `core_pose_global.csv` 里最近的参考帧，然后用这个 `source_frame` 更新 3DGS/CARLA actor。这样 actor 会跟随 ego 沿 route 的进度出现，而不是单纯按墙钟时间播放。

如果只是想不管 ego 开到哪里都让场景按时间播放，可以改成：

```bash
--source-frame-mode timeline
```

两种模式的区别：

```text
nearest:
  actor frame 由 ego 当前离哪一个 core pose 最近决定。
  适合 control/path-follow，因为 ego 速度可能和原始 nuScenes 时间轴不同。

timeline:
  actor frame 由仿真步数和 source_frame_fps 决定。
  适合 kinematic replay 或冻结 ego 时检查动态 actor。
```

因此，“XML 能不能跟着仿真一起移动”的准确答案是：

```text
XML 不移动，也不应该移动。
trigger/actor 被离线展开成 route-relative 的 frame schedule。
runtime 根据 source_frame 选择当前应该显示哪一帧 actor。
在 nearest 模式下，source_frame 跟着 ego 在 route 上的空间进度移动。
```

## CARLA 实时回放与校验

如果想单独让 CARLA 按生成轨迹 replay actors，并记录实际 CARLA actor pose：

```bash
python tools/bench2drive_real/traffic_scenario_runtime.py \
  --actor-trajectories outputs/bench2drive_real/003_global_control/generated/actor_trajectories_carla.json \
  --output-csv outputs/bench2drive_real/003_global_control/generated/carla_replay_actual.csv \
  --output-json outputs/bench2drive_real/003_global_control/generated/carla_replay_actual.json \
  --draw-debug
```

如果另一个 runtime 已经在 synchronous mode tick CARLA，不要加 `--tick-world`。
如果这个脚本单独测试，可以加：

```bash
--tick-world --real-time
```

如果要从 CARLA 实际输出反驱动 GS actor，把 `carla_replay_actual.csv` 转回 mapping/global `instances_info.json`：

```bash
python tools/bench2drive_real/carla_actual_to_instances.py \
  --csv outputs/bench2drive_real/003_global_control/generated/carla_replay_actual.csv \
  --output-dir outputs/bench2drive_real/003_global_control/generated/instances_from_carla_actual \
  --use actual \
  --route-start-frame 0
```

转换关系仍然是：

```text
x_global   =  x_carla
y_global   = -y_carla
yaw_global = -yaw_carla
```

## 当前层次和限制

当前实现属于 benchmark driver 第一版：

- 已实现：模板解析、route anchor、8 类 scenario schedule、CARLA/GS 双轨迹输出、scene package patch。
- 已实现：CARLA live replay 校验入口。
- 已实现：CARLA actual pose CSV 到 GS `instances_info.json` 的反向转换。
- 未实现：完整 Bench2Drive behavior tree。
- 未实现：Traffic Manager 背景车流。
- 未实现：动态 actor 的像素级合成；这里假设外部 3DGS actor 替换链已经能按 `instances_info.json` 渲染 actor。

## 方法概述

Bench2Drive-Real 将 Bench2Drive 中源于合成 CARLA 城镇的交互场景语义，重定向到由真实数据重建的 3DGS 场景地图中。该方法通过拓扑感知的 anchor 匹配，以及统一的 CARLA/GS actor 轨迹接口，同时保持 CARLA 闭环动力学、真实 3DGS 背景渲染和可控交通参与者运动。由此构建出的 benchmark 能够在 nuScenes 对齐的 OpenDRIVE 道路上进行可复现压力测试，使感知模块面对真实几何与纹理，使规划控制模块面对 Bench2Drive 风格的交互场景，并支持坐标一致性、可见性一致性和行为一致性的量化评估。

## UniAD-Tiny 六相机闭环接入

固定 `send_control.py` 只发送人工指定 control；真正 agent 闭环使用相同的 29001 JSON-line socket，由 UniAD 根据 3DGS 六相机和 CARLA 运动学逐 tick 输出 control。CARLA server、runtime、agent 应分别运行；runtime 使用 `gussian-sim` 环境，UniAD使用 `b2d_zoo` 环境。

runtime 命令中必须把原来的 `--cameras 0` 改为六相机，并启用 observation 发布：

```bash
conda activate gussian-sim
cd /home/classlab/TPAMI/3dgs-ue4-simulator-2

python -m gs_carla.nuscenes_global_runtime.run_nuscenes_global_scene \
  --xodr /media/classlab/2E0CECBF0CEC82E7/0922_test/boston-seaport_nuscenes_global.xodr \
  --checkpoint /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/checkpoint_final.pth \
  --config /media/classlab/2E0CECBF0CEC82E7/0922_test/0917_test/config.yaml \
  --processed-scene /media/classlab/2E0CECBF0CEC82E7/Classlab_Simulator_Data/nuscenes/mini/003 \
  --output-dir outputs/nuscenes_global_runtime/003_global_control_b2dr \
  --start-frame 0 --end-frame 48 \
  --cameras 0 1 2 3 4 5 \
  --render-every-n 1 --downscale 1 \
  --render-pose-source actor \
  --pygame --export-background \
  --motion-mode control --control-port 29001 \
  --publish-rendered-observations --no-save-frames \
  --instances-info outputs/bench2drive_real/003_global_control/generated_parkingcutin/instances/instances_info.json \
  --instance-transform-mode mapping-absolute \
  --no-render-dynamic-gaussians
```

第二个终端启动 UniAD-Tiny：

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

脚本自动设置 `PYTHONNOUSERSITE=1` 和`MPLCONFIGDIR=/tmp/uniad_gs_matplotlib`，默认加载`UniAD_GS_CARLA/checkpoints/uniad_tiny_b2d.pth`。检查 JSONL 中`state=inference`，并确认 runtime 日志收到非固定 control，即可证明 control 来自UniAD，而不是 warmup 制动。完整说明见[`UniAD_GS_CARLA/README_CN.md`](../../UniAD_GS_CARLA/README_CN.md)。

当前 mini/003 仅有 cam0 训练图，后向相机尤其 cam5 的 GS 覆盖不足，因此现状是“六相机接口闭环已跑通”，不是可靠的 360° 感知结果。正式评测应使用六相机共同训练的 checkpoint，或明确实现缺失视角的 CARLA RGB fallback。

## 替换 UniAD 与采集 PDM-Lite 专家数据

替换 TransFuser++、TCP、VAD 等 agent 时，runtime 保持不变，只需要新建模型适配器：

```text
observation JSON
  -> 模型自己的 RGB/LiDAR/GPS/IMU/route 预处理
  -> checkpoint inference
  -> 模型自己的 trajectory controller
  -> throttle/steer/brake JSON
```

必须严格匹配模型训练时的 sensor suite。尤其 TransFuser 系模型通常需要 LiDAR；当前接口只有 GS RGB 与 CARLA 运动学，应新增同步 LiDAR/depth，或使用 camera-only模型，不能用零点云代替。

PDM-Lite 专家采集入口位于仓库同级目录[`PDMLite_GS_CARLA`](../../../PDMLite_GS_CARLA/README.md)。采集时 runtime 再增加`--publish-expert-state`，由采集器作为唯一 control client：专家读取 CARLA 真值route/traffic state 产生 control，runtime 执行 control，采集器同时固化对应的六路3DGS 图像及标注。这样得到的数据可用于 imitation learning，但 student 训练输入必须去掉 route/traffic privileged truth，避免信息泄漏。

这里的 PDM-Lite 特指[OpenDriveLab/DriveLM 的 DriveLM-CARLA 实现](https://github.com/OpenDriveLab/DriveLM/tree/DriveLM-CARLA/pdm_lite)，不是 NAVSIM PDM-Closed。官方实现绑定 CARLA Leaderboard 2.0/CARLA 0.9.14 的`CarlaDataProvider`；接到本 XODR runtime 前仍需实现本地 backend wrapper，并验证CARLA 版本、路线注册、交通灯/stop sign 与同步 tick 的兼容性。
