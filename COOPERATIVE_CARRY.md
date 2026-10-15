# OLI 预连接搬箱 demo

## 单 OLI 双手托箱

单机模式复用已训练的 OLI HU_D04 gripper ONNX 和配套 JSON。初始化用带关节限位的
双臂 IK 求出箱子下侧把手位置，腿、腰和头保持策略默认姿态。箱子有自由关节和真实重力，
左右手各通过一个柔性约束连接箱子，不固定机器人根部，也不向箱子添加重力补偿。
这是预连接托箱，不包含接近、闭合夹爪、抬箱或依靠接触摩擦的真实抓取。

在服务器项目根目录，先确认已同步新增的 `solo_scene.py` 和 `oli_solo.json`，再运行：

```bash
uv run --frozen --extra train adapt-carry \
  --config configs/cooperative_carry/oli_solo.json \
  --checkpoint "${OLI_EXPORT_DIR:?请先设置策略导出目录}/oli.onnx" \
  --command 0 0 0 --seconds 10 --pause-on-failure
```

先观察负载下站立，再运行慢速行走：

```bash
uv run --frozen --extra train adapt-carry \
  --config configs/cooperative_carry/oli_solo.json \
  --checkpoint "${OLI_EXPORT_DIR:?请先设置策略导出目录}/oli.onnx" \
  --command 0.1 0 0 --seconds 60 --pause-on-failure
```

也可不加载配置，使用 `--mode solo`；默认箱子尺寸为 0.28 × 0.30 × 0.20 m、质量 0.5 kg，
中心坐标为初始机器人前方 0.32 m、世界高度 1.12 m，左右手位于箱子下侧。
单机默认关闭虚拟领队位置纠偏，直接给策略速度指令；若 0.1 m/s 不启动，可对照测试 0.2 m/s。

单机默认 `--arm-hold 1`，把双臂完整交给 IK 姿态保持，策略继续控制腿、腰和头。
单机 `--arm-hold 0.5` 是策略手臂目标与 **IK 托箱姿态** 各占一半；双机模式仍与原来的默认姿态混合。
保持导出的策略默认姿态、归一化和 ONNX 不变；观测历史与 MoMo 使用实际执行的目标。
但托箱姿态可能超出原策略训练分布，本地诊断成功不代表已学会负载行走；需用服务器的训练策略验证。

`solo_box_center` 和 `box_size` 可在 JSON 中调整。IK 在机器人初始位置固定时求解，
目标无法达到会报错，不通过移动根部或腿来凑姿态。朝向只是 IK 的软目标；约束姿态使用实际求解的手部朝向，
避免初始连接错位。模型沿用原有的自碰撞关闭设置，初始碰撞检查不能代替完整的网格自碰撞验证。

结果目录包含视频（添加 `--video`）、`run.json`、`telemetry.jsonl` 和 `summary.json`。
`run.json` 保存实际 IK 关节目标，遥测增加 `arm_max_position_error`（rad）和
`arm_max_effort_ratio`（相对于配置力矩上限），用于分析姿态跟踪和执行器饱和。
单机没有机器人间距，`formation_error` 与 `spacing_error` 固定为 0，不代表速度跟踪误差。
原成功条件仍要求至少 30 秒且机器人与箱子各移动至少 2 m；静止测试不会标记 `success: true`。

无训练权重的结构诊断（不能当作行走验证）：

```bash
uv run --frozen --extra train adapt-carry \
  --config configs/cooperative_carry/oli_solo.json --pd-check --headless --seconds 0.5
```

单机新增文件为 `deployment/cooperative/solo_scene.py`，复用原来的单机器人地址映射、策略代理、
连接力监测、共享时钟和日志流程。回归检查：

```bash
uv run --frozen --extra train python -m unittest \
  checks.test_solo_carry checks.test_cooperative checks.test_robot_migration
```

## 双 OLI 搬箱

第一版使用两台同向并排的 OLI HU_D04 夹爪模型。左侧机器人使用左手，右侧机器人使用
右手，分别连接中间箱子的两侧把手。两份独立的 ONNX 策略共享一个 MuJoCo 物理世界，
各自维护观测历史、上一动作和 MoMo。没有新增协同强化学习训练。

## 先准备模型

需要你已经在单机 `adapt-sim` 中验证过的 OLI ONNX 和同名 JSON。原 G1 模型不能用于此 demo。
如果目前只有 PT 检查点，先按 `OLI_MIGRATION.md` 导出。复制 ONNX 时必须带上配套 JSON。

在 Linux 的项目根目录安装/同步环境：

```bash
cd ~/workspace/ADAPT
uv sync --frozen --extra train
```

新增命令为 `adapt-carry`，也可使用 `python -m scripts.cooperative_carry`。
运行本身使用 MuJoCo CPU 物理和 ONNX Runtime CPU 推理，不需要选择 CUDA GPU。

## 建议的运行顺序

以下路径替换成服务器上实际导出的路径。变量必须在当前终端或 tmux 窗口中设置。

```bash
OLI_CARRY_POLICY="$HOME/workspace/ADAPT/outputs/你的导出目录/oli.onnx"
ls "$OLI_CARRY_POLICY" "${OLI_CARRY_POLICY%.onnx}.json"
```

### 1. 双机不连接，检查是否能同行

```bash
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_CARRY_POLICY" \
  --connection none --seconds 20
```

此模式保留箱子，但不连接手部，箱子会落在地面。它用于排查双机状态与策略映射，
不承担负载，也不会被判定为搬运成功。默认运动时序包含站立、加速、减速和停止。

### 2. 连接 0.5 kg 箱子，低速直线搬运

```bash
uv run --frozen --extra train adapt-carry \
  --config configs/cooperative_carry/oli_pair.json \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" \
  --pause-on-failure
```

默认参数：机器人间距 1.10 m、箱子质量 0.5 kg、箱体尺寸 0.30×0.44×0.20 m，
前进速度 0.10 m/s，总时长 60 秒。开始站立 2 秒，用 3 秒加速；结束前平滑减速并站立。
窗口显示时间、阶段、队形误差、箱体倾角、抓持误差；红色箭头显示连接作用在手部的
反作用力方向，箭头长度仅用于可视化并有限幅。

相机自动跟随两机中心，可通过鼠标旋转视角。MuJoCo 原生鼠标扰动也可用于推机器人或箱子。
失败后不自动重置；使用 `--pause-on-failure` 时保留现场直到关闭窗口，再写出最终摘要。

### 3. 如搬运侧手臂摆动引起明显拉扯

先比较原始策略，再逐渐增加搬运侧手臂保持权重：

```bash
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" \
  --arm-hold 0.5 --pause-on-failure
```

`--arm-hold 0` 保留原策略，`1` 将搬运侧七个手臂关节的目标保持在策略默认姿态，
中间值为线性混合。非搬运侧手臂和其余关节仍由策略控制。
上一动作观测和 MoMo 使用混合后实际施加的控制。改变手臂控制可能降低行走稳定性，
需要通过对照确定合适值，不能保证保持权重越高越好。

### 4. 录制视频或无界面评测

```bash
MUJOCO_GL=egl uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_CARRY_POLICY" \
  --headless --video
```

视频录制需要 imageio 和 ffmpeg（train 依赖的 MJLab 包含相关依赖）。视频为 1280×720、
30 FPS 的原始场景画面，不包含窗口 HUD 和额外受力箭头。
无界面模式默认尽快运行；增加 `--realtime` 可按物理时间播放。

### 5. 直线搬运通过后再测试转向和负载

```bash
# 缓慢转向；这比直线搬运更难。
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx"  --command 0.1 0 0.05

# 增加箱子质量。
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" --box-mass 1.0

# 比较点连接，允许箱体更自由地转动。
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" --connection connect

# 关闭队形反馈，比较相互拉扯和队形偏离。
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" --no-formation

  uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" \
  --command 0.2 0 0 \
  --seconds 20 \
  --no-formation \
  --pause-on-failure

uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" \
  --command 0.2 0 0 \
  --seconds 60 \
  --no-formation \
  --pause-on-failure
```

## 连接与物理建模

- 两个夹爪均固定在现有 URDF 零位。抓持点位于对应夹爪基座局部 `[0.055,0,0]` m。
  箱子和把手根据策略默认站姿的正运动学对齐，而不是让约束强行拉近手与把手。
  抓持点是初步几何标定，可在配置中调整。
- 箱子有一个真正的自由关节，全部质量由重力和手部约束作用。
  没有世界锚定、悬浮补偿、运行时重写箱子位置或强制机器人跟随。
- 默认使用柔顺 weld，`grasp_time_constant=0.08` s、`grasp_torque_scale=0.03` m。
  `connect` 是可选对照模式。
- 已连接的手部与箱子排除接触，避免虚拟抓持和手指碰撞互相竞争；其余启用的机器人
  碰撞几何、箱体与把手支持跨机器人/物体碰撞，保持原 OLI 的自碰撞关闭设置。
- 这验证预连接搬运，不验证手指摩擦力闭合、自主抓取或脱手。它仍使用 OLI 的等效
  串联模型，不能据此直接用于真机闭链电机控制。

## 运行记录与失败判定

每次运行创建新的 `outputs/carry_时间戳/`，也可用 `--output 新目录` 指定。
已有目录不会覆盖。

```text
run.json           配置、模型哈希、版本与诊断标记
telemetry.jsonl    每控制周期的双机、箱体、抓持和 MoMo 数据
summary.json      完成/成功状态、失败原因、距离和最大误差
demo.mp4          使用 --video 时生成
```

力记录 `payload_grasp_wrench_world` 为每个抓持约束施加到箱子上的世界坐标系力/力矩，
力矩参考点为箱体原点（本模型也为质心）。通过单个等式约束的 `J.T @ lambda` 投影到
箱子自由关节计算，不能把原始等式乘子或整个 `qfrc_constraint` 当作手部三维力。
`box_velocity_world` 的线速度与角速度均在世界系；机器人速度及输入指令在各自基座系。

默认在初始化宽限 1 秒后监测：基座高度、机器人倾角、相对队形误差、箱体离地高度、
箱体倾角、抓持位置误差和连接力。越界时停止，保留失败现场，命令返回非零状态。
此处是 demo 的工程阈值，不是机器人能力的标定结果。

`completed=true` 只表示运行完指定时长。只有连接有效、无失败、至少运行 30 秒，
且箱子和队伍均移动至少 2 米，摘要才报告 `success=true`。固定 PD 诊断永不报告搬运成功。

## 本地检查

```bash
uv run --frozen --extra train python -m unittest discover -s checks -v

# 无需任何训练模型：检查连接与 PD/MoMo 链路，0.5 秒内不进行行走验收。
uv run --frozen --extra train adapt-carry --pd-check --headless --seconds 0.5
```

已完成本地测试：抓持初始对齐、双机状态/控制隔离、单一物理时钟、手臂混合后的动作
反馈、转向内外侧速度差、约束承重与释放、无连接模式、错误配置拒绝；短时双 ONNX
推理、MP4 录制和失败停止链路也已运行。

训练好的 OLI 权重目前仅在用户 Linux 服务器上，本地测试使用固定 PD 与此前的短训练
验证模型。**尚未验证用户的稳定行走策略能否持续搬运箱子。** 服务器测试需按上述顺序
执行，若失败请提供 `run.json`、`summary.json`、日志或视频，再据失败类型调整连接、
手臂保持和队形参数。

## 代码组织

```text
deployment/cooperative/
  config.py          场景和协调参数
  scene_builder.py   双机命名空间、箱体把手、碰撞和约束
  robot_view.py      共享世界中的单机地址映射，不推进物理
  agent.py           每台机器人的策略、历史、MoMo、手臂处理
  formation.py       公共时序、虚拟队形与有限幅速度反馈
  grasp.py           抓持误差和实际约束传力
  simulation.py      统一物理步进、失败监测与运行摘要
  telemetry.py       新目录中的遥测记录
scripts/cooperative_carry.py       GUI、视频和命令行入口
configs/cooperative_carry/oli_pair.json
checks/test_cooperative.py
```

原有单机 `adapt-sim`、训练任务和 G1 模型无需修改。第一版只接入现有 OLI ONNX/JSON
策略接口，尚未实现 CHIP 策略适配、自主抓取、四手搬运或协同策略训练。

@@ test box weight 1-7 sucess >8 fail

matt14@server-Super-Server:~/workspace/ADAPT$ uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_EXPORT_DIR/oli.onnx" --box-mass 10
2026-10-15 16:50:08.318210325 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
t=0.0s settle | formation=0.000m box tilt=0.0deg grasp=0.000m
t=1.0s settle | formation=0.055m box tilt=1.2deg grasp=0.016m
t=2.0s settle | formation=0.054m box tilt=0.2deg grasp=0.016m
t=3.0s accelerate | formation=0.051m box tilt=0.5deg grasp=0.016m
t=4.0s accelerate | formation=0.049m box tilt=0.1deg grasp=0.016m
t=5.0s accelerate | formation=0.042m box tilt=0.2deg grasp=0.016m
t=6.0s carry | formation=0.038m box tilt=0.2deg grasp=0.016m
t=7.0s carry | formation=0.109m box tilt=3.7deg grasp=0.017m
t=8.0s carry | formation=0.169m box tilt=0.5deg grasp=0.017m
t=9.0s carry | formation=0.280m box tilt=0.9deg grasp=0.017m
t=10.0s carry | formation=0.383m box tilt=5.2deg grasp=0.018m
t=11.0s carry | formation=0.403m box tilt=5.5deg grasp=0.017m
t=12.0s carry | formation=0.414m box tilt=2.4deg grasp=0.017m
t=13.0s carry | formation=0.446m box tilt=8.1deg grasp=0.017m
t=14.0s carry | formation=0.425m box tilt=6.5deg grasp=0.018m
t=15.0s carry | formation=0.410m box tilt=2.3deg grasp=0.017m
t=16.0s carry | formation=0.415m box tilt=1.8deg grasp=0.017m
t=17.0s carry | formation=0.414m box tilt=0.5deg grasp=0.017m
t=18.0s carry | formation=0.489m box tilt=0.9deg grasp=0.017m
STOP: formation_separation_exceeded; failed state preserved, no automatic reset.
{
  "completed": false,
  "success": false,
  "diagnostic": false,
  "failure": "formation_separation_exceeded",
  "simulated_seconds": 18.200000000000664,
  "box_displacement": 1.2093613207248912,
  "team_displacement": 1.2032047531878867,
  "min_box_clearance": 0.41270370111363636,
  "max_grasp_error": 0.018695126026148062,
  "max_spacing_error": 0.10546258167397649,
  "max_formation_error": 0.5003584583155318,
  "error": null,
  "interrupted": false
}
Saved: outputs/carry_20261015_165008_674770
X connection to :11 broken (explicit kill or server shutdown).
free(): invalid pointer
matt14@server-Super-Server:~/workspace/ADAPT$ uv run --frozen --extra train adapt-carry   --checkpoint "$OLI_EXPORT_DIR/oli.onnx" --box-mass 20
2026-10-15 16:50:47.070946057 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
t=0.0s settle | formation=0.000m box tilt=0.0deg grasp=0.000m
t=1.0s settle | formation=0.080m box tilt=0.9deg grasp=0.031m
t=2.0s settle | formation=0.088m box tilt=0.5deg grasp=0.031m
t=3.0s accelerate | formation=0.085m box tilt=0.3deg grasp=0.031m
t=4.0s accelerate | formation=0.086m box tilt=0.2deg grasp=0.031m
t=5.0s accelerate | formation=0.093m box tilt=0.4deg grasp=0.031m
t=6.0s carry | formation=0.364m box tilt=2.6deg grasp=0.032m
STOP: formation_separation_exceeded; failed state preserved, no automatic reset.
{
  "completed": false,
  "success": false,
  "diagnostic": false,
  "failure": "formation_separation_exceeded",
  "simulated_seconds": 6.539999999999883,
  "box_displacement": 0.42421481182290915,
  "team_displacement": 0.3696382250259587,
  "min_box_clearance": 0.37374830156459615,
  "max_grasp_error": 0.033136501663897165,
  "max_spacing_error": 0.06710359987153902,
  "max_formation_error": 0.5035095009770213,
  "error": null,
  "interrupted": false
}
Saved: outputs/carry_20261015_165047_462688
matt14@server-Super-Server:~/workspace/ADAPT$ uv run --frozen --extra train adapt-carry   --checkpoint "$OLI_EXPORT_DIR/oli.onnx" --box-mass 12
2026-10-15 16:51:00.977119955 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
t=0.0s settle | formation=0.000m box tilt=0.0deg grasp=0.000m
t=1.0s settle | formation=0.062m box tilt=1.2deg grasp=0.019m
t=2.0s settle | formation=0.062m box tilt=0.3deg grasp=0.019m
t=3.0s accelerate | formation=0.058m box tilt=0.5deg grasp=0.019m
t=4.0s accelerate | formation=0.057m box tilt=0.1deg grasp=0.019m
t=5.0s accelerate | formation=0.050m box tilt=0.2deg grasp=0.019m
t=6.0s carry | formation=0.049m box tilt=0.2deg grasp=0.019m
t=7.0s carry | formation=0.140m box tilt=3.0deg grasp=0.019m
t=8.0s carry | formation=0.206m box tilt=1.0deg grasp=0.020m
t=9.0s carry | formation=0.332m box tilt=1.0deg grasp=0.020m
t=10.0s carry | formation=0.332m box tilt=0.2deg grasp=0.020m
t=11.0s carry | formation=0.390m box tilt=2.4deg grasp=0.019m
t=12.0s carry | formation=0.399m box tilt=2.5deg grasp=0.020m
t=13.0s carry | formation=0.474m box tilt=0.8deg grasp=0.019m
STOP: formation_separation_exceeded; failed state preserved, no automatic reset.
{
  "completed": false,
  "success": false,
  "diagnostic": false,
  "failure": "formation_separation_exceeded",
  "simulated_seconds": 13.440000000000701,
  "box_displacement": 0.7445072693892354,
  "team_displacement": 0.7419401702826038,
  "min_box_clearance": 0.40813177274425577,
  "max_grasp_error": 0.021127265512182577,
  "max_spacing_error": 0.08894545492003414,
  "max_formation_error": 0.5001418445139248,
  "error": null,
  "interrupted": false
}
Saved: outputs/carry_20261015_165101_363329
matt14@server-Super-Server:~/workspace/ADAPT$ uv run --frozen --extra train adapt-carry   --checkpoint "$OLI_EXPORT_DIR/oli.onnx" --box-mass 6
2026-10-15 16:51:20.307716308 [W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform] GPU device discovery failed: device_discovery.cc:89 ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"
t=0.0s settle | formation=0.000m box tilt=0.0deg grasp=0.000m
t=1.0s settle | formation=0.038m box tilt=0.9deg grasp=0.010m
t=2.0s settle | formation=0.036m box tilt=0.2deg grasp=0.010m
t=3.0s accelerate | formation=0.034m box tilt=0.3deg grasp=0.010m
t=4.0s accelerate | formation=0.032m box tilt=0.3deg grasp=0.010m
t=5.0s accelerate | formation=0.025m box tilt=0.2deg grasp=0.010m
t=6.0s carry | formation=0.019m box tilt=0.5deg grasp=0.010m
t=7.0s carry | formation=0.033m box tilt=8.6deg grasp=0.013m
t=8.0s carry | formation=0.109m box tilt=1.5deg grasp=0.011m
t=9.0s carry | formation=0.185m box tilt=1.4deg grasp=0.012m
t=10.0s carry | formation=0.231m box tilt=5.3deg grasp=0.011m
t=11.0s carry | formation=0.251m box tilt=6.2deg grasp=0.012m
t=12.0s carry | formation=0.283m box tilt=4.4deg grasp=0.012m
t=13.0s carry | formation=0.289m box tilt=8.2deg grasp=0.012m
t=14.0s carry | formation=0.287m box tilt=1.8deg grasp=0.011m
t=15.0s carry | formation=0.263m box tilt=0.4deg grasp=0.011m
t=16.0s carry | formation=0.252m box tilt=3.1deg grasp=0.011m
t=17.0s carry | formation=0.271m box tilt=4.0deg grasp=0.012m
t=18.0s carry | formation=0.277m box tilt=1.6deg grasp=0.012m
t=19.0s carry | formation=0.268m box tilt=2.2deg grasp=0.013m
t=20.0s carry | formation=0.258m box tilt=4.7deg grasp=0.012m
t=21.0s carry | formation=0.266m box tilt=0.7deg grasp=0.012m
t=22.0s carry | formation=0.271m box tilt=5.5deg grasp=0.012m
t=23.0s carry | formation=0.250m box tilt=5.0deg grasp=0.012m
t=24.0s carry | formation=0.247m box tilt=1.9deg grasp=0.013m
t=25.0s carry | formation=0.267m box tilt=6.7deg grasp=0.012m
t=26.0s carry | formation=0.279m box tilt=1.9deg grasp=0.011m
t=27.0s carry | formation=0.293m box tilt=4.5deg grasp=0.012m
t=28.0s carry | formation=0.298m box tilt=2.2deg grasp=0.012m
t=29.0s carry | formation=0.303m box tilt=2.5deg grasp=0.011m
t=30.0s carry | formation=0.296m box tilt=0.4deg grasp=0.012m
t=31.0s carry | formation=0.296m box tilt=5.1deg grasp=0.012m
t=32.0s carry | formation=0.330m box tilt=3.0deg grasp=0.012m
t=33.0s carry | formation=0.341m box tilt=6.0deg grasp=0.012m
t=34.0s carry | formation=0.334m box tilt=5.3deg grasp=0.012m
t=35.0s carry | formation=0.374m box tilt=6.6deg grasp=0.012m
t=36.0s carry | formation=0.349m box tilt=5.6deg grasp=0.013m
t=37.0s carry | formation=0.351m box tilt=1.0deg grasp=0.012m
t=38.0s carry | formation=0.343m box tilt=1.7deg grasp=0.011m
t=39.0s carry | formation=0.352m box tilt=3.7deg grasp=0.012m
t=40.0s carry | formation=0.358m box tilt=3.4deg grasp=0.012m
t=41.0s carry | formation=0.345m box tilt=4.3deg grasp=0.012m
t=42.0s carry | formation=0.337m box tilt=1.1deg grasp=0.012m
t=43.0s carry | formation=0.330m box tilt=6.4deg grasp=0.012m
t=44.0s carry | formation=0.334m box tilt=2.6deg grasp=0.012m
t=45.0s carry | formation=0.321m box tilt=4.7deg grasp=0.012m
t=46.0s carry | formation=0.300m box tilt=6.0deg grasp=0.012m
t=47.0s carry | formation=0.291m box tilt=0.5deg grasp=0.012m
t=48.0s carry | formation=0.285m box tilt=8.2deg grasp=0.013m
t=49.0s carry | formation=0.287m box tilt=3.4deg grasp=0.012m
t=50.0s carry | formation=0.272m box tilt=2.3deg grasp=0.011m
t=51.0s carry | formation=0.277m box tilt=5.5deg grasp=0.012m
t=52.0s carry | formation=0.256m box tilt=1.5deg grasp=0.012m
t=53.0s carry | formation=0.245m box tilt=2.2deg grasp=0.012m
t=54.0s carry | formation=0.240m box tilt=1.2deg grasp=0.011m
t=55.0s decelerate | formation=0.233m box tilt=1.9deg grasp=0.011m
t=56.0s decelerate | formation=0.267m box tilt=1.6deg grasp=0.011m
t=57.0s decelerate | formation=0.266m box tilt=1.3deg grasp=0.010m
t=58.0s hold | formation=0.267m box tilt=0.2deg grasp=0.010m
t=59.0s hold | formation=0.268m box tilt=0.2deg grasp=0.010m
t=60.0s hold | formation=0.267m box tilt=0.2deg grasp=0.010m
{
  "completed": true,
  "success": true,
  "diagnostic": false,
  "failure": null,
  "simulated_seconds": 60.00000000001224,
  "box_displacement": 4.964152361777193,
  "team_displacement": 4.958889068403855,
  "min_box_clearance": 0.42113125353716535,
  "max_grasp_error": 0.013248157414391637,
  "max_spacing_error": 0.07044536000962065,
  "max_formation_error": 0.37624870370145663,
  "error": null,
  "interrupted": false
}
Saved: outputs/carry_20261015_165120_686681
^C
