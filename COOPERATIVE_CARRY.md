# 双 OLI 预连接搬箱 demo

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
  --checkpoint "$OLI_CARRY_POLICY" \
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
  --checkpoint "$OLI_CARRY_POLICY" \
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
  --checkpoint "$OLI_CARRY_POLICY" --command 0.1 0 0.05

# 增加箱子质量。
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_CARRY_POLICY" --box-mass 1.0

# 比较点连接，允许箱体更自由地转动。
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_CARRY_POLICY" --connection connect

# 关闭队形反馈，比较相互拉扯和队形偏离。
uv run --frozen --extra train adapt-carry \
  --checkpoint "$OLI_CARRY_POLICY" --no-formation
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
