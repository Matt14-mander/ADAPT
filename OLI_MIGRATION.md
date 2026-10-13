# OLI HU_D04 夹爪版：初步仿真迁移

本分支增加 OLI 的 Stage 1 平地行走任务，保留原 G1 任务和已发布模型。
这是模型、训练接口和部署接口的迁移；没有可直接运行的 OLI 行走检查点。
必须重新训练。G1 的 PT、ONNX、JSON 不能用于 OLI。

## 已实现范围

- 使用官方 HU_D04 夹爪 URDF，8 个夹爪关节固定在 URDF 零位；保留各部件质量和惯量。
- 保留 31 个身体关节（包含两个头部关节），每关节一个位置 PD 执行器。
- 单浮动基座：`nq=38`、`nv=37`、`nu=31`。
- baseline actor 输入 401 维，ADAPT actor 输入 586 维；MoMo 残差为 37 维，5 帧历史。
- 原始 URDF 的足底碰撞，以及官方 MJCF 中与连杆名称匹配的身体局部基本碰撞几何。
  仅启用机器人与地面的碰撞，初版不启用自碰撞。
- GPU 批量 MoMo、共享编码器观测、模型导出、独立 MuJoCo/ONNX 部署。
- 默认初始速度范围：前后 ±0.5 m/s、侧向 ±0.3 m/s、转向 ±0.5 rad/s。

## 模型与参数来源

官方模型：<https://github.com/limxdynamics/humanoid-description>

固定提交：`80fbd2d650c43c38916979c2a6eab7a4c79e38de`。

官方行走参数：<https://github.com/limxdynamics/humanoid-rl-deploy-python>

固定提交：`bc76b0fe8399ecb84ce5492a6c385feb7d7a17a5`。

源 URDF、SRDF、MJCF 和行走 YAML 在 `models/oli/gripper/source/`。
模型仓库的 Apache-2.0 许可证在 `models/oli/gripper/LICENSE`。
部署配置的 Apache-2.0 许可证在 `models/oli/gripper/source/DEPLOY_LICENSE`。

PD 增益和站姿参考官方行走配置；按关节名称生成，避免将官方 SDK 顺序直接当作
MuJoCo 关节顺序。反射惯量暂用 SRDF 的 `mass * gear_ratio**2` 约定。
执行器力矩上限使用 URDF 的 `effort`；动作尺度采用 ADAPT 的
`0.25 * effort_limit / stiffness`。因此腰踝的尺度可能与官方行走 YAML 不同：
YAML 的部分上限为 80 Nm，URDF 为 42 Nm，初版采用 URDF 的较低值。
URDF 的速度上限保存在元数据中，当前位置执行器没有模拟电机转速相关的力矩衰减。

**重要的建模边界：** 官方原生 HU_D04 MJCF 包含踝部/腰部闭链机构、被动关节和
电机坐标。本分支采用官方 URDF 的等效串联关节模型，不包含闭链传动的状态相关
映射。它用于初步训练与仿真接口验证；实际电机控制、传动映射和参数校准需要另行完成。
冻结夹爪的 URDF 零位也应与实际机械姿态核对。

## Linux 使用步骤

在此分支项目根目录中安装依赖：

```bash
uv sync --frozen --extra train
```

模型已包含在分支中，无需首次运行就下载。需要从固定官方源重新生成时：

```bash
uv run --frozen --extra train python -m scripts.prepare_oli --variant gripper
```

检查模型、坐标系、动作映射和 MoMo：

```bash
uv run --frozen --extra train python -m unittest discover -s checks -v
uv run --frozen --extra train python -m scripts.check_oli --seconds 5
uv run --frozen --extra train python -m checks.check_oli_tasks --device cuda:0
```

`check_oli` 使用固定位置目标，不运行行走策略。它检查有限值与观测维度，并报告
最低基座高度；不保证站立平衡，机器人可能倒下。不能把此命令通过理解为行走成功。

先做小规模 baseline 训练：

```bash
uv run --frozen --extra train adapt-train \
  Ref-NoMoMo-Flat-LimX-OLI-HU-D04 \
  --env.scene.num-envs 128 \
  --agent.max-iterations 20 \
  --agent.run-name smoke \
  --gpu-ids "[0]"
```

再检查 ADAPT 训练：

```bash
uv run --frozen --extra train adapt-train \
  MOMO-Lumped-Flat-LimX-OLI-HU-D04-EffortScale \
  --env.scene.num-envs 128 \
  --agent.max-iterations 20 \
  --agent.run-name smoke \
  --gpu-ids "[0]"
```

完成流程检查后，根据显存增大环境数（例如 4096），将迭代数改为 15000。
训练输出分别位于 `logs/rsl_rl/oli_baseline/` 和 `logs/rsl_rl/oli_adapt/`。
短训练只能验证执行流程，不能验证行走效果。

回放训练后的检查点，替换示例路径：

```bash
uv run --frozen --extra train adapt-play \
  MOMO-Lumped-Flat-LimX-OLI-HU-D04-EffortScale \
  --checkpoint /path/to/oli/model.pt \
  --viewer none --steps 500
```

导出与独立仿真：

```bash
uv run --frozen --extra train adapt-export \
  MOMO-Lumped-Flat-LimX-OLI-HU-D04-EffortScale \
  --checkpoint /path/to/oli/model.pt \
  --output /path/to/new/export/oli.onnx

uv run --frozen --extra train adapt-sim \
  --checkpoint /path/to/new/export/oli.onnx \
  --command 0.3 0 0 --headless --seconds 10
```

导出的 JSON 包含机器人型号、模型选择、基座名称、初始高度、关节顺序和腿部映射。
ONNX 和同名 JSON 必须配套。原有 `--policy adapt/baseline/lightstep` 仍指向已发布的
G1 策略；OLI 使用 `--checkpoint`。

## 暂未迁移

- Stage 2 手部负载、CleanMix 与轻步训练，`adapt-two-stage` 仍为 G1 流程。
- 灵巧手训练任务（生成脚本预留 hand 模型选项，但当前注册任务固定为夹爪版本）。
- OLI 真机 SDK、模式切换、闭链电机映射、真实速度估计和控制参数校准。
- 当前 `adapt-real` 只支持 G1，并会拒绝 OLI 配置。

## 本地验证记录

已通过：5 项 CPU 回归检查、OLI baseline / ADAPT GPU 环境步进、训练/部署关节
与惯量对应检查、1 次 PPO 更新、检查点恢复、5 步 PyTorch/ONNX 输出一致性检查，
以及导出策略在独立 MuJoCo 部署端的步进。原 G1 的 ADAPT、baseline、lightstep
ONNX 仿真也通过回归检查。

本地验证使用 Windows、Python 3.12、PyTorch 2.7、MuJoCo 3.6 和本项目固定版本的
MJLab / MuJoCo Warp / RSL-RL。Windows 的中文路径问题仅在临时验证环境处理。
仍需在 Linux 的 Python 3.10 / PyTorch 2.9.1 锁定环境运行上述小规模检查。
这些结果验证执行链路，不代表策略已学会稳定行走，也不代表真机已经验证。

## 主要代码入口

- `scripts/prepare_oli.py`：固定官方资产下载和模型转换。
- `tasks/oli_task/robot.py`：MJLab 机器人配置。
- `tasks/oli_task/env_cfg.py`：OLI 平地任务。
- `tasks/oli_task/__init__.py`：baseline / ADAPT 注册及 OLI 环境类。
- `deployment/config_manager.py`：配置导出与通用关节校验。
- `scripts/check_oli.py`：无神经网络的 PD/MoMo 流程检查。
- `checks/test_robot_migration.py`：质量、关节顺序、传感器坐标系、G1 兼容性、批量/单机 MoMo 一致性检查。
