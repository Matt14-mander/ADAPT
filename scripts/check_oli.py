"""Check OLI model, deployment mapping and MoMo with fixed PD targets (no trained policy)."""

import argparse
import json

import mujoco
import numpy as np

from common.paths import robot_asset_dir, mujoco_path
from deployment.config_manager import DeploymentConfig
from deployment.mujoco_robot import MujocoRobot
from deployment.obs_processor import MoMoObservationGenerator
from observer import MomentumObserver


def make_config(variant="gripper"):
    asset_dir = robot_asset_dir("oli", variant)
    metadata = json.loads((asset_dir / "robot.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(mujoco_path(asset_dir / "scene.xml"))
    names = [model.joint(i).name for i in range(1, model.njnt)]
    params = [metadata["joints"][name] for name in names]
    mg = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
    terms = [("base_lin_vel", 3, 5), ("base_ang_vel", 3, 5), ("projected_gravity", 3, 5),
        ("joint_pos", len(names), 5), ("joint_vel", len(names), 5),
        ("actions", len(names), 1), ("command", 3, 5), ("momo_disturbance", model.nv, 5)]
    return DeploymentConfig(
        task="MOMO-Lumped-Flat-LimX-OLI-HU-D04-EffortScale", joint_names=names,
        timestep=0.005, decimation=4,
        stiffness=[p["stiffness"] for p in params], damping=[p["damping"] for p in params],
        armature=[p["armature"] for p in params], effort_limits=[p["effort_limit"] for p in params],
        default_position=[p["default_position"] for p in params], action_scale=[p["action_scale"] for p in params],
        observations=[dict(name=name, dim=dim, history=history, scale=1.0) for name, dim, history in terms],
        residual_mode="full", residual_scale=[mg] * 3 + [mg * 0.5] * 3 + [p["effort_limit"] for p in params],
        robot="oli", variant=variant, base_body=metadata["base_body"],
        default_base_position=metadata["default_base_position"],
        leg_joint_names=[[name for name in names if name.startswith(side + "_") and any(
            part in name for part in ("hip_", "knee", "ankle_"))] for side in ("left", "right")],
    )


def check(variant="gripper", seconds=5):
    cfg = make_config(variant)
    robot = MujocoRobot(cfg)
    nominal = mujoco.MjModel.from_xml_path(mujoco_path(cfg.scene_path))
    from deployment.config_manager import configure_model
    actuators = configure_model(nominal, cfg)
    observer = MomentumObserver(nominal, dt=cfg.control_dt, reference_body=cfg.base_body)
    state = robot.reset()
    observer.reset(state.qpos, state.qvel)
    history = MoMoObservationGenerator(cfg)
    min_height = float(state.qpos[2])
    for _ in range(max(1, int(seconds / cfg.control_dt))):
        targets = np.asarray(cfg.default_position)
        state = robot.step(targets)
        ctrl = np.zeros(nominal.nu)
        ctrl[actuators] = targets
        observer.update(state.qpos, state.qvel, ctrl)
        residual = observer.local_residual(state.body_rotation)
        obs = history.get_obs(state, np.zeros(3), residual)
        if obs.shape != (1, cfg.observation_dim) or not np.isfinite(obs).all():
            raise RuntimeError("Invalid OLI observation")
        if not np.isfinite(state.qpos).all():
            raise RuntimeError("Non-finite OLI dynamics")
        min_height = min(min_height, float(state.qpos[2]))
    print(f"OLI {variant}: nq={robot.model.nq}, nv={robot.model.nv}, nu={robot.model.nu}; "
          f"actor={cfg.observation_dim}; PD+MoMo={robot.data.time:.2f}s; min height={min_height:.3f}m")
    print("This checks model/control plumbing; it does not validate a locomotion policy or real hardware.")
    return cfg, state, observer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("gripper", "hand"), default="gripper")
    parser.add_argument("--seconds", type=float, default=5)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    check(args.variant, args.seconds)


if __name__ == "__main__":
    main()
