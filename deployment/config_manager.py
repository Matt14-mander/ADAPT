"""Portable deployment settings. Loading these does not import the training stack."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np

from common.paths import robot_asset_dir, task_robot


@dataclass
class DeploymentConfig:
    task: str
    joint_names: list[str]
    timestep: float
    decimation: int
    stiffness: list[float]
    damping: list[float]
    armature: list[float]
    effort_limits: list[float]
    default_position: list[float]
    action_scale: list[float]
    observations: list[dict]
    residual_mode: str
    residual_scale: list[float]
    observer_gain: float = 3.0
    filter_order: int = 2
    cutoff_hz: float = 1.0
    model_sha256: str = ""
    robot: str = "g1"
    variant: str = "gripper"
    base_body: str = "pelvis"
    default_base_position: list[float] = field(default_factory=lambda: [0, 0, 0.76])
    leg_joint_names: list[list[str]] = field(default_factory=list)

    @property
    def scene_path(self):
        return robot_asset_dir(self.robot, self.variant) / "scene.xml"

    @property
    def leg_residual_indices(self):
        if self.leg_joint_names:
            return [[6 + self.joint_names.index(name) for name in side] for side in self.leg_joint_names]
        # Compatibility with the original G1 exports.
        return [list(range(6, 12)), list(range(12, 18))]

    @property
    def control_dt(self):
        return self.timestep * self.decimation

    @property
    def observation_dim(self):
        return sum(term["dim"] * term["history"] for term in self.observations)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path, model_path):
        self.model_sha256 = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_env(cls, task, env, model):
        """Extract the single-free-base, one-actuator-per-joint deployment contract."""
        names = [
            model.joint(i).name
            for i in range(model.njnt)
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        ]
        short = [name.split("/")[-1] for name in names]
        robot = env.scene.entities["robot"]

        def gains(field):
            return np.asarray(
                [
                    next(
                        getattr(group, field)
                        for group in robot.articulation.actuators
                        if any(re.fullmatch(p, name) for p in group.target_names_expr)
                    )
                    for name in short
                ],
                dtype=np.float32,
            ).tolist()

        def patterns(values, default=0.0):
            if not isinstance(values, dict):
                return np.full(len(names), values, dtype=np.float32).tolist()
            return np.asarray(
                [
                    next((v for p, v in values.items() if re.fullmatch(p, name)), default)
                    for name in short
                ],
                dtype=np.float32,
            ).tolist()

        effort = gains("effort_limit")
        mg = float(model.body_mass.sum() * abs(model.opt.gravity[2]))
        scales = np.array([mg] * 3 + [mg * 0.5] * 3 + effort, dtype=np.float32).tolist()
        dims = dict(
            base_lin_vel=3,
            base_ang_vel=3,
            projected_gravity=3,
            joint_pos=len(names),
            joint_vel=len(names),
            actions=len(names),
            command=3,
        )
        terms = env.observations["actor"].terms
        mode = "none"
        if "momo_disturbance" in terms:
            term = terms["momo_disturbance"]
            func = term.func.__name__
            if func == "momo_dist_effort_scaled_localframe":
                mode = "full"
                dims["momo_disturbance"] = model.nv
                scales[3:6] = np.full(3, mg * term.params["l_ref"], dtype=np.float32).tolist()
            elif func == "zero_momo_leg_effort_scaled_envelope":
                mode = "zero_leg"
                dims["momo_disturbance"] = 14
            else:
                raise ValueError(f"Not a released observer observation: {func}")
        observations = [
            dict(
                name=name,
                dim=dims[name],
                history=int(term.history_length or 1),
                scale=np.asarray(1.0 if term.scale is None else term.scale).tolist(),
            )
            for name, term in terms.items()
        ]
        cfg = cls(
            task,
            names,
            env.sim.mujoco.timestep,
            env.decimation,
            gains("stiffness"),
            gains("damping"),
            gains("armature"),
            effort,
            patterns(robot.init_state.joint_pos),
            patterns(env.actions["joint_pos"].scale, 1.0),
            observations,
            mode,
            scales,
        )
        cfg.robot = task_robot(task)
        cfg.base_body = "base_link" if cfg.robot == "oli" else "pelvis"
        cfg.default_base_position = list(robot.init_state.pos)
        cfg.leg_joint_names = [
            [name for name in names if name.startswith(side + "_") and any(
                part in name for part in ("hip_", "knee", "ankle_"))]
            for side in ("left", "right")
        ]
        return cfg


def configure_model(model, cfg):
    """Synchronize PD and armature, returning joint-order -> actuator-order mapping."""
    names = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    ]
    n = len(cfg.joint_names)
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if (names != cfg.joint_names or (model.nq, model.nv, model.nu) != (n + 7, n + 6, n)
            or len(free) != 1 or free[0] != 0
            or np.any(~np.isin(model.jnt_type[1:], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]))):
        raise ValueError("Expected one free base followed by scalar joints, each with one actuator, in exported order")
    for field_name in ("stiffness", "damping", "armature", "effort_limits", "default_position", "action_scale"):
        values = np.asarray(getattr(cfg, field_name))
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {field_name}: expected {n} finite values")
    actuators = []
    for i, name in enumerate(names):
        joint = model.joint(name)
        ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint.id)
        if len(ids) != 1:
            raise ValueError(f"Expected one position actuator for {name}")
        act = int(ids[0])
        actuators.append(act)
        model.actuator_gainprm[act, 0] = cfg.stiffness[i]
        model.actuator_biasprm[act, 1:3] = [-cfg.stiffness[i], -cfg.damping[i]]
        model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_forcelimited[act] = True
        model.actuator_forcerange[act] = [-cfg.effort_limits[i], cfg.effort_limits[i]]
        model.dof_armature[joint.dofadr[0]] = cfg.armature[i]
    model.opt.timestep = cfg.timestep
    return np.asarray(actuators)
