"""Address-based single-robot views into a shared MjData; never step physics here."""

import mujoco
import numpy as np

from deployment.obs_processor import inverse_rotate
from deployment.robot_interface import RobotState


class RobotView:
    def __init__(self, model, data, cfg, prefix, *, rng=None, encoder_noise=False):
        self.model, self.data, self.cfg, self.prefix = model, data, cfg, prefix
        self.rng = rng or np.random.default_rng(0)
        self.encoder_noise = encoder_noise
        self.body_id = model.body(prefix + cfg.base_body).id
        roots = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                 and model.jnt_bodyid[j] == self.body_id]
        if len(roots) != 1:
            raise ValueError(f"Expected one free root for {prefix}")
        root = roots[0]
        self.root_q = int(model.jnt_qposadr[root])
        self.root_v = int(model.jnt_dofadr[root])
        ids = [model.joint(prefix + name).id for name in cfg.joint_names]
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in ids):
            raise ValueError("The OLI demo requires scalar hinge joints")
        self.qadr = model.jnt_qposadr[ids].copy()
        self.vadr = model.jnt_dofadr[ids].copy()
        self.actuators = []
        for index, joint_id in enumerate(ids):
            matches = np.flatnonzero((model.actuator_trnid[:, 0] == joint_id)
                                    & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            if len(matches) != 1:
                raise ValueError(f"Expected one actuator for {prefix + cfg.joint_names[index]}")
            act = int(matches[0])
            self.actuators.append(act)
            model.actuator_gainprm[act, 0] = cfg.stiffness[index]
            model.actuator_biasprm[act, 1:3] = [-cfg.stiffness[index], -cfg.damping[index]]
            model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_forcelimited[act] = True
            model.actuator_forcerange[act] = [-cfg.effort_limits[index], cfg.effort_limits[index]]
            model.dof_armature[model.jnt_dofadr[joint_id]] = cfg.armature[index]
        self.actuators = np.asarray(self.actuators, dtype=int)
        self.bias = np.zeros(len(ids))

    def reset(self, position, yaw=0.0):
        self.data.qpos[self.root_q:self.root_q + 3] = position
        self.data.qpos[self.root_q + 3:self.root_q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[self.root_v:self.root_v + 6] = 0
        self.data.qpos[self.qadr] = self.cfg.default_position
        self.data.qvel[self.vadr] = 0
        self.set_targets(self.cfg.default_position)
        self.bias = self.rng.uniform(-0.015, 0.015, len(self.qadr)) if self.encoder_noise else np.zeros(len(self.qadr))

    def set_targets(self, targets):
        targets = np.asarray(targets)
        if targets.shape != (len(self.actuators),) or not np.isfinite(targets).all():
            raise ValueError(f"Invalid joint targets for {self.prefix}")
        self.data.ctrl[self.actuators] = targets

    def state(self):
        q = np.concatenate((self.data.qpos[self.root_q:self.root_q + 7], self.data.qpos[self.qadr])).copy()
        v = np.concatenate((self.data.qvel[self.root_v:self.root_v + 6], self.data.qvel[self.vadr])).copy()
        if self.encoder_noise:
            q[7:] += self.bias + self.rng.uniform(-1e-4, 1e-4, len(self.qadr))
            v[6:] += self.rng.uniform(-0.03, 0.03, len(self.vadr))
        return RobotState(q, v, inverse_rotate(q[3:7], v[:3]), v[3:6].copy(),
                          self.data.xmat[self.body_id].reshape(3, 3).copy())
