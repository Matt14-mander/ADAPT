"""Per-robot policy, history and nominal dynamics; no shared observer state."""

import mujoco
import numpy as np

from common.paths import mujoco_path


class CarryAgent:
    def __init__(self, policy, view, arm_side, arm_hold):
        self.policy, self.view = policy, view
        self.cfg = policy.cfg
        self.arm_hold = arm_hold
        self.arm_indices = np.asarray([i for i, name in enumerate(self.cfg.joint_names)
            if name.startswith(arm_side + "_") and any(part in name for part in ("shoulder", "elbow", "wrist"))], dtype=int)
        if len(self.arm_indices) != 7:
            raise ValueError("Expected seven joints in the carrying arm")
        if np.any(np.asarray(self.cfg.action_scale) <= 0):
            raise ValueError("Action scales must be positive for applied-action feedback")
        nominal = mujoco.MjModel.from_xml_path(mujoco_path(self.cfg.scene_path))
        self.policy.attach_observer(nominal)
        self.targets = np.asarray(self.cfg.default_position, dtype=float)

    def reset(self, state):
        self.targets = np.asarray(self.cfg.default_position, dtype=float).copy()
        self.policy.reset(state)

    def decide(self, state, command):
        targets = self.policy.step(state, command).copy()
        indices = self.arm_indices
        targets[indices] = ((1 - self.arm_hold) * targets[indices]
                            + self.arm_hold * np.asarray(self.cfg.default_position)[indices])
        # History and nominal actuator force must describe the applied control.
        applied_action = (targets - np.asarray(self.cfg.default_position)) / np.asarray(self.cfg.action_scale)
        self.policy.obs_gen.update_last_action(applied_action)
        self.targets = targets
        self.view.set_targets(targets)
        return targets

    def observe(self, state):
        self.policy.refresh_encoder(state)
        self.policy.update_momo(state, self.targets)
