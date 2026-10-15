"""Grasp errors and equality-specific payload wrenches from J.T @ lambda."""

import mujoco
import numpy as np


class GraspMonitor:
    def __init__(self, model, data, prefixes, connected=True, grasp_sites=None):
        self.model, self.data = model, data
        self.connected = connected
        self.grips = [model.site(name).id for name in (grasp_sites or [prefix + "carry_grasp" for prefix in prefixes])]
        self.handles = [model.site(f"payload/handle_{i}").id for i in range(2)]
        self.equalities = [model.equality(f"grasp_{i}").id for i in range(2)] if connected else []
        self.box_body = model.body("payload").id
        self.box_dof = int(model.joint("payload/free").dofadr[0])

    def errors(self):
        positions, angles = [], []
        for grip, handle in zip(self.grips, self.handles):
            positions.append(float(np.linalg.norm(self.data.site_xpos[grip] - self.data.site_xpos[handle])))
            relative = self.data.site_xmat[grip].reshape(3, 3).T @ self.data.site_xmat[handle].reshape(3, 3)
            angles.append(float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
        return positions, angles

    def payload_wrenches(self):
        # Equality multipliers are not directly Cartesian forces. Project only
        # each grasp's rows onto the payload free joint, then rotate its torque.
        wrenches = np.zeros((2, 6))
        if not self.connected:
            return wrenches
        n = self.data.nefc
        rotation = self.data.xmat[self.box_body].reshape(3, 3)
        for index, equality in enumerate(self.equalities):
            selected = ((self.data.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                        & (self.data.efc_id[:n] == equality))
            multiplier = np.where(selected, self.data.efc_force[:n], 0.0)
            generalized = np.zeros(self.model.nv)
            mujoco.mj_mulJacTVec(self.model, self.data, generalized, multiplier)
            wrenches[index, :3] = generalized[self.box_dof:self.box_dof + 3]
            wrenches[index, 3:] = rotation @ generalized[self.box_dof + 3:self.box_dof + 6]
        return wrenches

    def arrow_segments(self, scale=0.002):
        forces = self.payload_wrenches()[:, :3]
        return [(self.data.site_xpos[grip].copy(), self.data.site_xpos[grip] - force * scale)
                for grip, force in zip(self.grips, forces)]
