"""Solo pose feasibility, applied-control feedback and shared demo regression."""

import unittest

import mujoco
import numpy as np

from deployment.cooperative.config import CarryConfig
from deployment.cooperative.simulation import CarrySimulation
from deployment.policy_manager import ZeroPolicy
from scripts.check_oli import make_config


def make_solo(**overrides):
    cfg = CarryConfig.load("configs/cooperative_carry/oli_solo.json")
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return CarrySimulation([ZeroPolicy(make_config())], cfg)


class SoloCarryTests(unittest.TestCase):
    def test_ik_pose_limits_alignment_and_free_payload(self):
        sim = make_solo()
        self.assertEqual((sim.model.nq, sim.model.nv, sim.model.nu, sim.model.neq), (45, 43, 31, 2))
        self.assertEqual(len(sim.views), 1)
        np.testing.assert_allclose(sim.grasp.errors()[0], 0, atol=1e-10)
        np.testing.assert_allclose(sim.grasp.errors()[1], 0, atol=1e-7)
        view, agent = sim.views[0], sim.agents[0]
        self.assertEqual(len(agent.arm_indices), 14)
        other = np.setdiff1d(np.arange(31), agent.arm_indices)
        np.testing.assert_allclose(sim.states[0].qpos[7:][other], np.asarray(view.cfg.default_position)[other])
        for i, name in enumerate(view.cfg.joint_names):
            low, high = sim.model.joint(view.prefix + name).range
            self.assertTrue(low <= agent.hold_pose[i] <= high)
        self.assertEqual(sim.data.ncon, 0)  # enabled initial collision primitives
        self.assertAlmostEqual(sim.model.body_mass[sim.box_body], 0.5)
        for i, grip in enumerate(sim.grasp.grips):
            target = np.array(sim.cfg.solo_box_center) + [0, (1 if i == 0 else -1) * 0.185, -0.1]
            np.testing.assert_allclose(sim.data.site_xpos[grip], target, atol=0.003)

    def test_step_holds_both_arms_and_reports_actual_actions(self):
        sim = make_solo()
        agent = sim.agents[0]
        for _ in range(25):
            record = sim.step()
        self.assertIsNone(sim.failure)
        self.assertAlmostEqual(record["time"], 0.5)
        np.testing.assert_allclose(agent.targets[agent.arm_indices], agent.hold_pose[agent.arm_indices])
        action = (agent.targets - np.array(agent.cfg.default_position)) / np.array(agent.cfg.action_scale)
        np.testing.assert_allclose(agent.policy.obs_gen.last_action, action, atol=1e-6)
        np.testing.assert_allclose(agent.policy.observer.data.ctrl[agent.policy.actuators], agent.targets)
        self.assertLess(max(record["grasp_position_error"]), 0.02)
        self.assertEqual(np.shape(record["commands"]), (1, 3))
        commands, _, phase = sim.formation.commands(sim.states, 6.0)
        self.assertEqual(phase, "carry")
        self.assertGreater(commands[0, 0], 0)
        self.assertFalse(sim.summary(True, diagnostic=True)["success"])

    def test_reset_restores_ik_pose_and_unreachable_target_is_rejected(self):
        sim = make_solo()
        pose = sim.states[0].qpos.copy()
        sim.step()
        sim.reset()
        np.testing.assert_allclose(sim.states[0].qpos, pose)
        agent = sim.agents[0]
        expected = (agent.targets - np.array(agent.cfg.default_position)) / np.array(agent.cfg.action_scale)
        np.testing.assert_allclose(agent.policy.obs_gen.last_action, expected, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "unreachable"):
            make_solo(solo_box_center=(2, 0, 1.12))

    def test_disconnected_payload_has_no_constraints_or_support(self):
        sim = make_solo(connection="none")
        self.assertEqual(sim.model.neq, 0)
        sim.step()
        self.assertLess(sim.data.qvel[sim.box_v + 2], -0.1)
        np.testing.assert_allclose(sim.grasp.payload_wrenches(), 0)
        self.assertFalse(sim.summary(True)["success"])


if __name__ == "__main__":
    unittest.main()
