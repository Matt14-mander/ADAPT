"""Shared-clock, mapping, aligned grasp and physical load-transfer regression checks."""

import json
import unittest

import mujoco
import numpy as np

from deployment.cooperative.config import CarryConfig
from deployment.cooperative.formation import FormationController
from deployment.cooperative.grasp import GraspMonitor
from deployment.cooperative.robot_view import RobotView
from deployment.cooperative.scene_builder import build_scene
from deployment.cooperative.simulation import CarrySimulation
from deployment.policy_manager import ZeroPolicy
from deployment.robot_interface import RobotState
from scripts.check_oli import make_config


def make_sim(**kwargs):
    return CarrySimulation([ZeroPolicy(make_config(), name=str(i)) for i in range(2)], CarryConfig(**kwargs))


class CooperativeTests(unittest.TestCase):
    def test_grasps_start_aligned_and_payload_mass_preserved(self):
        sim = make_sim()
        self.assertEqual((sim.model.nq, sim.model.nv, sim.model.nu, sim.model.neq), (83, 80, 62, 2))
        positions, angles = sim.grasp.errors()
        np.testing.assert_allclose(positions, 0, atol=1e-10)
        np.testing.assert_allclose(angles, 0, atol=1e-7)
        self.assertAlmostEqual(sim.model.body_mass[sim.box_body], sim.cfg.box_mass, places=8)
        self.assertGreater(sim.record()["box_clearance"], 0.40)
        # Robots cannot accidentally share controls, joint states or histories.
        self.assertFalse(set(sim.views[0].actuators) & set(sim.views[1].actuators))
        self.assertFalse(set(sim.views[0].qadr) & set(sim.views[1].qadr))
        self.assertIsNot(sim.agents[0].policy.obs_gen, sim.agents[1].policy.obs_gen)
        self.assertIsNot(sim.agents[0].policy.observer.data, sim.agents[1].policy.observer.data)

    def test_namespaced_state_and_control_mapping(self):
        sim = make_sim()
        a, b = sim.views
        before = b.state()
        sim.data.qpos[a.root_q + 3:a.root_q + 7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        sim.data.qvel[a.root_v:a.root_v + 3] = [1, 0, 0]
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose(a.state().linear_velocity, [0, -1, 0], atol=1e-6)
        np.testing.assert_array_equal(b.state().qpos, before.qpos)
        targets = np.arange(31) / 100
        a.set_targets(targets)
        np.testing.assert_array_equal(sim.data.ctrl[a.actuators], targets)
        np.testing.assert_allclose(sim.data.ctrl[b.actuators], b.cfg.default_position)

    def test_one_clock_and_applied_action_feedback(self):
        class ConstantPolicy(ZeroPolicy):
            def infer(self, observation):
                return np.full(31, 0.2, dtype=np.float32)
        policies = [ConstantPolicy(make_config(), name=str(i)) for i in range(2)]
        sim = CarrySimulation(policies, CarryConfig(arm_hold=0.6))
        record = sim.step()
        self.assertAlmostEqual(sim.data.time, 0.02, places=10)
        self.assertTrue(np.isfinite(record["payload_grasp_wrench_world"]).all())
        for agent in sim.agents:
            np.testing.assert_allclose(agent.policy.obs_gen.last_action[agent.arm_indices], 0.08, atol=1e-6)
            np.testing.assert_allclose(agent.policy.observer.data.ctrl[agent.policy.actuators], agent.targets)
        sim.failure = "test_failure"
        qpos = sim.data.qpos.copy()
        with self.assertRaises(RuntimeError):
            sim.step()
        np.testing.assert_array_equal(sim.data.qpos, qpos)
        self.assertFalse(sim.summary(completed=True)["success"])

    def test_turn_gives_outer_robot_faster_velocity(self):
        cfg = CarryConfig(command=(0.1, 0, 0.1), linear_acceleration=100, angular_acceleration=100)
        controller = FormationController(cfg, 0.02)
        states = [RobotState(np.r_[0, y, 0.9, 1, 0, 0, 0, np.zeros(31)], np.zeros(37),
                             np.zeros(3), np.zeros(3)) for y in (-0.55, 0.55)]
        commands, _, _ = controller.commands(states, 5.0)
        self.assertGreater(commands[0, 0], commands[1, 0])
        np.testing.assert_allclose(commands[:, 2], 0.1)

    def test_numerical_failure_keeps_last_valid_summary(self):
        sim = make_sim()
        sim.step()
        last_time = sim.last_record["time"]
        sim.data.qvel[sim.box_v] = np.nan
        with self.assertRaises(FloatingPointError):
            sim.record()
        summary = sim.summary(completed=False)
        self.assertEqual(summary["failure"], "non_finite_telemetry")
        self.assertEqual(summary["simulated_seconds"], last_time)
        json.dumps(summary, allow_nan=False)

    def test_constraint_wrench_supports_weight_and_release_removes_it(self):
        # Static test fixtures only: real demo robots are never fixed to the world.
        model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002"/>
          <worldbody><body name="anchor_a" pos="0 -0.2 1.1"><site name="a/carry_grasp"/></body>
          <body name="anchor_b" pos="0 0.2 1.1"><site name="b/carry_grasp"/></body>
          <body name="payload" pos="0 0 1"><freejoint name="payload/free"/>
          <geom type="box" size="0.1 0.2 0.1" mass="1" contype="0" conaffinity="0"/>
          <site name="payload/handle_0" pos="0 -0.2 0.1"/>
          <site name="payload/handle_1" pos="0 0.2 0.1"/></body></worldbody>
          <equality><weld name="grasp_0" site1="a/carry_grasp" site2="payload/handle_0" solref="0.03 1"/>
          <weld name="grasp_1" site1="b/carry_grasp" site2="payload/handle_1" solref="0.03 1"/></equality></mujoco>''')
        data = mujoco.MjData(model)
        monitor = GraspMonitor(model, data, ("a/", "b/"))
        for _ in range(500):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        wrench = monitor.payload_wrenches()
        self.assertAlmostEqual(wrench[:, 2].sum(), 9.81, delta=0.05)
        self.assertAlmostEqual(wrench[0, 2], wrench[1, 2], delta=0.05)
        data.eq_active[:] = False
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(monitor.payload_wrenches(), 0, atol=1e-10)
        self.assertLess(data.qacc[2], -9.0)

    def test_no_connection_and_configuration_rejection(self):
        sim = make_sim(connection="none")
        self.assertEqual(sim.model.neq, 0)
        np.testing.assert_allclose(sim.grasp.payload_wrenches(), 0)
        sim.step()
        for kwargs in (dict(box_mass=-1), dict(arm_hold=1.1), dict(command=(1, 0, 0)), dict(separation=0.5)):
            with self.assertRaises(ValueError):
                cfg = CarryConfig(**kwargs)
                cfg.validate()
                build_scene([make_config(), make_config()], cfg)


if __name__ == "__main__":
    unittest.main()
