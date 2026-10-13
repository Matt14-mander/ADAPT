"""CPU regression checks for robot dimensions, joint order and observer semantics."""

import importlib.util
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.paths import PROJECT_ROOT, checkpoint_path, mujoco_path
from deployment.config_manager import DeploymentConfig, configure_model
from deployment.mujoco_robot import MujocoRobot
from deployment.obs_processor import MoMoObservationGenerator
from scripts.check_oli import make_config
from observer import MomentumObserver


class RobotMigrationTests(unittest.TestCase):
    def test_oli_mass_and_frozen_grippers(self):
        cfg = make_config()
        model = mujoco.MjModel.from_xml_path(mujoco_path(cfg.scene_path))
        self.assertEqual((model.nq, model.nv, model.nu), (38, 37, 31))
        self.assertFalse(any("gripper" in model.joint(i).name for i in range(model.njnt)))
        source = cfg.scene_path.parent / "source" / "HU_D04_01_with_gripper.urdf"
        mass = sum(float(element.attrib["value"]) for element in ET.parse(source).getroot().iter("mass"))
        # MJCF serialization rounds individual inertial masses to six digits.
        self.assertAlmostEqual(float(model.body_mass.sum()), mass, delta=1e-4)

    def test_oli_sensor_frame_and_action_mapping(self):
        cfg = make_config()
        robot = MujocoRobot(cfg)
        robot.reset()
        # Use a yaw-rotated base to distinguish world velocity from body velocity.
        robot.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        robot.data.qvel[:3] = [1, 0, 0]
        mujoco.mj_forward(robot.model, robot.data)
        state = robot.read_state()
        np.testing.assert_allclose(state.linear_velocity, [0, -1, 0], atol=1e-6)
        targets = np.arange(31) * 0.001
        robot.set_targets(targets)
        for index, name in enumerate(cfg.joint_names):
            actuator = robot.actuators[index]
            self.assertEqual(robot.model.actuator_trnid[actuator, 0], robot.model.joint(name).id)
            self.assertEqual(robot.data.ctrl[actuator], targets[index])
        obs = MoMoObservationGenerator(cfg).get_obs(state, np.zeros(3), np.zeros(37))
        self.assertEqual(obs.shape, (1, 586))

    def test_g1_export_compatibility_and_cross_robot_rejection(self):
        cfg = DeploymentConfig.load(checkpoint_path("adapt", ".json"))
        robot = MujocoRobot(cfg)
        robot.reset()
        self.assertEqual((robot.model.nq, robot.model.nv, robot.model.nu), (36, 35, 29))
        self.assertEqual(cfg.observation_dim, 554)
        with self.assertRaises(ValueError):
            robot.configure(make_config())

    def test_config_rejects_wrong_joint_order(self):
        cfg = make_config()
        model = mujoco.MjModel.from_xml_path(mujoco_path(cfg.scene_path))
        cfg.joint_names[0], cfg.joint_names[1] = cfg.joint_names[1], cfg.joint_names[0]
        with self.assertRaises(ValueError):
            configure_model(model, cfg)

    def test_training_and_deployment_observer_agree(self):
        import torch
        # Import the pure tensor observer without constructing a training task.
        path = PROJECT_ROOT / "tasks" / "momo_task" / "mdp" / "momo_observer.py"
        spec = importlib.util.spec_from_file_location("batch_observer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = make_config()
        robot = MujocoRobot(cfg)
        state = robot.reset()
        nominal = mujoco.MjModel.from_xml_path(mujoco_path(cfg.scene_path))
        actuators = configure_model(nominal, cfg)
        single = MomentumObserver(nominal, reference_body=cfg.base_body)
        single.reset(state.qpos, state.qvel)
        batch = module.VectorizedMoMo(1, nominal.nv, 3.0, 0.02, "cpu",
            nominal_mg=single.nominal_mg, butter_order=2, butter_cutoff=1.0)
        batch.initialize_state()
        batch.p_prev[:] = torch.tensor(single.previous_momentum)
        for _ in range(50):
            state = robot.step(np.asarray(cfg.default_position))
            ctrl = np.zeros(nominal.nu)
            ctrl[actuators] = cfg.default_position
            single.update(state.qpos, state.qvel, ctrl)
            momentum = np.zeros(nominal.nv)
            mujoco.mj_mulM(nominal, single.data, momentum, single.data.qvel)
            tensor = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            batch.update(tensor(momentum), tensor(-single.data.qfrc_bias),
                tensor(single.data.qfrc_actuator), tensor(single.data.qfrc_passive))
            np.testing.assert_allclose(batch.r_filtered[0].numpy(), single.filtered, atol=2e-3, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
