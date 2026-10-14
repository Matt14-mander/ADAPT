"""One shared simulation clock, two independent policy and MoMo instances."""

import mujoco
import numpy as np

from .agent import CarryAgent
from .formation import FormationController, rotation2
from .grasp import GraspMonitor
from .robot_view import RobotView
from .scene_builder import build_scene, SIDES


class CarrySimulation:
    def __init__(self, policies, cfg):
        if len(policies) != 2 or policies[0] is policies[1]:
            raise ValueError("Provide two independent policy instances")
        self.cfg = cfg
        self.scene = build_scene([p.cfg for p in policies], cfg)
        self.model, self.data = self.scene.model, self.scene.data
        self.dt = policies[0].cfg.control_dt
        if cfg.seconds < self.dt:
            raise ValueError("seconds must cover at least one policy step")
        seeds = np.random.SeedSequence(cfg.seed).spawn(2)
        self.views = [RobotView(self.model, self.data, p.cfg, prefix,
                      rng=np.random.default_rng(seed), encoder_noise=cfg.encoder_noise)
                      for p, prefix, seed in zip(policies, self.scene.prefixes, seeds)]
        self.agents = [CarryAgent(p, view, side, cfg.arm_hold) for p, view, side in zip(policies, self.views, SIDES)]
        self.formation = FormationController(cfg, self.dt)
        self.grasp = GraspMonitor(self.model, self.data, self.scene.prefixes, cfg.connection != "none")
        self.box_body = self.model.body("payload").id
        self.box_q = int(self.model.joint("payload/free").qposadr[0])
        self.box_v = int(self.model.joint("payload/free").dofadr[0])
        self.decimation = policies[0].cfg.decimation
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for view, position in zip(self.views, self.scene.robot_positions):
            view.reset(position)
        self.data.qpos[self.box_q:self.box_q + 3] = self.scene.box_position
        self.data.qpos[self.box_q + 3:self.box_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.states = [view.state() for view in self.views]
        for agent, state in zip(self.agents, self.states):
            agent.reset(state)
        self.formation.reset()
        self.initial_box = self.data.xpos[self.box_body].copy()
        self.initial_center = np.mean([s.qpos[:2] for s in self.states], axis=0)
        self.failure = None
        self.phase = "settle"
        self.minimum_clearance = float("inf")
        self.max_grasp_error = 0.0
        self.max_spacing_error = 0.0
        self.max_formation_error = 0.0
        self.commands = np.zeros((2, 3))
        self._numerical_warnings = self.data.warning.number[4:].copy()
        self.last_record = self.record()
        return self.last_record

    def step(self):
        if self.failure is not None:
            raise RuntimeError("Simulation has failed; reset explicitly before stepping again")
        self.commands, _, self.phase = self.formation.commands(self.states, self.data.time)
        # Both decisions use the same timestamp; RobotView never calls mj_step.
        for agent, state, command in zip(self.agents, self.states, self.commands):
            agent.decide(state, command)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
            # MuJoCo can reset state internally after an unstable acceleration;
            # treat that as failure rather than continuing a seemingly valid run.
            if np.any(self.data.warning.number[4:] > self._numerical_warnings):
                self.failure = "mujoco_numerical_warning"
                raise FloatingPointError(self.failure)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.failure = "non_finite_dynamics"
                raise FloatingPointError(self.failure)
        mujoco.mj_forward(self.model, self.data)
        self.states = [view.state() for view in self.views]
        for agent, state in zip(self.agents, self.states):
            agent.observe(state)
        record = self.record()
        if self.data.time >= self.cfg.failure_grace_seconds:
            self.failure = self.failure_reason(record)
        record["failure"] = self.failure
        self.last_record = record
        return record

    def failure_reason(self, record):
        c = self.cfg
        if min(record["base_heights"]) < c.min_base_height:
            return "robot_base_too_low"
        if max(record["robot_tilt_deg"]) > c.max_robot_tilt_deg:
            return "robot_fell_over"
        if record["formation_error"] > c.max_formation_error:
            return "formation_separation_exceeded"
        if c.connection != "none":
            if record["box_clearance"] < c.min_box_clearance:
                return "payload_touched_ground"
            if record["box_tilt_deg"] > c.max_box_tilt_deg:
                return "payload_tilt_exceeded"
            if max(record["grasp_position_error"]) > c.max_grasp_error:
                return "grasp_separation_exceeded"
            if max(np.linalg.norm(np.asarray(record["payload_grasp_wrench_world"])[:, :3], axis=1)) > c.max_grasp_force:
                return "grasp_force_exceeded"
        return None

    def record(self):
        box_rotation = self.data.xmat[self.box_body].reshape(3, 3)
        half = np.asarray(self.cfg.box_size) / 2
        clearance = float(self.data.xpos[self.box_body, 2] - np.abs(box_rotation[2]) @ half)
        position_errors, angle_errors = self.grasp.errors()
        separation = float(np.linalg.norm(self.states[1].qpos[:2] - self.states[0].qpos[:2]))
        spacing_error = abs(separation - self.cfg.separation)
        delta = self.states[1].qpos[:2] - self.states[0].qpos[:2]
        formation_error = float(np.linalg.norm(delta - rotation2(self.formation.yaw) @ [0, self.cfg.separation]))
        self.minimum_clearance = min(self.minimum_clearance, clearance)
        self.max_grasp_error = max(self.max_grasp_error, max(position_errors))
        self.max_spacing_error = max(self.max_spacing_error, spacing_error)
        self.max_formation_error = max(self.max_formation_error, formation_error)
        record = dict(time=float(self.data.time), phase=self.phase, failure=self.failure,
            commands=self.commands.tolist(), base_positions=[s.qpos[:3].tolist() for s in self.states],
            base_heights=[float(s.qpos[2]) for s in self.states],
            robot_tilt_deg=[float(np.degrees(np.arccos(np.clip(s.body_rotation[2, 2], -1, 1)))) for s in self.states],
            linear_velocities=[s.linear_velocity.tolist() for s in self.states],
            angular_velocities=[s.angular_velocity.tolist() for s in self.states],
            separation=separation, spacing_error=spacing_error, formation_error=formation_error,
            box_position=self.data.xpos[self.box_body].tolist(),
            box_quaternion=self.data.qpos[self.box_q + 3:self.box_q + 7].tolist(),
            box_velocity_world=np.r_[self.data.qvel[self.box_v:self.box_v + 3],
                box_rotation @ self.data.qvel[self.box_v + 3:self.box_v + 6]].tolist(),
            box_tilt_deg=float(np.degrees(np.arccos(np.clip(box_rotation[2, 2], -1, 1)))),
            box_clearance=clearance,
            box_displacement=float(np.linalg.norm(self.data.xpos[self.box_body, :2] - self.initial_box[:2])),
            team_displacement=float(np.linalg.norm(np.mean([s.qpos[:2] for s in self.states], axis=0) - self.initial_center)),
            grasp_position_error=position_errors, grasp_angle_error=angle_errors,
            payload_grasp_wrench_world=self.grasp.payload_wrenches().tolist(),
            momo_local=[a.policy.observer.local_residual(s.body_rotation).tolist() for a, s in zip(self.agents, self.states)])
        if any(not np.isfinite(value).all() for value in record.values() if isinstance(value, (float, int, list))):
            self.failure = "non_finite_telemetry"
            raise FloatingPointError(self.failure)
        return record

    def summary(self, completed, *, diagnostic=False):
        # Keep a usable summary even when native dynamics reset or become NaN.
        record = self.last_record
        success = (completed and not diagnostic and self.cfg.connection != "none" and self.failure is None
                   and record["time"] >= self.cfg.success_duration
                   and record["box_displacement"] >= self.cfg.success_distance
                   and record["team_displacement"] >= self.cfg.success_distance)
        return dict(completed=completed, success=bool(success), diagnostic=diagnostic,
            failure=self.failure, simulated_seconds=record["time"],
            box_displacement=record["box_displacement"], team_displacement=record["team_displacement"],
            min_box_clearance=self.minimum_clearance, max_grasp_error=self.max_grasp_error,
            max_spacing_error=self.max_spacing_error, max_formation_error=self.max_formation_error)
