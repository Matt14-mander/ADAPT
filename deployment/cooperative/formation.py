"""Bounded formation corrections around an integrated virtual team frame."""

import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quaternion(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rotation2(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


class FormationController:
    def __init__(self, cfg, dt):
        self.cfg, self.dt = cfg, dt
        self.reset()

    def reset(self):
        self.center = np.zeros(2)
        self.yaw = 0.0
        self.last_commands = np.zeros((2, 3))

    def phase(self, time):
        c = self.cfg
        if time < c.stand_seconds:
            return "settle", 0.0
        stop = max(c.stand_seconds, c.seconds - c.final_hold_seconds - c.ramp_seconds)
        if time >= c.seconds - c.final_hold_seconds:
            return "hold", 0.0
        up = np.clip((time - c.stand_seconds) / c.ramp_seconds, 0, 1)
        down = np.clip((stop + c.ramp_seconds - time) / c.ramp_seconds, 0, 1)
        scale = float(min(up, down))
        if time >= stop:
            return "decelerate", scale
        return ("accelerate" if up < 1 else "carry"), scale

    def commands(self, states, time):
        phase, scale = self.phase(time)
        requested = np.asarray(self.cfg.command) * scale
        team_rotation = rotation2(self.yaw)
        velocity_world = team_rotation @ requested[:2]
        errors, commands = [], []
        for i, (state, lateral) in enumerate(zip(states, (-self.cfg.separation / 2, self.cfg.separation / 2))):
            offset = team_rotation @ [0, lateral]
            desired = self.center + offset
            error = desired - state.qpos[:2]
            errors.append(error)
            correction = self.cfg.position_gain * error if self.cfg.formation else np.zeros(2)
            norm = np.linalg.norm(correction)
            if norm > self.cfg.max_velocity_correction:
                correction *= self.cfg.max_velocity_correction / norm
            # Outside robot travels faster in a turn: omega x offset.
            world = velocity_world + requested[2] * np.array([-offset[1], offset[0]]) + correction
            yaw = yaw_from_quaternion(state.qpos[3:7])
            yaw_correction = self.cfg.yaw_gain * wrap_angle(self.yaw - yaw) if self.cfg.formation else 0.0
            command = np.r_[rotation2(yaw).T @ world, requested[2] + yaw_correction]
            command = np.clip(command, -np.asarray(self.cfg.command_limits), self.cfg.command_limits)
            rate = np.asarray([self.cfg.linear_acceleration] * 2 + [self.cfg.angular_acceleration]) * self.dt
            command = self.last_commands[i] + np.clip(command - self.last_commands[i], -rate, rate)
            commands.append(command)
        self.last_commands[:] = commands
        self.center += self.dt * velocity_world
        self.yaw = wrap_angle(self.yaw + self.dt * requested[2])
        return np.asarray(commands, dtype=np.float32), np.asarray(errors), phase
