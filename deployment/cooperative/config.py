"""Demo settings are separate from the ONNX policy's deployment contract."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class CarryConfig:
    mode: str = "pair"
    solo_box_center: tuple[float, float, float] = (0.32, 0.0, 1.12)
    seconds: float = 60.0
    separation: float = 1.10
    command: tuple[float, float, float] = (0.10, 0.0, 0.0)
    stand_seconds: float = 2.0
    ramp_seconds: float = 3.0
    final_hold_seconds: float = 3.0
    box_mass: float = 0.5
    box_size: tuple[float, float, float] = (0.30, 0.44, 0.20)
    handle_rise: float = 0.06
    grasp_offset: tuple[float, float, float] = (0.055, 0.0, 0.0)
    connection: str = "weld"
    grasp_time_constant: float = 0.08
    grasp_damping_ratio: float = 1.0
    grasp_torque_scale: float = 0.03
    formation: bool = True
    position_gain: float = 0.6
    yaw_gain: float = 1.0
    max_velocity_correction: float = 0.10
    command_limits: tuple[float, float, float] = (0.5, 0.3, 0.5)
    linear_acceleration: float = 0.25
    angular_acceleration: float = 0.35
    arm_hold: float = 0.0
    seed: int = 0
    encoder_noise: bool = False
    min_base_height: float = 0.50
    max_robot_tilt_deg: float = 60.0
    max_box_tilt_deg: float = 45.0
    min_box_clearance: float = 0.015
    max_grasp_error: float = 0.20
    max_formation_error: float = 0.50
    max_grasp_force: float = 300.0
    failure_grace_seconds: float = 1.0
    success_distance: float = 2.0
    success_duration: float = 30.0

    @classmethod
    def load(cls, path):
        cfg = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        cfg.validate()
        return cfg

    def as_dict(self):
        return asdict(self)

    def validate(self):
        if self.mode not in ("pair", "solo"):
            raise ValueError("mode must be pair or solo")
        for name, value in self.as_dict().items():
            if isinstance(value, (int, float, list, tuple)) and not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        for name in ("seconds", "separation", "ramp_seconds", "box_mass", "handle_rise",
                     "grasp_time_constant", "grasp_damping_ratio", "linear_acceleration",
                     "angular_acceleration", "min_base_height", "max_grasp_error",
                     "max_formation_error", "max_grasp_force", "success_duration", "success_distance"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("stand_seconds", "final_hold_seconds", "grasp_torque_scale", "position_gain",
                     "yaw_gain", "max_velocity_correction", "failure_grace_seconds", "min_box_clearance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("command", "box_size", "grasp_offset", "command_limits", "solo_box_center"):
            if np.asarray(getattr(self, name)).shape != (3,):
                raise ValueError(f"{name} requires three values")
        if np.any(np.asarray(self.box_size) <= 0) or np.any(np.asarray(self.command_limits) <= 0):
            raise ValueError("box_size and command_limits must be positive")
        if np.any(np.abs(self.command) > np.asarray(self.command_limits)):
            raise ValueError("Requested command exceeds configured locomotion limits")
        if self.connection not in ("weld", "connect", "none"):
            raise ValueError("connection must be weld, connect, or none")
        if not 0 <= self.arm_hold <= 1:
            raise ValueError("arm_hold must be in [0, 1]")
        if not 0 < self.max_robot_tilt_deg < 90 or not 0 < self.max_box_tilt_deg < 90:
            raise ValueError("Tilt limits must be between 0 and 90 degrees")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
