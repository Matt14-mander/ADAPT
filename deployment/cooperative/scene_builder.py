"""Build two named robot instances, a real dynamic box and pre-aligned grasps."""

from dataclasses import dataclass

import mujoco
import numpy as np

from common.paths import robot_asset_dir, mujoco_path
from deployment.config_manager import configure_model
from .robot_view import RobotView

PREFIXES = ("robot_a/", "robot_b/")
SIDES = ("left", "right")


@dataclass
class CarryScene:
    model: object
    data: object
    prefixes: tuple
    box_position: np.ndarray
    robot_positions: list


def build_scene(policy_cfgs, cfg):
    cfg.validate()
    if len(policy_cfgs) != 2:
        raise ValueError("Exactly two robot configurations are required")
    for pc in policy_cfgs:
        if (pc.robot, pc.variant) != ("oli", "gripper"):
            raise ValueError("This first demo supports OLI HU_D04 gripper policies only")
        if not np.isfinite(pc.timestep) or pc.timestep <= 0 or not isinstance(pc.decimation, int) or pc.decimation < 1:
            raise ValueError("Policies require a positive timestep and integer decimation")
        # Validate the original single-robot contract before constructing a shared world.
        nominal = mujoco.MjModel.from_xml_path(mujoco_path(pc.scene_path))
        configure_model(nominal, pc)
    if policy_cfgs[0].timestep != policy_cfgs[1].timestep or policy_cfgs[0].decimation != policy_cfgs[1].decimation:
        raise ValueError("Both policies must use identical physics and control timing")
    spec = mujoco.MjSpec.from_string('''<mujoco model="oli_cooperative_carry">
      <option timestep="0.005" integrator="Euler" iterations="100"/>
      <visual><global offwidth="1280" offheight="720"/><headlight diffuse="0.7 0.7 0.7"/></visual>
      <asset><texture name="checker" type="2d" builtin="checker" width="256" height="256"
          rgb1="0.19 0.22 0.26" rgb2="0.28 0.31 0.35"/>
        <material name="floor_mat" texture="checker" texrepeat="12 12" texuniform="true"/></asset>
      <worldbody><geom name="floor" type="plane" size="0 0 0.1" material="floor_mat"
          contype="1" conaffinity="15" friction="0.6 0.005 0.0001"/>
        <light pos="0 -2 4" dir="0 0 -1"/></worldbody></mujoco>''')
    spec.option.timestep = policy_cfgs[0].timestep
    for index, (pc, prefix, side) in enumerate(zip(policy_cfgs, PREFIXES, SIDES)):
        child = mujoco.MjSpec.from_file(mujoco_path(robot_asset_dir(pc.robot, pc.variant) / "robot.xml"))
        child.body(f"{side}_gripper_link_0").add_site(
            name="carry_grasp", pos=cfg.grasp_offset, size=[0.012, 0.012, 0.012], rgba=[0.2, 1, 0.2, 1])
        # Preserve disabled self-collision; enable robot/robot and robot/payload contacts.
        own_bit, other_bit = (2, 4) if index == 0 else (4, 2)
        for geom in child.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype = own_bit
                geom.conaffinity = 1 | other_bit | 8
        frame = spec.worldbody.add_frame(name=f"mount_{index}")
        spec.attach(child, prefix=prefix, frame=frame)
    positions = [np.asarray(pc.default_base_position, dtype=float) + [0, offset, 0]
                 for pc, offset in zip(policy_cfgs, (-cfg.separation / 2, cfg.separation / 2))]
    # Compute grasp locations in the actual policy default pose, not in qpos0.
    preliminary = spec.compile()
    preliminary_data = mujoco.MjData(preliminary)
    for pc, prefix, position in zip(policy_cfgs, PREFIXES, positions):
        RobotView(preliminary, preliminary_data, pc, prefix).reset(position)
    mujoco.mj_forward(preliminary, preliminary_data)
    points = [preliminary_data.site_xpos[preliminary.site(prefix + "carry_grasp").id].copy() for prefix in PREFIXES]
    rotations = [preliminary_data.site_xmat[preliminary.site(prefix + "carry_grasp").id].reshape(3, 3).copy()
                 for prefix in PREFIXES]
    if abs(points[1][1] - points[0][1]) < cfg.box_size[1] + 0.04:
        raise ValueError("Box is too wide for the initial hand separation; increase separation or reduce box width")
    center = np.mean(points, axis=0) - [0, 0, cfg.box_size[2] / 2 + cfg.handle_rise]
    if center[2] - cfg.box_size[2] / 2 <= 0.05:
        raise ValueError("Initial payload would intersect the floor")
    box = spec.worldbody.add_body(name="payload", pos=center)
    box.add_freejoint(name="payload/free")
    half = np.asarray(cfg.box_size) / 2
    box.add_geom(name="payload/box", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
        mass=cfg.box_mass, rgba=[0.85, 0.48, 0.12, 1], contype=8, conaffinity=1 | 2 | 4,
        friction=[0.6, 0.005, 0.0001])
    for index, (point, rotation, prefix, side) in enumerate(zip(points, rotations, PREFIXES, SIDES)):
        local = point - center
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        handle_name = f"payload/handle_{index}"
        box.add_site(name=handle_name, pos=local, quat=quat, size=[0.012] * 3, rgba=[1, 0.1, 0.1, 1])
        # Massless handle fixtures: the specified box mass includes handles.
        for number, (start, end) in enumerate((
            (local + [-0.045, 0, 0], local + [0.045, 0, 0]),
            (local + [-0.045, 0, 0], [-0.045 + local[0], np.sign(local[1]) * half[1], half[2]]),
            (local + [0.045, 0, 0], [0.045 + local[0], np.sign(local[1]) * half[1], half[2]]),
        )):
            box.add_geom(name=f"payload/handle_{index}_{number}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=np.concatenate((start, end)), size=[0.012, 0, 0], mass=0,
                contype=8, conaffinity=1 | 2 | 4, rgba=[0.18, 0.2, 0.22, 1])
        if cfg.connection != "none":
            equality = spec.add_equality(name=f"grasp_{index}",
                type=mujoco.mjtEq.mjEQ_WELD if cfg.connection == "weld" else mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_SITE, name1=prefix + "carry_grasp", name2=handle_name,
                solref=[cfg.grasp_time_constant, cfg.grasp_damping_ratio], solimp=[0.8, 0.95, 0.01, 0.5, 2])
            equality.data[10] = cfg.grasp_torque_scale
            # The virtual grasp replaces finger/handle contact at the attached hand.
            for body in (f"{side}_wrist_roll_link", *(f"{side}_gripper_link_{j}" for j in range(5))):
                spec.add_exclude(name=f"grasp_exclude_{index}_{body}", bodyname1=prefix + body, bodyname2="payload")
    model = spec.compile()
    data = mujoco.MjData(model)
    return CarryScene(model, data, PREFIXES, center, positions)
