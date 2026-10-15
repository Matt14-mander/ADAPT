"""Single OLI with two IK-positioned arms and pre-attached lower box handles.

This is a virtual grasp, not friction-based grasp acquisition. The free payload
is supported only through the two hands; the robot root remains free.
"""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from common.paths import robot_asset_dir, mujoco_path
from deployment.config_manager import configure_model
from .robot_view import RobotView
from .scene_builder import CarryScene


def solve_arms(model, data, view, targets, site_names):
    """Bounded pose IK, leaving the pelvis, legs and head exactly at defaults."""
    pose = np.asarray(view.cfg.default_position, dtype=float).copy()
    for side, target, name in zip(("left", "right"), targets, site_names):
        indices = np.array([i for i, joint in enumerate(view.cfg.joint_names)
            if joint.startswith(side + "_") and any(part in joint for part in ("shoulder", "elbow", "wrist"))])
        joint_ids = [model.joint(view.prefix + view.cfg.joint_names[i]).id for i in indices]
        ranges = model.jnt_range[joint_ids]
        qadr = view.qadr[indices]
        site = model.site(name).id
        seed = pose[indices].copy()
        seed[0], seed[3] = -0.45, -1.1
        seed = np.clip(seed, ranges[:, 0] + 1e-5, ranges[:, 1] - 1e-5)

        def residual(q):
            data.qpos[qadr] = q
            mujoco.mj_forward(model, data)
            orientation = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec()
            return np.r_[data.site_xpos[site] - target, 0.015 * orientation, 0.002 * (q - seed)]

        result = least_squares(residual, seed, bounds=(ranges[:, 0] + 1e-4, ranges[:, 1] - 1e-4),
                               max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        residual(result.x)
        error = np.linalg.norm(data.site_xpos[site] - target)
        if error > 0.003:
            raise ValueError(f"{side} hand target is unreachable (IK error {error:.4f} m); adjust solo_box_center/box_size")
        pose[indices] = result.x
    data.qpos[view.qadr] = pose
    mujoco.mj_forward(model, data)
    return pose


def build_solo_scene(policy_cfgs, cfg):
    if len(policy_cfgs) != 1:
        raise ValueError("Solo carry requires one policy")
    pc = policy_cfgs[0]
    if (pc.robot, pc.variant) != ("oli", "gripper"):
        raise ValueError("Solo carry requires an OLI HU_D04 gripper policy")
    nominal = mujoco.MjModel.from_xml_path(mujoco_path(pc.scene_path))
    configure_model(nominal, pc)
    spec = mujoco.MjSpec.from_string('''<mujoco model="oli_solo_carry">
      <option timestep="0.005" iterations="100"/>
      <visual><global offwidth="1280" offheight="720"/>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/></visual>
      <asset><texture name="checker" type="2d" builtin="checker" width="256" height="256"
        rgb1="0.19 0.22 0.26" rgb2="0.28 0.31 0.35"/>
      <material name="floor_mat" texture="checker" texrepeat="12 12" texuniform="true"/></asset>
      <worldbody><geom name="floor" type="plane" size="0 0 0.1" material="floor_mat"
        contype="1" conaffinity="10" friction="0.6 0.005 0.0001"/>
      <light pos="0 -2 4" dir="0 0 -1"/></worldbody></mujoco>''')
    spec.option.timestep = pc.timestep
    prefix = "robot_a/"
    child = mujoco.MjSpec.from_file(mujoco_path(robot_asset_dir(pc.robot, pc.variant) / "robot.xml"))
    names = [prefix + side + "_carry_grasp" for side in ("left", "right")]
    for side in ("left", "right"):
        child.body(side + "_gripper_link_0").add_site(name=side + "_carry_grasp", pos=cfg.grasp_offset,
            size=[0.012] * 3, rgba=[0.2, 1, 0.2, 1])
    for geom in child.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 2, 1 | 8
    spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame(name="robot_mount"))
    preliminary = spec.compile()
    data = mujoco.MjData(preliminary)
    view = RobotView(preliminary, data, pc, prefix)
    position = np.asarray(pc.default_base_position, dtype=float)
    view.reset(position)
    center = np.asarray(cfg.solo_box_center, dtype=float) + [position[0], position[1], 0]
    half = np.asarray(cfg.box_size) / 2
    if center[2] - half[2] <= 0.05:
        raise ValueError("Solo payload would intersect the floor")
    targets = [center + [0, sign * (half[1] + 0.035), -half[2]] for sign in (1, -1)]
    pose = solve_arms(preliminary, data, view, targets, names)
    points = [data.site_xpos[preliminary.site(name).id].copy() for name in names]
    rotations = [data.site_xmat[preliminary.site(name).id].copy() for name in names]
    if cfg.solo_arm_test:
        # Diagnostic: preserve the exact same IK arm pose, but park the real box
        # beside the robot, resting on the floor without any hand connection.
        center = np.array([position[0] + 1.5, position[1] + 1.5, half[2] + 0.002])
    box = spec.worldbody.add_body(name="payload", pos=center)
    box.add_freejoint(name="payload/free")
    box.add_geom(name="payload/box", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
        mass=cfg.box_mass, rgba=[0.85, 0.48, 0.12, 1], contype=8, conaffinity=1 | 2,
        friction=[0.6, 0.005, 0.0001])
    for i, (point, rotation, name, side) in enumerate(zip(points, rotations, names, ("left", "right"))):
        local = (np.array([0, (1 if i == 0 else -1) * (half[1] + 0.035), -half[2]])
                 if cfg.solo_arm_test else point - center)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation)
        box.add_site(name=f"payload/handle_{i}", pos=local, quat=quat,
                     size=[0.012] * 3, rgba=[1, 0.1, 0.1, 1])
        # Visible side rails at the lower box edge, included in the specified mass.
        box.add_geom(name=f"payload/rail_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=np.r_[local + [-0.07, 0, 0], local + [0.07, 0, 0]], size=[0.015, 0, 0],
            mass=0, contype=0, conaffinity=0, rgba=[0.18, 0.2, 0.22, 1])
        if cfg.connection != "none":
            eq = spec.add_equality(name=f"grasp_{i}",
                type=mujoco.mjtEq.mjEQ_WELD if cfg.connection == "weld" else mujoco.mjtEq.mjEQ_CONNECT,
                objtype=mujoco.mjtObj.mjOBJ_SITE, name1=name, name2=f"payload/handle_{i}",
                solref=[cfg.grasp_time_constant, cfg.grasp_damping_ratio], solimp=[0.8, 0.95, 0.01, 0.5, 2])
            eq.data[10] = cfg.grasp_torque_scale
            for body in (side + "_wrist_roll_link", *(f"{side}_gripper_link_{j}" for j in range(5))):
                spec.add_exclude(name=f"grasp_exclude_{i}_{body}", bodyname1=prefix + body, bodyname2="payload")
    model = spec.compile()
    return CarryScene(model, mujoco.MjData(model), (prefix,), center, [position], [pose], names)
