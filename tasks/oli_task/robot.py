"""MJLab asset for HU_D04, with frozen grippers and serial virtual joints."""

import json

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from common.paths import robot_asset_dir, mujoco_path

ASSET_DIR = robot_asset_dir("oli")


def robot_metadata():
    path = ASSET_DIR / "robot.json"
    if not path.exists():
        raise FileNotFoundError("Prepare OLI assets first: python -m scripts.prepare_oli --variant gripper")
    return json.loads(path.read_text(encoding="utf-8"))


def get_spec():
    # The training scene supplies terrain; robot.xml deliberately has no floor.
    spec = mujoco.MjSpec.from_file(mujoco_path(ASSET_DIR / "robot.xml"))
    # MJLab articulation builds the position actuators from the same metadata.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def get_oli_robot_cfg():
    metadata = robot_metadata()
    actuators = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,), stiffness=params["stiffness"],
            damping=params["damping"], effort_limit=params["effort_limit"], armature=params["armature"],
        )
        for name, params in metadata["joints"].items()
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(metadata["default_base_position"]),
            joint_pos={name: params["default_position"] for name, params in metadata["joints"].items()},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=0.9),
        collisions=(CollisionCfg(
            geom_names_expr=(".*_collision_.*",), contype=0, conaffinity=1,
            condim={".*_ankle_roll_link_collision_.*": 3, ".*_collision_.*": 1},
            priority=1, friction=(0.6,),
        ),),
    )


def action_scale():
    return {name: params["action_scale"] for name, params in robot_metadata()["joints"].items()}
