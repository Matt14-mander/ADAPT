"""Initial OLI Stage 1 locomotion tasks; no hand-load or light-step curriculum."""

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg

from tasks.momo_task.env_cfg import (
    make_momo_lumped_env_cfg, _use_effort_scaled_momo_obs, _use_shared_joint_obs,
    _remove_momo_obs_from_groups,
)
from .robot import get_oli_robot_cfg, action_scale


def oli_flat_env_cfg(play=False, observer=True):
    cfg = make_momo_lumped_env_cfg()
    cfg.scene.entities = {"robot": get_oli_robot_cfg()}
    cfg.actions["joint_pos"].scale = action_scale()
    feet = ("left_foot", "right_foot")
    contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
    )
    heights = TerrainHeightSensorCfg(
        name="foot_height_scan", frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in feet),
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6), ray_alignment="yaw",
        max_distance=1.0, exclude_parent_body=True, include_geom_groups=(0,),
    )
    cfg.scene.sensors = (contact, heights)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r"left_ankle_roll_link_collision_.*", r"right_ankle_roll_link_collision_.*",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_pitch_link",)
    for reward in ("upright", "body_ang_vel"):
        cfg.rewards[reward].params["asset_cfg"].body_names = ("waist_pitch_link",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    for reward in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward].params["asset_cfg"].site_names = feet
    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = feet
    cfg.rewards["pose"].params.update(
        std_standing={".*": 0.05},
        std_walking={".*hip_pitch.*": 0.3, ".*hip_roll.*": 0.15, ".*hip_yaw.*": 0.15,
            ".*knee.*": 0.35, ".*ankle_pitch.*": 0.25, ".*ankle_roll.*": 0.1,
            ".*waist.*": 0.1, ".*shoulder.*": 0.15, ".*elbow.*": 0.15,
            ".*wrist.*": 0.1, ".*head.*": 0.05},
        std_running={".*hip_pitch.*": 0.5, ".*hip_roll.*": 0.2, ".*hip_yaw.*": 0.2,
            ".*knee.*": 0.6, ".*ankle_pitch.*": 0.35, ".*ankle_roll.*": 0.15,
            ".*waist.*": 0.1, ".*shoulder.*": 0.2, ".*elbow.*": 0.2,
            ".*wrist.*": 0.1, ".*head.*": 0.05},
    )
    cfg.viewer.body_name = "waist_pitch_link"
    # Conservative initial OLI velocity range. Remove G1's high-speed curriculum.
    cfg.commands["twist"].ranges.lin_vel_x = (-0.5, 0.5)
    cfg.commands["twist"].ranges.lin_vel_y = (-0.3, 0.3)
    cfg.commands["twist"].ranges.ang_vel_z = (-0.5, 0.5)
    cfg.curriculum.pop("command_vel", None)
    _use_shared_joint_obs(cfg)
    if observer:
        _use_effort_scaled_momo_obs(cfg)
    else:
        _remove_momo_obs_from_groups(cfg, ("actor", "critic"))
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        for event in ("push_robot", "foot_friction", "base_com", "encoder_bias"):
            cfg.events.pop(event, None)
        cfg.commands["twist"].ranges.lin_vel_x = (0.3, 0.3)
        cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        cfg.commands["twist"].heading_command = False
        cfg.commands["twist"].ranges.heading = None
        cfg.commands["twist"].rel_heading_envs = 0.0
        cfg.commands["twist"].rel_standing_envs = 0.0
    return cfg
