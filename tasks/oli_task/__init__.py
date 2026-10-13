"""Experimental HU_D04 gripper Stage 1 tasks, alongside the original G1 tasks."""

from tasks.registry import register_mjlab_task
from tasks.momo_task.momo_env import MoMoEnv
from tasks.momo_task.effort_scale_runner import MoMoEffortScaleRunner
from tasks.momo_task.rl_cfg import momo_g1_ppo_runner_cfg
from .env_cfg import oli_flat_env_cfg


class OliMoMoEnv(MoMoEnv):
    _torso_body_name = "waist_pitch_link"
    _hand_body_names = ("left_wrist_roll_link", "right_wrist_roll_link")
    _qpos_noise_range = 1e-4
    _qvel_noise_range = 0.03
    # A scalar gain covers all 37 generalized velocities, including the head.
    _momo_ko = 3.0


for task, observer in (
    ("Ref-NoMoMo-Flat-LimX-OLI-HU-D04", False),
    ("MOMO-Lumped-Flat-LimX-OLI-HU-D04-EffortScale", True),
):
    rl = momo_g1_ppo_runner_cfg()
    rl.experiment_name = "oli_adapt" if observer else "oli_baseline"
    rl.run_name = "stage1"
    rl.logger = "tensorboard"
    rl.upload_model = False
    register_mjlab_task(
        task_id=task, env_cfg=oli_flat_env_cfg(observer=observer),
        play_env_cfg=oli_flat_env_cfg(play=True, observer=observer), rl_cfg=rl,
        env_cls=OliMoMoEnv, runner_cls=MoMoEffortScaleRunner if observer else None,
    )
