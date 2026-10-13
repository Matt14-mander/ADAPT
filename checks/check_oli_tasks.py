"""GPU smoke check for OLI training/play managers and deployment dimensions."""

import argparse

import mujoco
import torch

from common.paths import robot_asset_dir, mujoco_path
from deployment.config_manager import DeploymentConfig
from tasks.registry import load_env_cfg, load_env_cls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for task in ("Ref-NoMoMo-Flat-LimX-OLI-HU-D04", "MOMO-Lumped-Flat-LimX-OLI-HU-D04-EffortScale"):
        for play in (False, True):
            cfg = load_env_cfg(task, play=play)
            cfg.scene.num_envs = 2
            env = load_env_cls(task)(cfg=cfg, device=args.device)
            try:
                env.reset()
                for _ in range(10):
                    obs, reward, _, _, _ = env.step(torch.zeros((2, 31), device=args.device))
                    if not torch.isfinite(reward).all() or not torch.isfinite(obs["actor"]).all():
                        raise RuntimeError(f"Non-finite rollout: {task}, play={play}")
                model = mujoco.MjModel.from_xml_path(mujoco_path(robot_asset_dir("oli") / "scene.xml"))
                deployment = DeploymentConfig.from_env(task, cfg, model)
                if deployment.observation_dim != obs["actor"].shape[-1]:
                    raise RuntimeError("Training/deployment observation dimensions differ")
                print(f"PASS {task} play={play}: actor={tuple(obs['actor'].shape)}")
            finally:
                env.close()


if __name__ == "__main__":
    main()
