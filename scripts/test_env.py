from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # Register custom ManiSkill environments
import numpy as np
import torch
import tyro

@dataclass
class Args:
    seed: int = 0
    """seed of the experiment"""
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment"""
    reward_mode: str = "dense"
    """reward type"""
    randomize_init_config: bool = True
    """whether or not to randomize initial configuration of objects"""
    obj_noise: float = 0.0
    """obj observation noise"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environment setup
    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu",
        reward_mode=args.reward_mode,
        randomize_init_config=args.randomize_init_config,
        obj_noise=args.obj_noise
    )

    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    env = gym.make(
        args.env_id,
        num_envs=1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs
    )

    print(f"Testing environment: {args.env_id}")
    print(f"Observation noise: {env_kwargs['obj_noise']}")

    env.reset()
    obs = env.step(env.action_space.sample())
    print(f"Observation size = {env.observation_space.shape}")
    print(f"Action size = {env.action_space.shape}")
    print(f"Example observation: {obs}")
    env.close()
    print("Environment test completed successfully!")