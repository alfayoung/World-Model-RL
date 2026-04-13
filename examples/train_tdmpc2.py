"""Main training loop for TD-MPC2 baseline on LIBERO."""
from __future__ import annotations
import os
import pathlib
import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from tdmpc2.agent import TDMPC2Agent
from tdmpc2.data.replay_buffer import ReplayBuffer
from tdmpc2.utils.wandb_logger import WandBLogger


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _preprocess_obs(raw_obs: dict, variant: dict) -> dict:
    """Extract pixels and optional state from LIBERO raw obs dict."""
    # LIBERO obs keys: 'agentview_image', 'robot0_eef_pos', etc.
    cam_key = 'agentview_image'
    pixels = raw_obs[cam_key]  # (H, W, 3) uint8

    processed = {'pixels': pixels}
    if variant.get('add_states', True):
        state = np.concatenate([
            raw_obs.get('robot0_eef_pos', np.zeros(3)),
            raw_obs.get('robot0_eef_quat', np.zeros(4)),
            raw_obs.get('robot0_gripper_qpos', np.zeros(2)),
        ]).astype(np.float32)
        processed['state'] = state
    return processed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(agent: TDMPC2Agent, env, variant: dict, num_episodes: int) -> dict:
    successes, ep_returns = [], []
    for _ in range(num_episodes):
        raw_obs = env.reset()
        obs = _preprocess_obs(raw_obs, variant)
        done = False
        ep_ret = 0.0
        prev_mean = None
        while not done:
            action, prev_mean = agent.select_action(obs, eval_mode=True, prev_mean=prev_mean)
            raw_obs, reward, done, info = env.step(action)
            obs = _preprocess_obs(raw_obs, variant)
            ep_ret += reward
        successes.append(float(info.get('success', 0.0)))
        ep_returns.append(ep_ret)
    return {
        'eval/success_rate': float(np.mean(successes)),
        'eval/episode_return': float(np.mean(ep_returns)),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(variant: dict):
    seed = variant['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------- Environment --------------------------------------------------
    libero_suite = variant.get('libero_suite', 'libero_90')
    task_id = variant.get('task_id', 0)
    resolution = variant.get('resize_image', 64)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[libero_suite]()
    task = task_suite.get_task(task_id)
    env, task_description = _get_libero_env(task, resolution, seed)
    print(f'Task: {task_description}')

    # ---------- Observation / action spaces ----------------------------------
    raw_obs = env.reset()
    cam_key = 'agentview_image'
    h, w, c = raw_obs[cam_key].shape
    obs_shape = (c, h, w)           # CHW for the encoder

    state_dim = 0
    if variant.get('add_states', True):
        state_dim = 9               # eef_pos(3) + eef_quat(4) + gripper(2)

    action_dim = env.action_space.shape[0]
    obs_space = {'pixels': (h, w, c)}
    if state_dim > 0:
        obs_space['state'] = (state_dim,)

    # ---------- Agent --------------------------------------------------------
    agent = TDMPC2Agent(
        obs_shape=obs_shape,
        action_dim=action_dim,
        state_dim=state_dim,
        latent_dim=variant.get('latent_dim', 256),
        hidden_dim=variant.get('hidden_dim', 256),
        num_q=variant.get('num_q', 5),
        encoder_feature_dim=variant.get('encoder_feature_dim', 256),
        lr=variant.get('lr', 3e-4),
        tau=variant.get('tau', 0.005),
        discount=variant.get('discount', 0.99),
        horizon=variant.get('horizon', 5),
        num_samples=variant.get('num_samples', 512),
        num_elites=variant.get('num_elites', 64),
        num_pi_trajs=variant.get('num_pi_trajs', 24),
        temperature=variant.get('temperature', 0.5),
        std_max=variant.get('std_max', 2.0),
        std_min=variant.get('std_min', 0.05),
        consistency_coef=variant.get('consistency_coef', 20.0),
        reward_coef=variant.get('reward_coef', 0.5),
        value_coef=variant.get('value_coef', 0.1),
        seed=seed,
        device=variant.get('device', 'cuda'),
    )

    # ---------- Replay buffer ------------------------------------------------
    replay = ReplayBuffer(
        capacity=variant.get('replay_capacity', 200_000),
        obs_space=obs_space,
        action_dim=action_dim,
    )

    # ---------- Logger -------------------------------------------------------
    exp_name = (
        f"tdmpc2_{libero_suite}_task{task_id}_seed{seed}"
    )
    if variant.get('prefix'):
        exp_name = variant['prefix'] + '_' + exp_name
    logger = WandBLogger(
        project=variant.get('wandb_project', 'tdmpc2_libero'),
        name=exp_name,
        config=variant,
        enabled=variant.get('use_wandb', True),
    )

    # ---------- Training loop ------------------------------------------------
    max_steps = variant.get('max_steps', 500_000)
    start_updates = variant.get('start_online_updates', 500)
    batch_size = variant.get('batch_size', 256)
    utd = variant.get('multi_grad_step', 1)
    eval_interval = variant.get('eval_interval', 10_000)
    log_interval = variant.get('log_interval', 500)
    eval_episodes = variant.get('eval_episodes', 10)
    query_freq = variant.get('query_freq', 1)  # env steps per action

    raw_obs = env.reset()
    obs = _preprocess_obs(raw_obs, variant)
    prev_mean = None
    ep_step = 0
    ep_ret = 0.0

    for global_step in range(1, max_steps + 1):
        # ---------- Collect one transition -----------------------------------
        if global_step < start_updates:
            action = env.action_space.sample()
            prev_mean = None
        else:
            if ep_step % query_freq == 0:
                action, prev_mean = agent.select_action(obs, eval_mode=False, prev_mean=prev_mean)

        raw_next_obs, reward, done, info = env.step(action)
        next_obs = _preprocess_obs(raw_next_obs, variant)
        replay.insert(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_ret += reward
        ep_step += 1

        if done:
            raw_obs = env.reset()
            obs = _preprocess_obs(raw_obs, variant)
            prev_mean = None
            ep_step = 0
            ep_ret = 0.0

        # ---------- Learning updates -----------------------------------------
        metrics: dict = {}
        if global_step >= start_updates and len(replay) >= batch_size:
            for _ in range(utd):
                batch = replay.sample(batch_size)
                update_info = agent.update(batch)
            metrics.update(update_info)

        # ---------- Logging --------------------------------------------------
        if global_step % log_interval == 0 and metrics:
            logger.log(metrics, step=global_step)
            print(f'[{global_step}/{max_steps}] ' +
                  ' | '.join(f'{k}={v:.4f}' for k, v in metrics.items()))

        # ---------- Evaluation -----------------------------------------------
        if global_step % eval_interval == 0:
            eval_metrics = evaluate(agent, env, variant, num_episodes=eval_episodes)
            logger.log(eval_metrics, step=global_step)
            print(f'[eval @{global_step}] ' +
                  ' | '.join(f'{k}={v:.4f}' for k, v in eval_metrics.items()))

    logger.finish()
    env.close()
