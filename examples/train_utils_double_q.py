"""Training utilities for Real-World Digital-Twin Double-Q Learning."""

from tqdm import tqdm
import numpy as np
import wandb
import jax
import jax.numpy as jnp
from openpi_client import image_tools
import math
from typing import Dict, List, Tuple, Optional
import copy
import imageio
import os
from pathlib import Path

from examples.train_utils_sim import (
    obs_to_img, obs_to_pi_zero_input, obs_to_qpos,
    add_online_data_to_buffer, perform_control_eval,
)


def beta_schedule(step: int, warmup_steps: int = 5000, beta_max: float = 0.5) -> float:
    """
    Compute mixing coefficient β for trajectory scoring.

    Early training: β ≈ 0 (rely on Monte Carlo returns from twin rollouts)
    Late training: β → beta_max (use value function estimates)

    Args:
        step: Current training step
        warmup_steps: Number of steps to reach beta_max
        beta_max: Maximum value of β

    Returns:
        Current β value
    """
    return min(step / warmup_steps, 1.0) * beta_max


def compute_trajectory_score(
    twin_return: float,
    terminal_q_value: float,
    beta: float,
    gamma: float,
    horizon: int
) -> float:
    """
    Compute trajectory score as a mix of Monte Carlo return and value estimate.

    score = (1 - β) * G + β * Q_twin(s_H, a_H)

    Args:
        twin_return: Discounted return from twin rollout G_k
        terminal_q_value: Q_twin value at terminal state (or min over ensemble)
        beta: Mixing coefficient
        gamma: Discount factor
        horizon: Rollout horizon

    Returns:
        Combined score
    """
    score = (1.0 - beta) * twin_return + beta * terminal_q_value
    return score


def collect_single_twin_trajectory(
    variant,
    agent,
    env_twin,
    initial_obs,
    seed_key: jax.Array,
    agent_dp,
    training_step: int,
    horizon: Optional[int] = None,
    save_video: bool = False,
    video_dir: Optional[str] = None,
    episode: Optional[int] = None,
) -> Dict:
    """
    Collect a single trajectory in the twin environment using a specific seed.

    Args:
        variant: Configuration dictionary
        agent: Twin SAC agent
        env_twin: Twin environment
        initial_obs: Initial observation to start from
        seed: RNG seed for deterministic rollout
        agent_dp: Diffusion policy (π₀)
        training_step: Current training step
        horizon: Maximum rollout horizon (defaults to variant.max_timesteps)
        save_video: Whether to save video of the rollout
        video_base_dir: Directory to save videos (required if save_video=True)

    Returns:
        Dictionary containing trajectory data and computed return
    """
    query_frequency = variant.query_freq
    max_timesteps = horizon if horizon is not None else variant.max_timesteps
    discount = variant.get('discount', 0.999)

    # Create deterministic RNG for this seed
    agent._rng = seed_key
    agent._rng, rng = jax.random.split(agent._rng)

    obs = initial_obs
    action_list = []
    obs_list = []
    rewards = []
    dones = []
    image_list = []  # Collect images for video
    twin_return = 0.0
    discount_factor = 1.0

    for t in range(max_timesteps):
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)

        # Save image for video
        if save_video:
            image_list.append(curr_image)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:
            # Sample noise from SAC policy
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)

            if training_step == 0:
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

            # Generate actions from diffusion policy
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)

        # Execute action in twin environment
        action_t = actions[t % query_frequency]

        if 'libero' in variant.env:
            obs, reward, done, _ = env_twin.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env_twin.step(action_t)
            done = terminated or truncated
        else:
            raise NotImplementedError()

        rewards.append(reward)
        dones.append(done)

        # Accumulate discounted return
        twin_return += discount_factor * reward
        discount_factor *= discount

        if done:
            break

    # Add final observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict_final = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict_final)

    # Save final image for video
    if save_video:
        image_list.append(curr_image)

    # Compute episode metrics
    rewards_array = np.array(rewards)
    episode_return = np.sum(rewards_array[rewards_array != None])
    is_success = (rewards[-1] == variant.env_max_reward) if len(rewards) > 0 else False

    # Save video if requested
    if save_video and video_dir is not None:
        # Create video directory structure with clear naming
        video_subdir = Path(video_dir) / f"step_{training_step:07d}"
        video_subdir.mkdir(parents=True, exist_ok=True)

        # Create informative filename
        success_tag = "success" if is_success else "fail"
        video_filename = f"twin_ep_{episode}_seed_{seed_key}_ret_{twin_return:.2f}_{success_tag}.mp4"
        video_path = video_subdir / video_filename

        # Save video using imageio
        print(f"[DEBUG] Saving twin trajectory video to: {video_path}")
        images_uint8 = [img.astype(np.uint8) for img in image_list]
        imageio.mimsave(str(video_path), images_uint8, fps=20)
        print(f"[DEBUG] Video saved successfully: {video_path}")

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards_array,
        'twin_return': twin_return,
        'is_success': is_success,
        'episode_return': episode_return,
        'timesteps': t + 1,
        'dones': dones,
    }


def collect_K_twin_trajectories(
    variant,
    agent,
    env_twin,
    env_real,
    agent_dp,
    seed_keys: list[jax.Array],
    beta: float,
    training_step: int,
    twin_replay_buffer,
    save_video: bool = False,
    video_base_dir: Optional[str] = None,
    episode: Optional[int] = None,
) -> Tuple[int, Dict, List[Dict]]:
    """
    Collect K candidate trajectories in twin environment and select the best one.

    This implements the core Double-Q twin planning mechanism:
    1. Reset env_twin to match env_real's current state
    2. Generate K trajectories with different seeds
    3. Score each trajectory: (1-β)*G_k + β*Q_twin(s_H, a_H)
    4. Select best trajectory
    5. Execute only the first action in env_real
    6. Store all twin transitions in twin buffer

    Args:
        variant: Configuration dictionary
        agent: Twin SAC agent
        env_twin: Twin environment
        env_real: Real environment (for state synchronization)
        agent_dp: Diffusion policy
        K: Number of candidate trajectories
        beta: Mixing coefficient for scoring
        training_step: Current training step
        twin_replay_buffer: Replay buffer for twin transitions
        save_videos: Whether to save videos of twin trajectories
        video_base_dir: Base directory for saving videos

    Returns:
        Tuple of (selected_seed, best_trajectory, all_trajectories)
    """
    # Get current state from real environment
    # Note: For pure simulation, we can synchronize by resetting twin to same state
    # For real robots, this would involve approximate state reconstruction

    candidate_trajectories = []
    scores = []

    # Set up video directory with clear structure
    if save_video:
        if video_base_dir is None:
            video_base_dir = os.path.join(variant.outputdir, "twin_trajectory_videos")
        video_dir = Path(video_base_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Video base directory: {video_dir}")

    print(f"\nGenerating {len(seed_keys)} candidate trajectories in twin environment (β={beta:.3f})...")

    for seed_key in seed_keys:
        # Reset twin environment to match real environment state
        if 'libero' in variant.env:
            # Get current state from real env
            real_state = env_real.get_sim_state()
            # Set twin to same state
            env_twin.reset()
            obs_twin = env_twin.set_init_state(real_state)
        elif 'aloha' in variant.env:
            # For Aloha, synchronize via reset with same initial condition
            obs_twin, _ = env_twin.reset()
        else:
            raise NotImplementedError()

        # Collect trajectory with this seed
        traj = collect_single_twin_trajectory(
            variant=variant,
            agent=agent,
            env_twin=env_twin,
            initial_obs=obs_twin,
            seed_key=seed_key,
            agent_dp=agent_dp,
            training_step=training_step,
            save_video=save_video,
            video_dir=str(video_dir) if save_video else None,
            episode=episode,
        )

        # Compute Q_twin value at terminal state for scoring
        if len(traj['actions']) > 0:
            terminal_obs = traj['observations'][-2]  # Last obs before final
            terminal_action = traj['actions'][-1]  # Last action

            # Get Q_twin values (ensemble)
            terminal_q = agent.get_twin_q_value(terminal_obs, terminal_action)
        else:
            terminal_q = 0.0

        # Compute trajectory score
        score = compute_trajectory_score(
            twin_return=traj['twin_return'],
            terminal_q_value=terminal_q,
            beta=beta,
            gamma=variant.get('discount', 0.999),
            horizon=traj['timesteps']
        )

        candidate_trajectories.append(traj)
        scores.append(score)

        print(f"  Seed {seed_key}: Return={traj['twin_return']:.3f}, Terminal Q Value={terminal_q:.3f}, "
              f"Score={score:.3f}, Success={traj['is_success']}")

        # Store all transitions from this trajectory into twin buffer
        store_twin_trajectory_in_buffer(variant, traj, twin_replay_buffer)

    # Select best trajectory
    best_idx = np.argmax(scores)
    best_seed = seed_keys[best_idx]
    best_trajectory = candidate_trajectories[best_idx]

    print(f"✓ Selected trajectory {best_idx} with score {scores[best_idx]:.3f}")

    return best_seed, best_trajectory, candidate_trajectories


def store_twin_trajectory_in_buffer(variant, traj, twin_replay_buffer):
    """
    Store all transitions from a twin trajectory into the twin replay buffer.

    Args:
        variant: Configuration dictionary
        traj: Trajectory dictionary from collect_single_twin_trajectory
        twin_replay_buffer: Replay buffer for twin data
    """
    query_steps = len(traj['actions'])

    # Convert sparse reward to -1/0 format
    if traj['is_success']:
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    # Insert transitions at query_freq intervals
    for j in range(query_steps):
        insert_dict = {
            'observations': traj['observations'][j],
            'next_observations': traj['observations'][j + 1],
            'actions': traj['actions'][j],
            'rewards': rewards[j],
            'masks': masks[j],
            'dones': float(masks[j] == 0),
        }

        # Apply discount scaling
        if variant.query_freq > 1:
            insert_dict['discounts'] = variant.get('discount', 0.999) ** variant.query_freq
        else:
            insert_dict['discounts'] = variant.get('discount', 0.999)

        twin_replay_buffer.insert(insert_dict)


def double_q_training_loop(
    variant,
    agent,
    env_real,
    env_twin,
    eval_env,
    real_replay_buffer,
    twin_replay_buffer,
    wandb_logger,
    perform_control_evals=True,
    shard_fn=None,
    agent_dp=None,
    proprio_tracker=None,
    proprio_plotter=None
):
    """
    Main training loop for Real-World Digital-Twin Double-Q Learning.

    This implements the algorithm from IDEA.md where:
    - env_real is used for final execution and performance evaluation
    - env_twin is used for K-seed lookahead planning
    - Q_real is trained on D_real (real environment data)
    - Q_twin is trained on D_twin (twin environment data)

    Args:
        variant: Configuration dictionary with Double-Q parameters
        agent: TwinPixelSACLearner agent
        env_real: Real (or primary) environment
        env_twin: Twin environment for planning
        eval_env: Evaluation environment
        real_replay_buffer: Replay buffer for real transitions (D_real)
        twin_replay_buffer: Replay buffer for twin transitions (D_twin)
        wandb_logger: WandB logger
        ... (other standard arguments)
    """
    # Get Double-Q specific parameters
    K_seeds = variant.get('K_seeds', 5)
    beta_warmup_steps = variant.get('beta_warmup_steps', 5000)
    beta_max = variant.get('beta_max', 0.5)
    twin_update_freq = variant.get('twin_update_freq', 1)  # Update twin critic every N steps

    # Create iterators for both buffers
    real_buffer_iterator = real_replay_buffer.get_iterator(variant.batch_size)
    twin_buffer_iterator = twin_replay_buffer.get_iterator(variant.batch_size)

    if shard_fn is not None:
        real_buffer_iterator = map(shard_fn, real_buffer_iterator)
        twin_buffer_iterator = map(shard_fn, twin_buffer_iterator)

    total_env_steps = 0
    i = 0
    episode_count = 0

    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        # perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
        
        rng = jax.random.PRNGKey(variant.seed)
        while i <= variant.max_steps:
            # ========== Episode-Level Twin Planning ==========
            env_real.reset() # Ensure the real_env is fixed for twin planning
            
            # Compute current β
            beta = beta_schedule(i, beta_warmup_steps, beta_max)

            # Compute unique seeds for K trajectories
            seed_keys = []
            rng, *step_keys = jax.random.split(rng, K_seeds + 1)
            for k, key in enumerate(step_keys):
                seed_key = jax.random.fold_in(key, i)
                seed_key = jax.random.fold_in(seed_key, k)
                seed_keys.append(seed_key)

            # Collect K twin trajectories and select best
            best_seed_key, best_traj, all_trajs = collect_K_twin_trajectories(
                variant=variant,
                agent=agent,
                env_twin=env_twin,
                env_real=env_real,
                agent_dp=agent_dp,
                seed_keys=seed_keys,
                beta=beta,
                training_step=i,
                twin_replay_buffer=twin_replay_buffer,
                save_video=False,
                episode=episode_count,
            )

            # ========== Real Environment Execution ==========
            # Execute the full best trajectory in real environment using the selected seed
            print(f"\nExecuting selected trajectory (seed {best_seed_key}) in REAL environment...")

            # Set up video directory for real trajectories (same structure as twin)
            real_traj = execute_real_trajectory_with_seed(
                variant=variant,
                agent=agent,
                env_real=env_real,
                seed_key=best_seed_key,
                agent_dp=agent_dp,
                training_step=i,
                proprio_tracker=proprio_tracker,
                save_video=True,
                episode=episode_count
            )

            # Store real trajectory in real buffer
            add_online_data_to_buffer(variant, real_traj, real_replay_buffer)
            total_env_steps += real_traj['env_steps']
            episode_count += 1

            print(f'Real buffer timesteps: {len(real_replay_buffer)}')
            print(f'Twin buffer timesteps: {len(twin_replay_buffer)}')
            print(f'Episode {episode_count}, Total env steps: {total_env_steps}')

            # ========== Gradient Updates ==========
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(real_traj["rewards"]) * variant.multi_grad_step

            if len(real_replay_buffer) > variant.start_online_updates:
                for grad_step in range(num_gradsteps):
                    # Perform initial evaluation before any updates
                    if i == 0:
                        print('Performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, real_replay_buffer,
                                             real_buffer_iterator, eval_env)

                    # Sample batches from both buffers
                    batch_real = next(real_buffer_iterator)

                    # Only update twin critic periodically and if twin buffer has data
                    update_twin = (i % twin_update_freq == 0) and (len(twin_replay_buffer) > variant.batch_size)

                    if update_twin:
                        batch_twin = next(twin_buffer_iterator)
                    else:
                        batch_twin = None

                    # Update agent (both real critic and conditionally twin critic)
                    # agent: TwinPixelSACLearner
                    update_info = agent.update(
                        batch_real=batch_real,
                        batch_twin=batch_twin,
                        update_twin=update_twin
                    )

                    pbar.update()
                    i += 1

                    # ========== Logging ==========
                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}

                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)

                        # Log Double-Q specific metrics
                        wandb_logger.log({
                            'real_buffer_size': len(real_replay_buffer),
                            'twin_buffer_size': len(twin_replay_buffer),
                            'episode_return (real)': real_traj['episode_return'],
                            'is_success (real)': int(real_traj['is_success']),
                            'beta': beta,
                            'K_seeds': K_seeds,
                            'selected_seed': best_seed_key % K_seeds,
                        }, i)

                        # Log trajectory selection statistics
                        twin_scores = [compute_trajectory_score(
                            t['twin_return'], 0.0, beta, variant.get('discount', 0.999), t['timesteps']
                        ) for t in all_trajs]
                        wandb_logger.log({
                            'twin_trajectories/mean_score': np.mean(twin_scores),
                            'twin_trajectories/max_score': np.max(twin_scores),
                            'twin_trajectories/min_score': np.min(twin_scores),
                            'twin_trajectories/mean_return': np.mean([t['twin_return'] for t in all_trajs]),
                            'twin_trajectories/success_rate': np.mean([t['is_success'] for t in all_trajs]),
                        }, i)

                    # ========== Evaluation ==========
                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(real_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': episode_count}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)

                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, real_replay_buffer,
                                             real_buffer_iterator, eval_env)

                        # Generate proprioceptive trajectory visualizations
                        if proprio_tracker is not None and proprio_plotter is not None:
                            if len(proprio_tracker.trajectories) > 0:
                                import matplotlib.pyplot as plt

                                # Get recent trajectories
                                recent_trajs = proprio_tracker.get_recent_trajectories(n=10)
                                recent_successful = proprio_tracker.get_recent_trajectories(n=5, only_successful=True)

                                # Plot all recent trajectories
                                fig_all = proprio_plotter.plot_trajectories(
                                    recent_trajs,
                                    title=f"Recent Proprioceptive Trajectories (Step {i})",
                                    label_trajectories=False
                                )

                                # Plot successful trajectories only (if any)
                                log_dict = {'proprio_trajectories/recent_all': wandb.Image(fig_all)}

                                if len(recent_successful) > 0:
                                    fig_success = proprio_plotter.plot_trajectories(
                                        recent_successful,
                                        title=f"Recent Successful Trajectories (Step {i})",
                                        colormap='viridis',
                                        label_trajectories=False
                                    )
                                    log_dict['proprio_trajectories/recent_successful'] = wandb.Image(fig_success)
                                    plt.close(fig_success)

                                    # Plot single latest successful trajectory with time gradient
                                    latest_successful = recent_successful[-1]
                                    fig_single = proprio_plotter.plot_single_trajectory(
                                        latest_successful,
                                        title=f"Latest Successful Trajectory (Episode {latest_successful.episode_id})"
                                    )
                                    log_dict['proprio_trajectories/latest_successful_single'] = wandb.Image(fig_single)
                                    plt.close(fig_single)

                                # Log to wandb
                                wandb_logger.log(log_dict, step=i)
                                plt.close(fig_all)

                                # Print stats
                                stats = proprio_tracker.get_stats()
                                print(f"[Proprio Viz] Generated trajectory plots at step {i}")
                                print(f"  Total trajectories: {stats['num_trajectories']}")
                                print(f"  Success rate: {stats['success_rate']:.2%}")
                                print(f"  Avg trajectory length: {stats['avg_length']:.1f}")

                    # ========== Checkpointing ==========
                    if i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                        print(f'Checkpoint saved at step {i}')

    print("Training complete!")


def execute_real_trajectory_with_seed(
    variant,
    agent,
    env_real,
    seed_key: jax.Array,
    agent_dp,
    training_step: int,
    proprio_tracker=None,
    save_video: bool = True,
    video_base_dir: Optional[str] = None,
    episode: Optional[int] = None,
) -> Dict:
    """
    Execute a full trajectory in the real environment using a specific seed.

    This ensures the real execution matches the selected twin trajectory by using
    the same RNG seed for policy sampling.

    Args:
        variant: Configuration dictionary
        agent: Twin SAC agent
        env_real: Real environment
        seed: RNG seed (should match the selected twin trajectory)
        agent_dp: Diffusion policy
        training_step: Current training step
        proprio_tracker: Optional proprioceptive state tracker
        save_video: Whether to save video of the rollout
        video_dir: Directory to save videos (required if save_video=True)

    Returns:
        Trajectory dictionary
    """
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    # Use the provided seed for deterministic execution
    agent._rng = seed_key
    agent._rng, rng = jax.random.split(agent._rng)

    # Reset environment
    if 'libero' in variant.env:
        # no reset here! reset is done in the outer loop
        obs = env_real.env._get_observations()
    elif 'aloha' in variant.env:
        obs, _ = env_real.reset()
    else:
        raise NotImplementedError()

    image_list = []
    video_image_list = []  # Collect images for video (original resolution)
    rewards = []
    action_list = []
    obs_list = []
    qpos_list = []

    # Start proprioceptive tracking
    if proprio_tracker is not None:
        proprio_tracker.start_trajectory(training_step=training_step)

    # Set up video directory with clear structure
    if save_video:
        if video_base_dir is None:
            video_base_dir = os.path.join(variant.outputdir, "real_trajectory_videos")
        video_dir = Path(video_base_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Video base directory: {video_dir}")

    for t in tqdm(range(max_timesteps), desc="Real execution"):
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)
        qpos_list.append(qpos)

        # Save image for video at the start of the timestep
        if save_video:
            video_image_list.append(curr_image)

        if proprio_tracker is not None:
            proprio_tracker.record_state(qpos, t)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)

            # Sample noise from SAC policy with deterministic seed
            if training_step == 0:
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

            # Generate actions from diffusion policy
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)

        # Execute action
        action_t = actions[t % query_frequency]

        if 'libero' in variant.env:
            obs, reward, done, _ = env_real.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env_real.step(action_t)
            done = terminated or truncated
        else:
            raise NotImplementedError()

        rewards.append(reward)
        image_list.append(curr_image)

        if done:
            break

    # Add final observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)

    # Save final image for video
    if save_video:
        video_image_list.append(curr_image)

    # Compute episode metrics
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards != None])
    is_success = (reward == env_max_reward)
    # Get Q_twin values (ensemble)
    terminal_q = agent.get_real_q_value(obs_list[-2], action_list[-1])
    print(f'Real Rollout Done: episode_return={episode_return}, Success: {is_success}, Terminal Q Value: {terminal_q}')

    # Save video if requested
    if save_video:
        # Create video directory structure with clear naming
        video_subdir = Path(video_dir) / f"step_{training_step:07d}"
        video_subdir.mkdir(parents=True, exist_ok=True)

        # Create informative filename
        success_tag = "success" if is_success else "fail"
        video_filename = f"real_ep_{episode}_seed_{seed_key}_q_{terminal_q:.3f}_ret_{episode_return:.2f}_{success_tag}.mp4"
        video_path = video_subdir / video_filename

        # Save video using imageio
        print(f"[DEBUG] Saving real trajectory video to: {video_path}")
        images_uint8 = [img.astype(np.uint8) for img in video_image_list]
        imageio.mimsave(str(video_path), images_uint8, fps=20)
        print(f"[DEBUG] Video saved successfully: {video_path}")

    # End proprioceptive tracking
    if proprio_tracker is not None:
        proprio_tracker.end_trajectory(success=is_success, episode_return=episode_return)

    # Convert to sparse -1/0 rewards
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'qpos_trajectory': np.array(qpos_list)
    }
