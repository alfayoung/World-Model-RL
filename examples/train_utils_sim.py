import pathlib
from tqdm import tqdm
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL
import imageio
import os
from pathlib import Path

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None,
                                       canonical_mgr=None, latent_tracker=None, evolution_plotter=None,
                                       proprio_tracker=None, proprio_plotter=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    episode_count = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            # Enable video saving based on variant settings
            traj = collect_traj(variant, agent, env, i, agent_dp, proprio_tracker,
                              save_video=True, episode=episode_count)
            traj_id = online_replay_buffer._traj_counter
            episode_count += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('success status:', traj['is_success'])
            print('total env steps:', total_env_steps)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        # wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                        # Record latent policy outputs for canonical states
                        INIT_CANONICAL_AT_STEP = variant.latent_viz_init_step
                        if canonical_mgr is not None and i >= INIT_CANONICAL_AT_STEP:
                            # Initialize canonical states on first eligible step
                            if not canonical_mgr.is_initialized:
                                canonical_mgr.initialize(online_replay_buffer, sample_method='random')
                                print(f"[Latent Viz] Initialized {canonical_mgr.get_num_states()} canonical states at step {i}")

                            # Record latent outputs
                            if latent_tracker is not None:
                                latent_tracker.record_checkpoint(i, agent, canonical_mgr.get_states())

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                        # Generate latent evolution visualization
                        INIT_CANONICAL_AT_STEP = variant.latent_viz_init_step
                        if evolution_plotter is not None and latent_tracker is not None and i >= INIT_CANONICAL_AT_STEP:
                            if latent_tracker.get_num_checkpoints() > 0:
                                import matplotlib.pyplot as plt

                                # Generate full evolution plot (all states combined)
                                fig_full = evolution_plotter.fit_and_plot(
                                    latent_tracker,
                                    current_step=i,
                                    show_arrows=True,
                                    show_labels=True
                                )

                                # Generate individual per-state evolution plots
                                state_figures = evolution_plotter.plot_per_state_evolution(
                                    latent_tracker,
                                    current_step=i,
                                    return_individual=True
                                )

                                # Prepare wandb log dict
                                log_dict = {'latent_evolution/trajectory_plot': wandb.Image(fig_full)}
                                # Add individual state plots
                                for state_idx, fig in state_figures.items():
                                    log_dict[f'latent_evolution/state_{state_idx}'] = wandb.Image(fig)

                                # Log to wandb
                                wandb_logger.log(log_dict, step=i)

                                # Close all figures
                                plt.close(fig_full)
                                for fig in state_figures.values():
                                    plt.close(fig)

                                print(f"[Latent Viz] Generated evolution plots at step {i} ({len(state_figures)} individual states)")

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

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None, proprio_tracker=None, save_video=False, video_base_dir=None, episode=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)

    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()

    image_list = [] # for visualization
    video_image_list = []  # Collect images for video (original resolution)
    rewards = []
    action_list = []
    obs_list = []
    qpos_list = []  # for proprioceptive tracking

    # Start tracking proprioceptive trajectory
    if proprio_tracker is not None:
        proprio_tracker.start_trajectory(training_step=i)

    # Set up video directory with clear structure
    if save_video:
        if video_base_dir is None:
            video_base_dir = os.path.join(variant.outputdir, "trajectory_videos")
        video_dir = Path(video_base_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Video base directory: {video_dir}")

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)

        # Save image for video at the start of the timestep
        if save_video:
            video_image_list.append(curr_image)

        qpos = obs_to_qpos(obs, variant)
        qpos_list.append(qpos)

        # Record proprioceptive state
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

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if i == 0:
                # for initial round of data collection, we sample from standard gaussian noise
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
            
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)
     
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
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

    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')

    # Save video if requested
    if save_video:
        # Create video directory structure with clear naming
        video_subdir = Path(video_dir) / f"step_{i:07d}"
        video_subdir.mkdir(parents=True, exist_ok=True)

        # Create informative filename
        success_tag = "success" if is_success else "fail"
        if episode is not None:
            video_filename = f"ep_{episode}_ret_{episode_return:.2f}_{success_tag}.mp4"
        else:
            video_filename = f"step_{i}_ret_{episode_return:.2f}_{success_tag}.mp4"
        video_path = video_subdir / video_filename

        # Save video using imageio
        print(f"[DEBUG] Saving trajectory video to: {video_path}")
        images_uint8 = [img.astype(np.uint8) for img in video_image_list]
        imageio.mimsave(str(video_path), images_uint8, fps=20)
        print(f"[DEBUG] Video saved successfully: {video_path}")

    # End proprioceptive trajectory tracking
    if proprio_tracker is not None:
        proprio_tracker.end_trajectory(success=is_success, episode_return=episode_return)

    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
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
        'qpos_trajectory': np.array(qpos_list)  # Add proprioceptive trajectory
    }

def _get_libero_env(variant):
    """Initializes and returns the LIBERO environment, along with the task description."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[variant.libero_suite]()
    task_id = variant.task_id
    task = task_suite.get_task(task_id)
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
    env = OffScreenRenderEnv(**env_args)
    env.seed(variant.seed)
    initial_states = task_suite.get_task_init_states(task_id)
    return env, initial_states

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)
    
    if 'libero' in variant.env:
        # ensure fair evaluation with fixed initial states
        env, initial_states = _get_libero_env(variant)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            env.reset()
            obs = env.set_init_state(initial_states[rollout_id])
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                rng, key = jax.random.split(rng)
                assert agent_dp is not None
                
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                
                
                if i == 0:
                    # for initial evaluation, we sample from standard gaussian noise to evaluate the base policy's performance
                    noise = jax.random.normal(rng, (1, 50, 32))
                else:
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                    
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
