import os
import time
from tqdm import tqdm
import time
import numpy as np
import jax
import sys
import select
import tty
import termios
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       shard_fn=None, agent_dp=None, robot_config=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)
        
    i = 0
    total_env_steps = 0
    total_num_traj = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
   
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, env, i, agent_dp, wandb_logger, total_num_traj, robot_config)
            total_num_traj += 1
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', total_num_traj)
            print('total env steps:', total_env_steps)
            
            if i == 0:
                num_gradsteps = 5000
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step
            print(f'num_gradsteps: {num_gradsteps}')
            if total_num_traj >= variant.num_initial_traj_collect:
                for _ in range(num_gradsteps):

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
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': total_num_traj}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1:
                        if i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):
    
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, 14)
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

def collect_traj(variant, agent, env, i, agent_dp=None, wandb_logger=None, traj_id=None, robot_config=None):
    query_frequency = variant.query_freq
    instruction = variant.instruction
    max_timesteps = robot_config['max_timesteps']
    agent._rng, rng = jax.random.split(agent._rng)
    try:
        env.reset()
    except Exception as e:
        print(f"Environment reset failed")
        import traceback
        traceback.print_exc() 
        import pdb; pdb.set_trace()
    step_time = 1 / 15 # 15 Hz
    last_step_time = time.time()
    old_settings = termios.tcgetattr(sys.stdin)
    
    rewards = []
    action_list = []
    obs_list = []
    image_list = []

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        for t in tqdm(range(max_timesteps)):    
            # Check for keyboard input
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input.lower() == 'q':
                    print("'q' pressed, stopping loop.")
                    break
            
            try:
                _env_obs = env.get_observation()
            except Exception as e:
                print(f"Environment get obs failed")
                import traceback
                traceback.print_exc()
                import pdb; pdb.set_trace()
            curr_obs = _extract_observation(
                    robot_config,
                    _env_obs,
            )
            image_list.append(curr_obs[robot_config['camera_to_use'] + "_image"])

            request_data = get_pi0_input(curr_obs, robot_config, instruction)
        
            if t % query_frequency == 0:

                rng, key = jax.random.split(rng)

                img_all = process_images(variant, curr_obs)
                
                # extract the feature from the pi0 VLM backbone and concat with the qpos as states
                img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
                img_rep_pi0 = img_rep_pi0[:, -1, :] # (1, 2048)
                qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])

                obs_dict = {
                    'pixels': img_all,
                    'state': qpos[np.newaxis, ..., np.newaxis],
                }
                if i == 0:
                    noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                    noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 10 - noise.shape[1], axis=1)
                    noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                    actions_noise = noise[0, :agent.action_chunk_shape[0], :]
                else:
                    # sac agent predicts the noise for diffusion model
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], 10 - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                action_list.append(actions_noise)
                obs_list.append(obs_dict)
                action = agent_dp.infer(request_data, noise=np.asarray(noise))["actions"]

            action_t = action[t % query_frequency]
            
            # binarize gripper action.
            if action_t[-1].item() > 0.5:
                action_t = np.concatenate([action_t[:-1], np.ones((1,))])
            else:
                action_t = np.concatenate([action_t[:-1], np.zeros((1,))])
            action_t = np.clip(action_t, -1, 1)
            
            try:
                env.step(action_t)
            except Exception as e:
                print(f"Environment step failed")
                import traceback
                traceback.print_exc()  # This prints the full traceback
                import pdb; pdb.set_trace()
        
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
            
        print("Trial finished. Mark as (1) Success or (0) Failure:")
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input == '1':
                    print("Trial marked as SUCCESS.")
                    is_success = True
                    break
                elif char_input == '0':
                    print("Trial marked as FAILURE.")                    
                    is_success = False
                    break
                else:
                    print("Invalid input. Please enter '1' for Success or '0' for Failure:")
            time.sleep(0.01) # Small sleep to prevent busy-waiting if no input

        try:
            _env_obs = env.get_observation()
        except Exception as e:
            print(f"Environment get obs failed")
            import traceback
            traceback.print_exc()
            import pdb; pdb.set_trace()
        
        # add last observation
        curr_obs = _extract_observation(
                    robot_config,
                    _env_obs,
            )
        image_list.append(curr_obs[robot_config['camera_to_use'] + "_image"])
        request_data = get_pi0_input(curr_obs, robot_config, instruction)
        img_all = process_images(variant, curr_obs)
        img_rep_pi0, _ = agent_dp.get_prefix_rep(request_data)
        img_rep_pi0 = img_rep_pi0[:, -1, :] # (1, 2048)
        qpos = np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"], img_rep_pi0.flatten()])
        obs_dict = {
            'pixels': img_all,
            'state': qpos[np.newaxis, ..., np.newaxis],
        }
        obs_list.append(obs_dict)
        print(f'Rollout Done')
        
    finally:
        if is_success:
            query_steps = len(action_list)
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            query_steps = len(action_list)
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)
            
        if wandb_logger is not None:
            wandb_logger.log({f'is_success': int(is_success)}, step=i)
            wandb_logger.log({f'total_num_traj': traj_id}, step=i)

        video_path = os.path.join(variant.outputdir, f'video_high_{traj_id}.mp4')
        video = np.stack(image_list)
        ImageSequenceClip(list(video), fps=15).write_videofile(video_path, codec="libx264")
       
        print("Episide Done! Press c after resetting the environment")
        try:
            env.reset()
        except Exception as e:
            print(f"Environment reset failed")
            import traceback
            traceback.print_exc()  # This prints the full traceback
            import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    traj = {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'env_steps': t + 1,
    }
    
    return traj


def _extract_observation(robot_config, obs_dict):
    '''
    from https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py
    '''
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations.keys():
        if robot_config['left_camera_id'] in key and "left" in key:
            left_image = image_observations[key]
        elif robot_config['right_camera_id'] in key and "left" in key:
            right_image = image_observations[key]
        elif robot_config['wrist_camera_id'] in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }
    
def get_pi0_input(obs, robot_config, instruction):
    external_image = obs[robot_config['camera_to_use'] + "_image"]
    request_data = {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            external_image, 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(obs["wrist_image"], 224, 224),
        "observation/joint_position": obs["joint_position"],
        "observation/gripper_position": obs["gripper_position"],
        "prompt": instruction,
    }
    return request_data
    

def process_images(variant, obs):
    '''
    concat the images from all cameras
    '''
    im1 = image_tools.resize_with_pad(obs["left_image"], variant.resize_image, variant.resize_image)
    im2 = image_tools.resize_with_pad(obs["right_image"], variant.resize_image, variant.resize_image)
    im3 = image_tools.resize_with_pad(obs["wrist_image"], variant.resize_image, variant.resize_image)
    img_all = np.concatenate([im1, im2, im3], axis=2)[np.newaxis, ..., np.newaxis]
    return img_all