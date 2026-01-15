import argparse
import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--libero_suite', default='libero_90', help='which libero suite to use')
    parser.add_argument('--task_id', default=57, help='task id.', type=int)
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)

    # VLAC reward densification arguments
    parser.add_argument('--use_vlac_rewards', default=False, type=bool,
                        help='Enable VLAC dense rewards (True) or use sparse rewards (False)')
    parser.add_argument('--vlac_model_path', default='', type=str,
                        help='Path to VLAC InternVL2 model checkpoint')
    parser.add_argument('--vlac_ref_images_path', default='', type=str,
                        help='Path to reference images (directory, .txt file, or comma-separated paths)')
    parser.add_argument('--vlac_batch_num', default=5, type=int,
                        help='Batch size for VLAC inference')
    parser.add_argument('--vlac_temperature', default=0.5, type=float,
                        help='Temperature for VLAC model sampling')
    parser.add_argument('--vlac_top_k', default=1, type=int,
                        help='Top-k sampling for VLAC')
    parser.add_argument('--vlac_reward_scale', default=1.0, type=float,
                        help='Scale factor for VLAC rewards (e.g., 1.0 for [0,1] range)')
    parser.add_argument('--vlac_device', default='cuda', type=str,
                        help='Device for VLAC model (e.g., cuda:0, cuda:1)')
    parser.add_argument('--main_device', default=-1, type=int,
                        help='JAX device ID to use (e.g., 0, 1, 2). Use -1 for all available devices')

    # Video recording arguments
    parser.add_argument('--save_video', default=True, type=bool,
                        help='Enable video recording of trajectories with VLAC reward visualization')

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    