"""Launch script for TD-MPC2 baseline on LIBERO."""
import argparse
import sys
from examples.train_tdmpc2 import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--env', default='libero', type=str)
    parser.add_argument('--libero_suite', default='libero_90', type=str)
    parser.add_argument('--task_id', default=0, type=int)
    parser.add_argument('--resize_image', default=64, type=int)
    parser.add_argument('--add_states', default=1, type=int)

    # Training
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max_steps', default=500_000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--start_online_updates', default=500, type=int)
    parser.add_argument('--multi_grad_step', default=1, type=int,
                        help='Gradient steps per env step (UTD ratio)')
    parser.add_argument('--replay_capacity', default=200_000, type=int)

    # Agent hyperparameters
    parser.add_argument('--latent_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--encoder_feature_dim', default=256, type=int)
    parser.add_argument('--num_q', default=5, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--discount', default=0.99, type=float)

    # MPPI planning
    parser.add_argument('--horizon', default=5, type=int)
    parser.add_argument('--num_samples', default=512, type=int)
    parser.add_argument('--num_elites', default=64, type=int)
    parser.add_argument('--num_pi_trajs', default=24, type=int)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--std_max', default=2.0, type=float)
    parser.add_argument('--std_min', default=0.05, type=float)
    parser.add_argument('--query_freq', default=1, type=int)

    # Loss coefficients
    parser.add_argument('--consistency_coef', default=20.0, type=float)
    parser.add_argument('--reward_coef', default=0.5, type=float)
    parser.add_argument('--value_coef', default=0.1, type=float)

    # Logging
    parser.add_argument('--wandb_project', default='tdmpc2_libero', type=str)
    parser.add_argument('--prefix', default='', type=str)
    parser.add_argument('--use_wandb', default=1, type=int)
    parser.add_argument('--log_interval', default=500, type=int)
    parser.add_argument('--eval_interval', default=10_000, type=int)
    parser.add_argument('--eval_episodes', default=10, type=int)

    # Hardware
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    variant = vars(args)
    variant['add_states'] = bool(variant['add_states'])
    variant['use_wandb'] = bool(variant['use_wandb'])
    print(variant)
    main(variant)
    sys.exit()
