"""Launch script for Real-World Digital-Twin Double-Q Learning."""

import argparse
import sys
from examples.train_double_q import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train DSRL with Real-World Digital-Twin Double-Q Learning"
    )

    # ========== Standard Training Arguments ==========
    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10, help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment (libero or aloha_cube)')
    parser.add_argument('--libero_suite', default='libero_90',
                       help='LIBERO benchmark suite name (e.g., libero_90, libero_10, etc.)')
    parser.add_argument('--task_id', default=57, help='Task ID within the LIBERO suite', type=int)
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--latent_viz_init_step', default=5000,
                       help='Step to initialize latent policy visualization.', type=int)
    parser.add_argument('--checkpoint_interval', default=10000, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1,
                       help='whether to add low-dim states to the observations', type=int)
    parser.add_argument('--wandb_project', default='DSRL_double_q', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000,
                       help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='twin_pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1,
                       help='Number of gradient steps to take per environment step (UTD)', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)

    # ========== Double-Q Specific Arguments ==========
    parser.add_argument('--K_seeds', default=5,
                       help='Number of candidate trajectories to generate in twin env', type=int)
    parser.add_argument('--beta_warmup_steps', default=5000,
                       help='Number of steps to linearly increase beta from 0 to beta_max', type=int)
    parser.add_argument('--beta_max', default=0.5,
                       help='Maximum value of beta (mixing coefficient for MC vs Q-value)', type=float)
    parser.add_argument('--twin_update_freq', default=1,
                       help='Update twin critic every N gradient steps', type=int)

    # ========== Network Architecture ==========
    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims=(128, 128, 128),
        cnn_features=(32, 32, 32, 32),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding='VALID',
        latent_dim=50,
        discount=0.999,
        tau=0.005,
        critic_reduction='mean',
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

    # Add Double-Q parameters to variant
    variant.K_seeds = args.K_seeds
    variant.beta_warmup_steps = args.beta_warmup_steps
    variant.beta_max = args.beta_max
    variant.twin_update_freq = args.twin_update_freq

    print("\n" + "="*60)
    print("DOUBLE-Q TRAINING CONFIGURATION")
    print("="*60)
    print(f"Environment: {variant.env}")
    if variant.env == 'libero':
        print(f"LIBERO Suite: {variant.libero_suite}, Task ID: {variant.task_id}")
    print(f"Seed: {variant.seed}")
    print(f"\nDouble-Q Parameters:")
    print(f"  K_seeds (candidate trajectories): {variant.K_seeds}")
    print(f"  beta_warmup_steps: {variant.beta_warmup_steps}")
    print(f"  beta_max: {variant.beta_max}")
    print(f"  twin_update_freq: {variant.twin_update_freq}")
    print(f"\nNetwork Configuration:")
    print(f"  critic_reduction: {variant.train_kwargs['critic_reduction']}")
    print(f"  num_qs per critic: {variant.train_kwargs['num_qs']}")
    print(f"  encoder_type: {variant.train_kwargs['encoder_type']}")
    print(f"\nTraining Configuration:")
    print(f"  max_steps: {variant.max_steps}")
    print(f"  batch_size: {variant.batch_size}")
    print(f"  multi_grad_step (UTD): {variant.multi_grad_step}")
    print(f"  query_freq: {variant.query_freq}")
    print("="*60 + "\n")

    main(variant)
    sys.exit()
