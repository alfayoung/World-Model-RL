"""Launch script for the Dreamer-v3 baseline.

Argument names mirror ``launch_train_sim.py`` wherever possible so that runs
can be compared in WandB using the same group/sweep keys.

Example
-------
  python -m examples.launch_train_dreamerv3 \\
      --task_id 0 --libero_suite libero_90 \\
      --max_steps 500000 --seed 42 --prefix dreamer_baseline
"""
import sys
import argparse
from examples.train_dreamerv3 import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---- Shared with launch_train_sim.py (identical names) ------------------
    parser.add_argument('--seed',                 default=42,                 type=int)
    parser.add_argument('--launch_group_id',      default='',                 type=str)
    parser.add_argument('--env',                  default='libero',           type=str)
    parser.add_argument('--libero_suite',         default='libero_90',        type=str)
    parser.add_argument('--task_id',              default=57,                 type=int)
    parser.add_argument('--log_interval',         default=1000,               type=int)
    parser.add_argument('--eval_interval',        default=5000,               type=int)
    parser.add_argument('--eval_episodes',        default=10,                 type=int)
    parser.add_argument('--batch_size',           default=16,                 type=int)
    parser.add_argument('--max_steps',            default=int(1e6),           type=int)
    parser.add_argument('--add_states',           default=1,                  type=int)
    parser.add_argument('--wandb_project',        default='dreamerv3_libero', type=str)
    parser.add_argument('--start_online_updates', default=1000,               type=int)
    parser.add_argument('--prefix',               default='',                 type=str)
    parser.add_argument('--suffix',               default='',                 type=str)
    parser.add_argument('--multi_grad_step',      default=1,                  type=int,
                        help='Gradient steps per env step (UTD ratio).')
    parser.add_argument('--query_freq',           default=1,                  type=int,
                        help='Env steps between Dreamer action queries.')
    parser.add_argument('--checkpoint_interval',  default=-1,                 type=int)
    parser.add_argument('--replay_capacity',      default=1_000_000,          type=int)
    parser.add_argument('--image_size',           default=64,                 type=int,
                        help='Pixel resolution fed to Dreamer encoder.')

    # ---- Dreamer-v3 architecture hyper-params -------------------------------
    train_args_dict = dict(
        # Dreamer network sizes
        encoder_depth  = 32,
        encoder_embed  = 1024,
        deter          = 1024,   # RSSM deterministic dim
        stoch          = 32,     # RSSM stochastic categories
        classes        = 32,     # classes per stochastic category
        mlp_units      = 512,
        mlp_layers     = 3,
        # Optimisation
        lr             = 3e-4,
        # Algorithm
        discount       = 0.997,
        imag_horizon   = 15,
        lam            = 0.95,
        kl_free        = 1.0,
        batch_length   = 64,     # BPTT sequence length
    )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit(0)
