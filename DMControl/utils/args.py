import argparse


def mind_parser():
    parser = argparse.ArgumentParser(description='MIND_DMC')

    # environment option
    parser.add_argument('--domain-name', type=bool, default=None)
    parser.add_argument('--task-name', type=bool, default=None)
    parser.add_argument('--env-name', type=str, default='cartpole/swingup',
                        choices=['finger/spin', 'cartpole/swingup', 'reacher/easy',
                                 'cheetah/run', 'walker/walk', 'ball_in_cup/catch'],
                        help='DeepMind Control Suite')
    parser.add_argument('--pre-transform-image-size', type=int, default=100)
    parser.add_argument('--image-size', type=int, default=84)
    parser.add_argument('--action-repeat', type=int, default=8)
    parser.add_argument('--frame-stack', type=int, default=3)

    # Replay memory parameter
    parser.add_argument('--memory-capacity', type=int, default=int(1e5),
                        metavar='CAPACITY',
                        help='Experience replay memory capacity')

    # Train
    parser.add_argument('--init-steps', type=int, default=1000)
    parser.add_argument('--num-env-steps', type=int, default=105000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=1024)

    # Eval
    parser.add_argument('--eval-freq', type=int, default=5000)
    parser.add_argument('--num-eval-episodes', type=int, default=10)

    # Actor-Critic
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--critic-beta', type=float, default=0.9)
    parser.add_argument('--critic-tau', type=float, default=0.99)
    parser.add_argument('--critic-target-update-freq', type=int, default=2)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--actor-beta', type=float, default=0.9)
    parser.add_argument('--actor-log-std-min', type=float, default=-10)
    parser.add_argument('--actor-log-std-max', type=float, default=2)
    parser.add_argument('--actor-update-freq', type=int, default=2)

    # Encoder
    parser.add_argument('--encoder-type', type=str, default='pixel')
    parser.add_argument('--encoder-feature-dim', type=int, default=50)
    parser.add_argument('--encoder-lr', type=float, default=1e-3)
    parser.add_argument('--encoder-tau', type=float, default=0.95)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-filters', type=int, default=64)
    parser.add_argument('--curl-latent-dim', type=int, default=128)

    # SAC
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--init-temperature', type=float, default=0.1)
    parser.add_argument('--alpha-lr', type=float, default=1e-4)
    parser.add_argument('--alpha-beta', type=float, default=0.5)

    # Additional Option
    parser.add_argument('--time-length', type=int, default=5)
    parser.add_argument('--auxiliary-task-batch-size', type=int, default=128)
    parser.add_argument('--warmup', type=bool, default=True)
    parser.add_argument('--adam-warmup-step', type=int, default=6000)
    parser.add_argument('--detach-encoder', type=bool, default=False)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-tb', type=bool, default=True)
    parser.add_argument('--use-wandb', type=bool, default=False)

    # Transformer parameters
    parser.add_argument('--embed-dim', type=int, default=50,
                        help='CNN Embedding Dimension.')
    parser.add_argument('--pos-max-length', type=int, default=128,
                        help='Positional Encoding Max Length.')
    parser.add_argument('--depth', type=int, default=2,
                        choices=[1, 2, 4, 6],
                        help='Transformer Depth.')
    parser.add_argument('--num-heads', type=int, default=1,
                        help='Transformer Number of Heads.')
    parser.add_argument('--mlp-ratio', type=float, default=2.0,
                        help='Transformer MLP Ratio.')
    parser.add_argument('--qkv-bias', type=bool, default=False,
                        help='Transformer QKV Bias.')

    # Data Augmentation Option (Reconstruction)
    parser.add_argument('--mask-scale', type=float, default=0.33,
                        choices=[0.02, 0.15, 0.33, 0.55, 0.7, 1.1, 1.7],
                        help='Mask Augmenation Scale Values.')
    parser.add_argument('--mask-ratio', type=float, default=3.3,
                        choices=[0.3, 1.5, 3.3, 3.5, 7.0, 9.0, 9.9],
                        help='Mask Augmenation Ratio Values.')

    # MIND
    parser.add_argument('--mind-lr', type=float, default=1e-3)

    # Experiment Option (Inverse Dynamics, Reconstruction)
    parser.add_argument('--ssl-option', type=str,
                        default='multi_task', metavar='ARCH',
                        choices=['recon', 'inv_dynamics', 'multi_task', 'none'],
                        help='Self-Supervised Learning Method Switch')
    parser.add_argument('--momentum', type=float, default=0.95,
                        help='Momentum rate to BYOL target network update ')

    # cuda and seed
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Ables CUDA training (default: cuda:0)')
    parser.add_argument('--enable-cudnn', action='store_true',
                        help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: auto)')
    
    # Checkpoint
    parser.add_argument('--checkpoint-interval', default=10000,
                        help='How often to checkpoint the model, defaults to 0 (never checkpoint)')

    return parser
