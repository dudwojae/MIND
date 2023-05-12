# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
import atari_py
import argparse


def mind_parser():
    parser = argparse.ArgumentParser(description='MIND')

    # environment option (Don't Change)
    parser.add_argument('--id', type=str, default='default',
                        help='Experiment ID')
    parser.add_argument('--max-episode-length', type=int,
                        default=int(108e3), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--game', type=str, default='ms_pacman',
                        choices=atari_py.list_games(),
                        help='Atari game')
    parser.add_argument('--history-length', type=int,
                        default=4, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--atoms', type=int,
                        default=51, metavar='C',
                        help='Discretized size of value distribution')

    # Rainbow parameter (Don't Change)
    # architecture data_effieicnt -> canonical (Same as SPR)
    parser.add_argument('--architecture', type=str,
                        default='canonical', metavar='ARCH',
                        choices=['canonical', 'data_efficient'],
                        help='Network architecture')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V',
                        help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V',
                        help='Maximum of value distribution support')

    # state option (Don't Change)
    parser.add_argument('--resize', type=int, default=84,
                        help='Resize state information')

    # Noisy network parameter (Don't Change)
    parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE',
                        help='Network hidden size')

    # Replay memory parameter (Don't Change)
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                        help='Prioritized experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=1, metavar='β',
                        help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--memory', help='Path to save/load the memory from')
    parser.add_argument('--disable_bzip_memory', action='store_true',
                        help='Don\'t zip the memory file. '
                             'Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e5),
                        metavar='CAPACITY',
                        help='Experience replay memory capacity')
    # Add Real Time Step Hyperparameter (Can Change)
    parser.add_argument('--time-length', type=int, default=6,
                        choices=[2, 4, 6, 8, 10],
                        help='Consider Real Time Length (History + Sequence Length / 4)')

    # Training hyperparamters (Don't Change)
    # multi_step 20 -> 10 (Same as SPR)
    parser.add_argument('--multi-step', type=int, default=10, metavar='n',
                        help='Number of steps for multi-step return')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Reward discount factor')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE',
                        help='Batch size')
    parser.add_argument('--model', type=str, metavar='PARAMS',
                        help='Pretrained model (state dict)')
    # learn_start 1600 -> 2000 (Same as SPR)
    parser.add_argument('--learn-start', type=int, default=int(2e3), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--T-max', type=int, default=int(1e5), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--replay-frequency', type=int, default=1, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE',
                        help='Reward clipping (0 to disable)')
    # target_update 2000 -> 1 (Same as SPR)
    parser.add_argument('--target-update', type=int, default=1, metavar='τ',
                        help='Number of steps after which to update target network')
    parser.add_argument('--lambda-coef', type=float, default=1.,
                        help='Weighted contrastive loss coefficient')

    # optimizer parameters (Don't Change)
    parser.add_argument('--clip-value', type=float, default=10, metavar='NORM',
                        help='Max L2 norm for gradient clipping')
    parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε',
                        help='Adam epsilon')

    # Evaluate parameter (Don't Change)
    parser.add_argument('--train_mode', type=bool, default=True,
                        help='Train or Test Mode')
    parser.add_argument('--evaluate', type=bool, default=False,
                        help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS',
                        help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
                        help='Number of evaluation episodes to average over')
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
                        help='Number of transitions to use for validating Q')
    parser.add_argument('--render', type=bool, default=False,
                        help='Display screen (testing only)')
    parser.add_argument('--checkpoint-interval', default=5000,
                        help='How often to checkpoint the model, defaults to 0 (never checkpoint)')

    # Transformer parameters
    parser.add_argument('--cnn-embed-dim', type=int, default=128,
                        help='CNN Embedding Dimension.')
    parser.add_argument('--pos-max-length', type=int, default=128,
                        help='Positional Encoding Max Length.')
    parser.add_argument('--depth', type=int, default=2,
                        choices=[1, 2, 4, 6],
                        help='Transformer Depth.')
    parser.add_argument('--num-heads', type=int, default=2,
                        choices=[2],
                        help='Transformer Number of Heads.')
    parser.add_argument('--mlp-ratio', type=float, default=1.0,
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

    # Experiment Option (Inverse Dynamics, Reconstruction)
    parser.add_argument('--ssl-option', type=str,
                        default='recon', metavar='ARCH',
                        choices=['recon', 'inv_dynamics', 'multi_task', 'none'],
                        help='Self-Supervised Learning Method Switch')
    parser.add_argument('--projection-size', type=int, default=128,
                        help='MLP Projection Size.')
    parser.add_argument('--momentum', type=float, default=0.999,
                        help='Momentum rate to BYOL target network update ')

    # cuda and seed
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Ables CUDA training (default: cuda:0)')
    parser.add_argument('--enable-cudnn', action='store_true',
                        help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: auto)')

    return parser
