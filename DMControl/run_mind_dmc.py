from __future__ import division

import os
import json
import argparse
import numpy as np

import torch

from utils.args import mind_parser
from tasks.mind_dmc import MIND
from utils.mypath import mypath

from itertools import product

import warnings

warnings.filterwarnings(action='ignore')


def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def main(args: argparse, exp_num: int = None):
    result_path = mypath(args, exp_num)

    env_id = f'{args.domain_name}-{args.task_name}-{str(args.seed)}'

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    results_dir = os.path.join(result_path, env_id)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save parameter argument dictionary
    with open(os.path.join(results_dir, 'arg_parser.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn

    dmcmind = MIND(args, results_dir)
    dmcmind.run_mind_dmc()


if __name__ == '__main__':
    parser = mind_parser()
    args = parser.parse_args()

    env_list = ['finger/spin', 'cartpole/swingup', 'reacher/easy',
                'cheetah/run', 'walker/walk', 'ball_in_cup/catch']

    experiments = list(np.random.randint(10000, size=10))

    for i, exp_seed in enumerate(experiments):
        args.seed = int(exp_seed)

        for env_name in env_list:
            args.domain_name = env_name.split('/')[0]
            args.task_name = env_name.split('/')[1]

            if env_name == 'finger/spin' or 'walker/walk':
                args.action_repeat = 2

            elif env_name == 'cartpole/swingup':
                args.action_repeat = 8

            else:
                args.action_repeat = 4

            if env_name == 'cheetah/run':
                args.critic_lr = 2e-4
                args.actor_lr = 2e-4
                args.mind_lr = 2e-4

            else:
                args.critic_lr = 1e-3
                args.actor_lr = 1e-3
                args.mind_lr = 1e-3

            if env_name == 'walker/walk':
                args.momentum = 0.9

            else:
                args.momentum = 0.95
                
            args.init_steps = 1000
            args.log_interval = 100
            args.actor_update_freq = 2
            args.critic_target_update_freq = 2
            args.init_steps *= args.action_repeat
            args.log_interval *= args.action_repeat
            args.actor_update_freq *= args.action_repeat
            args.critic_target_update_freq *= args.action_repeat

            main(args, exp_num=int(i + 1))
