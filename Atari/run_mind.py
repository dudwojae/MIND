from __future__ import division

import os
import json
import argparse
import numpy as np

import torch

from utils.args import mind_parser
from tasks.mind import MIND
from utils.mypath import mypath

from itertools import product

import warnings

warnings.filterwarnings(action='ignore')


def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def main(args: argparse, game_name: str = None, exp_num: int = None):
    # Define Save Path
    result_path = mypath(args, exp_num=exp_num)

    args.game = game_name
    xid = f'{args.game}-{str(args.seed)}'
    args.id = xid

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    results_dir = os.path.join(result_path, args.id)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save parameter argument dictionary
    with open(os.path.join(results_dir, 'arg_parser.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = args.enable_cudnn

    mind = MIND(args, results_dir)
    mind.run_mind()


if __name__ == '__main__':
    parser = mind_parser()
    args = parser.parse_args()

    game_list = ['alien', 'amidar', 'assault', 'asterix', 'bank_heist',
                 'battle_zone', 'boxing', 'breakout', 'chopper_command',
                 'crazy_climber', 'demon_attack', 'freeway', 'frostbite',
                 'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull',
                 'kung_fu_master', 'ms_pacman', 'pong', 'private_eye',
                 'qbert', 'road_runner', 'seaquest', 'up_n_down']
    
    experiments = list(np.random.randint(10000, size=10))

    for i, exp_seed in enumerate(experiments):
        args.seed = int(exp_seed)

        # Default Parameters
        args.ssl_option = 'multi_task'
        args.time_length = 6
        args.mask_scale = 0.33
        args.mask_ratio = 3.3
        args.depth = 2
        args.momentum = 0.999

        for game_name in game_list:
            main(args, game_name, exp_num=int(i + 1))
