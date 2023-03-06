import os
import argparse


def mypath(args: argparse, exp_num: int = None):

    if args.train_mode:
        result_path = f'./atari_mtssl_{args.ssl_option}_{args.momentum}_{args.time_length}_' \
                      f'{args.mask_scale}_{args.mask_ratio}_results/Experiment_{exp_num}'
    else:
        result_path = f'./mind_test_results/'

    return result_path
