import argparse


def mypath(args: argparse, exp_num: int = None):

    if args.train_mode:
        result_path = f'./atari_mind_{args.ssl_option}_{args.momentum}_{args.time_length}_' \
                      f'{args.mask_scale}_{args.mask_ratio}_{args.depth}_results/Experiment_{exp_num}'
    else:
        result_path = f'./mind_test_results/'

    return result_path
