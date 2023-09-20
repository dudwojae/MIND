import argparse


def mypath(args: argparse, exp_num: int = None):
    result_path = f'./dmc_mind_{args.ssl_option}_{args.momentum}_{args.time_length + 1}_' \
                  f'{args.mask_scale}_{args.mask_ratio}_{args.depth}_results/Experiment_{exp_num}'

    return result_path
