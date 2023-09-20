import argparse
import torch
import kornia.augmentation as aug


class Augmentations(object):
    def __init__(self, args: argparse):
        super(Augmentations, self).__init__()

        self.args = args
        self.mask_augmentation = aug.RandomErasing(
            scale=(0.02, args.mask_scale),
            ratio=(0.3, args.mask_ratio),
            value=0.0,
            same_on_batch=False,
            p=1.0)

    def masking(self, x: torch.Tensor):
        B, T, C, H, W = x.shape

        masked_x = [self.mask_augmentation(x[:, t]) for t in range(T)]

        return masked_x
