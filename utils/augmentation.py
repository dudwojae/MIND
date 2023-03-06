import argparse
import numpy as np

import torch
import torch.nn as nn

from collections import deque

import kornia.augmentation as aug


class Augmentations(object):
    def __init__(self, args: argparse):
        super(Augmentations, self).__init__()

        self.args = args

        # Image-based augmentation option
        self.augmentations = {
            'Shift': nn.Sequential(nn.ReplicationPad2d(4),
                                   aug.RandomCrop((args.resize, args.resize))),
            'Intensity': Intensity(scale=0.05)
        }

        self.aug_func_list = []
        for v in self.augmentations.values():
            self.aug_func_list.append(v)

        self.mask_augmentation = aug.RandomErasing(
            scale=(0.02, args.mask_scale),
            ratio=(0.3, args.mask_ratio),
            value=0.0,
            same_on_batch=False,
            p=1.0)

    def select_aug(self):

        return self.aug_func_list

    def masking(self, x: torch.Tensor):
        B, T, C, H, W = x.shape

        masked_x = [self.mask_augmentation(x[:, t]) for t in range(T)]

        return masked_x


class Intensity(nn.Module):
    def __init__(self, scale: float):
        super().__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))

        return x * noise
