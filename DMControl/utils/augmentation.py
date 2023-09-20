import argparse
import numpy as np
import torch
import torch.nn as nn
import kornia.augmentation as aug


class Augmentations(object):
    def __init__(self, args: argparse):
        super(Augmentations, self).__init__()

        self.args = args
        self.center_crop = aug.CenterCrop((84, 84))
        self.mask_augmentation = nn.Sequential(aug.RandomCrop((84, 84)),
                                               aug.RandomErasing(
            scale=(0.02, args.mask_scale),
            ratio=(0.3, args.mask_ratio),
            value=0.0,
            same_on_batch=False,
            p=1.0))

        self.random_crop = aug.RandomCrop((84, 84))



    def masking(self, x: torch.Tensor):
        TB, C, H, W = x.shape

        masked_x = self.mask_augmentation(x)

        # masked_x = [self.mask_augmentation(x[:, t]) for t in range(T)]

        return masked_x

    def centercrop(self, x: torch.Tensor):
        TB, C, H, W = x.shape

        crop_x = self.center_crop(x)

        return crop_x

    def center_crop_image(self, image, output_size):
        h, w = image.shape[1:]
        new_h, new_w = output_size, output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[:, top:top + new_h, left:left + new_w]
        return image
