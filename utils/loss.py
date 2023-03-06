import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.weight_init import weight_init


class InverseDynamics(nn.Module):
    def __init__(self,
                 args: argparse,
                 dynamics: nn.Module):
        super(InverseDynamics, self).__init__()

        self.args = args

        # MLP Head for Inverse Dynamics Modeling
        self.inverse = dynamics

        # Network Initial Weights
        self.apply(weight_init)

    def forward(self,
                online_obs: torch.Tensor,
                target_obs_next: torch.Tensor):

        actions = self.inverse(online_obs, target_obs_next)

        return actions
