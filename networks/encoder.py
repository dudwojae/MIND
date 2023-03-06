from functools import partial

import torch
import torch.nn as nn

from networks.modules import *
from utils.weight_init import weight_init, weight_transformer_init


class SSLEncoder(nn.Module):
    def __init__(self, args: argparse):
        super(SSLEncoder, self).__init__()

        self.transformer = TransformerEncoder(args=args)
        self.projector = MLPHead(args=args)

    def forward(self, x: torch.Tensor):
        x = self.transformer(x)
        x = self.projector(x)

        return x


class DQN(nn.Module):
    def __init__(self,
                 args: argparse,
                 action_space: int):
        super(DQN, self).__init__()

        self.args = args
        self.atoms = args.atoms
        self.action_space = action_space

        if args.architecture == 'canonical':
            self.convs = nn.Sequential(
                nn.Conv2d(in_channels=args.history_length,
                          out_channels=32,
                          kernel_size=8,
                          stride=4,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=4,
                          stride=2,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=3,
                          stride=1,
                          padding=0),
                nn.ReLU()
            )

            self.conv_output_size = 3136  # 64 * 7 * 7

        elif args.architecture == 'data_efficient':
            self.convs = nn.Sequential(
                nn.Conv2d(in_channels=args.history_length,
                          out_channels=32,
                          kernel_size=5,
                          stride=5,
                          padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=5,
                          stride=5,
                          padding=0),
                nn.ReLU()
            )

            self.conv_output_size = 576  # 64 * 3 * 3

        self.fc_h_v = NoisyLinear(self.conv_output_size,
                                  args.hidden_size,
                                  std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size,
                                  args.hidden_size,
                                  std_init=args.noisy_std)

        self.fc_z_v = NoisyLinear(args.hidden_size,
                                  self.atoms,
                                  std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size,
                                  action_space * self.atoms,
                                  std_init=args.noisy_std)

        # Linear layer for dimension reduction
        self.linear = nn.Linear(self.conv_output_size,
                                args.cnn_embed_dim)

        # Network Initial Weights
        self.apply(weight_init)

    def forward(self,
                x: torch.Tensor,
                log: bool = False):

        x = self.convs(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # for RL
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream

        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension

        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension

        # Dimension Reduction
        x = self.linear(x)

        return q, x

    def reset_noise(self):
        for name, module in self.named_children():

            if 'fc' in name:
                module.reset_noise()


class TransformerEncoder(nn.Module):
    def __init__(self, args: argparse):
        super(TransformerEncoder, self).__init__()

        self.args = args
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.position = PositionalEmbedding(d_model=args.cnn_embed_dim,
                                            max_length=args.pos_max_length)
        self.blocks = nn.Sequential(*[
            Block(
                embed_dim=args.cnn_embed_dim,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                qkv_bias=args.qkv_bias,
                norm_layer=norm_layer
            ) for i in range(args.depth)])

        self.apply(weight_transformer_init)

    def forward(self, x: torch.Tensor):
        length = x.shape[1]
        position = self.position(length)
        x = x + position
        x = self.blocks(x)

        return x
