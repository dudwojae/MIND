import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# MLP Head for BYOL projector & Predictor
class MLPHead(nn.Module):
    def __init__(self,
                 args: argparse):
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(args.cnn_embed_dim,
                      args.cnn_embed_dim,
                      bias=True),
            # nn.BatchNorm1d(in_features),  # FIXME
            nn.ReLU(inplace=True),
            nn.Linear(args.cnn_embed_dim,
                      args.projection_size,
                      bias=True))

    def forward(self, x: torch.Tensor):

        return self.mlp(x)


# MLP Head for Inverse Dynamics Modeling
class InverseHead(nn.Module):
    def __init__(self,
                 in_features: int,
                 action_space: int):
        super(InverseHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_features,
                      in_features,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features,
                      action_space,
                      bias=True))

    def forward(self,
                x: torch.Tensor,
                x_next: torch.Tensor):
        joint_x = torch.cat([x, x_next], dim=2)
        actions = self.mlp(joint_x)

        return actions


# Factorized NoisyLinear with bias
class NoisyLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 std_init: float = 0.5):
        super(NoisyLinear, self).__init__()

        self.module_name = 'noisy_linear'
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int):
        x = torch.randn(size)

        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input: torch.Tensor):
        if self.training:
            return F.linear(input,
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)

        else:
            return F.linear(input,
                            self.weight_mu,
                            self.bias_mu)


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 d_model: int = None,
                 max_length: int = None):
        super(PositionalEmbedding, self).__init__()

        """
        Generate positional encoding as described in original paper.
        Parameters
        ----------
        d_model:
            Dimension of the model vector.
        
        max_length:
            Time window length.
        
        Returns
        -------
            Tensor of shape (max_length, d_model)
        """

        pe = torch.zeros(max_length, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):

        return self.pe[:, :length]


class Attention(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 num_heads: int = 2,
                 qkv_bias: bool = False,
                 qk_scale: int = None):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = embed_dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim,
                             embed_dim * 3,
                             bias=qkv_bias)
        self.proj = nn.Linear(embed_dim,
                              embed_dim)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        # (3, batch size, num head, query length, head dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scaled dot product (batch size, num head, query length, head dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # (batch size, num head, query length, key length)
        # --> (batch size, num head, query length, head dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, attn


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 num_heads: int = 2,
                 mlp_ratio: float = 1.,
                 qkv_bias: bool = False,
                 qk_scale: int = None,
                 norm_layer: nn.Module = nn.LayerNorm):
        super(Block, self).__init__()

        self.norm1 = norm_layer(embed_dim)

        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale)

        self.norm2 = norm_layer(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,
                      mlp_hidden_dim,
                      bias=True),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim,
                      embed_dim,
                      bias=True))

    def forward(self, x: torch.Tensor):

        # Self-Attention
        residual = x
        x = self.norm1(residual + self.attn(x)[0])

        # Feed Forward
        residual = x
        x = self.norm2(residual + self.mlp(x))

        return x
