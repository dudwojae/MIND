import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from networks.encoder import make_encoder
from utils.weight_init import weight_init, weight_transformer_init


LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class Actor(nn.Module):
    def __init__(self,
                 args,
                 obs_shape,
                 action_shape):
        super().__init__()
        self.args = args

        self.encoder = make_encoder(
            encoder_type=self.args.encoder_type,
            obs_shape=obs_shape,
            feature_dim=self.args.encoder_feature_dim,  # 50
            num_layers=self.args.num_layers,
            num_filters=self.args.num_filters,
            output_logits=True
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, self.args.hidden_dim),  # 50, 1024
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),  # 1024, 1024
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, 2 * action_shape[0])  # 1024,
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self,
                obs,
                compute_pi=True,
                compute_log_pi=True,
                detach_encoder=False):
        obs = self.encoder(
            obs=obs,
            detach=detach_encoder
        )

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # Constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.args.actor_log_std_min + 0.5 * (
            self.args.actor_log_std_max - self.args.actor_log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std

        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(
                noise=noise,
                log_std=log_std
            )

        else:
            log_pi = None

        mu, pi, log_pi = squash(
            mu=mu,
            pi=pi,
            log_pi=log_pi
        )

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):

        if step % log_freq != 0:

            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim,
                      hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                obs,
                action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)

        return self.trunk(obs_action)


class Critic(nn.Module):
    """Employs two Q-functions"""
    def __init__(self,
                 args,
                 obs_shape,
                 action_shape):
        super().__init__()
        self.args = args

        self.encoder = make_encoder(
            encoder_type=self.args.encoder_type,
            obs_shape=obs_shape,
            feature_dim=self.args.encoder_feature_dim,  # 50
            num_layers=self.args.num_layers,
            num_filters=self.args.num_filters,
            output_logits=True
        )

        self.Q1 = QFunction(
            obs_dim=self.encoder.feature_dim,  # 50
            action_dim=action_shape[0],
            hidden_dim=self.args.hidden_dim  # 1024
        )
        self.Q2 = QFunction(
            obs_dim=self.encoder.feature_dim,  # 50
            action_dim=action_shape[0],
            hidden_dim=self.args.hidden_dim  # 1024
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self,
                obs,
                action,
                detach_encoder=False):
        # detach_encoder allows to stop gradient propagation to encoder
        obs = self.encoder(
            obs=obs,
            detach=detach_encoder
        )

        q1 = self.Q1(
            obs=obs,
            action=action
        )

        q2 = self.Q2(
            obs=obs,
            action=action
        )

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):

        if step % log_freq != 0:

            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


# MLP Head for BYOL projector & Predictor
class MLPHead(nn.Module):
    def __init__(self,
                 args):
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(args.encoder_feature_dim,  # 50
                      args.embed_dim * 2,  # 100
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(args.embed_dim * 2,  # 100
                      args.encoder_feature_dim,  # 50
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
            nn.Linear(embed_dim,  # 50
                      mlp_hidden_dim,  # 50 * 2
                      bias=True),
            nn.GELU(),  # FIXME: Change ReLU to GELU
            nn.Linear(mlp_hidden_dim,  # 50 * 2
                      embed_dim,  # 50
                      bias=True))

    def forward(self, x: torch.Tensor):

        # Self-Attention
        residual = x
        x = self.norm1(residual + self.attn(x)[0])

        # Feed Forward
        residual = x
        x = self.norm2(residual + self.mlp(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()

        self.args = args
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.position = PositionalEmbedding(d_model=args.encoder_feature_dim,
                                            max_length=args.pos_max_length)
        self.blocks = nn.Sequential(*[
            Block(
                embed_dim=args.embed_dim,
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
