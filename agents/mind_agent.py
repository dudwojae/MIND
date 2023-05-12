# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from __future__ import division

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.modules import MLPHead, InverseHead
from networks.encoder import DQN, TransformerEncoder
from utils.augmentation import Augmentations
from utils.loss import InverseDynamics


class MINDAgent:
    def __init__(self,
                 args: argparse,
                 env,
                 result_path: str):

        self.args = args
        self.result_path = result_path
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max

        # Support (range) of z
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.cuda)
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.coeff = args.lambda_coef if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.

        # Masking Augmentation
        self.augmentations = Augmentations(args=args)

        # Define Model (Default: Off-Policy Reinforcement Learning)
        self.online_net = DQN(args, self.action_space).to(device=args.cuda)
        self.target_net = DQN(args, self.action_space).to(device=args.cuda)

        # Load Pre-trained Model If Provided
        if args.model:

            if os.path.isfile(args.model):
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                state_dict = torch.load(args.model, map_location='cpu')

                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'),
                                             ('conv1.bias', 'convs.0.bias'),
                                             ('conv2.weight', 'convs.2.weight'),
                                             ('conv2.bias', 'convs.2.bias'),
                                             ('conv3.weight', 'convs.4.weight'),
                                             ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict

                self.online_net.load_state_dict(state_dict)
                print(f"Loading pretrained model: {args.model}")

            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        # Define Multi-Task Self-Supervised Learning Common Online Networks
        self.online_transformer = TransformerEncoder(args=args).to(device=args.cuda)
        self.online_projector = MLPHead(args=args).to(device=args.cuda)
        self.online_predictor = MLPHead(args=args).to(device=args.cuda)

        # Momentum CNN Encoder & projector for reconstruction
        self.recon_cnn = DQN(args, self.action_space).to(device=args.cuda)
        self.recon_projector = MLPHead(args=args).to(device=args.cuda)

        # Momentum CNN Encoder & Transformer Encoder & projector for inverse dynamics
        self.dyna_cnn = DQN(args, self.action_space).to(device=args.cuda)
        self.dyna_transformer = TransformerEncoder(args=args).to(device=args.cuda)
        self.dyna_projector = MLPHead(args=args).to(device=args.cuda)

        if args.ssl_option == 'recon':
            # Momentum Network Initialize (CNN & Transformer)
            self.initialize_momentum_net(online_net=self.online_net,
                                         momentum_net=self.recon_cnn)
            self.initialize_momentum_net(online_net=self.online_projector,
                                         momentum_net=self.recon_projector)

            # Define Masked Reconstruction Loss
            self.recon_loss = nn.MSELoss().to(device=args.cuda)

            # Train Mode
            self.online_transformer.train()
            self.online_projector.train()
            self.online_predictor.train()
            self.recon_cnn.train()
            self.recon_projector.train()

            # Define Optimizer
            self.optim_params = list(self.online_net.parameters()) + \
                                list(self.online_transformer.parameters()) + \
                                list(self.online_projector.parameters()) + \
                                list(self.online_predictor.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate,
                                              eps=args.adam_eps)

        elif args.ssl_option == 'inv_dynamics':
            # Momentum Network Initialize (CNN & Transformer & Projector)
            self.initialize_momentum_net(online_net=self.online_net,
                                         momentum_net=self.dyna_cnn)
            self.initialize_momentum_net(online_net=self.online_transformer,
                                         momentum_net=self.dyna_transformer)
            self.initialize_momentum_net(online_net=self.online_projector,
                                         momentum_net=self.dyna_projector)

            # MLP Head for Inverse Dynamics Modeling
            self.inverse = InverseHead(in_features=args.projection_size,
                                       action_space=self.action_space).to(device=args.cuda)

            # Define Inverse Dynamics Loss
            self.inverse_loss = InverseDynamics(args=args,
                                                dynamics=self.inverse)

            # Train Mode
            self.online_transformer.train()
            self.online_projector.train()
            self.online_predictor.train()
            self.dyna_cnn.train()
            self.dyna_transformer.train()
            self.dyna_projector.train()
            self.inverse.train()

            # Define Optimizer
            self.optim_params = list(self.online_net.parameters()) + \
                                list(self.online_transformer.parameters()) + \
                                list(self.online_projector.parameters()) + \
                                list(self.online_predictor.parameters()) + \
                                list(self.inverse.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate,
                                              eps=args.adam_eps)

        elif args.ssl_option == 'multi_task':
            # Reconstruction Momentum Network Initialize (CNN & Projector)
            self.initialize_momentum_net(online_net=self.online_net,
                                         momentum_net=self.recon_cnn)
            self.initialize_momentum_net(online_net=self.online_projector,
                                         momentum_net=self.recon_projector)

            # Momentum Network Initialize (CNN & Transformer & Projector)
            self.initialize_momentum_net(online_net=self.online_net,
                                         momentum_net=self.dyna_cnn)
            self.initialize_momentum_net(online_net=self.online_transformer,
                                         momentum_net=self.dyna_transformer)
            self.initialize_momentum_net(online_net=self.online_projector,
                                         momentum_net=self.dyna_projector)

            # MLP Head for Inverse Dynamics Modeling
            self.inverse = InverseHead(in_features=args.projection_size,
                                       action_space=self.action_space).to(device=args.cuda)

            # Define Masked Reconstruction Loss
            self.recon_loss = nn.MSELoss().to(device=args.cuda)

            # Define Inverse Dynamics Loss
            self.inverse_loss = InverseDynamics(args=args,
                                                dynamics=self.inverse)

            # Train Mode
            self.online_transformer.train()
            self.online_projector.train()
            self.online_predictor.train()
            self.recon_cnn.train()
            self.recon_projector.train()
            self.dyna_cnn.train()
            self.dyna_transformer.train()
            self.dyna_projector.train()
            self.inverse.train()

            # Define Optimizer
            self.optim_params = list(self.online_net.parameters()) + \
                                list(self.online_transformer.parameters()) + \
                                list(self.online_projector.parameters()) + \
                                list(self.online_predictor.parameters()) + \
                                list(self.inverse.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate,
                                              eps=args.adam_eps)

        else:
            self.optim_params = list(self.online_net.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate,
                                              eps=args.adam_eps)

        # RL Online & Target Network (Default: Off-Policy Reinforcement Learning)
        self.online_net.train()
        self.update_target_net()
        self.target_net.train()
        for param_t in self.target_net.parameters():
            param_t.requires_grad = False

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state: torch.Tensor):
        with torch.no_grad():
            self.online_net.eval()
            a, _ = self.online_net(state.unsqueeze(0))

        return (a * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_epsilon_greedy(self,
                           state: torch.Tensor,
                           epsilon: float = 0.001):

        # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) \
            if np.random.random() < epsilon else self.act(state)

    @staticmethod
    # For Inverse Dynamics Modeling (Same as BYOL)
    def initialize_momentum_net(online_net, momentum_net):
        for param_q, param_k in zip(online_net.parameters(), momentum_net.parameters()):
            param_k.data.copy_(param_q.data)  # update
            param_k.requires_grad = False  # not update by gradient

    # For RL target network
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.no_grad()
    def update_momentum_net(self):
        """
        Exponential Moving Average Update (Same as MoCo Momentum Update)
        """
        momentum = self.args.momentum
        if self.args.ssl_option == 'recon':
            for param_q, param_k in zip(self.online_net.parameters(), self.recon_cnn.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

            for param_q, param_k in zip(self.online_projector.parameters(), self.recon_projector.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

        elif self.args.ssl_option == 'inv_dynamics':
            for param_q, param_k in zip(self.online_net.parameters(), self.dyna_cnn.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

            for param_q, param_k in zip(self.online_transformer.parameters(), self.dyna_transformer.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

            for param_q, param_k in zip(self.online_projector.parameters(), self.dyna_projector.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

        elif self.args.ssl_option == 'multi_task':
            for param_q, param_k in zip(self.online_net.parameters(), self.recon_cnn.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

            for param_q, param_k in zip(self.online_net.parameters(), self.dyna_cnn.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

            for param_q, param_k in zip(self.online_transformer.parameters(), self.dyna_transformer.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

            for param_q, param_k in zip(self.online_projector.parameters(), self.recon_projector.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

            for param_q, param_k in zip(self.online_projector.parameters(), self.dyna_projector.parameters()):
                param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

    def optimize(self,
                 memory,
                 timesteps: int):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights, \
        sequential_states, sequential_next_states, sequential_actions = memory.sample(self.args.batch_size)

        # Multi-Task Self-Supervised Learning Online Network Flow
        masked_sequential_states = self.augmentations.masking(sequential_states)

        c_masked_states = [self.online_net(masked_state, log=True)[1]
                           for masked_state in masked_sequential_states]
        c_masked_states = torch.stack(c_masked_states, dim=1)
        t_masked_states = self.online_transformer(c_masked_states)
        proj_masked_states = self.online_projector(t_masked_states)
        pred_masked_states = F.normalize(self.online_predictor(proj_masked_states), dim=2)

        if self.args.ssl_option == 'recon':

            # Reconstruction Momentum Network Flow
            with torch.no_grad():
                c_original_states = [self.recon_cnn(sequential_states[:, t], log=True)[1]
                                     for t in range(self.args.time_length)]
                c_original_states = torch.stack(c_original_states, dim=1)
                proj_original_states = F.normalize(self.recon_projector(c_original_states), dim=2)

            recon_loss = self.recon_loss(pred_masked_states, proj_original_states)

        elif self.args.ssl_option == 'inv_dynamics':

            # Inverse Dynamics Momentum Network Flow
            with torch.no_grad():
                c_original_next_states = [self.dyna_cnn(sequential_next_states[:, t], log=True)[1]
                                          for t in range(self.args.time_length)]
                c_original_next_states = torch.stack(c_original_next_states, dim=1)
                t_original_next_states = self.dyna_transformer(c_original_next_states)
                proj_original_next_states = F.normalize(self.dyna_projector(t_original_next_states), dim=2)

            pred_actions = self.inverse_loss(online_obs=pred_masked_states,
                                             target_obs_next=proj_original_next_states)

            # Reshape Output & Target
            pred_actions = pred_actions.contiguous().view(-1, self.action_space)
            sequential_actions = sequential_actions.contiguous().view(-1)

            inverse_loss = (nn.CrossEntropyLoss(ignore_index=-1)
                            (pred_actions, sequential_actions)).to(device=self.args.cuda)

        elif self.args.ssl_option == 'multi_task':

            # Multi-Task Self-Supervised Learning Flow
            with torch.no_grad():
                # Reconstruction
                c_original_states = [self.recon_cnn(sequential_states[:, t], log=True)[1]
                                     for t in range(self.args.time_length)]
                c_original_states = torch.stack(c_original_states, dim=1)
                proj_original_states = F.normalize(self.recon_projector(c_original_states), dim=2)

                # Inverse Dynamics
                c_original_next_states = [self.dyna_cnn(sequential_next_states[:, t], log=True)[1]
                                          for t in range(self.args.time_length)]
                c_original_next_states = torch.stack(c_original_next_states, dim=1)
                t_original_next_states = self.dyna_transformer(c_original_next_states)
                proj_original_next_states = F.normalize(self.dyna_projector(t_original_next_states), dim=2)

            recon_loss = self.recon_loss(pred_masked_states, proj_original_states)

            pred_actions = self.inverse_loss(online_obs=pred_masked_states,
                                             target_obs_next=proj_original_next_states)

            # Reshape Output & Target
            pred_actions = pred_actions.contiguous().view(-1, self.action_space)
            sequential_actions = sequential_actions.contiguous().view(-1)

            inverse_loss = (nn.CrossEntropyLoss(ignore_index=-1)
                            (pred_actions, sequential_actions)).to(device=self.args.cuda)

        # RL update
        log_ps, _ = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.args.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate n-th next state probabilities
            pns, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))

            # Perform argmax action selection using online network argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)

            # Sample new target net noise
            self.target_net.reset_noise()

            pns, _ = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)

            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.args.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(1) + nonterminals * \
                 (self.args.gamma ** self.args.multi_step) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values

            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.args.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.args.batch_size - 1) * self.atoms),
                                    self.args.batch_size).unsqueeze(1).expand(
                self.args.batch_size, self.atoms).to(actions)

            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

        rl_loss = -torch.sum(m * log_ps_a, 1)

        if self.args.ssl_option == 'recon':
            loss = rl_loss + (self.coeff * recon_loss)

        elif self.args.ssl_option == 'inv_dynamics':
            loss = rl_loss + (self.coeff * inverse_loss)

        elif self.args.ssl_option == 'multi_task':
            loss = rl_loss + (self.coeff * (recon_loss + inverse_loss))

        else:
            loss = rl_loss

        self.optimizer.zero_grad()
        total_loss = (weights * loss).mean()
        total_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        # Clip gradients by L2 norm
        torch.nn.utils.clip_grad_norm_(self.optim_params, self.args.clip_value)
        self.optimizer.step()

        # Save SSL Loss with RL Loss & Save number of action labels
        if timesteps % 500 == 0:
            if self.args.ssl_option == 'recon':
                self.log_loss_action(f'| Reinforcement Learning = {rl_loss.mean().item()}'
                                     f'| Masked Reconstruction = {recon_loss.item()}')

            elif self.args.ssl_option == 'inv_dynamics':
                self.log_loss_action(f'| Reinforcement Learning = {rl_loss.mean().item()}'
                                     f'| Inverse Dynamics = {inverse_loss.item()}'
                                     f'| Batch of Action Labels = {actions}'
                                     f'| Batch of Sequential Target Action Labels = {sequential_actions}'
                                     f'| Batch of Sequential Pred Action Labels = {torch.argmax(pred_actions, dim=1)}')

            elif self.args.ssl_option == 'multi_task':
                self.log_loss_action(f'| Reinforcement Learning = {rl_loss.mean().item()}'
                                     f'| Masked Reconstruction = {recon_loss.item()}'
                                     f'| Inverse Dynamics = {inverse_loss.item()}'
                                     f'| Batch of Action Labels = {actions}'
                                     f'| Batch of Sequential Target Action Labels = {sequential_actions}'
                                     f'| Batch of Sequential Pred Action Labels = {torch.argmax(pred_actions, dim=1)}')

            else:
                self.log_loss_action(f'| Reinforcement Learning = {rl_loss.mean().item()}'
                                     f'| Batch of Action Labels = {actions}')

        # Update priorities of sampled transitions
        memory.update_priorities(idxs, loss.detach().cpu().numpy())

    # Save model parameters on current device (don't move model between devices)
    def save(self,
             path: str,
             name: str = 'rainbow.pt'):

        torch.save(self.online_net.state_dict(), os.path.join(path, name))
        torch.save(self.online_transformer.state_dict(), os.path.join(path, f'transformer_{name}'))

    def log_loss_action(self,
                        s: str,
                        name='loss_and_action.txt'):

        filename = os.path.join(self.result_path, name)

        if not os.path.exists(filename) or s is None:
            f = open(filename, 'w')

        else:
            f = open(filename, 'a')

        f.write(str(s) + '\n')
        f.close()

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        a, _ = self.online_net(state.unsqueeze(0))

        return (a * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    @staticmethod
    def count_parameters(net: nn.Module,
                         as_int: bool = False):
        """
        Returns number of params in network.
        """

        count = sum(p.numel() for p in net.parameters())
        if as_int:
            return count

        return f'{count:,}'
