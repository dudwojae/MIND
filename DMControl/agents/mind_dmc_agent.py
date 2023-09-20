from __future__ import division

import os
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.augmentation import (CenterCrop, RandomCrop)
from networks.modules import Actor, Critic, MLPHead, InverseHead, TransformerEncoder
from utils.augmentation import Augmentations
from utils.loss import InverseDynamics
from utils.scheduler import InverseSquareRootSchedule


class MINDDMCAgent:
    def __init__(self,
                 args: argparse,
                 obs_shape,
                 action_shape):

        self.args = args
        self.image_size = obs_shape[-1]

        # Masking Augmentation
        self.augmentations = Augmentations(args=args)

        # Define Actor-Critic (SAC)
        self.actor = Actor(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape
        ).to(device=args.cuda)
        self.critic = Critic(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape
        ).to(device=args.cuda)
        self.target_critic = Critic(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape
        ).to(device=args.cuda)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(device=args.cuda)
        self.log_alpha.requires_grad = True

        # Set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # Define Multi-Task Self-Supervised Learning Common Online Networks
        self.online_transformer = TransformerEncoder(args=args).to(device=args.cuda)
        self.online_projector = MLPHead(args=args).to(device=args.cuda)
        self.online_predictor = MLPHead(args=args).to(device=args.cuda)

        # Momentum CNN & projector for reconstruction
        self.recon_cnn = copy.deepcopy(self.critic.encoder)
        self.recon_projector = MLPHead(args=args).to(device=args.cuda)

        # Momentum CNN & Transformer Encoder & projector for inverse dynamics
        self.dyna_cnn = copy.deepcopy(self.critic.encoder)
        self.dyna_transformer = TransformerEncoder(args=args).to(device=args.cuda)
        self.dyna_projector = MLPHead(args=args).to(device=args.cuda)

        # MLP Head for Inverse Dynamics Modeling
        self.inverse = InverseHead(in_features=args.encoder_feature_dim,
                                   action_space=action_shape[0]).to(device=args.cuda)

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

        # Define Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=args.actor_lr,
                                                betas=(args.actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=args.critic_lr,
                                                 betas=(args.critic_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=args.alpha_lr,
                                                    betas=(args.alpha_beta, 0.999))

        # MIND network optimizers
        self.optim_params = list(self.online_transformer.parameters()) + \
                            list(self.online_projector.parameters()) + \
                            list(self.online_predictor.parameters()) + \
                            list(self.inverse.parameters())
        self.mind_optimizer = torch.optim.Adam(self.optim_params,
                                               lr=args.mind_lr * 0.5)

        if args.warmup:
            lr_scheduler = InverseSquareRootSchedule(args.adam_warmup_step)
            lr_scheduler_lambda = lambda x: lr_scheduler.step(x)
            self.mind_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.mind_optimizer, lr_scheduler_lambda)

        else:
            self.mind_lr_scheduler = None

        self.train()
        self.target_critic.train()
        for param_t in self.target_critic.parameters():
            param_t.requires_grad = False

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):

        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.args.cuda)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(obs,
                                     compute_pi=False,
                                     compute_log_pi=False)

            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = self.augmentations.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.args.cuda)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)

            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.target_critic(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.args.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.args.detach_encoder
        )
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)

        if step % self.args.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.args.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * \
                  (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        if step % self.args.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        if step % self.args.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)

        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_mind(self, mind_kwargs, L, step):
        observation = mind_kwargs['observation']
        action = mind_kwargs['action']
        reward = mind_kwargs['reward']
        next_observation = mind_kwargs['next_observation']

        T, B, C = observation.size()[:3]
        Z = self.args.encoder_feature_dim

        # Online Network
        online_observation = observation.squeeze(-3).flatten(0, 1)
        online_observation = self.augmentations.masking(online_observation)
        masked_observation = self.critic.encoder(online_observation)
        c_masked_observation = masked_observation.view(T, B, Z)
        t_masked_observation = self.online_transformer(c_masked_observation)
        proj_masked_observation = self.online_projector(t_masked_observation)
        pred_masked_observation = F.normalize(self.online_predictor(proj_masked_observation))

        # Target Networks
        target_observation = observation.squeeze(-3).flatten(0, 1)
        target_observation = self.augmentations.center_crop(target_observation)
        target_next_observation = next_observation.squeeze(-3).flatten(0, 1)
        target_next_observation = self.augmentations.center_crop(target_next_observation)

        with torch.no_grad():
            # Reconstruction
            c_original_observation = self.recon_cnn(target_observation)
            c_original_observation = c_original_observation.view(T, B, Z)
            proj_orginal_observation = F.normalize(self.recon_projector(c_original_observation))

            # Inverse Dynamics
            c_original_next_observation = self.dyna_cnn(target_next_observation)
            c_original_next_observation = c_original_next_observation.view(T, B, Z)
            t_original_next_observation = self.dyna_transformer(c_original_next_observation)
            proj_original_next_observation = F.normalize(self.dyna_projector(t_original_next_observation))

        recon_loss = self.recon_loss(pred_masked_observation, proj_orginal_observation)

        pred_actions = self.inverse_loss(online_obs=pred_masked_observation,
                                         target_obs_next=proj_original_next_observation)

        inverse_loss = (nn.MSELoss()(pred_actions, action)).to(device=self.args.cuda)

        mind_loss = recon_loss + inverse_loss

        self.mind_optimizer.zero_grad()
        mind_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.optim_params, 10
        )
        self.mind_optimizer.step()

        if step % self.args.log_interval == 0:
            L.log('train/mind_loss', mind_loss, step)

        if self.mind_lr_scheduler is not None:
            self.mind_lr_scheduler.step()
            L.log('train/mind_lr', self.mind_optimizer.param_groups[0]['lr'], step)

    def update(self, memory, L, step):
        elements = memory.sample_spr()
        obs, action, reward, next_obs, not_done, mind_kwargs = elements

        if step % self.args.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_mind(mind_kwargs, L, step)

        if step % self.args.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.args.critic_target_update_freq == 0:
            self.soft_update_params(self.critic.Q1,
                                    self.target_critic.Q1,
                                    self.args.critic_tau)
            self.soft_update_params(self.critic.Q2,
                                    self.target_critic.Q2,
                                    self.args.critic_tau)
            self.soft_update_params(self.critic.encoder,
                                    self.target_critic.encoder,
                                    self.args.encoder_tau)
            self.soft_update_params(self.critic.encoder,
                                    self.recon_cnn,
                                    self.args.momentum)
            self.soft_update_params(self.critic.encoder,
                                    self.dyna_cnn,
                                    self.args.momentum)
            self.soft_update_params(self.online_transformer,
                                    self.dyna_transformer,
                                    self.args.momentum)
            self.soft_update_params(self.online_projector,
                                    self.recon_projector,
                                    self.args.momentum)
            self.soft_update_params(self.online_projector,
                                    self.dyna_projector,
                                    self.args.momentum)

    @staticmethod
    def soft_update_params(online_net, target_net, tau):

        for param, target_param in zip(online_net.parameters(), target_net.parameters()):
            target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)

    def save(self,
             path: str,
             step: int):

        torch.save(self.actor.state_dict(), '%s/actor_%s.pt' % (path, step))
        torch.save(self.critic.state_dict(), '%s/critic_%s.pt' % (path, step))

    def load(self,
             path: str,
             step: int):

        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (path, step)))
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (path, step)))
