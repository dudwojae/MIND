import time

import dmc2gym
import gym
import numpy as np
import torch

from collections import deque
from agents.mind_dmc_agent import MINDDMCAgent
from tasks.mind_dmc_test import evaluate, eval_mode
from utils.recorder import VideoRecorder
from utils.memory import ReplayBuffer
from utils.logger import Logger


class MIND:
    def __init__(self,
                 args,
                 result_path: str):

        self.args = args
        self.result_path = result_path

        self.env = dmc2gym.make(domain_name=args.domain_name,
                                task_name=args.task_name,
                                seed=args.seed,
                                visualize_reward=False,
                                from_pixels=args.encoder_type,
                                height=args.pre_transform_image_size,
                                width=args.pre_transform_image_size,
                                frame_skip=args.action_repeat)
        self.env.seed(args.seed)

        # Stack several consecutive frames together
        if args.encoder_type == 'pixel':
            self.env = FrameStack(self.env, k=args.frame_stack)

        self.video = VideoRecorder(dir_name=None)

        self.action_shape = self.env.action_space.shape

        if args.encoder_type == 'pixel':
            self.obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
            self.pre_aug_obs_shape = (3 * args.frame_stack,
                                      args.pre_transform_image_size,
                                      args.pre_transform_image_size)

        else:
            self.obs_shape = self.env.observation_space.shape
            self.pre_aug_obs_shape = self.obs_shape

        self.learner = MINDDMCAgent(
            args=args,
            obs_shape=self.obs_shape,
            action_shape=self.action_shape
        )

    def run_mind_dmc(self):
        memory = ReplayBuffer(
            obs_shape=self.pre_aug_obs_shape,
            action_shape=self.action_shape,
            capacity=self.args.memory_capacity,
            batch_size=self.args.batch_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            image_size=self.args.image_size,
            auxiliary_task_batch_size=self.args.auxiliary_task_batch_size,
            jumps=self.args.time_length
        )

        L = Logger(self.result_path, use_tb=self.args.save_tb, use_wandb=False)

        episode, episode_reward, done = 0, 0, True
        start_time = time.time()

        for step in range(0, self.args.num_env_steps, self.args.action_repeat):
            # Evaluate agent periodically
            if step % self.args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                evaluate(self.env,
                         self.learner,
                         self.video,
                         self.args.num_eval_episodes,
                         L,
                         step,
                         self.args)

            if done:

                if step > 0:

                    if step % self.args.log_interval == 0:
                        L.log('train/duration', time.time() - start_time, step)
                        L.dump(step)

                    start_time = time.time()

                if step % self.args.log_interval == 0:
                    L.log('train/episode_reward', episode_reward, step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                if step % self.args.log_interval == 0:
                    L.log('train/episode', episode, step)

            # Sample action for data collection
            if step < self.args.init_steps:
                action = self.env.action_space.sample()

            else:
                with eval_mode(self.learner):
                    action = self.learner.sample_action(obs)

            # run training update
            if step >= self.args.init_steps:
                num_updates = 1

                for _ in range(num_updates):
                    self.learner.update(memory, L, step)

            next_obs, reward, done, _ = self.env.step(action)

            done_bool = 0 if episode_step + 1 == self.env._max_episode_steps else float(
                done)
            episode_reward += reward
            memory.add(obs, action, reward, next_obs, done_bool)

            obs = next_obs
            episode_step += 1

            # Save models
            if (self.args.checkpoint_interval != 0) and (step % self.args.checkpoint_interval == 0):
                self.learner.save(self.result_path,
                                  step)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k, ) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
