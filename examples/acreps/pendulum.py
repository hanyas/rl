#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: pendulum.py
# @Date: 2019-07-15-15-17
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
import gym

from rl.acreps import ACREPS

env = gym.make('Pendulum-v0')
env._max_episode_steps = 250

acreps = ACREPS(env=env, n_samples=3000, n_keep=0, n_rollouts=25,
                kl_bound=0.1, discount=0.98, lmbda=0.95,
                vreg=1e-12, preg=1e-12, cov0=16.0,
                n_vfeat=75, n_pfeat=75,
                s_band=np.array([0.5, 0.5, 4.0]),
                sa_band=np.array([0.5, 0.5, 4.0, 1.0]),
                mult=1.0)

acreps.run(verbose=True)
