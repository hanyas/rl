#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: pendulum.py
# @Date: 2019-07-15-15-37
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
import gym

from rl.reps import REPS

env = gym.make('Pendulum-v0')
env._max_episode_steps = 5000

reps = REPS(env=env, n_samples=3000, n_keep=0,
            n_rollouts=25, n_steps=250,
            kl_bound=0.1, discount=0.99,
            vreg=1e-12, preg=1e-12, cov0=16.0,
            n_vfeat=75, n_pfeat=75,
            band=np.array([0.5, 0.5, 4.0]),
            mult=1.0)

reps.run(nb_iter=10, verbose=True)

# # save data
# reps.ctl.cov = 0.1 * np.eye(reps.dim_action)
# rollouts, data = reps.evaluate(n_rollouts=25, n_steps=100, stoch=True)
# np.savez('reps_pendulum_rollouts', rollouts)
