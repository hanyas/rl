#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: pendulum.py
# @Date: 2019-07-15-15-37
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np
import gym

from rl.reps import REPS

env = gym.make('Pendulum-RL-v0')
env._max_episode_steps = 25000

reps = REPS(env=env, nb_samples=15000, nb_keep=0,
            nb_rollouts=25, nb_steps=1250,
            kl_bound=0.2, discount=0.998,
            vreg=1e-16, preg=1e-16, cov0=16.0,
            nb_vfeat=75, nb_pfeat=75,
            band=np.array([0.5, 0.5, 4.0]),
            mult=1.0)

reps.run(nb_iter=3, verbose=True)

# # save data
# reps.ctl.cov = 0.1 * np.eye(reps.dm_act)
# rollouts, data = reps.evaluate(nb_rollouts=25, nb_steps=100, stoch=True)
# np.savez('reps_pendulum_rollouts', rollouts)
