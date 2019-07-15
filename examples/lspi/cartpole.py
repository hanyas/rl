#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: cartpole.py
# @Date: 2019-07-15-14-36
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
from rl.lspi import LSPI

# env = gym.make('Lagoudakis-v0')
env = gym.make('CartPole-v0')
env._max_episode_steps = 100

lspi = LSPI(env=env, n_samples=10000, n_actions=2,
            discount=0.95, lmbda=.25,
            alpha=1e-12, beta=1e-8,
            qdict={'type': 'poly',
                   'degree': 2},
            pdict={'type': 'greedy',
                   'eps': 1.0},
            )

lspi.run(delta=1e-3)
