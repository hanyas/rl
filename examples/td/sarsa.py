#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: sarsa.py
# @Date: 2019-07-10-14-47
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.td import SARSA
import gym

env = gym.make('FrozenLake-v0')

sarsa = SARSA(env, discount=0.95,
              lmbda=0.25, alpha=0.1,
              pdict={'type': 'softmax', 'beta': 0.98})

sarsa.run(n_samples=10000 * 200)
