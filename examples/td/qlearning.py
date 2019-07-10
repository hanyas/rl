#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: qlearning.py
# @Date: 2019-07-10-14-46
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.td import QLearning
import gym

env = gym.make('FrozenLake-v0')

qlearning = QLearning(env,
                      discount=0.95,
                      alpha=0.1,
                      pdict={'type': 'softmax', 'beta': 0.98})

qlearning.run(n_samples=10000 * 200)
