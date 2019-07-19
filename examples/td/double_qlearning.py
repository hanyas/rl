#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: double_qlearning.py
# @Date: 2019-07-10-14-49
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


from rl.td import DoubleQLearning
import gym

env = gym.make('FrozenLake-v0')

dqlearning = DoubleQLearning(env, discount=0.95,
                             alpha=0.1, pdict={'type': 'softmax', 'beta': 0.98})

dqlearning.run(nb_samples=10000*200)
