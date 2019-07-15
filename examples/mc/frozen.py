#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: frozen.py
# @Date: 2019-07-15-13-47
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de


import gym

from rl.mc import MC

env = gym.make('FrozenLake-v0')

mc = MC(env,
        n_episodes=1000,
        discount=0.99)

mc.eval()
