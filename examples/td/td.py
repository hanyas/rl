#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: td.py
# @Date: 2019-07-10-14-48
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.td import TD
import gym
from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')

td = TD(env, discount=0.95, alpha=0.25)
td.eval(nb_samples=10000)

print(td.vfunc)

plt.plot(td.td_error)
plt.show()
