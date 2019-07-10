#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: pi.py
# @Date: 2019-07-10-03-23
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.envs import Grid
from rl.dp import PI

import matplotlib.pyplot as plt

env = Grid()

pi = PI(env)
pi.run(type='fin', horizon=25, loop=25)

ax = pi.env.world('Environment Finite Horizon')
pi.env.policy(pi.A[0, ...], ax)
plt.show()

pi = PI(env)
pi.run(type='inf', discount=0.9, loop=25)

ax = pi.env.world('Environment Infinite Horizon')
pi.env.policy(pi.A, ax)
plt.show()
