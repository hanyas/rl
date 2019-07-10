#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: pi.py
# @Date: 2019-07-10-03-23
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.envs import Grid
from rl.dp import VI

import matplotlib.pyplot as plt


env = Grid()

vi = VI(env)
vi.run(type='fin', horizon=25, loop=25)

ax = vi.env.world('Environment Finite Horizon')
vi.env.policy(vi.A[0, ...], ax)
plt.show()

vi = VI(env)
vi.run(type='inf', discount=0.9, loop=25)

ax = vi.env.world('Environment Infinite Horizon')
vi.env.policy(vi.A, ax)
plt.show()
