#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-07-15-12-59
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
import matplotlib.pyplot as plt

from rl.gpomdp import GPOMDP

env = gym.make('LQR-RL-v0')
env._max_episode_steps = 100

gpomdp = GPOMDP(env, n_episodes=25, nb_steps=100,
                discount=0.995, alpha=1e-5)

trace = gpomdp.run(nb_iter=15, verbose=True)

rollouts = gpomdp.sample(25, 100, stoch=False)

fig = plt.figure()
for r in rollouts:
    plt.plot(r['x'][:, 0])
plt.show()
