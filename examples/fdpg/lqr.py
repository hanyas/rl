#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr
# @Date: 2019-07-15-14-43
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
import matplotlib.pyplot as plt

from rl.fdpg import FDPG

env = gym.make('LQR-v0')
env._max_episode_steps = 100

fdpg = FDPG(env, n_episodes=100,
            discount=0.995, alpha=1e-4,
            pdict={'type': 'poly', 'degree': 1, 'cov0': 0.01})

trace = fdpg.run(nb_iter=10, verbose=True)

rollouts = fdpg.sample(25)

fig = plt.figure()
for r in rollouts:
    plt.plot(r['x'][:, 0])
plt.show()
