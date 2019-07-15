#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-07-15-12-59
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
import matplotlib.pyplot as plt

from rl.reinforce import REINFORCE

env = gym.make('LQR-v0')
env._max_episode_steps = 100

reinforce = REINFORCE(env, n_samples=2500,
                      discount=0.995,
                      alpha=1e-5,
                      pdict={'type': 'poly', 'degree': 1})

trace = reinforce.run(nb_iter=10, verbose=True)

rollouts = reinforce.sample(2500, stoch=False)

fig = plt.figure()
for r in rollouts:
    plt.plot(r['x'][:, 0])
plt.show()
