#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: lqr.py
# @Date: 2019-07-15-12-59
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
import matplotlib.pyplot as plt

from rl.pgpe import PGPE

env = gym.make('LQR-v0')
env._max_episode_steps = 100

pgpe = PGPE(env, n_episodes=100,
            discount=0.995,
            alpha=1e-5, beta=1e-8,
            pdict={'type': 'poly', 'degree': 1, 'cov0': 0.1})

trace = pgpe.run(nb_iter=10, verbose=True)

rollouts = pgpe.sample(25)

fig = plt.figure()
for r in rollouts:
    plt.plot(r['x'][:, 0])
plt.show()
