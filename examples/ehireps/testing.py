#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: testing.py
# @Date: 2019-07-17-17-29
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.ehireps import eHiREPS
from rl.envs import Himmelblau


ehireps = eHiREPS(func=Himmelblau(),
                 n_comp=5, n_episodes=2500,
                 kl_bound=0.1)

trace = ehireps.run(nb_iter=10, verbose=True)

