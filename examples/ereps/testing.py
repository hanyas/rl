#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: testing
# @Date: 2019-07-08-21-45
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.ereps import eREPS
from rl.envs import Sphere


ereps = eREPS(func=Sphere(d_action=5),
              n_episodes=10,
              kl_bound=0.1,
              cov0=10.0)

ereps.run(nb_iter=250, verbose=True)
