#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: testing
# @Date: 2019-07-08-21-45
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.more import MORE
from rl.envs import Sphere

more = MORE(func=Sphere(dm_act=2),
            nb_samples=1000,
            kl_bound=0.05,
            ent_rate=0.99,
            cov0=100.0,
            h0=75.0)

more.run(nb_iter=250, verbose=True)
