#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: testing
# @Date: 2019-07-08-21-45
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.cmore import CMORE
from rl.envs import CSphere


cmore = CMORE(func=CSphere(d_cntxt=1, d_action=1),
              n_episodes=1000,
              kl_bound=0.05, ent_rate=0.99,
              cov0=100.0, h0=75.0, cdgr=1)

trace = cmore.run(nb_iter=250, verbose=True)
