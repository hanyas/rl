#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: test.py
# @Date: 2019-07-08-21-43
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np


class CSphere:

    def __init__(self, dm_cntxt, dm_act):
        self.dm_cntxt = dm_cntxt
        self.dm_act = dm_act

        M = np.random.randn(self.dm_act, self.dm_act)
        M = 0.5 * (M + M.T)
        self.Q = M @ M.T

    def context(self, nb_episodes):
        return np.random.uniform(-1.0, 1.0, size=(nb_episodes, self.dm_cntxt))

    def eval(self, x, c):
        diff = x - c
        return - np.einsum('nk,kh,nh->n', diff, self.Q, diff)
