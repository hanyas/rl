#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: test.py
# @Date: 2019-07-08-21-43
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np


class CSphere:

    def __init__(self, d_cntxt, d_action):
        self.d_cntxt = d_cntxt
        self.d_action = d_action

        M = np.random.randn(self.d_action, self.d_action)
        M = 0.5 * (M + M.T)
        self.Q = M @ M.T

    def context(self, n_episodes):
        return np.random.uniform(-1.0, 1.0, size=(n_episodes, self.d_cntxt))

    def eval(self, x, c):
        diff = x - c
        return - np.einsum('nk,kh,nh->n', diff, self.Q, diff)
