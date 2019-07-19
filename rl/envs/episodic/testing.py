#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: test.py
# @Date: 2019-07-08-21-43
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import autograd.numpy as np

from sklearn.preprocessing import PolynomialFeatures


class Himmelblau:
    def __init__(self, dm_act=2):
        self.dm_act = dm_act

    def eval(self, x):
        a = x[:, 0] * x[:, 0] + x[:, 1] - 11.0
        b = x[:, 0] + x[:, 1] * x[:, 1] - 7.0
        return -1.0 * (a * a + b * b)


class Sphere:

    def __init__(self, dm_act):
        self.dm_act = dm_act

        M = np.random.randn(dm_act, dm_act)
        tmp = M @ M.T
        Q = tmp[np.nonzero(np.triu(tmp))]

        q = 0.0 * np.random.rand(dm_act)
        q0 = 0.0 * np.random.rand()

        self.param = np.hstack((q0, q, Q))
        self.basis = PolynomialFeatures(degree=2)

    def eval(self, x):
        feat = self.basis.fit_transform(x)
        return - np.dot(feat, self.param)


class Rosenbrock:

    def __init__(self, dm_act):
        self.dm_act = dm_act

    def eval(self, x):
        return - np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0 +
                        (1 - x[:, :-1]) ** 2.0, axis=-1)


class Styblinski:
    def __init__(self, dm_act):
        self.dm_act = dm_act

    def eval(self, x):
        return - 0.5 * np.sum(x**4.0 - 16.0 * x**2 + 5 * x, axis=-1)


class Rastrigin:
    def __init__(self, dm_act):
        self.dm_act = dm_act

    def eval(self, x):
        return - (10.0 * self.dm_act +
                  np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1))
