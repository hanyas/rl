#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: cartpole
# @Date: 2019-05-27-22-23
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class CartPole(gym.Env):

    def __init__(self):

        self.g = 9.81
        self.m = 2.0
        self.l = 0.5
        self.M = 8.0

        self.dt = 0.1

        self.max_torque = 50.0

        high = np.array([np.pi, np.inf])

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        if u == 0:
            u = -50.0
        elif u == 1:
            u = 50.0
        else:
            u = 0.0

        done = False
        if np.fabs(self.state[0]) > 0.5 * np.pi:
            done = True

        rwrd = self._reward()

        xn = self.state + self.dt * self._dynamics(self.state, u)
        xn[0] = angle_normalize(xn[0])

        self.state = xn

        return self._get_obs(), rwrd, done, {}

    def _reward(self):
        if np.fabs(self.state[0]) <= 0.5 * np.pi:
            return np.array([0.0])
        else:
            return np.array([-1.0])

    def _dynamics(self, x, u):
        a = 1.0 / (self.m + self.M)
        return np.array([x[1],
                         (self.g * np.sin(x[0]) -
                          0.5 * a * self.m * self.l * x[1] **2 * np.sin(2 * x[0]) -
                          a * np.cos(x[0]) * u) /
                         (4.0 * self.l / 3.0 - a * self.m * self.l * np.cos(x[0]) ** 2)])

    def reset(self):
        self.state = np.random.uniform(-0.1, 0.1, size=2)
        return self._get_obs()

    def _get_obs(self):
        return self.state

def angle_normalize(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi
