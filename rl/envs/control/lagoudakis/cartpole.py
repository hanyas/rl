import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


def normalize(x):
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Cartpole(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 2

        self._dt = 0.1

        self._sigma = 1.e-64 * np.eye(self.dm_state)

        self._state_max = np.array([np.pi, np.inf])

        self._obs_max = np.array([np.pi, np.inf])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_max = 50.
        self.action_space = spaces.Box(low=-self._act_max,
                                       high=self._act_max, shape=(1,))

        self.state = None
        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self._state_max

    @property
    def ulim(self):
        return self._act_max

    @property
    def dt(self):
        return self._dt

    def dynamics(self, x, u):
        g, m, l, M = 9.81, 2.0, 0.5, 8.0
        a = 1.0 / (m + M)

        dx = np.array([x[1],
                       (g * np.sin(x[0]) - 0.5 * a * m * l * x[1]**2 * np.sin(2 * x[0]) -
                          a * np.cos(x[0]) * u) / (4.0 * l / 3.0 - a * m * l * np.cos(x[0])**2)])

        return x + self._dt * dx

    def action(self, u):
        if u == 0:
            return -50.0
        elif u == 1:
            return 50.0
        else:
            return 0.0

    def observe(self, x):
        return x

    def noise(self, x=None, u=None):
        return self._sigma

    def reward(self, x):
        if np.fabs(x[0]) <= 0.5 * np.pi:
            return np.array([0.0])
        else:
            return np.array([-1.0])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        _u = self.action(u)

        done = False
        if np.fabs(self.state[0]) > 0.5 * np.pi:
            done = True

        rwrd = self.reward(self.state)

        self.state = self.dynamics(self.state, _u)
        self.state[0] = normalize(self.state[0])

        return self.observe(self.state), rwrd, done, {}

    def reset(self):
        self.state = np.random.uniform(-0.1, 0.1, size=2)
        return self.observe(self.state)
