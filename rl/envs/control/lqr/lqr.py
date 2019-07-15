import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class _Base:

    def __init__(self, dt, goal=np.zeros((2, ))):
        self._dt = dt
        self._g = goal

        self._Q = - np.array([[1e2, 0.0], [0.0, 1e1]])
        self._H = - 1e-2

        self._A = np.array([[0.0, 1.0], [0.0, 0.0]])
        self._B = np.array([[0.0, 1.0]])
        self._c = np.zeros((2, ))

    def dynamics(self, x, u):
        dx = np.einsum('kh,h->k', self._A, x) + \
             np.einsum('kh,h->k', self._B, u) + self._c

        return x + self._dt * dx

    def reward(self, x, u):
        err = x - self._g
        return err.T @ self._Q @ err + self._H * u**2


class LQR(gym.Env):

    def __init__(self):
        self.max_state = np.array([1., 1.])
        self.max_action = np.inf

        self.low_state = - np.array([1., 1.])
        self.high_state = np.array([1., 1.])

        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action, shape=(1,))

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)

        self._dt = 0.1
        self._lqr_model = _Base(self._dt)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        u = np.clip(action, -self.max_action, self.max_action)

        rwrd = self._lqr_model.reward(self.state, u)
        self.state = self._lqr_model.dynamics(self.state, u)

        self.state = np.clip(self.state, -self.max_state, self.max_state)

        return self.state, rwrd.flatten(), False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=np.array([-1.0, -1e-2]),
                                            high=np.array([1.0, 1e-2]))
        return self.state
