import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


def normalize(x):
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Pendulum(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 3

        self._dt = 0.01

        # damping
        self._k = 0.  # 1.e-3

        self._sigma = 1.e-64 * np.eye(self.dm_state)

        # g = [th, thd]
        self._goal = np.array([0., 0.])
        self._goal_weight = - np.array([1.e0, 1.e-1])

        # x = [th, thd]
        self._state_max = np.array([np.inf, 8.0])

        # o = [cos, sin, thd]
        self._obs_max = np.array([1.0, 1.0, 8.0])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_weight = - np.array([1.e-3])
        self._act_max = 2.
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

    @property
    def goal(self):
        return self._goal

    def dynamics(self, x, u):
        g, m, l = 9.80665, 1., 1.

        def f(x, u):
            th, dth = x
            return np.hstack((dth, 3. * g / (2. * l) * np.sin(th) +
                              3. / (m * l ** 2) * (u - self._k * dth)))

        k1 = f(x, u)
        k2 = f(x + 0.5 * self.dt * k1, u)
        k3 = f(x + 0.5 * self.dt * k2, u)
        k4 = f(x + self.dt * k3, u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        return xn

    def observe(self, x):
        return np.array([np.cos(x[0]),
                         np.sin(x[0]),
                         x[1]])

    def noise(self, x=None, u=None):
        return self._sigma

    def rewrad(self, x, u):
        _x = normalize(x)
        return (_x - self._goal).T @ np.diag(self._goal_weight) @ (_x - self._goal)\
               + u.T @ np.diag(self._act_weight) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # apply action constraints
        _u = np.clip(u, -self._act_max, self._act_max)

        # state-action dependent noise
        _sigma = self.noise(self.state, _u)

        # evolve deterministic dynamics
        _xn = self.dynamics(self.state, _u)

        # apply state constraints
        _xn = np.clip(_xn, -self._state_max, self._state_max)

        # compute reward
        rwrd = self.rewrad(self.state, _u)

        # add noise
        self.state = self.np_random.multivariate_normal(mean=_xn, cov=_sigma)

        return self.observe(self.state), rwrd, False, {}

    def reset(self):
        high = np.array([np.pi, 1.0])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self.observe(self.state)

    # following functions for plotting
    def fake_reset(self, state):
        self.state = state

    def fake_step(self, u):
        # apply action constraints
        _u = np.clip(u, -self._act_max, self._act_max)

        # state-action dependent noise
        _sigma = self.noise(self.state, _u)

        # evolve deterministic dynamics
        _xn = self.dynamics(self.state, _u)

        # apply state constraints
        _xn = np.clip(_xn, -self._state_max, self._state_max)

        # add noise
        self.state = self.np_random.multivariate_normal(mean=_xn, cov=_sigma)

        return self.state
