import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np


class LQR(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 2

        self._dt = 0.1

        self._sigma = 1.e-64 * np.eye(self.dm_state)

        self._goal = np.array([0., 0.])
        self._goal_weight = - np.array([1.e2, 1.e1])

        self._state_max = np.array([1., 1.])

        self._obs_max = np.array([1., 1.])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_weight = - np.array([1.e-2])
        self._act_max = np.inf
        self.action_space = spaces.Box(low=-self._act_max,
                                       high=self._act_max, shape=(1,))

        self._A = np.array([[0., 1.], [0., 0.]])
        self._B = np.array([[0., 1.]])
        self._c = np.zeros((2, ))

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
        def f(x, u):
            return np.einsum('kh,h->k', self._A, x)\
                   + np.einsum('kh,h->k', self._B, u)\
                   + self._c

        k1 = f(x, u)
        k2 = f(x + 0.5 * self.dt * k1, u)
        k3 = f(x + 0.5 * self.dt * k2, u)
        k4 = f(x + self.dt * k3, u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        return xn

    def observe(self, x):
        return x

    def noise(self, x=None, u=None):
        return self._sigma

    def rewrad(self, x, u):
        return (x - self._goal).T @ np.diag(self._goal_weight) @ (x - self._goal)\
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
        _xn = np.clip(_xn, -self._obs_max, self._obs_max)

        # compute reward
        rwrd = self.rewrad(self.state, _u)

        # add noise
        self.state = self.np_random.multivariate_normal(mean=_xn, cov=_sigma)

        return self.observe(self.state), rwrd, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=np.array([-1.0, -1e-2]),
                                            high=np.array([1.0, 1e-2]))
        return self.state
