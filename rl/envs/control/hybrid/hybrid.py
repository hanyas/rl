import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from sds import rARHMM


def mass_spring_damper(param):
    k, m, d, s, const = param

    A = np.array([[0.0, 1.0], [- k / m, - d / m]])
    B = np.array([[0.0, 1.0 / m]]).T
    c = np.array([const, k * s / m])

    return A, B, c


class Hybrid(gym.Env):

    def __init__(self):

        self.max_state = np.array([10.0, 10.0])
        self.max_action = np.inf

        self.goal = np.array([1.0, 0.0])

        self.low_state = - self.max_state
        self.high_state = self.max_state

        self.action_space = spaces.Box(low=-self.max_action,
                                       high=self.max_action, shape=(1,))

        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)

        # define the swithching dynamics
        # k, m, d, s, const
        param = ([0.5, 0.25, 0.25, -5.0, 0.0],
                 [-0.5, 0.25, 0.25, 5.0, 0.0])

        self.rarhmm = rARHMM(nb_states=2, dim_obs=2, dim_act=1, type='recurrent')

        dt = 0.01
        for k in range(2):
            A, B, c = mass_spring_damper(param[k])
            self.rarhmm.observations.A[k, ...] = np.eye(2) + dt * A
            self.rarhmm.observations.B[k, ...] = dt * B
            self.rarhmm.observations.c[k, ...] = dt * c
            self.rarhmm.observations.cov[k, ...] = dt * 1e-4 * np.eye(2)

        self.rarhmm.init_observation.mu[0, ...] = np.zeros((2, ))
        self.rarhmm.init_observation.cov[0, ...] = 1. * np.eye(2)

        self.rarhmm.transitions.par = np.array([[1.0, 1.0, 0.0],
                                                [5.0, 5.0, 0.0]])

        # reward params
        self._Q = - np.array([[1.0, 0.0], [0.0, 0.01]])
        self._H = - 1e-2

        self.alpha = None
        self.state = None
        self.np_random = None

        self.seed()
        self.reset()

    def filter(self, x, u, alpha, stoch=False):
        trans = np.exp(self.rarhmm.transitions.log_likelihood([x], [u]))[0]
        alphan = np.einsum('...mk,...m->...k', trans, alpha)

        if stoch:
            zn = np.random.choice(self.rarhmm.nb_states, p=alphan)
        else:
            zn = np.argmax(alphan, axis=-1)

        return zn, alphan

    def evolve(self, x, u, alpha, max=True, stoch=True):
        if stoch:
            _xn = np.array([self.rarhmm.observations.sample(k, x, u)
                            for k in range(self.rarhmm.nb_states)])
        else:
            _xn = np.array([self.rarhmm.observations.mean(k, x, u)
                            for k in range(self.rarhmm.nb_states)])

        if max:
            xn = _xn[np.argmax(alpha)]
        else:
            xn = np.sum(alpha * _xn.T, axis=-1).T

        return xn

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        u = np.clip(action, -self.max_action, self.max_action)

        err = self.state - self.goal
        rwrd = err.T @ self._Q @ err + self._H * u**2

        # filter
        _, self.alpha = self.filter(self.state, u, self.alpha, stoch=True)

        # evolve
        self.state = self.evolve(self.state, action, self.alpha, True, True)

        return self.state, rwrd.flatten(), False, {}

    def reset(self):
        self.state = self.rarhmm.init_observation.sample(z=0)
        self.alpha = self.rarhmm.init_state.likelihood()
        return np.array(self.state)

    def fake_reset(self, state):
        self.state = state
        self.alpha = self.rarhmm.init_state.likelihood()
        return self.state
