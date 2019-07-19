import gym
from gym import spaces
from gym.utils import seeding

import autograd.numpy as np

from sds import rARHMM


def mass_spring_damper(param):
    k, m, d, s, const = param

    A = np.array([[0.0, 1.0], [- k / m, - d / m]])
    B = np.array([[0.0, 1.0 / m]]).T
    c = np.array([const, k * s / m])

    return A, B, c


class MassSpringDamper(gym.Env):

    def __init__(self):
        self.dm_state = 2
        self.dm_act = 1
        self.dm_obs = 2

        self._dt = 0.01

        self._goal = np.array([1., 0.])
        self._goal_weight = - np.array([1.e0, 1.e-1])

        self._state_max = np.array([10., 10.])

        self._obs_max = np.array([10., 10.])
        self.observation_space = spaces.Box(low=-self._obs_max,
                                            high=self._obs_max)

        self._act_weight = - np.array([1.e-3])
        self._act_max = np.inf
        self.action_space = spaces.Box(low=-self._act_max,
                                       high=self._act_max, shape=(1,))

        # define the swithching dynamics
        # k, m, d, s, const
        param = ([0.5, 0.25, 0.25, -5.0, 0.0],
                 [-0.5, 0.25, 0.25, 5.0, 0.0])

        self.rarhmm = rARHMM(nb_states=2,
                             dm_obs=self.dm_obs,
                             dm_act=self.dm_act,
                             type='recurrent')

        _sigma = 1.e-64 * np.eye(self.dm_state)
        for k in range(2):
            A, B, c = mass_spring_damper(param[k])
            self.rarhmm.observations.A[k, ...] = np.eye(self.dm_obs) + self._dt * A
            self.rarhmm.observations.B[k, ...] = self._dt * B
            self.rarhmm.observations.c[k, ...] = self._dt * c
            self.rarhmm.observations.cov[k, ...] = self._dt * _sigma

        self.rarhmm.init_observation.mu[0, ...] = np.zeros((self.dm_obs, ))
        self.rarhmm.init_observation.cov[0, ...] = 1. * np.eye(self.dm_obs)

        self.rarhmm.transitions.par = np.array([[1.0, 1.0, 0.0],
                                                [5.0, 5.0, 0.0]])

        self.belief = None
        self.obs = None
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

    def filter(self, b, x, u):
        trans = np.exp(self.rarhmm.transitions.log_likelihood(x, u)[0])
        bn = np.einsum('mk,m->k', trans, b)
        zn = np.random.choice(self.rarhmm.nb_states, p=bn)

        # for plotting
        # zn = np.argmax(bn)
        return zn, bn

    def evolve(self, z, x, u):
        xn = self.rarhmm.observations.sample(z, x, u)

        # for plotting
        # xn = self.rarhmm.observations.mean(z, x, u)
        return xn

    def dynamics(self, b, x, u):
        # filter
        zn, bn = self.filter(b, x, u)

        # evolve
        xn = self.evolve(zn, x, u)

        return bn, xn

    def rewrad(self, x, u):
        return (x - self._goal).T @ np.diag(self._goal_weight) @ (x - self._goal)\
               + u.T @ np.diag(self._act_weight) @ u

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # apply action constraints
        _u = np.clip(u, -self._act_max, self._act_max)

        # compute reward
        rwrd = self.rewrad(self.obs, _u)

        # evolve dynamics
        self.belief, self.obs = self.dynamics(self.belief, self.obs, _u)

        return self.obs, rwrd, False, {}

    def reset(self):
        self.obs = self.rarhmm.init_observation.sample(z=0)
        self.belief = self.rarhmm.init_state.likelihood()
        return self.obs

    # following functions for plotting
    def fake_reset(self, obs):
        self.obs = obs
        self.belief = self.rarhmm.init_state.likelihood()

    def fake_step(self, u):
        # apply action constraints
        _u = np.clip(u, -self._act_max, self._act_max)

        # evolve dynamics
        self.belief, self.obs = self.dynamics(self.belief, self.obs, _u)

        return self.obs
