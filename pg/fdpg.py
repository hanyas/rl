import numpy as np
import scipy as sc
from scipy import special

import copy
from sklearn.preprocessing import PolynomialFeatures


class Policy:

    def __init__(self, d_state, d_action, **kwargs):
        self.d_state = d_state
        self.d_action = d_action

        self.degree = kwargs.get('degree', False)
        self.n_feat = int(sc.special.comb(self.degree + self.d_state, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.n_param = self.d_action * self.n_feat
        self.K = 1e-8 * np.random.randn(self.n_param)
        self.cov = np.eye(self.n_param)

    def features(self, x):
        return self.basis.fit_transform(x.reshape(-1, self.d_state)).squeeze()

    def mean(self, x):
        feat = self.features(x)
        return np.einsum('...k,mk->...m', feat, self.K.reshape(self.d_action, self.d_state))

    def actions(self, x):
        return self.mean(x)

    def perturb(self):
        pert = copy.deepcopy(self)
        pert.K = np.random.multivariate_normal(self.K, self.cov)
        return pert


class FDPG:

    def __init__(self, env, n_episodes, n_steps, discount, alpha, cov):
        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.d_action = self.env.action_space.shape[0]

        self.action_limit = self.env.action_space.high

        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.discount = discount

        self.alpha = alpha

        self.ctl = Policy(self.d_state, self.d_action, degree=1)
        self.ctl.cov = cov * self.ctl.cov

        self.rollouts = []

    def sample(self, n_episodes, n_steps, ctl=None):
        rollouts = []

        for _ in range(n_episodes):
            roll = {'x': np.empty((0, self.d_state)),
                    'u': np.empty((0, self.d_action)),
                    'r': np.empty((0,))}

            x = self.env.reset()
            for _ in range(n_steps):
                if ctl is None:
                    u = self.ctl.actions(x)
                else:
                    u = ctl.actions(x)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.action_limit, self.action_limit))
                roll['r'] = np.hstack((roll['r'], r))

            rollouts.append(roll)

        return rollouts

    def run(self):
        self.rollouts = self.sample(n_episodes=self.n_episodes, n_steps=self.n_steps)

        _disc = np.hstack((1.0, np.cumprod(self.discount * np.ones((self.n_steps,))[:-1])))

        _reward = []
        for roll in self.rollouts:
            _reward.append(np.sum(_disc * roll['r']))

        _meanr = np.mean(_reward)

        _par, _reward = [], []
        for n in range(self.n_episodes):
            # perturbed policy
            _pert = self.ctl.perturb()

            # return of perturbed policy
            _roll = self.sample(n_episodes=1, n_steps=self.n_steps, ctl=_pert)

            _reward.append(np.sum(_disc * _roll[-1]['r']))
            _par.append(_pert.K)

        # param diff
        _dpar = np.squeeze(np.asarray(_par) - self.ctl.K)

        # rwrd diff
        _dr = np.asarray(_reward) - _meanr

        # gradient
        _grad = np.linalg.inv(_dpar.T @ _dpar + 1e-8 * np.eye(self.ctl.n_feat)) @ _dpar.T @ _dr

        # update
        self.ctl.K += self.alpha * _grad / (self.n_episodes * self.n_steps)

        return _meanr


if __name__ == "__main__":
    import gym
    import lab

    import matplotlib.pyplot as plt

    env = gym.make('LQR-v0')
    env._max_episode_steps = 100

    fdpg = FDPG(env, n_episodes=50, n_steps=100,
                discount=0.995, alpha=1e-4, cov=0.01)

    for it in range(10):
        ret = fdpg.run()
        print('it=', it, f'ret={ret:{5}.{4}}')

    rollouts = fdpg.sample(25, 100)

    fig = plt.figure()
    for r in rollouts:
        plt.plot(r['x'][:, 0])
    plt.show()
