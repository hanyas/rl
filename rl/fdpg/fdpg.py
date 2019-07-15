import numpy as np
import scipy as sc
from scipy import special

import copy
from sklearn.preprocessing import PolynomialFeatures


class Policy:

    def __init__(self, d_state, d_action, pdict):
        self.d_state = d_state
        self.d_action = d_action

        self.type = 'poly'
        self.degree = pdict['degree']
        self.n_feat = int(sc.special.comb(self.degree + self.d_state, self.degree)) - 1
        self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.n_param = self.d_action * self.n_feat
        self.K = 1e-8 * np.random.randn(self.n_param)
        self.cov = pdict['cov0'] * np.eye(self.n_param)

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

    def __init__(self, env, n_episodes, discount,
                 alpha, pdict):
        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.d_action = self.env.action_space.shape[0]

        self.alim = self.env.action_space.high

        self.n_episodes = n_episodes
        self.discount = discount

        self.alpha = alpha

        self.ctl = Policy(self.d_state, self.d_action, pdict)

        self.rollouts = None

    def sample(self, n_episodes, ctl=None):
        rollouts = []

        for _ in range(n_episodes):
            roll = {'x': np.empty((0, self.d_state)),
                    'u': np.empty((0, self.d_action)),
                    'xn': np.empty((0, self.d_state)),
                    'done': np.empty((0,), np.int64),
                    'r': np.empty((0,))}

            x = self.env.reset()

            done = False
            while not done:
                if ctl is None:
                    u = self.ctl.actions(x)
                else:
                    u = ctl.actions(x)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.alim, self.alim))
                roll['xn'] = np.vstack((roll['xn'], x))
                roll['done'] = np.hstack((roll['done'], done))
                roll['r'] = np.hstack((roll['r'], r))

            rollouts.append(roll)

        return rollouts

    def run(self, nb_iter=100, verbose=False):
        _trace = {'ret': []}

        for it in range(nb_iter):
            self.rollouts = self.sample(n_episodes=self.n_episodes)

            _return = []
            for roll in self.rollouts:
                _gamma = self.discount * np.ones((len(roll['r']), ))
                _disc = np.hstack((1.0, np.cumprod(_gamma[:-1])))
                _return.append(np.sum(_disc * roll['r']))

            _meanr = np.mean(_return)

            _par, _return = [], []
            for n in range(self.n_episodes):
                # perturbed policy
                _pert = self.ctl.perturb()

                # return of perturbed policy
                _roll = self.sample(n_episodes=1, ctl=_pert)

                _gamma = self.discount * np.ones((len(_roll[-1]['r']), ))
                _disc = np.hstack((1.0, np.cumprod(_gamma[:-1])))
                _return.append(np.sum(_disc * _roll[-1]['r']))
                _par.append(_pert.K)

            # param diff
            _dpar = np.squeeze(np.asarray(_par) - self.ctl.K)

            # rwrd diff
            _dr = np.asarray(_return) - _meanr

            # gradient
            _grad = np.linalg.inv(_dpar.T @ _dpar + 1e-8 * np.eye(self.ctl.n_feat)) @ _dpar.T @ _dr

            # update
            self.ctl.K += self.alpha * _grad / self.n_episodes

            _trace['ret'].append(_meanr)

            if verbose:
                print('it=', it, f'ret={_meanr:{5}.{4}}')

        return _trace
