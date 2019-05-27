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


class PGPE:

    def __init__(self, env, n_episodes, discount,
                 alpha, beta, pdict):
        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.d_action = self.env.action_space.shape[0]

        self.alim = self.env.action_space.high

        self.n_episodes = n_episodes
        self.discount = discount

        self.alpha = alpha
        self.beta = beta

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

    def run(self):
        self.rollouts = self.sample(n_episodes=self.n_episodes)

        _reward = []
        for roll in self.rollouts:
            _gamma = self.discount * np.ones((len(roll['r']), ))
            _disc = np.hstack((1.0, np.cumprod(_gamma[:-1])))
            _reward.append(np.sum(_disc * roll['r']))

        _meanr = np.mean(_reward)

        _par, _reward = [], []
        for n in range(self.n_episodes):
            # perturbed policy
            _pert = self.ctl.perturb()

            # return of perturbed policy
            _roll = self.sample(n_episodes=1, ctl=_pert)

            _gamma = self.discount * np.ones((len(_roll[-1]['r']), ))
            _disc = np.hstack((1.0, np.cumprod(_gamma[:-1])))
            _reward.append(np.sum(_disc * _roll[-1]['r']))
            _par.append(_pert.K)

        _sigma = np.sqrt(np.diag(self.ctl.cov))
        _T = np.squeeze(np.asarray(_par) - self.ctl.K)
        _S = (np.square(_T) - _sigma**2) / _sigma

        _b = np.mean(_reward)
        _r = np.asarray(_reward) - _b

        # update
        _mult = 1. / self.n_episodes
        self.ctl.K += self.alpha * _mult * _r @ _T
        self.ctl.cov += np.diag((self.beta * _mult * _r @ _S))**2

        return _meanr


if __name__ == "__main__":
    import gym
    import lab

    import matplotlib.pyplot as plt

    env = gym.make('LQR-v0')
    env._max_episode_steps = 100

    fdpg = PGPE(env, n_episodes=100, discount=0.995,
                alpha=1e-6, beta=1e-8,
                pdict={'type': 'poly', 'degree': 1, 'cov0': 0.025})

    for it in range(15):
        ret = fdpg.run()
        print('it=', it, f'ret={ret:{5}.{4}}')

    rollouts = fdpg.sample(25)

    fig = plt.figure()
    for r in rollouts:
        plt.plot(r['x'][:, 0])
    plt.show()
