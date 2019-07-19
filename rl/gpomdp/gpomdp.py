import numpy as np
import scipy as sc
from scipy import special

from sklearn.preprocessing import PolynomialFeatures


class FourierFeatures:

    def __init__(self, d_state, nb_feat, band):
        self.d_state = d_state
        self.nb_feat = nb_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.d_state),
                                                  cov=np.diag(1.0 / band),
                                                  size=self.nb_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.nb_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        return phi


class Policy:

    def __init__(self, d_state, dm_act, **kwargs):
        self.d_state = d_state
        self.dm_act = dm_act

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.nb_feat = kwargs.get('nb_feat', False)
            self.basis = FourierFeatures(self.d_state, self.nb_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.nb_feat = int(sc.special.comb(self.degree + self.d_state, self.degree)) - 1
            self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.K = 1e-8 * np.random.randn(self.dm_act, self.nb_feat)
        self.cov = np.eye(dm_act)

    def features(self, x):
        return self.basis.fit_transform(x.reshape(-1, self.d_state)).squeeze()

    def mean(self, x):
        feat = self.features(x)
        return np.einsum('...k,mk->...m', feat, self.K)

    def actions(self, x, stoch):
        mean = self.mean(x)
        if stoch:
            return np.random.multivariate_normal(mean, self.cov)
        else:
            return mean

    def grad(self, x, u):
        return np.einsum('nl,ll,nk->nk', u - self.mean(x), np.linalg.inv(self.cov), self.features(x))


class GPOMDP:

    def __init__(self, env, n_episodes, nb_steps, discount, alpha):
        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.dm_act = self.env.action_space.shape[0]

        self.ulim = self.env.action_space.high

        self.n_episodes = n_episodes
        self.nb_steps = nb_steps
        self.discount = discount

        self.alpha = alpha

        self.ctl = Policy(self.d_state, self.dm_act, degree=1)
        self.ctl.cov = 0.01 * self.ctl.cov

        self.rollouts = []

    def sample(self, n_episodes, nb_steps, stoch=True):
        rollouts = []

        for _ in range(n_episodes):
            roll = {'x': np.empty((0, self.d_state)),
                    'u': np.empty((0, self.dm_act)),
                    'r': np.empty((0,))}

            x = self.env.reset()

            for _ in range(nb_steps):
                u = self.ctl.actions(x, stoch)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.ulim, self.ulim))
                roll['r'] = np.hstack((roll['r'], r))

            rollouts.append(roll)

        return rollouts

    def baseline(self, rewards, gradient):
        _norm = np.zeros((self.nb_steps, self.ctl.nb_feat))
        for g in gradient:
            _norm += np.cumsum(g**2, axis=0)

        _b = np.zeros((self.nb_steps, self.ctl.nb_feat))
        for r, g in zip(rewards, gradient):
            _b += np.cumsum(g**2, axis=0) * r[:, np.newaxis] / _norm

        return _b

    def run(self, nb_iter=100, verbose=False):
        _trace = {'ret': []}

        for it in range(nb_iter):
            self.rollouts = self.sample(n_episodes=self.n_episodes, nb_steps=self.nb_steps)

            _disc = np.hstack((1.0, np.cumprod(self.discount * np.ones((self.nb_steps,))[:-1])))

            _reward = []
            for roll in self.rollouts:
                _reward.append(_disc * roll['r'])

            _grad = []
            for roll in self.rollouts:
                _g = self.ctl.grad(roll['x'], roll['u'])
                _grad.append(_g)

            _baseline = self.baseline(_reward, _grad)

            _wgrad = np.zeros((self.ctl.nb_feat, ))
            for r, g in zip(_reward, _grad):
                _wgrad += np.sum(np.cumsum(g, axis=0) * (r[:, np.newaxis] - _baseline), axis=0) / len(_reward)

            self.ctl.K += self.alpha * _wgrad

            ret = 0.0
            for r in _reward:
                ret += r.sum() / len(_reward)

            _trace['ret'].append(ret)

            if verbose:
                print('it=', it, f'ret={ret:{5}.{4}}')

        return _trace
