import numpy as np
import scipy as sc
from scipy import stats
from scipy import special

from sklearn.preprocessing import PolynomialFeatures

from rl.hyreps.logistic import logistic

import time
import pickle


class FourierFeatures:

    def __init__(self, n_states, n_feat, band):
        self.n_states = n_states
        self.n_feat = n_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.n_states),
                                                  cov=np.diag(1.0 / band),
                                                  size=self.n_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.n_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        phi = np.insert(phi, 0, 1.0, axis=-1)
        return phi


class Policy:

    def __init__(self, n_states, n_actions, prior, reg=1e-16):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_feat = self.n_states + 1

        self.reg = reg

        self.cov = sc.stats.invwishart.rvs(prior["nu"],
                                           prior["psi"] * np.eye(n_actions))

        self.K = np.random.randn(self.n_actions, self.n_feat)

        self.perc = 1.0 / self.cov
        self.const = 1.0 / np.sqrt(np.linalg.det(
            2.0 * np.pi * (self.cov + self.reg * np.eye(n_actions))))
        self.pdf = None

    def update(self):
        self.perc = 1.0 / (self.cov + self.reg * np.eye(self.n_actions))
        self.const = 1.0 / np.sqrt(np.linalg.det(
            2.0 * np.pi * (self.cov + self.reg * np.eye(self.n_actions))))

    def mean(self, x):
        aux = np.concatenate((np.ones(x.shape[:-1] + (1, )), x), axis=-1)
        return np.einsum('kh,...th->...tk', self.K, aux)

    def prob(self, x, u):
        err = u - self.mean(x)
        return self.const * np.exp(
            -0.5 * np.einsum('...tk,kh,...th->...t', err, self.perc, err))


class LinearGaussian:

    def __init__(self, n_states, prior, reg=1e-16):
        self.n_states = n_states

        self.reg = reg

        self.cov = sc.stats.invwishart.rvs(prior["nu"],
                                           prior["psi"] * np.eye(n_states))

        self.A = sc.stats.matrix_normal.rvs(mean=None, rowcov=self.cov,
                                            colcov=self.cov)
        self.B = sc.stats.matrix_normal.rvs(mean=None, rowcov=self.cov,
                                            colcov=self.cov)[:, [0]]
        self.C = sc.stats.matrix_normal.rvs(mean=None, rowcov=self.cov,
                                            colcov=self.cov)[:, 0]

        self.perc = np.linalg.inv(self.cov + self.reg * np.eye(n_states))
        self.const = 1.0 / np.sqrt(np.linalg.det(
            2.0 * np.pi * (self.cov + self.reg * np.eye(n_states))))
        self.pdf = None

    def update(self):
        self.perc = np.linalg.inv(self.cov + self.reg * np.eye(self.n_states))
        self.const = 1.0 / np.sqrt(np.linalg.det(
            2.0 * np.pi * (self.cov + self.reg * np.eye(self.n_states))))

    def mean(self, x, u):
        return np.einsum('kh,...th->...tk', self.A, x) + np.einsum(
            'kh,...th->...tk', self.B, u) + self.C

    def prob(self, x, u):
        err = x[..., 1:, :] - self.mean(x[..., :-1, :], u[..., :-1, :])
        return self.const * np.exp(
            -0.5 * np.einsum('...tk,kh,...th->...t', err, self.perc, err))


class MultiLogistic:

    def __init__(self, n_states, n_regions, **kwargs):
        self.n_states = n_states
        self.n_regions = n_regions

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.n_feat = int(
                sc.special.comb(self.degree + self.n_states, self.degree))
            self.basis = PolynomialFeatures(self.degree)

        self.par = np.random.randn(self.n_regions,
                                   self.n_regions * self.n_feat)

    def features(self, x):
        feat = self.basis.fit_transform(x.reshape(-1, self.n_states))
        return np.reshape(feat, x.shape[:-1] + (self.n_feat,))

    def transitions(self, x):
        feat = self.features(x)

        trans = np.zeros(x.shape[:-1] + (self.n_regions, self.n_regions, ))
        for i in range(self.n_regions):
            trans[..., i, :] = logistic(self.par[i, :], feat)

        return trans


class rSLDS:
    def __init__(self, n_states, n_actions, n_regions, dyn_prior, ctl_prior):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_regions = n_regions

        self.init_state = sc.stats.multivariate_normal(
            mean=np.random.randn(self.n_states),
            cov=sc.stats.invwishart.rvs(dyn_prior["nu"],
                                        dyn_prior["psi"] * np.eye(self.n_states)))

        self.linear_models = [LinearGaussian(self.n_states, dyn_prior) for _ in
                              range(n_regions)]

        self.logistic_model = MultiLogistic(self.n_states,
                                            self.n_regions, degree=1)

        self.linear_policy = [Policy(self.n_states, self.n_actions, ctl_prior)
                              for _ in range(n_regions)]

    def filter(self, xn, alpha):
        trans = self.logistic_model.transitions(xn)

        alphan = np.einsum('...mk,...m->...k', trans, alpha)
        alphan = alphan + np.finfo(float).tiny
        alphan = alphan / np.sum(alphan, axis=-1, keepdims=True)

        zn = np.argmax(alphan, axis=-1)

        return zn, alphan

    def evolve(self, z, x, u, alpha):
        x_next = np.array([dist.mean(x, u) for dist in self.linear_models])
        xn = np.sum(alpha * x_next.T, axis=-1).T
        return xn

    def act(self, z, x, alpha):
        u = np.array([ctl.mean(x) for ctl in self.linear_policy])
        return np.sum(alpha * u.T, axis=-1).T

    def step(self, z, x, u, alpha):
        # evolve mean
        xn = self.evolve(z, x, u, alpha)

        # filter
        zn, alphan = self.filter(xn, alpha)

        return zn, xn, alphan

    def sim(self, x0, u):
        n_rollouts, n_steps = u.shape[0], u.shape[1]

        alpha = np.zeros((n_rollouts, n_steps, self.n_regions))
        z = np.zeros((n_rollouts, n_steps,), np.int64)
        x = np.zeros((n_rollouts, n_steps, self.n_states))

        x[:, 0] = x0
        p = np.ones(self.n_regions) / self.n_regions
        z[:, 0], alpha[:, 0] = self.filter(x[:, 0], p)

        for t in range(1, n_steps):
            z[:, t], x[:, t], alpha[:, t] = self.step(z[:, t - 1], x[:, t - 1],
                                                      u[:, t - 1], alpha[:, t - 1])

        return z, x

    def save(self, path):
        time_stamp = time.strftime('%X')
        file = open(path + "rslds_" + time_stamp + ".pickle", "wb")

        pickle.dump(self.n_states, file)
        pickle.dump(self.n_actions, file)
        pickle.dump(self.n_regions, file)

        pickle.dump(self.init_state.mean, file)
        pickle.dump(self.init_state.cov, file)

        pickle.dump(self.linear_models, file)
        pickle.dump(self.logistic_model, file)

        file.close()

    def load(self, path):
        file = open(path, "rb")

        self.n_states = pickle.load(file)
        self.n_actions = pickle.load(file)
        self.n_regions = pickle.load(file)

        self.init_state.mean = pickle.load(file)
        self.init_state.cov = pickle.load(file)

        self.linear_models = pickle.load(file)
        self.logistic_model = pickle.load(file)

        file.close()


if __name__ == "__main__":
    n_rollouts, n_steps = 100, 50
    n_states, n_actions = 3, 1
    n_regions = 3

    x = np.random.randn(n_rollouts, n_steps, n_states)
    u = np.random.randn(n_rollouts, n_steps, n_actions)

    dyn_prior = {"nu": n_states + 1, "psi": 1e-4}
    ctl_prior = {"nu": n_actions + 1, "psi": 1.0}

    rslds = rSLDS(n_states, n_actions, n_regions, dyn_prior, ctl_prior)

    prob = rslds.linear_models[0].prob(x, u)
    trans = rslds.logistic_model.transitions(x)
