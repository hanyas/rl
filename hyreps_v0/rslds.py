import autograd.numpy as np
import scipy as sc
from scipy import stats
from scipy import special

import numexpr as ne

from sklearn.preprocessing import PolynomialFeatures

from rl.hyreps_v0 import dlogistic
from rl.hyreps_v0 import normalize

import time
import pickle

EXP_MAX = 700.0
EXP_MIN = -700.0


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
        return np.einsum('kh,...h->...k', self.K, aux)

    def sample(self, x):
        um = self.mean(x)
        u = um + sc.random.multivariate_normal(mean=np.zeros((self.n_actions, )),
                                                cov=self.cov)
        return u

    def prob(self, x, u):
        err = u - self.mean(x)
        return self.const * np.exp(
            -0.5 * np.einsum('...k,kh,...h->...', err, self.perc, err))


class LinearGaussian:

    def __init__(self, n_states, prior, reg=1e-16):
        self.n_states = n_states

        self.reg = reg

        self.cov = sc.stats.invwishart.rvs(prior['nu'],
                                           prior['psi'] * np.eye(n_states))

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

    def mean(self, x, u, A=None, B=None, C=None):
        if all(v is None for v in [A, B, C]):
            A, B, C = self.A, self.B, self.C

        return np.einsum('kh,...h->...k', A, x) + np.einsum('kh,...h->...k', B, u) + C

    def sample(self, x, u):
        xm = self.mean(x, u)
        xn = xm + sc.random.multivariate_normal(mean=np.zeros((self.n_states, )),
                                                cov=self.cov)
        return xn

    def prob(self, x, u):
        err = x[..., 1:, :] - self.mean(x[..., :-1, :], u[..., :-1, :])
        return self.const * np.exp(
            -0.5 * np.einsum('...k,kh,...h->...', err, self.perc, err))


class MultiLogistic:

    def __init__(self, n_states, n_actions, n_regions, **kwargs):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_regions = n_regions

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.n_states + self.n_actions, self.n_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.n_feat = int(
                sc.special.comb(self.degree + self.n_states + n_actions, self.degree))
            self.basis = PolynomialFeatures(self.degree)

        self.par = np.random.randn(self.n_regions,
                                   self.n_regions * self.n_feat)

    def features(self, x, u):
        xu = np.concatenate((x, u), axis=-1)
        feat = self.basis.fit_transform(xu.reshape(-1, self.n_states + self.n_actions))
        return np.reshape(feat, xu.shape[:-1] + (self.n_feat, ))

    def transitions(self, x, u, par=None):
        if par is None:
            par = self.par

        feat = self.features(x, u)
        trans = self.logistic(par.reshape((self.n_regions, self.n_regions, -1)), feat)

        return trans

    def logistic(self, p, feat):
        a = np.einsum('...k,mlk->...ml', feat, p)
        a = np.clip(a, EXP_MIN, EXP_MAX)
        expa = ne.evaluate('exp(a)')
        l = expa / np.sum(expa, axis=-1, keepdims=True)
        return np.squeeze(l)

    def logistic_err(self, p, feat, f):
        par = np.reshape(p, (1, self.n_regions, -1))
        err = np.sum(np.square(self.logistic(par, feat) - f), axis=0)
        return err

    def dlogistic_err(self, p, feat, f):
        par = np.reshape(p, (1, self.n_regions, -1))
        lgstc = self.logistic(par, feat)
        feat_noview = feat.copy()  # cython memoryview issue
        derr = 2.0 * np.einsum('tl,tlp->lp', lgstc - f, dlogistic(lgstc, feat_noview))
        return derr


class rSLDS:
    def __init__(self, n_states, n_actions, n_regions, priors):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_regions = n_regions

        dyn_prior, ctl_prior = priors[0], priors[1]

        alpha = np.ones(self.n_regions) / self.n_regions
        self.init_region = sc.stats.multinomial(1, alpha)

        self.init_state = sc.stats.multivariate_normal(
            mean=np.random.randn(self.n_states),
            cov=sc.stats.invwishart.rvs(dyn_prior["nu"],
                                        dyn_prior["psi"] * np.eye(self.n_states)))

        self.linear_models = [LinearGaussian(self.n_states, dyn_prior) for _ in
                              range(n_regions)]

        self.logistic_model = MultiLogistic(self.n_states, self.n_actions,
                                            self.n_regions, degree=1)

        self.linear_ctls = [Policy(self.n_states, self.n_actions, ctl_prior)
                              for _ in range(n_regions)]

    def filter(self, x, u, alpha, stoch=False):
        trans = self.logistic_model.transitions(x, u)

        alphan = np.einsum('...mk,...m->...k', trans, alpha)
        alphan, _ = normalize(alphan + np.finfo(float).tiny, dim=(-1, ))

        if stoch:
            zn = np.argmax(sc.stats.multinomial(1, alphan).rvs())
        else:
            zn = np.argmax(alphan, axis=-1)

        return zn, alphan

    def evolve(self, x, u, alpha, max=True, stoch=False):
        if stoch:
            x_next = np.array([dist.sample(x, u) for dist in self.linear_models])
        else:
            x_next = np.array([dist.mean(x, u) for dist in self.linear_models])

        if max:
            xn = x_next[np.argmax(alpha)]
        else:
            xn = np.sum(alpha * x_next.T, axis=-1).T

        return xn

    def act(self, x, alpha, max=True, stoch=False):
        if stoch:
            tmp = np.array([ctl.sample(x) for ctl in self.linear_ctls])
        else:
            tmp = np.array([ctl.mean(x) for ctl in self.linear_ctls])

        if max:
            u = tmp[np.argmax(alpha)]
        else:
            u = np.sum(alpha * tmp.T, axis=-1).T

        return u

    def step(self, x, u, alpha, stoch=False):
        # filter
        zn, alphan = self.filter(x, u, alpha, stoch)

        # evolve mean
        xn = self.evolve(x, u, alphan, stoch)

        return zn, xn, alphan

    def sim(self, x0, u, stoch=False):
        n_rollouts, n_steps = u.shape[0], u.shape[1]

        alpha = np.zeros((n_rollouts, n_steps, self.n_regions))
        z = np.zeros((n_rollouts, n_steps,), np.int64)
        x = np.zeros((n_rollouts, n_steps, self.n_states))

        # init continuous state
        x[:, 0] = x0

        # init discrete region
        alpha[:, 0] = self.init_region.p
        if stoch:
            z[:, 0] = np.argmax(sc.stats.multinomial(1, alpha[:, 0]).rvs())
        else:
            z[:, 0] = np.argmax(alpha[:, 0], axis=-1)

        # simulate
        for t in range(1, n_steps):
            z[:, t], x[:, t], alpha[:, t] = self.step(x[:, t - 1], u[:, t - 1],
                                                      alpha[:, t - 1], stoch)

        return z, x

    def save(self, path, custom=False):
        if custom:
            file = open(path + ".pickle", "wb")
        else:
            time_stamp = time.strftime('%X')
            file = open(path + "rslds_" + time_stamp + ".pickle", "wb")

        pickle.dump(self.n_states, file)
        pickle.dump(self.n_actions, file)
        pickle.dump(self.n_regions, file)

        pickle.dump(self.init_region.p, file)

        pickle.dump(self.init_state.mean, file)
        pickle.dump(self.init_state.cov, file)

        pickle.dump(self.linear_models, file)
        pickle.dump(self.logistic_model, file)
        pickle.dump(self.linear_ctls, file)

        file.close()

    def load(self, path):
        file = open(path, "rb")

        self.n_states = pickle.load(file)
        self.n_actions = pickle.load(file)
        self.n_regions = pickle.load(file)

        self.init_region.p = pickle.load(file)

        self.init_state.mean = pickle.load(file)
        self.init_state.cov = pickle.load(file)

        self.linear_models = pickle.load(file)
        self.logistic_model = pickle.load(file)
        self.linear_ctls = pickle.load(file)

        file.close()


if __name__ == "__main__":
    n_rollouts, n_steps = 50, 100
    n_states, n_actions = 2, 1
    n_regions = 2

    x = np.random.randn(n_rollouts, n_steps, n_states)
    u = np.random.randn(n_rollouts, n_steps, n_actions)

    dyn_prior = {"nu": n_states + 1, "psi": 1e-4}
    ctl_prior = {"nu": n_actions + 1, "psi": 1.0}
    priors = [dyn_prior, ctl_prior]

    rslds = rSLDS(n_states, n_actions, n_regions, priors)

    prob = rslds.linear_models[0].prob(x, u)
    trans = rslds.logistic_model.transitions(x, u)
