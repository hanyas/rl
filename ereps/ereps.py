import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import stats

from sklearn.preprocessing import PolynomialFeatures

import copy


class Quad:

    def __init__(self, n_states):
        self.n_states = n_states

        M = 10.0 * np.random.randn(n_states, n_states)
        tmp = M @ M.T
        Q = tmp[np.nonzero(np.triu(tmp))]
        q = np.random.rand(n_states)
        q0 = 0.0 * np.random.rand()

        self.param = np.hstack((q0, q, Q))
        self.basis = PolynomialFeatures(degree=2)

    def eval(self, x):
        feat = self.basis.fit_transform(x)
        return -np.dot(feat, self.param)


class Beta:

    def __init__(self, n_states, loc, scale):
        self.n_states = n_states

        self.alpha = np.random.rand(self.n_states)
        self.beta = np.random.rand(self.n_states)

        self.loc = loc
        self.scale = scale

    def sample(self, n):
        aux = sc.stats.beta(self.alpha, self.beta, self.loc, self.scale).rvs(n)
        return aux.reshape((n, self.n_states))

    def loglik(self, var, x, w):
        alpha, beta = var
        lik = sc.stats.beta.pdf(x, alpha, beta, self.loc, self.scale)
        return - np.sum(np.log(lik * w))


class Gauss:

    def __init__(self, n_states, cov0):
        self.n_states = n_states

        self.mu = np.random.randn(n_states)
        self.cov = cov0 * np.eye(n_states)

    def sample(self, n):
        aux = sc.stats.multivariate_normal(mean=self.mu, cov=self.cov).rvs(n)
        return aux.reshape((n, self.n_states))


class EREPS:

    def __init__(self, func,
                 n_samples, n_iter,
                 kl_bound, **kwargs):

        self.func = func
        self.n_states = self.func.n_states

        self.n_samples = n_samples
        self.n_iter = n_iter

        self.kl_bound = kl_bound

        if 'cov0' in kwargs:
            cov0 = kwargs.get('cov0', False)
            self.ctl = Gauss(self.n_states, cov0)
        else:
            loc = kwargs.get('loc', False)
            scale = kwargs.get('scale', False)
            self.ctl = Beta(self.n_states, loc, scale)

        self.data = {}

    def sample(self, n_samples):
        data = {}

        data['x'] = self.ctl.sample(n_samples)
        data['r'] = self.func.eval(data['x'])

        return data

    def dual(self, eta, eps, r):
        adv = r - np.max(r)
        g = eta * eps + np.max(r) + eta * np.log(np.mean(np.exp(adv / eta), axis=0))
        return g

    def ml_policy_gauss(self):
        pol = copy.deepcopy(self.ctl)

        adv = self.data['r'] - np.max(self.data['r'])
        w = np.exp(adv / self.eta)

        pol.mu = np.sum(self.data['x'] * w[:, np.newaxis], axis=0) / np.sum(w, axis=0)
        tmp = self.data['x'] - pol.mu
        pol.cov = np.einsum('nk,n,nh->kh', tmp, w, tmp) / np.sum(w, axis=0)

        return pol

    def ml_policy_beta(self):
        pol = copy.deepcopy(self.ctl)

        adv = self.data['r'] - np.max(self.data['r'])
        w = np.exp(adv / self.eta)

        res = sc.optimize.minimize(self.ctl.loglik,
                                   np.array([1e-4, 1e-4]),
                                   method='SLSQP', jac=None,
                                   args=(self.data['x'], w),
                                   bounds=((1e-8, 1e8), (1e-8, 1e8)))

        pol.alpha, pol.beta = res.x[0], res.x[1]
        return pol

    def kl_samples(self):
        adv = self.data['r'] - np.max(self.data['r'])
        w = np.exp(adv / self.eta)
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.ctl.cov) * (2.0 * np.pi * np.exp(1.0))**self.n_states)

    def run(self, n_iter):
        for it in range(n_iter):
            self.data = self.sample(self.n_samples)

            res = sc.optimize.minimize(self.dual, 1.0,
                                       method='SLSQP',
                                       jac=grad(self.dual),
                                       args=(
                                           self.kl_bound,
                                           self.data['r']),
                                       bounds=((1e-8, 1e8),))
            self.eta = res.x

            kl_samples = self.kl_samples()

            self.ctl = self.ml_policy_gauss()

            rwrd = np.mean(self.data['r'])

            print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kl_s={kl_samples:{5}.{4}}')

            if np.linalg.det(self.ctl.cov) < 1e-64:
                break


if __name__ == "__main__":

    ereps = EREPS(func=Quad(n_states=10),
                  n_samples=250, n_iter=250,
                  kl_bound=0.05, cov0=100.0)

    ereps.run(n_iter=ereps.n_iter)
