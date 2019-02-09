import os

os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import special

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  Ridge

import copy


class Sphere:

    def __init__(self, dim_action):
        self.dim_action = dim_action

        M = np.random.randn(self.dim_action, self.dim_action)
        M = 0.5 * (M + M.T)
        tmp = M @ M.T

        Q = tmp[np.nonzero(np.triu(tmp))]

        q = 0.0 * np.random.rand(self.dim_action)
        q0 = 0.0 * np.random.rand()

        self.param = np.hstack((q0, q, Q))
        self.basis = PolynomialFeatures(degree=2)

    def eval(self, x):
        feat = self.basis.fit_transform(x)
        return - np.dot(feat, self.param)


class Rosenbrock:

    def __init__(self, dim_action):
        self.dim_action = dim_action

    def eval(self, x):
        return - np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0 +
                        (1 - x[:, :-1]) ** 2.0, axis=-1)


class Styblinski:
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def eval(self, x):
        return - 0.5 * np.sum(x ** 4.0 - 16.0 * x ** 2 + 5 * x, axis=-1)


class Rastrigin:
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def eval(self, x):
        return - (10.0 * self.dim_action +
                  np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1))


class Policy:

    def __init__(self, dim_action, cov0):
        self.dim_action = dim_action

        self.mu = 0.0 * np.random.randn(dim_action)
        self.cov = cov0 * np.eye(dim_action)

    def action(self, n):
        return np.random.multivariate_normal(self.mu, self.cov, size=(n))

    def kli(self, pi):
        diff = self.mu - pi.mu

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) + diff.T @ np.linalg.inv(self.cov) @ diff
                    - self.dim_action + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def update(self, eta, omega, model):
        pol = copy.deepcopy(self)

        Raa, ra, r0 = model.Raa, model.ra, model.r0

        b, Q = self.mu, self.cov
        invQ = np.linalg.inv(Q)

        F = np.linalg.inv(eta * invQ - 2.0 * Raa)
        f = eta * invQ @ b + ra

        pol.mu = F @ f
        pol.cov = F * (eta + omega) + np.eye(self.dim_action) * 1e-24

        return pol


class Model:

    def __init__(self, dim_action):
        self.dim_action = dim_action

        self.Raa = np.zeros((dim_action, dim_action))
        self.ra = np.zeros((dim_action,))
        self.r0 = np.zeros((1, ))

    def fit(self, x, r):
        poly = PolynomialFeatures(2)
        feat = poly.fit_transform(x)

        reg = Ridge(alpha=1e-8, fit_intercept=False)
        reg.fit(feat, r)
        par = reg.coef_

        uid = np.triu_indices(self.dim_action)
        self.Raa[uid] = par[1 + self.dim_action:]
        self.Raa.T[uid] = self.Raa[uid]

        self.ra = par[1: 1 + self.dim_action]
        self.r0 = par[0]

        # check for negative definitness
        w, v = np.linalg.eig(self.Raa)
        w[w >= 0.0] = -1e-12
        self.Raa = v @ np.diag(w) @ v.T
        self.Raa = 0.5 * (self.Raa + self.Raa.T)

        # refit quadratic
        # poly = PolynomialFeatures(1)
        # feat = poly.fit_transform(x)
        #
        # aux = r - np.einsum('nk,kh,nh->n', x, self.Raa, x)
        #
        # reg = Ridge(alpha=1e-8, fit_intercept=False)
        # reg.fit(feat, aux)
        # par = reg.coef_
        #
        # self.ra = par[1:]
        # self.r0 = par[0]


class MORE:

    def __init__(self, func, n_samples,
                 kl_bound, ent_rate, **kwargs):

        self.func = func
        self.dim_action = self.func.dim_action

        self.n_samples = n_samples

        self.kl_bound = kl_bound
        self.ent_rate = ent_rate

        if 'cov0' in kwargs:
            cov0 = kwargs.get('cov0', False)
            self.ctl = Policy(self.dim_action, cov0)
        else:
            self.ctl = Policy(self.dim_action, 100.0)

        if 'h0' in kwargs:
            self.h0 = kwargs.get('h0', False)
        else:
            self.h0 = 75.0

        self.model = Model(self.dim_action)

        self.eta = np.array([1.0])
        self.omega = np.array([1.0])

        self.data = {}

    def sample(self, n_samples):
        data = {}

        data['x'] = self.ctl.action(n_samples)
        data['r'] = self.func.eval(data['x'])

        return data

    def dual(self, var, eps, beta, ctl, model):
        eta = var[0]
        omega = var[1]

        Raa, ra, r0 = model.Raa, model.ra, model.r0

        b, Q = ctl.mu, ctl.cov
        invQ = np.linalg.inv(Q)

        F = np.linalg.inv(eta * invQ - 2.0 * Raa)
        f = eta * invQ @ b + ra

        _, q_lgdt = np.linalg.slogdet(2.0 * np.pi * Q)
        _, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * F)

        g = eta * eps - omega * beta + 0.5 * (f.T @ F @ f - eta * b.T @ invQ @ b
                                              - eta * q_lgdt
                                              + (eta + omega) * f_lgdt)
        return g

    def grad(self, var, eps, beta, ctl, model):
        eta = var[0]
        omega = var[1]

        Raa, ra, r0 = model.Raa, model.ra, model.r0

        b, Q = ctl.mu, ctl.cov
        invQ = np.linalg.inv(Q)

        F = np.linalg.inv(eta * invQ - 2.0 * Raa)
        f = eta * invQ @ b + ra

        _, q_lgdt = np.linalg.slogdet(2.0 * np.pi * Q)
        _, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * F)

        dF_deta = - F.T @ invQ @ F
        df_deta = invQ @ b

        deta = eps + 0.5 * (2.0 * f.T @ F @ df_deta + f.T @ dF_deta @ f
                            - b.T @ invQ @ b - q_lgdt
                            + f_lgdt + self.dim_action - (eta + omega) * np.trace(F @ invQ))

        domega = - beta + 0.5 * (f_lgdt + self.dim_action)

        return np.hstack((deta, domega))

    def run(self):
        # update entropy bound
        ent_bound = self.ent_rate * (self.ctl.entropy() + self.h0) - self.h0

        # sample current policy
        self.data = self.sample(self.n_samples)
        rwrd = np.mean(self.data['r'])

        # fit quadratic model
        self.model.fit(self.data['x'], self.data['r'])

        # optimize dual
        var = np.stack((100.0, 1000.0))
        bnds = ((1e-8, 1e8), (1e-8, 1e8))

        res = sc.optimize.minimize(self.dual, var,
                                   method='L-BFGS-B',
                                   jac=self.grad,
                                   args=(self.kl_bound, ent_bound,
                                         self.ctl, self.model),
                                   bounds=bnds)
        self.eta = res.x[0]
        self.omega = res.x[1]

        # update policy
        pi = self.ctl.update(self.eta, self.omega, self.model)

        # check kl
        kl = self.ctl.kli(pi)

        self.ctl = pi
        ent = self.ctl.entropy()

        return rwrd, kl, ent


if __name__ == "__main__":

    # np.random.seed(1337)

    more = MORE(func=Sphere(dim_action=2), n_samples=1000,
                kl_bound=0.05, ent_rate=0.99,
                cov0=100.0, h0=75.0)

    for it in range(1500):
        rwrd, kl, ent = more.run()

        print('it=', it, f'rwrd={rwrd:{5}.{4}}',
              f'kl={kl:{5}.{4}}', f'ent={ent:{5}.{4}}')
