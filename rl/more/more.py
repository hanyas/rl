import autograd.numpy as np

import scipy as sc
from scipy import optimize

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import copy


class Policy:

    def __init__(self, dm_act, cov0):
        self.dm_act = dm_act

        self.mu = 0.0 * np.random.randn(dm_act)
        self.cov = cov0 * np.eye(dm_act)

    def action(self, n):
        return np.random.multivariate_normal(self.mu, self.cov, size=(n))

    def kli(self, pi):
        diff = self.mu - pi.mu

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) + diff.T @ np.linalg.inv(self.cov) @ diff
                    - self.dm_act + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
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
        pol.cov = F * (eta + omega) + np.eye(self.dm_act) * 1e-24

        return pol


class Model:

    def __init__(self, dm_act):
        self.dm_act = dm_act

        self.Raa = np.zeros((dm_act, dm_act))
        self.ra = np.zeros((dm_act,))
        self.r0 = np.zeros((1, ))

    def fit(self, x, r):
        poly = PolynomialFeatures(2)
        feat = poly.fit_transform(x)

        reg = Ridge(alpha=1e-8, fit_intercept=False)
        reg.fit(feat, r)
        par = reg.coef_

        uid = np.triu_indices(self.dm_act)
        self.Raa[uid] = par[1 + self.dm_act:]
        self.Raa.T[uid] = self.Raa[uid]

        self.ra = par[1: 1 + self.dm_act]
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

    def __init__(self, func, nb_samples,
                 kl_bound, ent_rate, **kwargs):

        self.func = func
        self.dm_act = self.func.dm_act

        self.nb_samples = nb_samples

        self.kl_bound = kl_bound
        self.ent_rate = ent_rate

        if 'cov0' in kwargs:
            cov0 = kwargs.get('cov0', False)
            self.ctl = Policy(self.dm_act, cov0)
        else:
            self.ctl = Policy(self.dm_act, 100.0)

        if 'h0' in kwargs:
            self.h0 = kwargs.get('h0', False)
        else:
            self.h0 = 75.0

        self.model = Model(self.dm_act)

        self.eta = np.array([1.0])
        self.omega = np.array([1.0])

        self.data = {}

    def sample(self, nb_samples):
        data = {'x': self.ctl.action(nb_samples)}
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
                            + f_lgdt + self.dm_act - (eta + omega) * np.trace(F @ invQ))

        domega = - beta + 0.5 * (f_lgdt + self.dm_act)

        return np.hstack((deta, domega))

    def run(self, nb_iter=100, verbose=False):
        _trace = {'rwrd': [],
                  'kl': [],
                  'ent': []}

        for it in range(nb_iter):
            # update entropy bound
            ent_bound = self.ent_rate * (self.ctl.entropy() + self.h0) - self.h0

            # sample current policy
            self.data = self.sample(self.nb_samples)
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

            _trace['rwrd'].append(rwrd)
            _trace['kl'].append(kl)
            _trace['ent'].append(ent)

            if verbose:
                print('it=', it,
                      f'rwrd={rwrd:{5}.{4}}',
                      f'kl={kl:{5}.{4}}',
                      f'ent={ent:{5}.{4}}')

            if ent < -3e2:
                break

        return _trace
