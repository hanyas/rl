import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import stats

from sklearn.preprocessing import PolynomialFeatures

import copy

EXP_MAX = 700.0
EXP_MIN = -700.0


class Sphere:

    def __init__(self, dim_action):
        self.dim_action = dim_action

        M = np.random.randn(dim_action, dim_action)
        tmp = M @ M.T
        Q = tmp[np.nonzero(np.triu(tmp))]

        q = 0.0 * np.random.rand(dim_action)
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
                        (1 - x[: ,:-1]) ** 2.0, axis=-1)


class Styblinski:
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def eval(self, x):
        return - 0.5 * np.sum(x**4.0 - 16.0 * x**2 + 5 * x, axis=-1)


class Rastrigin:
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def eval(self, x):
        return - (10.0 * self.dim_action +
                  np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1))


class Policy:

    def __init__(self, dim_action, cov0):
        self.dim_action = dim_action

        self.mu = np.random.randn(self.dim_action)
        self.cov = cov0 * np.eye(self.dim_action)

    def action(self, n):
        aux = sc.stats.multivariate_normal(mean=self.mu, cov=self.cov).rvs(n)
        return aux.reshape((n, self.dim_action))

    def loglik(self, pi, x):
        mu, cov = pi.mu, pi.cov
        c = mu.shape[0] * np.log(2.0 * np.pi)

        ans = - 0.5 * (np.einsum('nk,kh,nh->n', mu - x, np.linalg.inv(cov), mu - x) +
                       np.log(np.linalg.det(cov)) + c)
        return ans

    def kli(self, pi):
        diff = self.mu - pi.mu

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) + diff.T @ np.linalg.inv(self.cov) @ diff
                    - self.dim_action + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def klm(self, pi):
        diff = pi.mu - self.mu

        kl = 0.5 * (np.trace(np.linalg.inv(pi.cov) @ self.cov) + diff.T @ np.linalg.inv(pi.cov) @ diff
                    - self.dim_action + np.log(np.linalg.det(pi.cov) / np.linalg.det(self.cov)))
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def wml(self, x, w, eta=np.array([0.0])):
        pol = copy.deepcopy(self)

        pol.mu = (np.sum(w[:, np.newaxis] * x, axis=0) + eta * self.mu) / (np.sum(w, axis=0) + eta)

        diff = x - pol.mu
        tmp = np.einsum('nk,n,nh->nkh', diff, w, diff)
        pol.cov = (np.sum(tmp, axis=0) + eta * self.cov +
               eta * np.outer(pol.mu - self.mu, pol.mu - self.mu)) / (np.sum(w, axis=0) + eta)

        return pol

    def dual(self, eta, x, w, eps):
        pol = self.wml(x, w, eta)
        return np.sum(w * self.loglik(pol, x)) + eta * (eps - self.klm(pol))

    def wmap(self, x, w, eps=np.array([0.1])):
        res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                   method='SLSQP',
                                   jac=grad(self.dual),
                                   args=(x, w, eps),
                                   bounds=((1e-8, 1e8),))
        eta = res['x']
        pol = self.wml(x, w, eta)

        return pol


class EREPS:

    def __init__(self, func, n_samples,
                 kl_bound, **kwargs):

        self.func = func
        self.dim_action =self.func.dim_action

        self.n_samples = n_samples
        self.kl_bound = kl_bound

        if 'cov0' in kwargs:
            cov0 = kwargs.get('cov0', False)
            self.ctl = Policy(self.dim_action, cov0)
        else:
            self.ctl = Policy(self.dim_action, 100.0)

        self.data = {}
        self.w = None

    def sample(self, n_samples):
        data = {}

        data['x'] = self.ctl.action(n_samples)
        data['r'] = self.func.eval(data['x'])

        return data

    def weights(self, r, eta):
        adv = r - np.max(r)
        w = np.exp(np.clip(adv / eta, EXP_MIN, EXP_MAX))
        return w, adv

    def dual(self, eta, eps, r):
        w, _ = self.weights(r, eta)
        g = eta * eps + np.max(r) + eta * np.log(np.mean(w, axis=0))
        return g

    def grad(self, eta, eps, r):
        w, adv = self.weights(r, eta)
        dg = eps + np.log(np.mean(w, axis=0)) - \
            np.sum(w * adv, axis=0) / (eta * np.sum(w, axis=0))
        return dg

    def kl_samples(self, w):
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self):
        self.data = self.sample(self.n_samples)
        rwrd = np.mean(self.data['r'])

        res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                   method='SLSQP',
                                   jac=self.grad,
                                   # jac=grad(self.dual),
                                   args=(
                                       self.kl_bound,
                                       self.data['r']),
                                   bounds=((1e-8, 1e8),))
        self.eta = res.x
        self.w, _ = self.weights(self.data['r'], self.eta)

        # pol = self.ctl.wml(self.data['x'], self.w)
        pol = self.ctl.wmap(self.data['x'], self.w, eps=self.kl_bound)

        kls = self.kl_samples(self.w)
        kli = self.ctl.kli(pol)
        klm = self.ctl.klm(pol)

        self.ctl = pol
        ent = self.ctl.entropy()

        return rwrd, kls, kli, klm, ent


if __name__ == "__main__":

    # np.random.seed(1337)

    ereps = EREPS(func=Sphere(dim_action=5),
                  n_samples=25, kl_bound=0.1,
                  cov0=100.0)

    # ereps = EREPS(func=Rosenbrock(dim_action=3),
    #               n_samples=1000, kl_bound=0.05,
    #               cov0=100.0)
    #
    # ereps = EREPS(func=Styblinski(dim_action=3),
    #               n_samples=1000, kl_bound=0.05,
    #               cov0=100.0)
    #
    # ereps = EREPS(func=Rastrigin(dim_action=3),
    #               n_samples=1000, kl_bound=0.05,
    #               cov0=100.0)

    for it in range(250):
        rwrd, kls, kli, klm, ent = ereps.run()

        print('it=', it, f'rwrd={rwrd:{5}.{4}}',
              f'kls={kls:{5}.{4}}', f'kli={kli:{5}.{4}}',
              f'klm={klm:{5}.{4}}', f'ent={ent:{5}.{4}}')
