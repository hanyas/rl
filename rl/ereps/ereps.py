import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import stats

import copy

EXP_MAX = 700.0
EXP_MIN = -700.0


class Policy:

    def __init__(self, dm_act, cov0):
        self.dm_act = dm_act

        self.mu = np.random.randn(self.dm_act)
        self.cov = cov0 * np.eye(self.dm_act)

    def action(self, n):
        aux = sc.stats.multivariate_normal(mean=self.mu, cov=self.cov).rvs(n)
        return aux.reshape((n, self.dm_act))

    def loglik(self, pi, x):
        mu, cov = pi.mu, pi.cov
        c = mu.shape[0] * np.log(2.0 * np.pi)

        ans = - 0.5 * (np.einsum('nk,kh,nh->n', mu - x, np.linalg.inv(cov), mu - x) +
                       np.log(np.linalg.det(cov)) + c)
        return ans

    def kli(self, pi):
        diff = self.mu - pi.mu

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) + diff.T @ np.linalg.inv(self.cov) @ diff
                    - self.dm_act + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def klm(self, pi):
        diff = pi.mu - self.mu

        kl = 0.5 * (np.trace(np.linalg.inv(pi.cov) @ self.cov) + diff.T @ np.linalg.inv(pi.cov) @ diff
                    - self.dm_act + np.log(np.linalg.det(pi.cov) / np.linalg.det(self.cov)))
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


class eREPS:

    def __init__(self, func, n_episodes,
                 kl_bound, **kwargs):

        self.func = func
        self.dm_act = self.func.dm_act

        self.n_episodes = n_episodes
        self.kl_bound = kl_bound

        cov0 = kwargs.get('cov0', 100.0)
        self.ctl = Policy(self.dm_act, cov0)

        self.data = None
        self.w = None
        self.eta = np.array([1.0])

    def sample(self, n_episodes):
        data = {'x': self.ctl.action(n_episodes)}
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

    def run(self, nb_iter=100, verbose=False):
        _trace = {'rwrd': [],
                  'kls': [], 'kli': [], 'klm': [],
                  'ent': []}

        for it in range(nb_iter):
            self.data = self.sample(self.n_episodes)
            rwrd = np.mean(self.data['r'])

            res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                       method='SLSQP',
                                       jac=self.grad,
                                       # jac=grad(self.dual),
                                       args=(self.kl_bound,
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

            _trace['rwrd'].append(rwrd)
            _trace['kls'].append(kls)
            _trace['kli'].append(kli)
            _trace['klm'].append(klm)
            _trace['ent'].append(ent)

            if verbose:
                print('it=', it,
                      f'rwrd={rwrd:{5}.{4}}',
                      f'kls={kls:{5}.{4}}',
                      f'kli={kli:{5}.{4}}',
                      f'klm={klm:{5}.{4}}',
                      f'ent={ent:{5}.{4}}')

            if ent < -3e2:
                break

        return _trace
