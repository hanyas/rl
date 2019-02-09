import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import stats
from scipy import special

from sklearn.preprocessing import PolynomialFeatures

import copy

EXP_MAX = 700.0
EXP_MIN = -700.0


class Sphere:

    def __init__(self, dim_cntxt, dim_action):
        self.dim_cntxt = dim_cntxt
        self.dim_action = dim_action

        M = np.random.randn(self.dim_action, self.dim_action)
        M = 0.5 * (M + M.T)
        self.Q = M @ M.T

    def context(self, n_samples):
        return np.random.uniform(-1.0, 1.0, size=(n_samples, self.dim_cntxt))

    def eval(self, x, c):
        diff = x - c
        return - np.einsum('nk,kh,nh->n', diff, self.Q, diff)


class Policy:

    def __init__(self, dim_cntxt, dim_action, degree, cov0):
        self.dim_cntxt = dim_cntxt
        self.dim_action = dim_action

        self.degree = degree
        self.basis = PolynomialFeatures(self.degree)
        self.n_feat = int(sc.special.comb(self.degree + self.dim_cntxt, self.degree))

        self.K = 1e-8 * np.random.randn(self.dim_action, self.n_feat)
        self.cov = cov0 * np.eye(dim_action)

    def features(self, c):
        return self.basis.fit_transform(c.reshape(-1, self.dim_cntxt))

    def mean(self, c):
        feat = self.features(c)
        return np.einsum('...k,mk->...m', feat, self.K)

    def action(self, c, stoch=True):
        mean = self.mean(c)
        if stoch:
            return np.array([np.random.multivariate_normal(mu, self.cov) for mu in mean])
        else:
            return mean

    def loglik(self, pi, c, x):
        mu, cov = pi.mean(c), pi.cov
        cnst = mu.shape[-1] * np.log(2.0 * np.pi)

        ans = - 0.5 * (np.einsum('nk,kh,nh->n', x - mu, np.linalg.inv(cov), x - mu) +
                       np.log(np.linalg.det(cov)) + cnst)
        return ans

    def kli(self, pi, c):
        diff = self.mean(c) - pi.mean(c)

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) +
                    np.mean(np.einsum('nk,kh,nh->n', diff, np.linalg.inv(self.cov), diff), axis=0) -
                    self.dim_action + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def klm(self, pi, c):
        diff = pi.mean(c) - self.mean(c)

        kl = 0.5 * (np.trace(np.linalg.inv(pi.cov) @ self.cov) +
                    np.mean(np.einsum('nk,kh,nh->n', diff, np.linalg.inv(pi.cov), diff), axis=0) -
                    self.dim_action + np.log(np.linalg.det(pi.cov) / np.linalg.det(self.cov)))
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def wml(self, c, x, w, preg):
        pol = copy.deepcopy(self)

        psi = self.features(c)

        _inv = np.linalg.inv(psi.T @ np.diag(w) @ psi + preg * np.eye(psi.shape[1]))
        pol.K = x.T @ np.diag(w) @ psi @ _inv

        std = x - pol.mean(c)
        pol.cov = np.sum(np.einsum('nk,n,nh->nkh', std, w, std), axis=0) / np.sum(w)

        return pol

    def rwml(self, c, x, w, preg=0.0, eta=np.array([0.0])):
        pol = copy.deepcopy(self)

        psi = self.features(c)

        _inv = np.linalg.inv(psi.T @ np.diag(w + eta) @ psi + preg * np.eye(psi.shape[1]))
        pol.K = (x.T @ np.diag(w) + eta * self.mean(c).T) @ psi @ _inv

        std = x - pol.mean(c)
        tmp = np.mean(np.einsum('nk,n,nh->nkh', std, w, std), axis=0)

        diff = self.mean(c) - pol.mean(c)
        aux = eta * np.mean(np.einsum('nk,nh->nkh', diff, diff), axis=0)

        pol.cov = (tmp + aux + eta * self.cov) / (np.mean(w) + eta)
        return pol

    def dual(self, eta, c, x, w, eps):
        pol = self.rwml(c, x, w, eta=eta)
        return np.mean(w * self.loglik(pol, c, x)) + eta * (eps - self.klm(pol, c))

    def wmap(self, c, x, w, eps):
        res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                   method='SLSQP',
                                   # jac=grad(self.dual),
                                   args=(c, x, w, eps),
                                   bounds=((1e-8, 1e8),),
                                   options={'maxiter': 250,
                                            'ftol': 1e-06})
        eta = res['x']
        pol = self.rwml(c, x, w, eta=eta)

        return pol


class Vfunction:

    def __init__(self, dim_cntxt, degree):
        self.dim_cntxt = dim_cntxt

        self.degree = degree
        self.n_feat = int(
            sc.special.comb(self.degree + self.dim_cntxt, self.degree))
        self.basis = PolynomialFeatures(self.degree)

        self.omega = 1e-8 * np.random.randn(self.n_feat)

    def features(self, c):
        return self.basis.fit_transform(c)

    def values(self, c):
        feat = self.features(c)
        return np.dot(feat, self.omega)


class CREPS:

    def __init__(self, func,
                 n_samples, kl_bound,
                 vdgr, pdgr,
                 vreg, preg, **kwargs):

        self.func = func
        self.dim_action = self.func.dim_action
        self.dim_cntxt = self.func.dim_cntxt

        self.n_samples = n_samples
        self.kl_bound = kl_bound

        self.vreg = vreg
        self.preg = preg

        self.vdgr = vdgr
        self.pdgr = pdgr

        if 'cov0' in kwargs:
            cov0 = kwargs.get('cov0', False)
            self.ctl = Policy(self.dim_action, self.dim_cntxt,
                              self.pdgr, cov0)
        else:
            self.ctl = Policy(self.dim_action, self.dim_cntxt,
                              self.pdgr, 100.0)

        self.n_pfeat = self.ctl.n_feat

        self.vfunc = Vfunction(self.dim_cntxt, self.vdgr)
        self.n_vfeat = self.vfunc.n_feat

        self.eta = np.array([1.0])

        self.data = {}
        self.vfeatures = None
        self.w = None

    def sample(self, n_samples):
        data = {}

        data['c'] = self.func.context(n_samples)
        data['x'] = self.ctl.action(data['c'])
        data['r'] = self.func.eval(data['c'], data['x'])

        return data

    def weights(self, eta, omega, r, phi):
        adv = r - np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return w, delta, np.max(adv)

    def dual(self, var, eps, r, phi):
        eta, omega = var[0], var[1:]
        w, delta, max_adv = self.weights(eta, omega, r, phi)
        g = eta * eps + max_adv + np.dot(np.mean(phi, axis=0), omega) +\
            eta * np.log(np.mean(w, axis=0))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def grad(self, var, eps, r, phi):
        eta, omega = var[0], var[1:]
        w, delta, max_adv = self.weights(eta, omega, r, phi)

        deta = eps + np.log(np.mean(w, axis=0)) - \
               np.sum(w * delta, axis=0) / (eta * np.sum(w, axis=0))

        domega = np.mean(phi, axis=0) - \
                 np.sum(w[:, np.newaxis] * phi, axis=0) / np.sum(w, axis=0)
        domega = domega + self.vreg * 2 * omega

        return np.hstack((deta, domega))

    def kl_samples(self, w):
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self):
        self.data = self.sample(self.n_samples)
        rwrd = np.mean(self.data['r'])

        self.vfeatures = self.vfunc.features(self.data['c'])

        res = sc.optimize.minimize(self.dual,
                                   np.hstack((1.0, 1e-8 * np.random.randn(self.n_vfeat))),
                                   method='L-BFGS-B',
                                   jac=self.grad,
                                   # jac=grad(self.dual),
                                   args=(
                                       self.kl_bound,
                                       self.data['r'],
                                       self.vfeatures),
                                   bounds=((1e-8, 1e8), ) + ((-np.inf, np.inf), ) * self.n_vfeat)

        self.eta, self.vfunc.omega = res.x[0], res.x[1:]
        self.w, _, _ = self.weights(self.eta, self.vfunc.omega,
                                    self.data['r'], self.vfeatures)

        # pol = self.ctl.wml(self.data['c'], self.data['x'], self.w, self.preg)
        pol = self.ctl.wmap(self.data['c'], self.data['x'], self.w, eps=self.kl_bound)

        kls = self.kl_samples(self.w)
        kli = self.ctl.kli(pol, self.data['c'])
        klm = self.ctl.klm(pol, self.data['c'])

        self.ctl = pol
        ent = self.ctl.entropy()

        return rwrd, kls, kli, klm, ent


if __name__ == "__main__":

    # np.random.seed(1337)

    creps = CREPS(func=Sphere(dim_cntxt=3, dim_action=3),
                  n_samples=100, kl_bound=0.1,
                  vdgr=2, pdgr=1,
                  vreg=1e-16, preg=1e-16,
                  cov0=100.0)

    for it in range(250):
        rwrd, kls, kli, klm, ent = creps.run()

        print('it=', it, f'rwrd={rwrd:{5}.{4}}',
              f'kls={kls:{5}.{4}}', f'kli={kli:{5}.{4}}',
              f'klm={klm:{5}.{4}}', f'ent={ent:{5}.{4}}')