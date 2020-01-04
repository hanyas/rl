import autograd.numpy as np
from autograd import grad, jacobian

import scipy as sc
from scipy import optimize
from scipy import stats
from scipy import special

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import random
import copy


EXP_MAX = 700.0
EXP_MIN = -700.0


def merge(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]

    for key in d:
        d[key] = np.concatenate(d[key])

    return d


class FourierFeatures:

    def __init__(self, dm_state, nb_feat, band, mult):
        self.dm_state = dm_state
        self.nb_feat = nb_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.dm_state),
                                                  cov=np.diag(1.0 / (mult * band)),
                                                  size=self.nb_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.nb_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        return phi


class Policy:

    def __init__(self, dm_state, dm_act, **kwargs):
        self.dm_state = dm_state
        self.dm_act = dm_act

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.mult = kwargs.get('mult', False)
            self.nb_feat = kwargs.get('nb_feat', False)
            self.basis = FourierFeatures(self.dm_state, self.nb_feat,
                                         self.band, self.mult)
        else:
            self.degree = kwargs.get('degree', False)
            self.nb_feat = int(sc.special.comb(self.degree + self.dm_state, self.degree))
            self.basis = PolynomialFeatures(self.degree)

        self.K = 1e-8 * np.random.randn(self.dm_act, self.nb_feat)
        self.cov = np.eye(dm_act)

    def features(self, x):
        return self.basis.fit_transform(x.reshape(-1, self.dm_state)).squeeze()

    def mean(self, x):
        feat = self.features(x)
        return np.einsum('...k,mk->...m', feat, self.K)

    def actions(self, x, stoch):
        mean = self.mean(x)
        if stoch:
            return np.random.multivariate_normal(mean, self.cov)
        else:
            return mean

    def loglik(self, pi, x, u):
        mu, cov = pi.mean(x), pi.cov
        c = mu.shape[-1] * np.log(2.0 * np.pi)

        ans = - 0.5 * (np.einsum('nk,kh,nh->n', u - mu, np.linalg.inv(cov), u - mu) +
                       np.log(np.linalg.det(cov)) + c)
        return ans

    def kli(self, pi, x):
        diff = self.mean(x) - pi.mean(x)

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) +
                    np.mean(np.einsum('nk,kh,nh->n', diff, np.linalg.inv(self.cov), diff), axis=0) -
                    self.dm_act + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def klm(self, pi, x):
        diff = pi.mean(x) - self.mean(x)

        kl = 0.5 * (np.trace(np.linalg.inv(pi.cov) @ self.cov) +
                    np.mean(np.einsum('nk,kh,nh->n', diff, np.linalg.inv(pi.cov), diff), axis=0) -
                    self.dm_act + np.log(np.linalg.det(pi.cov) / np.linalg.det(self.cov)))
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def wml(self, x, u, w, preg):
        pol = copy.deepcopy(self)

        psi = self.features(x)

        reg = Ridge(alpha=preg, fit_intercept=False)
        reg.fit(psi, u, sample_weight=w)
        pol.K = reg.coef_

        # _inv = np.linalg.pinv(psi.T @ np.diag(w) @ psi + preg * np.eye(psi.shape[1]))
        # pol.K = u.T @ np.diag(w) @ psi @ _inv

        std = u - pol.mean(x)
        pol.cov = np.sum(np.einsum('nk,n,nh->nkh', std, w, std), axis=0) / np.sum(w)

        return pol

    def rwml(self, x, u, w, preg, eta=np.array([0.0])):
        pol = copy.deepcopy(self)

        psi = self.features(x)

        _inv = np.linalg.inv(psi.T @ np.diag(w + eta) @ psi + preg * np.eye(psi.shape[1]))
        pol.K = (u.T @ np.diag(w) + eta * self.mean(x).T) @ psi @ _inv

        std = u - pol.mean(x)
        tmp = np.mean(np.einsum('nk,n,nh->nkh', std, w, std), axis=0)

        diff = self.mean(x) - pol.mean(x)
        aux = eta * np.mean(np.einsum('nk,nh->nkh', diff, diff), axis=0)

        pol.cov = (tmp + aux + eta * self.cov) / (np.mean(w) + eta)
        return pol

    def dual(self, eta, x, u, w, preg, eps):
        pol = self.rwml(x, u, w, preg=preg, eta=eta)
        return np.mean(w * self.loglik(pol, x, u)) + eta * (eps - self.klm(pol, x))

    def wmap(self, x, u, w, preg, eps):
        res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                   method='SLSQP',
                                   # jac=grad(self.dual),
                                   args=(x, u, w, preg, eps),
                                   bounds=((1e-8, 1e8),),
                                   options={'maxiter': 250,
                                            'ftol': 1e-06})
        eta = res['x']
        pol = self.rwml(x, u, w, preg=preg, eta=eta)

        return pol


class Vfunction:

    def __init__(self, dm_state, **kwargs):
        self.dm_state = dm_state

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.mult = kwargs.get('mult', False)
            self.nb_feat = kwargs.get('nb_feat', False)
            self.basis = FourierFeatures(self.dm_state, self.nb_feat,
                                         self.band, self.mult)
        else:
            self.degree = kwargs.get('degree', False)
            self.nb_feat = int(sc.special.comb(self.degree + self.dm_state, self.degree))
            self.basis = PolynomialFeatures(self.degree)

        self.omega = 1e-8 * np.random.randn(self.nb_feat)

    def features(self, x):
        return self.basis.fit_transform(x)

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class REPS:

    def __init__(self, env,
                 nb_samples, nb_keep,
                 nb_rollouts, nb_steps,
                 kl_bound, discount,
                 vreg, preg, cov0,
                 **kwargs):

        self.env = env

        self.dm_state = self.env.observation_space.shape[0]
        self.dm_act = self.env.action_space.shape[0]

        self.nb_samples = nb_samples
        self.nb_keep = nb_keep

        self.nb_rollouts = nb_rollouts
        self.nb_steps = nb_steps

        self.kl_bound = kl_bound
        self.discount = discount

        self.vreg = vreg
        self.preg = preg

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.mult = kwargs.get('mult', False)

            self.nb_vfeat = kwargs.get('nb_vfeat', False)
            self.nb_pfeat = kwargs.get('nb_pfeat', False)

            self.vfunc = Vfunction(self.dm_state, nb_feat=self.nb_vfeat,
                                   band=self.band, mult=self.mult)

            self.ctl = Policy(self.dm_state, self.dm_act, nb_feat=self.nb_pfeat,
                              band=self.band,  mult=self.mult)
        else:
            self.vdgr = kwargs.get('vdgr', False)
            self.pdgr = kwargs.get('pdgr', False)

            self.vfunc = Vfunction(self.dm_state, degree=self.vdgr)
            self.nb_vfeat = self.vfunc.nb_feat

            self.ctl = Policy(self.dm_state, self.dm_act, degree=self.pdgr)
            self.nb_pfeat = self.ctl.nb_feat

        self.ctl.cov = cov0 * self.ctl.cov
        self.ulim = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.features = None
        self.w = None

        self.eta = np.array([1.0])

    def sample(self, nb_samples, nb_keep=0, reset=True, stoch=True, render=False):
        if len(self.rollouts) >= nb_keep:
            rollouts = random.sample(self.rollouts, nb_keep)
        else:
            rollouts = []

        coin = sc.stats.binom(1, 1.0 - self.discount)

        n = 0
        while True:
            roll = {'xi': np.empty((0, self.dm_state)),
                    'x': np.empty((0, self.dm_state)),
                    'u': np.empty((0, self.dm_act)),
                    'xn': np.empty((0, self.dm_state)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            x = self.env.reset()

            roll['xi'] = np.vstack((roll['xi'], x))
            roll['done'] = np.hstack((roll['done'], False))

            done = False
            while not done:
                u = self.ctl.actions(x, stoch)
                # u = np.clip(u, -self.ulim, self.ulim)

                if reset and coin.rvs():
                    done = True
                    roll['done'][-1] = done
                else:
                    roll['x'] = np.vstack((roll['x'], x))
                    roll['u'] = np.vstack((roll['u'], u))

                    x, r, done, _ = self.env.step(np.clip(u, -self.ulim, self.ulim))
                    if render:
                        self.env.render()

                    roll['xn'] = np.vstack((roll['xn'], x))
                    roll['r'] = np.hstack((roll['r'], r))
                    roll['done'] = np.hstack((roll['done'], done))

                    n = n + 1
                    if n >= nb_samples:
                        roll['done'][-1] = True
                        rollouts.append(roll)
                        data = merge(*rollouts)
                        return rollouts, data

            rollouts.append(roll)

    def evaluate(self, nb_rollouts, nb_steps, stoch=False, render=False):
        rollouts = []

        for n in range(nb_rollouts):
            roll = {'x': np.empty((0, self.dm_state)),
                    'u': np.empty((0, self.dm_act)),
                    'r': np.empty((0,))}

            x = self.env.reset()

            for t in range(nb_steps):
                u = self.ctl.actions(x, stoch)
                # u = np.clip(u, -self.ulim, self.ulim)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, -self.ulim, self.ulim))
                if render:
                    self.env.render()

                roll['r'] = np.hstack((roll['r'], r))

            rollouts.append(roll)

        data = merge(*rollouts)
        return rollouts, data

    def featurize(self, data):
        ivfeatures = np.mean(self.vfunc.features(data['xi']),
                             axis=0, keepdims=True)
        vfeatures = self.vfunc.features(data['x'])
        nvfeatures = self.vfunc.features(data['xn'])
        features = self.discount * nvfeatures - vfeatures\
                   + (1.0 - self.discount) * ivfeatures
        return features

    def weights(self, eta, omega, features, rwrd):
        adv = rwrd + np.dot(features, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return w, delta, np.max(adv)

    def dual(self, var, epsilon, phi, r):
        eta, omega = var[0], var[1:]
        w, _, max_adv = self.weights(eta, omega, phi, r)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(w, axis=0))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def grad(self, var, epsilon, phi, r):
        eta, omega = var[0], var[1:]
        w, delta, max_adv = self.weights(eta, omega, phi, r)

        deta = epsilon + np.log(np.mean(w, axis=0)) - \
               np.sum(w * delta, axis=0) / (eta * np.sum(w, axis=0))

        domega = np.sum(w[:, np.newaxis] * phi, axis=0) / np.sum(w, axis=0)
        domega = domega + self.vreg * 2 * omega

        return np.hstack((deta, domega))

    def dual_eta(self, eta, omega, epsilon, phi, r):
        w, _, max_adv = self.weights(eta, omega, phi, r)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(w, axis=0))
        return g

    def grad_eta(self, eta, omega, epsilon, phi, r):
        w, delta, max_adv = self.weights(eta, omega, phi, r)
        deta = epsilon + np.log(np.mean(w, axis=0)) - \
               np.sum(w * delta, axis=0) / (eta * np.sum(w, axis=0))
        return deta

    def dual_omega(self, omega, eta, phi, r):
        w, delta, max_adv = self.weights(eta, omega, phi, r)
        g = max_adv + eta * np.log(np.mean(w, axis=0))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def grad_omega(self, omega, eta, phi, r):
        w, delta, max_adv = self.weights(eta, omega, phi, r)
        domega = np.sum(w[:, np.newaxis] * phi, axis=0) / np.sum(w, axis=0)
        domega = domega + self.vreg * 2 * omega
        return domega

    def hess_omega(self, omega, eta, phi, r):
        w, delta, max_adv = self.weights(eta, omega, phi, r)
        w = w / np.sum(w, axis=0)
        aux = np.sum(w[:, np.newaxis] * phi, axis=0)
        tmp = phi - aux
        homega = 1.0 / eta * np.einsum('n,nk,nh->kh', w, tmp, tmp)
        return homega

    def kl_samples(self, w):
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self, nb_iter=10, verbose=False):
        _trace = {'rwrd': [],
                  'kls': [], 'kli': [], 'klm': [],
                  'ent': []}

        for it in range(nb_iter):
            _, eval = self.evaluate(self.nb_rollouts, self.nb_steps)

            self.rollouts, self.data = self.sample(self.nb_samples, self.nb_keep)
            self.features = self.featurize(self.data)

            res = sc.optimize.minimize(self.dual,
                                       np.hstack((1.0, 1e-8 * np.random.randn(self.nb_vfeat))),
                                       # np.hstack((1.0, self.vfunc.omega)),
                                       method='L-BFGS-B',
                                       jac=grad(self.dual),
                                       args=(self.kl_bound,
                                             self.features,
                                             self.data['r']),
                                       bounds=((1e-8, 1e8), ) + ((-np.inf, np.inf), ) * self.nb_vfeat)
            self.eta, self.vfunc.omega = res.x[0], res.x[1:]

            # self.eta, self.vfunc.omega = 1.0, 1e-8 * np.random.randn(self.nb_vfeat)
            # for _ in range(250):
            #     res = sc.optimize.minimize(self.dual_eta,
            #                                self.eta,
            #                                method='L-BFGS-B',
            #                                jac=grad(self.dual_eta),
            #                                args=(self.vfunc.omega,
            #                                      self.kl_bound,
            #                                      self.features,
            #                                      self.data['r']),
            #                                bounds=((1e-8, 1e8),),
            #                                options={'maxiter': 5})
            #     # print(res)
            #     #
            #     # check = sc.optimize.check_grad(self.dual_eta,
            #     #                                self.grad_eta,
            #     #                                res.x,
            #     #                                self.vfunc.omega,
            #     #                                self.kl_bound,
            #     #                                self.features,
            #     #                                self.data['r'])
            #     # print('Eta Error', check)
            #
            #     self.eta = res.x
            #
            #     res = sc.optimize.minimize(self.dual_omega,
            #                                self.vfunc.omega,
            #                                method='BFGS',
            #                                jac=grad(self.dual_omega),
            #                                args=(self.eta,
            #                                      self.features,
            #                                      self.data['r']),
            #                                options={'maxiter': 250})
            #
            #     # res = sc.optimize.minimize(self.dual_omega,
            #     #                            self.vfunc.omega,
            #     #                            method='trust-exact',
            #     #                            jac=grad(self.dual_omega),
            #     #                            hess=jacobian(grad(self.dual_omega)),
            #     #                            args=(self.eta,
            #     #                                  self.features,
            #     #                                  self.data['r']))
            #     # # print(res)
            #     #
            #     # check = sc.optimize.check_grad(self.dual_omega,
            #     #                                self.grad_omega,
            #     #                                res.x,
            #     #                                self.eta,
            #     #                                self.features,
            #     #                                self.data['r'])
            #     # print('Omega Error', check)
            #
            #     self.vfunc.omega = res.x

            self.w, _, _ = self.weights(self.eta, self.vfunc.omega, self.features, self.data['r'])

            pol = self.ctl.wml(self.data['x'], self.data['u'], self.w, preg=self.preg)
            # pol = self.ctl.wmap(self.data['x'], self.data['u'], self.w, preg=self.preg, eps=self.kl_bound)

            kls = self.kl_samples(self.w)
            kli = self.ctl.kli(pol, self.data['x'])
            klm = self.ctl.klm(pol, self.data['x'])

            self.ctl = pol
            ent = self.ctl.entropy()

            # rwrd = np.mean(self.data['r'])
            rwrd = np.mean(eval['r'])

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

        return _trace
