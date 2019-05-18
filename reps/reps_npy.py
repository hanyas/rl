import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import autograd.numpy as np
from autograd import grad, jacobian

import scipy as sc
from scipy import optimize
from scipy import stats
from scipy import special

from sklearn.preprocessing import PolynomialFeatures

import random
import copy


EXP_MAX = 700.0
EXP_MIN = -700.0


def merge_dicts(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]

    for key in d:
        d[key] = np.concatenate(d[key]).squeeze()

    return d


class FourierFeatures:

    def __init__(self, dim_state, n_feat, band):
        self.dim_state = dim_state
        self.n_feat = n_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.dim_state),
                                                  cov=np.diag(1.0 / band),
                                                  size=self.n_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.n_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        return phi


class Policy:

    def __init__(self, dim_state, dim_action, **kwargs):
        self.dim_state = dim_state
        self.dim_action = dim_action

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.dim_state, self.n_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.n_feat = int(
                sc.special.comb(self.degree + self.dim_state, self.degree))
            self.basis = PolynomialFeatures(self.degree)

        self.K = 1e-8 * np.random.randn(self.dim_action, self.n_feat)
        self.cov = np.eye(dim_action)

    def features(self, x):
        return self.basis.fit_transform(x.reshape(-1, self.dim_state)).squeeze()

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
                    self.dim_action + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def klm(self, pi, x):
        diff = pi.mean(x) - self.mean(x)

        kl = 0.5 * (np.trace(np.linalg.inv(pi.cov) @ self.cov) +
                    np.mean(np.einsum('nk,kh,nh->n', diff, np.linalg.inv(pi.cov), diff), axis=0) -
                    self.dim_action + np.log(np.linalg.det(pi.cov) / np.linalg.det(self.cov)))
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def wml(self, x, u, w, preg):
        pol = copy.deepcopy(self)

        psi = self.features(x)

        _inv = np.linalg.inv(psi.T @ np.diag(w) @ psi + preg * np.eye(psi.shape[1]))
        pol.K = u.T @ np.diag(w) @ psi @ _inv

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

    def __init__(self, dim_state, **kwargs):
        self.dim_state = dim_state

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.dim_state, self.n_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.n_feat = int(
                sc.special.comb(self.degree + self.dim_state, self.degree))
            self.basis = PolynomialFeatures(self.degree)

        self.omega = 1e-8 * np.random.randn(self.n_feat)

    def features(self, x):
        return self.basis.fit_transform(x)

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class REPS:

    def __init__(self, env,
                 n_samples, n_keep,
                 n_rollouts, n_steps,
                 kl_bound, discount,
                 vreg, preg, cov0,
                 **kwargs):

        self.env = env

        self.dim_state = self.env.observation_space.shape[0]
        self.dim_action = self.env.action_space.shape[0]

        self.n_samples = n_samples
        self.n_keep = n_keep

        self.n_rollouts = n_rollouts
        self.n_steps = n_steps

        self.kl_bound = kl_bound
        self.discount = discount

        self.vreg = vreg
        self.preg = preg

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)

            self.n_vfeat = kwargs.get('n_vfeat', False)
            self.n_pfeat = kwargs.get('n_pfeat', False)

            self.vfunc = Vfunction(self.dim_state, n_feat=self.n_vfeat, band=self.band)

            self.ctl = Policy(self.dim_state, self.dim_action, n_feat=self.n_pfeat, band=self.band)
        else:
            self.vdgr = kwargs.get('vdgr', False)
            self.pdgr = kwargs.get('pdgr', False)

            self.vfunc = Vfunction(self.dim_state, degree=self.vdgr)
            self.n_vfeat = self.vfunc.n_feat

            self.ctl = Policy(self.dim_state, self.dim_action, degree=self.pdgr)
            self.n_pfeat = self.ctl.n_feat

        self.ctl.cov = cov0 * self.ctl.cov
        self.action_limit = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.features = None
        self.w = None

        self.eta = np.array([1.0])

    def sample(self, n_samples, n_keep=0, reset=True, stoch=True, render=False):
        if len(self.rollouts) >= n_keep:
            rollouts = random.sample(self.rollouts, n_keep)
        else:
            rollouts = []

        coin = sc.stats.binom(1, 1.0 - self.discount)

        n = 0
        while True:
            roll = {'xi': np.empty((0, self.dim_state)),
                    'x': np.empty((0, self.dim_state)),
                    'u': np.empty((0, self.dim_action)),
                    'xn': np.empty((0, self.dim_state)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            rollouts.append(roll)

            x = self.env.reset()

            rollouts[-1]['xi'] = np.vstack((rollouts[-1]['xi'], x))
            rollouts[-1]['done'] = np.hstack((rollouts[-1]['done'], False))

            while True:
                u = self.ctl.actions(x, stoch)

                if reset and coin.rvs():
                    rollouts[-1]['done'][-1] = True
                    break
                else:
                    rollouts[-1]['x'] = np.vstack((rollouts[-1]['x'], x))
                    rollouts[-1]['u'] = np.vstack((rollouts[-1]['u'], u))

                    x, r, done, _ = self.env.step(
                        np.clip(u, - self.action_limit, self.action_limit))
                    if render:
                        self.env.render()

                    rollouts[-1]['xn'] = np.vstack((rollouts[-1]['xn'], x))
                    rollouts[-1]['r'] = np.hstack((rollouts[-1]['r'], r))
                    rollouts[-1]['done'] = np.hstack((rollouts[-1]['done'], done))

                    n = n + 1
                    if n >= n_samples:
                        rollouts[-1]['done'][-1] = True

                        data = merge_dicts(*rollouts)
                        data['u'] = np.reshape(data['u'], (-1, self.dim_action))

                        return rollouts, data

                    if done:
                        break

    def evaluate(self, n_rollouts, n_steps, stoch=False, render=False):
        rollouts = []

        for n in range(n_rollouts):
            roll = {'x': np.empty((0, self.dim_state)),
                    'u': np.empty((0, self.dim_action)),
                    'r': np.empty((0,))}

            x = self.env.reset()

            for t in range(n_steps):
                u = self.ctl.actions(x, stoch)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                roll['r'] = np.hstack((roll['r'], r))

                if done:
                    break

            rollouts.append(roll)

        data = merge_dicts(*rollouts)
        data['u'] = np.reshape(data['u'], (-1, self.dim_action))

        return rollouts, data

    def featurize(self, data):
        ivfeatures = np.mean(self.vfunc.features(data['xi']),
                                  axis=0, keepdims=True)
        vfeatures = self.vfunc.features(data['x'])
        nvfeatures = self.vfunc.features(data['xn'])
        features = self.discount * nvfeatures - vfeatures + \
                        (1.0 - self.discount) * ivfeatures
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

    def run(self):
        _, eval = self.evaluate(self.n_rollouts, self.n_steps)

        self.rollouts, self.data = self.sample(self.n_samples, self.n_keep)
        self.features = self.featurize(self.data)

        res = sc.optimize.minimize(self.dual,
                                   np.hstack((1.0, 1e-8 * np.random.randn(self.n_vfeat))),
                                   # np.hstack((1.0, self.vfunc.omega)),
                                   method='L-BFGS-B',
                                   jac=grad(self.dual),
                                   args=(
                                       self.kl_bound,
                                       self.features,
                                       self.data['r']),
                                   bounds=((1e-8, 1e8), ) + ((-np.inf, np.inf), ) * self.n_vfeat)

        self.eta, self.vfunc.omega = res.x[0], res.x[1:]

        # self.eta, self.vfunc.omega = 1.0, 1e-8 * np.random.randn(self.n_vfeat)
        # for _ in range(250):
        #     res = sc.optimize.minimize(self.dual_eta,
        #                                self.eta,
        #                                method='SLSQP',
        #                                jac=grad(self.dual_eta),
        #                                args=(
        #                                     self.vfunc.omega,
        #                                     self.kl_bound,
        #                                     self.features,
        #                                     self.data['r']),
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
        #                                args=(
        #                                    self.eta,
        #                                    self.features,
        #                                    self.data['r']),
        #                                options={'maxiter': 250})
        #
        #     # res = sc.optimize.minimize(self.dual_omega,
        #     #                            self.vfunc.omega,
        #     #                            method='trust-exact',
        #     #                            jac=grad(self.dual_omega),
        #     #                            hess=jacobian(grad(self.dual_omega)),
        #     #                            args=(
        #     #                                self.eta,
        #     #                                self.features,
        #     #                                self.data['r']))
        #     #
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

        # pol = self.ctl.wml(self.data['x'], self.data['u'], self.w, preg=self.preg)
        pol = self.ctl.wmap(self.data['x'], self.data['u'], self.w, preg=self.preg, eps=self.kl_bound)

        kls = self.kl_samples(self.w)
        kli = self.ctl.kli(pol, self.data['x'])
        klm = self.ctl.klm(pol, self.data['x'])

        self.ctl = pol
        ent = self.ctl.entropy()

        rwrd = np.mean(self.data['r'])
        # rwrd = np.mean(eval['r'])

        return rwrd, kls, kli, klm, ent


if __name__ == "__main__":

    import gym
    import lab

    # np.random.seed(0)
    env = gym.make('Pendulum-v0')
    env._max_episode_steps = 5000
    # env.seed(0)

    reps = REPS(env=env,
                n_samples=3000, n_keep=0,
                n_rollouts=25, n_steps=250,
                kl_bound=0.1, discount=0.99,
                vreg=1e-12, preg=1e-12, cov0=8.0,
                n_vfeat=75, n_pfeat=75,
                band=np.array([0.5, 0.5, 4.0]))

    for it in range(15):
        rwrd, kls, kli, klm, ent = reps.run()

        print('it=', it, f'rwrd={rwrd:{5}.{4}}',
              f'kls={kls:{5}.{4}}', f'kli={kli:{5}.{4}}',
              f'klm={klm:{5}.{4}}', f'ent={ent:{5}.{4}}')

    # # np.random.seed(0)
    # env = gym.make('Linear-v0')
    # env._max_episode_steps = 5000
    # # env.seed(0)
    #
    # reps = REPS(env=env,
    #             n_samples=2500, n_keep=0,
    #             n_rollouts=25, n_steps=100,
    #             kl_bound=0.05, discount=0.975,
    #             vreg=1e-16, preg=1e-16, cov0=5.0,
    #             vdgr=2, pdgr=1)
    #
    # for it in range(10):
    #     rwrd, kls, kli, klm, ent = reps.run()
    #
    #     print('it=', it, f'rwrd={rwrd:{5}.{4}}',
    #           f'kls={kls:{5}.{4}}', f'kli={kli:{5}.{4}}',
    #           f'klm={klm:{5}.{4}}', f'ent={ent:{5}.{4}}')
