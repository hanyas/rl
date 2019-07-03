import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import special

from sklearn.preprocessing import PolynomialFeatures

from rl.hyreps.v0 import BaumWelch
from rl.hyreps.v0 import merge

import random
import copy
from joblib import Parallel, delayed


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
        return phi


class Policy:

    def __init__(self, n_states, n_actions, n_regions, **kwargs):
        self.n_states = n_states
        self.n_actions = n_actions
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

        self.K = [1e-8 * np.random.randn(self.n_actions, self.n_feat) for _ in range(self.n_regions)]
        self.cov = [np.eye(n_actions) for _ in range(self.n_regions)]

    def features(self, x):
        return self.basis.fit_transform(x.reshape(1, -1)).squeeze()

    def actions(self, z, x, stoch):
        feat = self.features(x)
        mean = np.dot(self.K[z], feat)
        if stoch:
            return np.random.normal(mean, np.sqrt(self.cov[z])).flatten()
        else:
            return mean


class Vfunction:

    def __init__(self, n_states, n_regions, **kwargs):
        self.n_states = n_states
        self.n_regions = n_regions

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)
            self.omega = 1e-8 * np.random.randn(self.n_feat)
        else:
            self.degree = kwargs.get('degree', False)
            self.n_feat = int(
                sc.special.comb(self.degree + self.n_states, self.degree))
            self.basis = PolynomialFeatures(self.degree)
            self.omega = 1e-8 * np.random.randn(self.n_regions * self.n_feat)

    def features(self, x):
        return self.basis.fit_transform(x)

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class HyREPS_AC:

    def __init__(self, env, n_regions,
                 n_iter, n_rollouts,
                 n_steps, n_keep,
                 kl_bound, discount, trace,
                 vreg, preg, cov0,
                 rslds, priors, **kwargs):

        self.env = env

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.n_regions = n_regions

        self.n_iter = n_iter

        self.n_rollouts = n_rollouts
        self.n_steps = n_steps
        self.n_keep = n_keep

        self.n_samples = self.n_rollouts * self.n_steps

        self.kl_bound = kl_bound
        self.discount = discount
        self.trace = trace

        self.vreg = vreg

        self.rslds = copy.deepcopy(rslds)
        self.priors = priors
        self.preg = preg

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_vfeat = kwargs.get('n_vfeat', False)
            self.vfunc = Vfunction(self.n_states, self.n_regions, n_feat=self.n_vfeat,
                                   band=self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.vfunc = Vfunction(self.n_states, self.n_regions, degree=self.degree)
            self.n_vfeat = self.vfunc.n_feat

        self.ctl = Policy(self.n_states, self.n_actions, n_regions, degree=1)
        for n in range(self.n_regions):
            self.ctl.cov[n] = cov0 * self.ctl.cov[n]

        for n in range(self.n_regions):
            self.rslds.linear_policy[n].K = self.ctl.K[n]
            self.rslds.linear_policy[n].cov = self.ctl.cov[n]

        self.n_pfeat = self.ctl.n_feat

        self.action_limit = self.env.action_space.high

        self.data = {'z': np.empty((0,), np.int64),
                     'x': np.empty((0, self.n_states)),
                     'g': np.empty((0, self.n_regions)),
                     'u': np.empty((0, self.n_actions)),
                     'zn': np.empty((0,), np.int64),
                     'xn': np.empty((0, self.n_states)),
                     'gn': np.empty((0, self.n_regions)),
                     'r': np.empty((0,)),
                     'done': np.empty((0,), np.int64)}

        self.data = {}
        self.rollouts = []

        self.vfeatures = None
        self.qfeatures = None
        self.nqfeatures = None

        self.targets = None

        self.eta = np.array([1.0])

    def sample(self, n_samples, n_keep=0, stoch=True, render=False):
        if len(self.rollouts) >= n_keep:
            rollouts = random.sample(self.rollouts, n_keep)
        else:
            rollouts = []

        n = 0
        while True:
            roll = {'z': np.empty((0,), np.int64),
                    'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'zn': np.empty((0,), np.int64),
                    'xn': np.empty((0, self.n_states)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            rollouts.append(roll)

            x = self.env.reset()
            p = np.ones(self.n_regions) / self.n_regions
            z, a = self.rslds.filter(x, p)

            while True:
                u = self.ctl.actions(z, x, stoch)

                rollouts[-1]['z'] = np.hstack((rollouts[-1]['z'], z))
                rollouts[-1]['x'] = np.vstack((rollouts[-1]['x'], x))
                rollouts[-1]['u'] = np.vstack((rollouts[-1]['u'], u))

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                # filter discrete state
                z, a = self.rslds.filter(x, a)

                rollouts[-1]['zn'] = np.hstack((rollouts[-1]['zn'], z))
                rollouts[-1]['xn'] = np.vstack((rollouts[-1]['xn'], x))
                rollouts[-1]['r'] = np.hstack((rollouts[-1]['r'], r))
                rollouts[-1]['done'] = np.hstack((rollouts[-1]['done'], done))

                n = n + 1
                if n >= n_samples:
                    rollouts[-1]['done'][-1] = True

                    data = merge(*rollouts)
                    data['u'] = np.reshape(data['u'], (-1, self.n_actions))

                    return rollouts, data

                if done:
                    break

    def evaluate(self, n_rollouts, n_steps, stoch=False, render=False):
        rollouts = []

        for n in range(n_rollouts):
            roll = {'z': np.empty((0,), np.int64),
                    'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            x = self.env.reset()
            p = np.ones(self.n_regions) / self.n_regions
            z, a = self.rslds.filter(x, p)

            for t in range(n_steps):
                u = self.ctl.actions(z, x, stoch)

                roll['z'] = np.hstack((roll['z'], z))
                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                # filter discrete state
                z, a = self.rslds.filter(x, a)

                roll['r'] = np.hstack((roll['r'], r))

                if done:
                    break

            rollouts.append(roll)

        data = merge(*rollouts)
        data['u'] = np.reshape(data['u'], (-1, self.n_actions))

        return rollouts, data

    def gae(self, data, phi, omega, discount, trace):
        values = np.dot(phi, omega)
        adv = np.zeros_like(values)

        for rev_k, v in enumerate(reversed(values)):
            k = len(values) - rev_k - 1
            if data['done'][k]:
                adv[k] = data['r'][k] - values[k]
            else:
                adv[k] = data['r'][k] + discount * values[k + 1] - values[k] +\
                         discount * trace * adv[k + 1]

        targets = adv + values
        return targets

    def mc(self, data, discount):
        rewards = data['r']
        targets = np.zeros_like(rewards)

        for rev_k, v in enumerate(reversed(rewards)):
            k = len(rewards) - rev_k - 1
            if data['done'][k]:
                targets[k] = rewards[k]
            else:
                targets[k] = rewards[k] + discount * rewards[k + 1]
        return targets

    def featurize(self, data):
        vfeatures = self.vfunc.features(data['x'])
        mvfeatures = np.mean(vfeatures, axis=0, keepdims=True)
        return vfeatures - mvfeatures

    def weights(self, eta, omega, features, targets):
        adv = targets - np.dot(features, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return w, delta, np.max(adv)

    def dual(self, var, epsilon, phi, targets):
        eta, omega = var[0], var[1:]
        w, _, max_adv = self.weights(eta, omega, phi, targets)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(w, axis=0))
        g = g + self.vreg * (omega.T @ omega)
        return g

    def grad(self, var, epsilon, phi, targets):
        eta, omega = var[0], var[1:]
        w, delta, max_adv = self.weights(eta, omega, phi, targets)

        deta = epsilon + np.log(np.mean(w, axis=0)) - \
               np.sum(w * delta, axis=0) / (eta * np.sum(w, axis=0))

        domega = - np.sum(w[:, np.newaxis] * phi, axis=0) / np.sum(w, axis=0)
        domega = domega + self.vreg * 2 * omega

        return np.hstack((deta, domega))

    def dual_eta(self, eta, omega, epsilon, phi, targets):
        w, _, max_adv = self.weights(eta, omega, phi, targets)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(w, axis=0))
        return g

    def grad_eta(self, eta, omega, epsilon, phi, targets):
        w, delta, max_adv = self.weights(eta, omega, phi, targets)
        deta = epsilon + np.log(np.mean(w, axis=0)) - \
               np.sum(w * delta, axis=0) / (eta * np.sum(w, axis=0))
        return deta

    def dual_omega(self, omega, eta, phi, targets):
        w, _, max_adv = self.weights(eta, omega, phi, targets)
        g = max_adv + eta * np.log(np.mean(w, axis=0))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def grad_omega(self, omega, eta, phi, targets):
        w, _, max_adv = self.weights(eta, omega, phi, targets)
        domega = - np.sum(w[:, np.newaxis] * phi, axis=0) / np.sum(w, axis=0)
        domega = domega + self.vreg * 2 * omega
        return domega

    def kl_samples(self):
        w, _, _ = self.weights(self.eta, self.vfunc.omega, self.vfeatures, self.targets)
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def kl_params(self, p, q):
        n_samples = self.data['x'].shape[0]
        diff = q.features(self.data['x']) @ q.K.T - p.features(self.data['x']) @ p.K.T
        kl = 0.5 * (np.trace(np.linalg.inv(q.cov) @ p.cov) +
                   np.einsum('nk,kh,nh->', diff, np.linalg.inv(q.cov), diff) / n_samples -
                   self.n_actions + np.log(np.linalg.det(q.cov) / np.linalg.det(p.cov)))
        return kl

    def ml_policy(self):
        pol = copy.deepcopy(self.ctl)

        w, _, _ = self.weights(self.eta, self.vfunc.omega, self.vfeatures, self.targets)

        x = np.reshape(self.data['x'], (-1, self.n_steps, self.n_states))
        u = np.reshape(self.data['u'], (-1, self.n_steps, self.n_actions))
        w = np.reshape(w, (x.shape[0], x.shape[1]))

        def baumWelchFunc(args):
            x, u, w, n_regions, priors, regs, rslds, update = args
            rollouts = x.shape[0]
            choice = np.random.choice(rollouts, size=int(0.8 * rollouts), replace=False)
            x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
            bw = BaumWelch(x, u, w, n_regions, priors, regs, rslds, update)
            lklhd = bw.run(n_iter=100, save=False)
            return bw, lklhd

        n_jobs = 25
        args = [(x, u, w, self.n_regions, self.priors, self.preg, self.rslds, [False, True]) for _ in range(n_jobs)]
        results = Parallel(n_jobs=n_jobs, verbose=0)(map(delayed(baumWelchFunc), args))
        bwl, lklhd = list(map(list, zip(*results)))
        bw = bwl[np.argmax(lklhd)]

        for n in range(self.n_regions):
            pol.K[n] = bw.rslds.linear_policy[n].K
            pol.cov[n] = bw.rslds.linear_policy[n].cov

        return pol

    def run(self, n_iter):
        for it in range(n_iter):
            _, eval = self.evaluate(self.n_rollouts, self.n_steps)

            self.rollouts, self.data = self.sample(self.n_samples, n_keep=self.n_keep)
            self.vfeatures = self.featurize(self.data)

            self.targets = self.gae(self.data, self.vfeatures, self.vfunc.omega,
                                    self.discount, self.trace)

            res = sc.optimize.minimize(self.dual,
                                       np.hstack((1.0, 1e-8 * np.random.randn(self.n_vfeat))),
                                       # np.hstack((1.0, self.vfunc.omega)),
                                       method='SLSQP',
                                       jac=grad(self.dual),
                                       args=(
                                           self.kl_bound,
                                           self.vfeatures,
                                           self.targets),
                                       bounds=((1e-8, 1e8), ) + ((-np.inf, np.inf), ) * self.n_vfeat)

            self.eta, self.vfunc.omega = res.x[0], res.x[1:]

            # self.eta, self.vfunc.omega = 1.0, 1e-8 * np.random.randn(self.n_vfeat)
            # for _ in range(250):
            #     res = sc.optimize.minimize(self.dual_eta,
            #                                self.eta,
            #                                method='SLSQP',
            #                                jac=grad(self.dual_eta),
            #                                args=(
            #                                    self.vfunc.omega,
            #                                    self.kl_bound,
            #                                    self.vfeatures,
            #                                    self.targets),
            #                                bounds=((1e-8, 1e8),),
            #                                options={'maxiter': 5})
            #     # print(res)
            #     #
            #     # check = sc.optimize.check_grad(self.dual_eta,
            #     #                                self.grad_eta, res.x,
            #     #                                self.vfunc.omega,
            #     #                                self.kl_bound,
            #     #                                self.vfeatures,
            #     #                                self.targets)
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
            #                                    self.vfeatures,
            #                                    self.targets),
            #                                options={'maxiter': 250})
            #
            #     # print(res)
            #     #
            #     # check = sc.optimize.check_grad(self.dual_omega,
            #     #                                self.grad_omega, res.x,
            #     #                                self.eta,
            #     #                                self.vfeatures,
            #     #                                self.targets)
            #     # print('Omega Error', check)
            #
            #     self.vfunc.omega = res.x

            kl_samples = self.kl_samples()

            self.ctl = self.ml_policy()

            for n in range(self.n_regions):
                self.rslds.linear_policy[n].K = self.ctl.K[n]
                self.rslds.linear_policy[n].cov = self.ctl.cov[n]

            rwrd = np.sum(eval['r']) / self.n_rollouts
            # rwrd = np.sum(self.data['r']) / np.sum(self.data['done'])

            print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kl_s={kl_samples:{5}.{4}}')
