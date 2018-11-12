import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

import scipy as sc
from scipy import optimize
from scipy import special

from sklearn.preprocessing import PolynomialFeatures

import random

from rl.hyreps import BaumWelch
from rl.hyreps import merge_dicts

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

    def features(self, z, x):
        phi = np.zeros((x.shape[0], self.n_regions * self.n_feat, self.n_regions))
        for n in range(self.n_regions):
            idx = np.ix_(range(x.shape[0]),
                         range(n * self.n_feat, n * self.n_feat + self.n_feat),
                         range(n, n + 1))
            phi[idx] = self.basis.fit_transform(x)[:, :, np.newaxis]
        return phi

    def values(self, z, x):
        feat = self.features(z, x)
        return np.dot(feat, self.omega)


class HyREPS:

    def __init__(self, env, n_regions,
                 n_samples, n_iter,
                 n_rollouts, n_steps, n_keep,
                 kl_bound, discount,
                 vreg, preg, cov0,
                 rslds, priors, **kwargs):

        self.env = env

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.n_regions = n_regions

        self.n_samples = n_samples
        self.n_iter = n_iter

        self.n_rollouts = n_rollouts
        self.n_steps = n_steps
        self.n_keep = n_keep

        self.kl_bound = kl_bound
        self.discount = discount

        self.vreg = vreg
        self.preg = preg

        self.rslds = rslds
        self.priors = priors

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

        self.n_pfeat = self.ctl.n_feat

        self.action_limit = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.vfeatures = None
        self.ivfeatures = None
        self.qfeatures = None
        self.features = None

        self.eta = np.array([1.0])

    def sample(self, n_samples, n_keep=0, reset=True, stoch=True, render=False):
        if len(self.rollouts) >= n_keep:
            rollouts = random.sample(self.rollouts, n_keep)
        else:
            rollouts = []

        n = 0
        while True:
            roll = {'zi': np.empty((0, 1), np.int64),
                    'xi': np.empty((0, self.n_states)),
                    'ai': np.empty((0, self.n_regions)),
                    'gi': np.empty((0, self.n_regions)),
                    'z': np.empty((0, 1), np.int64),
                    'x': np.empty((0, self.n_states)),
                    'a': np.empty((0, self.n_regions)),
                    'g': np.empty((0, self.n_regions)),
                    'u': np.empty((0, self.n_actions)),
                    'zn': np.empty((0, 1), np.int64),
                    'xn': np.empty((0, self.n_states)),
                    'an': np.empty((0, self.n_regions)),
                    'gn': np.empty((0, self.n_regions)),
                    'r': np.empty((0, 1)),
                    'done': np.empty((0, 1), np.int64)}

            x = self.env.reset()
            p = np.ones(self.n_regions) / self.n_regions
            z, a = self.rslds.filter(x, p)

            roll['done'] = np.vstack((roll['done'], False))
            roll['zi'] = np.vstack((roll['zi'], z))
            roll['xi'] = np.vstack((roll['xi'], x))
            roll['ai'] = np.vstack((roll['ai'], a))

            u = self.ctl.actions(z, x, stoch)

            while True:
                coin = np.random.binomial(1, 1.0 - self.discount)
                if reset and coin:
                    roll['done'][-1] = True
                    rollouts.append(roll)
                    break
                else:
                    roll['z'] = np.vstack((roll['z'], z))
                    roll['x'] = np.vstack((roll['x'], x))
                    roll['u'] = np.vstack((roll['u'], u))
                    roll['a'] = np.vstack((roll['a'], a))

                    x, r, done, _ = self.env.step(
                        np.clip(u, - self.action_limit, self.action_limit))
                    if render:
                        self.env.render()

                    # filter discrete state
                    z, a = self.rslds.filter(x, a)

                    roll['zn'] = np.vstack((roll['zn'], z))
                    roll['xn'] = np.vstack((roll['xn'], x))
                    roll['r'] = np.vstack((roll['r'], r))
                    roll['done'] = np.vstack((roll['done'], done))
                    roll['an'] = np.vstack((roll['an'], a))

                    n = n + 1
                    if n >= n_samples:
                        roll['done'][-1] = True
                        rollouts.append(roll)
                        return rollouts, merge_dicts(*rollouts)

                    if done:
                        rollouts.append(roll)
                        break
                    else:
                        u = self.ctl.actions(z, x, stoch)

    def evaluate(self, n_rollouts, n_steps, stoch=False, render=False):
        data = {'zi': np.empty((0, 1), np.int64),
                'xi': np.empty((0, self.n_states)),
                'ai': np.empty((0, self.n_regions)),
                'z': np.empty((0, 1), np.int64),
                'x': np.empty((0, self.n_states)),
                'a': np.empty((0, self.n_regions)),
                'u': np.empty((0, self.n_actions)),
                'zn': np.empty((0, 1), np.int64),
                'xn': np.empty((0, self.n_states)),
                'an': np.empty((0, self.n_regions)),
                'r': np.empty((0, 1)),
                'done': np.empty((0, 1), np.int64)}

        for n in range(n_rollouts):
            x = self.env.reset()
            p = np.ones(self.n_regions) / self.n_regions
            z, a = self.rslds.filter(x, p)

            data['zi'] = np.vstack((data['zi'], z))
            data['xi'] = np.vstack((data['xi'], x))
            data['ai'] = np.vstack((data['ai'], a))

            u = self.ctl.actions(z, x, stoch)

            for t in range(n_steps):
                data['z'] = np.vstack((data['z'], z))
                data['x'] = np.vstack((data['x'], x))
                data['u'] = np.vstack((data['u'], u))
                data['a'] = np.vstack((data['a'], a))

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                # filter discrete state
                z, a = self.rslds.filter(x, a)

                data['zn'] = np.vstack((data['zn'], z))
                data['xn'] = np.vstack((data['xn'], x))
                data['r'] = np.vstack((data['r'], r))
                data['done'] = np.vstack((data['done'], done))
                data['an'] = np.vstack((data['an'], a))

                if done:
                    break
                else:
                    u = self.ctl.actions(z, x, stoch)

        return data

    def dual_eta(self, eta, omega, epsilon, phi, r):
        n_samples = phi.shape[0]
        adv = np.empty((n_samples, self.n_regions))
        for i in range(self.n_regions):
            adv = r + np.dot(phi, omega)
            delta = adv - np.max(adv)
            w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0))
        return g

    def grad_eta(self, eta, omega, epsilon, phi, r):
        adv = r + np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        deta = epsilon + np.log(np.mean(w, axis=0)) - \
               np.sum(w * delta, axis=0) / (eta * np.sum(w, axis=0))
        return deta

    def dual_omega(self, omega, eta, phi, r):
        adv = r + np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        g = np.max(adv) + eta * np.log(np.mean(w, axis=0))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def grad_omega(self, omega, eta, phi, r):
        adv = r + np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        domega = np.sum(w[:, np.newaxis] * phi, axis=0) / np.sum(w, axis=0)
        domega = domega + self.vreg * 2 * omega
        return domega

    def kl_samples(self):
        adv = self.data['r'] + np.dot(self.features, self.vfunc.omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def ml_policy(self):
        pol = copy.deepcopy(self.ctl)

        data = self.evaluate(self.n_rollouts, self.n_steps, stoch=True)

        ivfeatures = np.mean(self.vfunc.features(data['zi'], data['xi']),
                             axis=0, keepdims=True)
        vfeatures = self.vfunc.features(data['z'], data['x'])
        qfeatures = self.vfunc.features(data['zn'], data['xn'])
        features = self.discount * qfeatures - vfeatures +\
                   (1.0 - self.discount) * ivfeatures

        adv = data['r'] + np.dot(features, self.vfunc.omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))

        x = np.reshape(data['x'], (self.n_rollouts, self.n_steps, -1))
        u = np.reshape(data['u'], (self.n_rollouts, self.n_steps, -1))
        w = np.reshape(w, (self.n_rollouts, self.n_steps))

        def baumWelchFunc(args):
            choice = np.random.choice(25, size=20, replace=False)
            x, u, w, n_regions, priors, rslds = args
            x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
            bw = BaumWelch(x, u, w, n_regions, priors, rslds=rslds)
            lklhd = bw.run(n_iter=100, save=False)
            return bw, lklhd

        n_jobs = 25
        args = [(x, u, w, self.n_regions, self.priors, self.rslds) for _ in range(n_jobs)]
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            map(delayed(baumWelchFunc), args))
        bwl, lklhd = list(map(list, zip(*results)))

        em = bwl[np.argmax(lklhd)]

        for n in range(self.n_regions):
            pol.K[n] = em.rslds.linear_policy[n].K
            pol.cov[n] = em.rslds.linear_policy[n].cov

        return pol

    def run(self, n_iter):
        for it in range(n_iter):
            # eval = self.evaluate(self.n_rollouts, self.n_steps)

            self.data = self.sample(self.n_samples, self.n_keep)

            self.ivfeatures = np.mean(self.vfunc.features(self.data['zi'], self.data['xi']),
                                      axis=0, keepdims=True)
            self.vfeatures = self.vfunc.features(self.data['z'], self.data['x'])
            self.qfeatures = self.vfunc.features(self.data['zn'], self.data['xn'])

            # self.qfeatures = self.get_qfeatures(self.data['zn'],
            #                                     self.data['z'],
            #                                     self.data['xn'],
            #                                     self.data['x'],
            #                                     self.data['u'])

            self.features = self.discount * self.qfeatures - self.vfeatures +\
                            (1.0 - self.discount) * self.ivfeatures

            self.eta, self.vfunc.omega = 1.0, 1e-8 * np.random.randn(self.n_vfeat)
            for _ in range(250):
                res = sc.optimize.minimize(self.dual_eta,
                                           self.eta,
                                           method='SLSQP',
                                           jac=self.grad_eta,
                                           args=(
                                                self.vfunc.omega,
                                                self.kl_bound,
                                                self.features,
                                                self.data['r']),
                                           bounds=((1e-8, 1e8),),
                                           options={'maxiter': 5})
                # print(res)
                #
                # check = sc.optimize.check_grad(self.dual_eta,
                #                                self.grad_eta,
                #                                res.x,
                #                                self.vfunc.omega,
                #                                self.kl_bound,
                #                                self.features,
                #                                self.data['r'])
                # print('Eta Error', check)

                self.eta = res.x

                res = sc.optimize.minimize(self.dual_omega,
                                           self.vfunc.omega,
                                           method='BFGS',
                                           jac=self.grad_omega,
                                           args=(
                                               self.eta,
                                               self.features,
                                               self.data['r']),
                                           options={'maxiter': 250})
                # print(res)
                #
                # check = sc.optimize.check_grad(self.dual_omega,
                #                                self.grad_omega,
                #                                res.x,
                #                                self.eta,
                #                                self.features,
                #                                self.data['r'])
                # print('Omega Error', check)

                self.vfunc.omega = res.x

            kl_samples = self.kl_samples()

            pi = self.ml_policy()
            self.ctl = pi

            rwrd = np.sum(eval['r']) / self.n_rollouts
            # rwrd = np.sum(self.data['r']) / np.sum(self.data['done'])

            print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kl_s={kl_samples:{5}.{4}}')
