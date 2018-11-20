import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import special
from scipy import stats

from sklearn.preprocessing import PolynomialFeatures

from rl.hyreps import BaumWelch
from rl.hyreps import merge_dicts

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

    def features(self, z, x):
        phi = np.zeros((self.n_regions, x.shape[0], self.n_regions * self.n_feat))
        for n in range(self.n_regions):
            idx = np.ix_(range(n, n + 1),
                         range(x.shape[0]),
                         range(n * self.n_feat, n * self.n_feat + self.n_feat))
            phi[idx] = self.basis.fit_transform(x)[np.newaxis, :, :]
        return phi

    def values(self, z, x):
        feat = self.features(z, x)
        return np.dot(feat, self.omega)


class Qfunction:

    def __init__(self, lgstc_mdl, lin_mdl, vfunc, n_actions):
        self.lgstc_mdl = lgstc_mdl
        self.lin_mdl = lin_mdl
        self.vfunc = vfunc

        self.n_states = self.vfunc.n_states
        self.n_regions = self.vfunc.n_regions
        self.n_actions = n_actions

    def features(self, zn, xn):
        lgstc = self.lgstc_mdl.transitions(xn)
        vfeat = self.vfunc.features(zn, xn)
        qfeat = np.einsum('nlm,mnk->lnk', lgstc, vfeat)
        return qfeat

    def values(self, zn, x, u):
        feat = self.features(zn, x, u)
        return np.dot(feat, self.vfunc.omega)


class HyREPS_V:

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

        self.qfunc = Qfunction(self.rslds.logistic_model, self.rslds.linear_models,
                               self.vfunc, self.n_actions)

        self.ctl = Policy(self.n_states, self.n_actions, n_regions, degree=1)
        for n in range(self.n_regions):
            self.ctl.cov[n] = cov0 * self.ctl.cov[n]

        for n in range(self.n_regions):
            self.rslds.linear_policy[n].K = self.ctl.K[n]
            self.rslds.linear_policy[n].cov = self.ctl.cov[n]

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

        coin = sc.stats.binom(1, 1.0 - self.discount)

        n = 0
        while True:
            roll = {'zi': np.empty((0,), np.int64),
                    'ai': np.empty((0, self.n_regions)),
                    'xi': np.empty((0, self.n_states)),
                    'z': np.empty((0,), np.int64),
                    'a': np.empty((0, self.n_regions)),
                    'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'zn': np.empty((0,), np.int64),
                    'an': np.empty((0, self.n_regions)),
                    'xn': np.empty((0, self.n_states)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            rollouts.append(roll)

            x = self.env.reset()
            p = np.ones(self.n_regions) / self.n_regions
            z, a = self.rslds.filter(x, p)

            rollouts[-1]['zi'] = np.hstack((rollouts[-1]['zi'], z))
            rollouts[-1]['ai'] = np.vstack((rollouts[-1]['ai'], a))
            rollouts[-1]['xi'] = np.vstack((rollouts[-1]['xi'], x))
            rollouts[-1]['done'] = np.hstack((rollouts[-1]['done'], False))

            while True:
                u = self.ctl.actions(z, x, stoch)

                if reset and coin.rvs() and (rollouts[-1]['x'].shape[0] > 0):
                    rollouts[-1]['done'][-1] = True
                    break
                else:
                    rollouts[-1]['z'] = np.hstack((rollouts[-1]['z'], z))
                    rollouts[-1]['a'] = np.vstack((rollouts[-1]['a'], a))
                    rollouts[-1]['x'] = np.vstack((rollouts[-1]['x'], x))
                    rollouts[-1]['u'] = np.vstack((rollouts[-1]['u'], u))

                    x, r, done, _ = self.env.step(
                        np.clip(u, - self.action_limit, self.action_limit))
                    if render:
                        self.env.render()

                    # filter discrete state
                    z, a = self.rslds.filter(x, a)

                    rollouts[-1]['zn'] = np.hstack((rollouts[-1]['zn'], z))
                    rollouts[-1]['an'] = np.vstack((rollouts[-1]['an'], a))
                    rollouts[-1]['xn'] = np.vstack((rollouts[-1]['xn'], x))
                    rollouts[-1]['r'] = np.hstack((rollouts[-1]['r'], r))
                    rollouts[-1]['done'] = np.hstack((rollouts[-1]['done'], done))

                    n = n + 1
                    if n >= n_samples:
                        rollouts[-1]['done'][-1] = True

                        data = merge_dicts(*rollouts)
                        data['u'] = np.reshape(data['u'], (-1, self.n_actions))

                        return rollouts, data

                    if done:
                        break

    def evaluate(self, n_rollouts, n_steps, stoch=False, render=False):
        rollouts = []

        for n in range(n_rollouts):
            roll = {'zi': np.empty((0,), np.int64),
                    'ai': np.empty((0, self.n_regions)),
                    'xi': np.empty((0, self.n_states)),
                    'z': np.empty((0,), np.int64),
                    'a': np.empty((0, self.n_regions)),
                    'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'zn': np.empty((0,), np.int64),
                    'an': np.empty((0, self.n_regions)),
                    'xn': np.empty((0, self.n_states)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            x = self.env.reset()
            p = np.ones(self.n_regions) / self.n_regions
            z, a = self.rslds.filter(x, p)

            roll['zi'] = np.hstack((roll['zi'], z))
            roll['ai'] = np.vstack((roll['ai'], a))
            roll['xi'] = np.vstack((roll['xi'], x))

            for t in range(n_steps):
                u = self.ctl.actions(z, x, stoch)

                roll['z'] = np.hstack((roll['z'], z))
                roll['a'] = np.vstack((roll['a'], a))
                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                # filter discrete state
                z, a = self.rslds.filter(x, a)

                roll['zn'] = np.hstack((roll['zn'], z))
                roll['an'] = np.vstack((roll['an'], a))
                roll['xn'] = np.vstack((roll['xn'], x))
                roll['r'] = np.hstack((roll['r'], r))
                roll['done'] = np.hstack((roll['done'], done))

                if done:
                    break

            rollouts.append(roll)

        data = merge_dicts(*rollouts)
        data['u'] = np.reshape(data['u'], (-1, self.n_actions))

        return rollouts, data

    def dual(self, var, epsilon, phi, r, a):
        eta, omega = var[0], var[1:]
        adv = r + np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        g = eta * epsilon + np.max(adv) +\
            eta * np.log(np.mean(np.sum(a * w, axis=0)))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def dual_eta(self, eta, omega, epsilon, phi, r, a):
        adv = r + np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(np.sum(a * w, axis=0)))
        return g

    def dual_omega(self, omega, eta, phi, r, a):
        adv = r + np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        g = np.max(adv) + eta * np.log(np.mean(np.sum(a * w, axis=0)))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def kl_samples(self):
        adv = self.data['r'] + np.dot(self.features, self.vfunc.omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))
        w = np.clip(w, 1e-75, np.inf)
        w = np.sum(self.data['a'].T * w, axis=0)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def ml_policy(self):
        pol = copy.deepcopy(self.ctl)

        _, data = self.evaluate(self.n_rollouts, self.n_steps, stoch=True)

        ivfeatures = self.vfunc.features(data['zi'], data['xi'])
        ivfeatures = np.einsum('nz,znk->nk', data['ai'], ivfeatures)
        ivfeatures = np.mean(ivfeatures, axis=0)

        vfeatures = self.vfunc.features(data['z'], data['x'])
        qfeatures = self.qfunc.features(data['zn'], data['xn'])

        features = self.discount * qfeatures - vfeatures + \
                        (1.0 - self.discount) * ivfeatures

        adv = data['r'] + np.dot(features, self.vfunc.omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))
        w = np.sum(data['a'].T * w, axis=0)

        x = np.reshape(data['x'], (self.n_rollouts, self.n_steps, -1))
        u = np.reshape(data['u'], (self.n_rollouts, self.n_steps, -1))
        w = np.reshape(w, (self.n_rollouts, self.n_steps))

        def baumWelchFunc(args):
            x, u, w, n_regions, priors, regs, rslds = args
            rollouts = x.shape[0]
            choice = np.random.choice(rollouts, size=int(0.8 * rollouts), replace=False)
            x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
            bw = BaumWelch(x, u, w, n_regions, priors, regs, rslds)
            lklhd = bw.run(n_iter=100, save=False)
            return bw, lklhd

        n_jobs = 25
        args = [(x, u, w, self.n_regions, self.priors, self.preg, self.rslds) for _ in range(n_jobs)]
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

            self.rollouts, self.data = self.sample(self.n_samples, self.n_keep)

            self.ivfeatures = self.vfunc.features(self.data['zi'], self.data['xi'])
            self.ivfeatures = np.einsum('nz,znk->nk', self.data['ai'], self.ivfeatures)
            self.ivfeatures = np.mean(self.ivfeatures, axis=0)

            self.vfeatures = self.vfunc.features(self.data['z'], self.data['x'])
            self.qfeatures = self.qfunc.features(self.data['zn'], self.data['xn'])

            self.features = self.discount * self.qfeatures - self.vfeatures +\
                            (1.0 - self.discount) * self.ivfeatures

            res = sc.optimize.minimize(self.dual,
                                       np.hstack((1.0, 1e-8 * np.random.randn(self.n_vfeat * self.n_regions))),
                                       # np.hstack((1.0, self.vfunc.omega)),
                                       method='SLSQP',
                                       jac=grad(self.dual),
                                       args=(
                                           self.kl_bound,
                                           self.features,
                                           self.data['r'][np.newaxis, ...],
                                           self.data['a'].T
                                       ),
                                       bounds=((1e-8, 1e8), ) + ((-np.inf, np.inf), ) * self.n_vfeat * self.n_regions)

            self.eta, self.vfunc.omega = res.x[0], res.x[1:]

            # self.eta, self.vfunc.omega = 1.0, 1e-8 * np.random.randn(self.n_vfeat * self.n_regions)
            # for _ in range(250):
            #     res = sc.optimize.minimize(self.dual_eta,
            #                                self.eta,
            #                                method='SLSQP',
            #                                jac=grad(self.dual_eta),
            #                                args=(
            #                                    self.vfunc.omega,
            #                                    self.kl_bound,
            #                                    self.features,
            #                                    self.data['r'][np.newaxis, ...],
            #                                    self.data['a'].T
            #                                ),
            #                                bounds=((1e-8, 1e8),),
            #                                options={'maxiter': 3})
            #     self.eta = res.x
            #
            #     res = sc.optimize.minimize(self.dual_omega,
            #                                self.vfunc.omega,
            #                                method='BFGS',
            #                                jac=grad(self.dual_omega),
            #                                args=(
            #                                    self.eta,
            #                                    self.features,
            #                                    self.data['r'][np.newaxis, ...],
            #                                    self.data['a'].T
            #                                ),
            #                                options={'maxiter': 100})
            #     self.vfunc.omega = res.x

            kl_samples = self.kl_samples()

            pi = self.ml_policy()
            self.ctl = pi

            for n in range(self.n_regions):
                self.rslds.linear_policy[n].K = self.ctl.K[n]
                self.rslds.linear_policy[n].cov = self.ctl.cov[n]

            rwrd = np.sum(eval['r']) / self.n_rollouts
            # rwrd = np.sum(self.data['r']) / np.sum(self.data['done'])

            print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kl_s={kl_samples:{5}.{4}}')
