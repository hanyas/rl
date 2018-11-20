import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad, jacobian

import scipy as sc
from scipy import optimize
from scipy import stats

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

    def __init__(self, n_states, n_actions, n_feat, band):
        self.n_states = n_states
        self.n_actions = n_actions

        self.band = band
        self.n_feat = n_feat
        self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)

        self.K = 1e-8 * np.random.randn(self.n_actions, self.n_feat)
        self.cov = np.eye(n_actions)

    def features(self, x):
        return self.basis.fit_transform(x)

    def actions(self, x, stoch):
        feat = self.features(x)
        mean = np.einsum('...k,mk->...m', feat, self.K)
        if stoch:
            return np.random.normal(mean, np.sqrt(self.cov)).flatten()
        else:
            return mean


class Vfunction:

    def __init__(self, n_states, n_feat, band):
        self.n_states = n_states

        self.band = band
        self.n_feat = n_feat
        self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)

        self.omega = 1e-8 * np.random.randn(self.n_feat)

    def features(self, x):
        return self.basis.fit_transform(x)

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class REPS:

    def __init__(self, env,
                 n_samples, n_iter,
                 n_rollouts, n_steps, n_keep,
                 kl_bound, discount,
                 n_vfeat, n_pfeat,
                 vreg, preg, cov0,
                 band):

        self.env = env

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.n_samples = n_samples
        self.n_iter = n_iter

        self.n_rollouts = n_rollouts
        self.n_steps = n_steps
        self.n_keep = n_keep

        self.kl_bound = kl_bound
        self.discount = discount

        self.n_vfeat = n_vfeat
        self.n_pfeat = n_pfeat

        self.vreg = vreg
        self.preg = preg

        self.band = band

        self.vfunc = Vfunction(self.n_states, n_feat=self.n_vfeat,
                               band=self.band)

        self.ctl = Policy(self.n_states, self.n_actions, n_feat=self.n_pfeat,
                          band=self.band)

        self.ctl.cov = cov0 * self.ctl.cov

        self.render = False
        self.action_limit = self.env.action_space.high

        self.data = {}
        self.rollouts = []

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
            roll = {'xi': np.empty((0, self.n_states)),
                    'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'xn': np.empty((0, self.n_states)),
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
                        data['u'] = np.reshape(data['u'], (-1, self.n_actions))

                        return rollouts, data

                    if done:
                        break

    def evaluate(self, n_rollouts, n_steps, stoch=False, render=False):
        rollouts = []

        for n in range(n_rollouts):
            roll = {'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'r': np.empty((0,))}

            x = self.env.reset()

            for t in range(n_steps):
                u = self.ctl.actions(x, stoch)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                roll['r'] = np.hstack((roll['r'], r))

                if done:
                    break

            rollouts.append(roll)

        data = merge_dicts(*rollouts)
        data['u'] = np.reshape(data['u'], (-1, self.n_actions))

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

    def kl_samples(self):
        w, _, _ = self.weights(self.eta, self.vfunc.omega, self.features, self.data['r'])
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

    def log_lkhd(self, K, psi, u, w):
        l = np.sum(w * (u - K @ psi.T)**2, axis=0) + self.preg * np.sum(K**2)
        return l

    def ml_policy(self):
        pol = copy.deepcopy(self.ctl)

        w, _, _ = self.weights(self.eta, self.vfunc.omega, self.features, self.data['r'])
        psi = self.ctl.features(self.data['x'])

        res = sc.optimize.minimize(self.log_lkhd, pol.K, method='SLSQP',
                                   jac=grad(self.log_lkhd),
                                   args=(psi, self.data['u'].flatten(), w))
        pol.K = np.reshape(res.x, (self.n_actions, self.n_pfeat))

        # from sklearn.linear_model import Ridge
        # clf = Ridge(alpha=1e-8, fit_intercept=False)
        # clf.fit(psi, self.data['u'], sample_weight=w)
        # pol.K = clf.coef_

        Z = (np.square(np.sum(w, axis=0)) -
             np.sum(np.square(w), axis=0)) / np.sum(w, axis=0)
        tmp = self.data['u'] - psi @ pol.K.T
        pol.cov = np.einsum('t,tk,th->kh', w, tmp, tmp) / Z

        return pol

    def entropy(self):
        return np.log(np.sqrt(self.ctl.cov * 2.0 * np.pi * np.exp(1.0)))

    def run(self, n_iter):
        for it in range(n_iter):
            _, eval = self.evaluate(self.n_rollouts, self.n_steps)

            self.rollouts, self.data = self.sample(self.n_samples, self.n_keep)
            self.features = self.featurize(self.data)

            res = sc.optimize.minimize(self.dual,
                                       np.hstack((1.0, 1e-8 * np.random.randn(self.n_vfeat))),
                                       # np.hstack((1.0, self.vfunc.omega)),
                                       method='SLSQP',
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

            kl_samples = self.kl_samples()

            pi = self.ml_policy()
            kl_params = self.kl_params(pi, self.ctl)
            self.ctl = pi

            ent = self.entropy().squeeze()

            rwrd = np.sum(eval['r']) / self.n_rollouts
            # rwrd = np.sum(self.data['r']) / np.sum(self.data['done'])

            print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kl_s={kl_samples:{5}.{4}}',
                  f'kl_p={kl_params:{5}.{4}}', f'ent={ent:{5}.{4}}')


if __name__ == "__main__":

    import gym
    import lab

    # np.random.seed(0)
    env = gym.make('Pendulum-v1')
    env._max_episode_steps = 5000
    # env.seed(0)

    reps = REPS(env=env,
                n_samples=5000, n_iter=10,
                n_rollouts=25, n_steps=250, n_keep=0,
                kl_bound=0.1, discount=0.98,
                n_vfeat=75, n_pfeat=75,
                vreg=1e-16, preg=1e-16, cov0=8.0,
                band=np.array([0.5, 0.5, 4.0]))

    reps.run(reps.n_iter)
