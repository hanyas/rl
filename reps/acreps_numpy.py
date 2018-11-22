import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import autograd.numpy as np
from autograd import grad

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


class Qfunction:

    def __init__(self, n_states, n_actions, n_feat, band):
        self.n_states = n_states
        self.n_actions = n_actions

        self.band = band
        self.n_feat = n_feat
        self.basis = FourierFeatures(self.n_states + self.n_actions, self.n_feat, self.band)

        self.theta = 1e-8 * np.random.randn(self.n_feat)

    def features(self, x, a):
        return self.basis.fit_transform(np.concatenate((x, a), axis=-1))

    def values(self, x, a):
        feat = self.features(x, a)
        return np.dot(feat, self.theta)


class ACREPS:

    def __init__(self, env,
                 n_samples, n_keep,
                 n_rollouts, n_steps,
                 kl_bound, discount, trace,
                 n_vfeat, n_pfeat,
                 vreg, preg, cov0,
                 s_band, sa_band):

        self.env = env

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.n_samples = n_samples
        self.n_keep = n_keep

        self.n_rollouts = n_rollouts
        self.n_steps = n_steps

        self.kl_bound = kl_bound
        self.discount = discount
        self.trace = trace

        self.n_vfeat = n_vfeat
        self.n_pfeat = n_pfeat

        self.vreg = vreg
        self.preg = preg

        self.s_band = s_band
        self.sa_band = sa_band

        self.vfunc = Vfunction(self.n_states, n_feat=self.n_vfeat,
                               band=self.s_band)

        self.ctl = Policy(self.n_states, self.n_actions, n_feat=self.n_pfeat,
                          band=self.s_band)

        self.qfunc = Qfunction(self.n_states, self.n_actions, n_feat=self.n_vfeat,
                               band=self.sa_band)

        self.ctl.cov = cov0 * self.ctl.cov

        self.action_limit = self.env.action_space.high

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
            roll = {'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'xn': np.empty((0, self.n_states)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            rollouts.append(roll)

            x = self.env.reset()

            while True:
                u = self.ctl.actions(x, stoch)

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

    def lstd(self, phin, phi, discount, data):
        A = phi.T @ (phi - discount * phin)
        b = np.sum(phi * data['r'][:, np.newaxis], axis=0).T

        I = np.eye(phi.shape[1])

        C = np.linalg.solve(phi.T @ phi + 1e-8 * I, phi.T).T
        X = C @ (A + 1e-8 * I)
        y = C @ b

        theta = np.linalg.solve(X.T @ X + 1e-6 * I, X.T @ y)

        return theta, np.dot(phi, theta)

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

    def log_lkhd(self, K, psi, u, w):
        l = np.sum(w * (u - K @ psi.T)**2, axis=0) + self.preg * np.sum(K**2)
        return l

    def ml_policy(self):
        pol = copy.deepcopy(self.ctl)

        w, _, _ = self.weights(self.eta, self.vfunc.omega, self.vfeatures, self.targets)
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

    def run(self):
        _, eval = self.evaluate(self.n_rollouts, self.n_steps)

        self.rollouts, self.data = self.sample(self.n_samples, self.n_keep)
        self.vfeatures = self.featurize(self.data)

        # un = self.ctl.actions(self.data['xn'], False).reshape((-1, 1))
        # self.qfeatures = self.qfunc.features(self.data['x'], self.data['u'])
        # self.nqfeatures = self.qfunc.features(self.data['xn'], un)

        # self.qfunc.theta, self.targets = self.lstd(self.nqfeatures, self.qfeatures,
        #                                            self.discount, self.data)

        # self.targets = self.mc(self.data, self.discount)

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
        #                                options={'maxiter': 100})
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

        pi = self.ml_policy()
        kl_params = self.kl_params(pi, self.ctl)
        self.ctl = pi

        ent = self.entropy().squeeze()

        rwrd = np.sum(eval['r']) / self.n_rollouts
        # rwrd = np.sum(self.data['r']) / np.sum(self.data['done'])

        return rwrd, kl_samples, kl_params, ent


if __name__ == "__main__":

    import gym
    import lab

    # np.random.seed(0)
    env = gym.make('Pendulum-v1')
    env._max_episode_steps = 500
    # env.seed(0)

    acreps = ACREPS(env=env,
                    n_samples=5000, n_keep=0,
                    n_rollouts=20, n_steps=250,
                    kl_bound=0.1, discount=0.98, trace=0.95,
                    n_vfeat=75, n_pfeat=75,
                    vreg=1e-12, preg=1e-12, cov0=16.0,
                    s_band=np.array([0.5, 0.5, 4.0]),
                    sa_band=np.array([0.5, 0.5, 4.0, 1.0]))

    for it in range(15):
        rwrd, kl_samples, kl_params, ent = acreps.run()

        print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kl_s={kl_samples:{5}.{4}}',
              f'kl_p={kl_params:{5}.{4}}', f'ent={ent:{5}.{4}}')
