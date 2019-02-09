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

        ans = - 0.5 * (np.einsum('nk,kh,nh->n', mu - u, np.linalg.inv(cov), mu - u) +
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

    def rwml(self, x, u, w, preg=0.0, eta=np.array([0.0])):
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

    def dual(self, eta, x, u, w, eps):
        pol = self.rwml(x, u, w, eta=eta)
        return np.mean(w * self.loglik(pol, x, u)) + eta * (eps - self.klm(pol, x))

    def wmap(self, x, u, w, eps):
        res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                   method='SLSQP',
                                   # jac=grad(self.dual),
                                   args=(x, u, w, eps),
                                   bounds=((1e-8, 1e8),),
                                   options={'maxiter': 250,
                                            'ftol': 1e-06})
        eta = res['x']
        pol = self.rwml(x, u, w, eta=eta)

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


class Qfunction:

    def __init__(self, dim_state, dim_action, **kwargs):
        self.dim_state = dim_state
        self.dim_action = dim_action

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.dim_state + self.dim_action, self.n_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.n_feat = int(
                sc.special.comb(self.degree + (self.dim_state + self.dim_action), self.degree))
            self.basis = PolynomialFeatures(self.degree)

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
        self.trace = trace

        self.vreg = vreg
        self.preg = preg

        if 's_band' in kwargs:
            self.s_band = kwargs.get('s_band', False)
            self.sa_band = kwargs.get('sa_band', False)

            self.n_vfeat = kwargs.get('n_vfeat', False)
            self.n_pfeat = kwargs.get('n_pfeat', False)

            self.vfunc = Vfunction(self.dim_state, n_feat=self.n_vfeat, band=self.s_band)

            self.qfunc = Qfunction(self.dim_state, self.dim_action, n_feat=self.n_vfeat, band=self.sa_band)

            self.ctl = Policy(self.dim_state, self.dim_action, n_feat=self.n_pfeat, band=self.s_band)
        else:
            self.vdgr = kwargs.get('vdgr', False)
            self.pdgr = kwargs.get('pdgr', False)

            self.vfunc = Vfunction(self.dim_state, degree=self.vdgr)
            self.n_vfeat = self.vfunc.n_feat

            self.qfunc = Qfunction(self.dim_state, self.dim_action, degree=self.vdgr)

            self.ctl = Policy(self.dim_state, self.dim_action, degree=self.pdgr)
            self.n_pfeat = self.ctl.n_feat

        self.ctl.cov = cov0 * self.ctl.cov
        self.action_limit = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.vfeatures = None
        self.qfeatures = None
        self.nqfeatures = None

        self.targets = None
        self.w = None

        self.eta = np.array([1.0])

    def sample(self, n_samples, n_keep=0, stoch=True, render=False):
        if len(self.rollouts) >= n_keep:
            rollouts = random.sample(self.rollouts, n_keep)
        else:
            rollouts = []

        n = 0
        while True:
            roll = {'x': np.empty((0, self.dim_state)),
                    'u': np.empty((0, self.dim_action)),
                    'xn': np.empty((0, self.dim_state)),
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

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                roll['r'] = np.hstack((roll['r'], r))

                if done:
                    break

            rollouts.append(roll)

        data = merge_dicts(*rollouts)
        data['u'] = np.reshape(data['u'], (-1, self.dim_action))

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

    def kl_samples(self, w):
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

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
                                   method='L-BFGS-B',
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

        self.w, _, _ = self.weights(self.eta, self.vfunc.omega, self.vfeatures, self.targets)

        # pol = self.ctl.wml(self.data['x'], self.data['u'], self.w, preg=self.preg)
        pol = self.ctl.wmap(self.data['x'], self.data['u'], self.w, self.kl_bound)

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
    env._max_episode_steps = 500
    # env.seed(0)

    acreps = ACREPS(env=env,
                    n_samples=5000, n_keep=0,
                    n_rollouts=20, n_steps=250,
                    kl_bound=0.1, discount=0.98, trace=0.95,
                    vreg=1e-12, preg=1e-12, cov0=16.0,
                    n_vfeat=75, n_pfeat=75,
                    s_band=np.array([0.5, 0.5, 4.0]),
                    sa_band=np.array([0.5, 0.5, 4.0, 1.0]))

    for it in range(15):
        rwrd, kls, kli, klm, ent = acreps.run()

        print('it=', it, f'rwrd={rwrd:{5}.{4}}',
              f'kls={kls:{5}.{4}}', f'kli={kli:{5}.{4}}',
              f'klm={klm:{5}.{4}}', f'ent={ent:{5}.{4}}')


    # # np.random.seed(0)
    # env = gym.make('Linear-v0')
    # env._max_episode_steps = 500
    # # env.seed(0)
    #
    # acreps = ACREPS(env=env,
    #                 n_samples=2500, n_keep=0,
    #                 n_rollouts=25, n_steps=150,
    #                 kl_bound=0.1, discount=0.95, trace=0.95,
    #                 vreg=1e-32, preg=1e-32, cov0=25.0,
    #                 vdgr=2, pdgr=1)
    #
    # for it in range(15):
    #     rwrd, kls, kli, klm, ent = acreps.run()
    #
    #     print('it=', it, f'rwrd={rwrd:{5}.{4}}',
    #           f'kls={kls:{5}.{4}}', f'kli={kli:{5}.{4}}',
    #           f'klm={klm:{5}.{4}}', f'ent={ent:{5}.{4}}')
