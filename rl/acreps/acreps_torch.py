import numpy as np
import scipy as sc
from scipy import stats

import math

import torch
import torch.nn as nn

from itertools import islice
import random

to_torch = lambda arr: torch.from_numpy(arr).float()
to_npy = lambda arr: arr.detach().double().numpy()


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


def batches(batch_size, data_size):
    idx_all = random.sample(range(data_size), data_size)
    idx_iter = iter(idx_all)
    yield from iter(lambda: list(islice(idx_iter, batch_size)), [])


class FourierFeatures:

    def __init__(self, dm_state, nb_feat, band, mult):
        self.dm_state = dm_state
        self.nb_feat = nb_feat
        self.mult = mult

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.dm_state),
                                                  cov=np.diag(1.0 / (mult * band)),
                                                  size=self.nb_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.nb_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        return phi


class Policy(nn.Module):

    def __init__(self, dm_state, dm_act,
                 cov, band, nb_feat, mult,
                 nb_epochs=250, batch_size=64,
                 lr=1e-3):
        super(Policy, self).__init__()

        self.dm_state = dm_state
        self.dm_act = dm_act

        self.cov = cov
        self.band = band
        self.nb_feat = nb_feat
        self.mult = mult
        self.basis = FourierFeatures(self.dm_state, self.nb_feat,
                                     self.band, self.mult)

        self.mu = nn.Linear(self.nb_feat, self.dm_act, bias=False)

        _cov = cov * torch.eye(self.dm_act)
        self.log_std = nn.Parameter(self.dm_act * torch.log(torch.sqrt(_cov)))

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def features(self, x):
        return to_torch(self.basis.fit_transform(x))

    @torch.no_grad()
    def forward(self, x, stoch):
        feat = self.features(x)
        if stoch:
            return torch.normal(self.mu(feat), torch.exp(self.log_std))
        else:
            return self.mu(feat)

    @torch.no_grad()
    def entropy(self):
        return 0.5 * torch.log(torch.det(torch.exp(self.log_std)**2) * 2.0 * math.pi * math.e)

    def log_likelihood(self, u, feat, w):
        a = self.mu(feat)
        var = torch.exp(self.log_std) ** 2
        log_lik = - (u - a) ** 2 / (2.0 * var) - self.log_std
        ll = - torch.mean(w * log_lik)
        return ll

    def minimize(self, u, feat, w):
        loss = None
        self.mu.reset_parameters()
        for epoch in range(self.nb_epochs):
            for batch in batches(self.batch_size, feat.shape[0]):
                self.opt.zero_grad()
                loss = self.log_likelihood(u[batch], feat[batch], w[batch])
                loss.backward()
                self.opt.step()
        return loss


class Dual(nn.Module):

    def __init__(self, dm_state, epsi,
                 nb_feat, band, mult,
                 discount, lmbda,
                 nb_epochs=250, batch_size=64,
                 lr=1e-3):
        super(Dual, self).__init__()

        self.dm_state = dm_state
        self.epsi = epsi

        self.band = band
        self.mult = mult
        self.nb_feat = nb_feat
        self.basis = FourierFeatures(self.dm_state, self.nb_feat,
                                     self.band, self.mult)

        self.discount = discount
        self.lmbda = lmbda

        self.vfunc = nn.Linear(self.nb_feat, 1)
        self.kappa = nn.Parameter(torch.as_tensor(1.0))

        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def features(self, x):
        return to_torch(self.basis.fit_transform(x))

    @torch.no_grad()
    def values(self, feat):
        return self.vfunc(feat)

    @torch.no_grad()
    def eta(self):
        return torch.exp(self.kappa)

    @torch.no_grad()
    def gae(self, feat, data):
        values = self.values(feat)
        adv = torch.zeros_like(values)

        for rev_k, v in enumerate(reversed(values)):
            k = len(values) - rev_k - 1
            if data["done"][k]:
                adv[k] = data["r"][k] - values[k]
            else:
                adv[k] = data["r"][k] + self.discount * values[k + 1] - values[k] + \
                         self.discount * self.lmbda * adv[k + 1]
        return values + adv

    def loss(self, feat, mfeat, targets):
        adv = targets - self.vfunc(feat)
        w = torch.exp(torch.clamp((adv - torch.max(adv)) / torch.exp(self.kappa), EXP_MIN, EXP_MAX))
        g = torch.max(adv) + self.vfunc(mfeat) + torch.exp(self.kappa) * self.epsi + torch.exp(self.kappa) * torch.log(torch.mean(w))
        return g

    def minimize(self, feat, mfeat, targets):
        loss = None
        for epoch in range(self.nb_epochs):
            for batch in batches(self.batch_size, feat.shape[0]):
                self.opt.zero_grad()
                loss = self.loss(feat[batch], mfeat, targets[batch])
                loss.backward()
                self.opt.step()
        return loss


class acREPS:

    def __init__(self, env,
                 nb_samples, nb_keep,
                 nb_rollouts, nb_steps,
                 kl_bound, discount, lmbda,
                 nb_vfeat, nb_pfeat,
                 cov0, band, mult):

        self.env = env

        self.dm_state = self.env.observation_space.shape[0]
        self.dm_act = self.env.action_space.shape[0]

        self.nb_samples = nb_samples
        self.nb_keep = nb_keep

        self.nb_rollouts = nb_rollouts
        self.nb_steps = nb_steps

        self.kl_bound = kl_bound
        self.discount = discount
        self.lmbda = lmbda

        self.nb_vfeat = nb_vfeat
        self.nb_pfeat = nb_pfeat

        self.band = band
        self.mult = mult

        self.dual = Dual(self.dm_state, nb_feat=self.nb_vfeat, band=self.band, mult=self.mult,
                         discount=self.discount, lmbda=self.lmbda, epsi=self.kl_bound)

        self.ctl = Policy(self.dm_state, self.dm_act, cov=cov0,
                          nb_feat=self.nb_pfeat, band=self.band, mult=self.mult)

        self.ulim = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.vfeatures = None
        self.mvfeatures = None

        self.targets = None

        self.advantage = None
        self.weights = None

        self.pfeatures = None

    def sample(self, nb_samples, nb_keep=0, stoch=True, render=False):
        if len(self.rollouts) >= nb_keep:
            rollouts = random.sample(self.rollouts, nb_keep)
        else:
            rollouts = []

        n = 0
        while True:
            roll = {'x': np.empty((0, self.dm_state)),
                    'u': np.empty((0, self.dm_act)),
                    'xn': np.empty((0, self.dm_state)),
                    'r': np.empty((0, 1)),
                    'done': np.empty((0,), np.int64)}

            x = self.env.reset()

            done = False
            while not done:
                u = to_npy(self.ctl.forward(x, stoch))

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.ulim, self.ulim))
                if render:
                    self.env.render()

                roll['xn'] = np.vstack((roll['xn'], x))
                roll['r'] = np.vstack((roll['r'], r))
                roll['done'] = np.hstack((roll['done'], done))

                n = n + 1
                if n >= nb_samples:
                    roll['done'][-1] = True
                    rollouts.append(roll)
                    data = merge(*rollouts)

                    data['x'] = to_torch(data['x'])
                    data['u'] = to_torch(data['u'])
                    data['xn'] = to_torch(data['xn'])
                    data['r'] = to_torch(data['r'])

                    return rollouts, data

            rollouts.append(roll)

    def evaluate(self, nb_rollouts, nb_steps, render=False):
        rollouts = []

        for n in range(nb_rollouts):
            roll = {'x': np.empty((0, self.dm_state)),
                    'u': np.empty((0, self.dm_act)),
                    'r': np.empty((0, 1))}

            x = self.env.reset()

            for t in range(nb_steps):
                u = to_npy(self.ctl.forward(x, False))

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.ulim, self.ulim))
                if render:
                    self.env.render()

                roll['r'] = np.vstack((roll['r'], r))

            rollouts.append(roll)

        data = merge(*rollouts)

        data['x'] = to_torch(data['x'])
        data['u'] = to_torch(data['u'])
        data['r'] = to_torch(data['r'])

        return rollouts, data

    def kl_samples(self, weights):
        w = torch.clamp(weights, 1e-75, np.inf)
        w = w / torch.mean(w, dim=0)
        return torch.mean(w * torch.log(w))

    def run(self, nb_iter=10, verbose=False):
        _trace = {'rwrd': [],
                  'dual': [], 'ploss': [], 'kl': [],
                  'ent': []}

        for it in range(nb_iter):
            _, eval = self.evaluate(self.nb_rollouts, self.nb_steps)

            self.rollouts, self.data = self.sample(self.nb_samples)

            self.vfeatures = self.dual.features(self.data['x'])
            self.mvfeatures = torch.mean(self.vfeatures, dim=0)
            self.targets = self.dual.gae(self.vfeatures, self.data)

            dual = self.dual.minimize(self.vfeatures, self.mvfeatures, self.targets)

            self.advantage = self.targets - self.dual.values(self.vfeatures)
            self.weights = torch.exp(torch.clamp((self.advantage - torch.max(self.advantage)) /
                                                 self.dual.eta(), EXP_MIN, EXP_MAX))
            kl = self.kl_samples(self.weights)

            self.pfeatures = self.ctl.features(self.data['x'])
            ploss = self.ctl.minimize(self.data['u'], self.pfeatures, self.weights)

            ent = self.ctl.entropy()

            # rwrd = torch.mean(self.data['r'])
            rwrd = torch.mean(eval['r'])

            _trace['rwrd'].append(to_npy(rwrd))
            _trace['dual'].append(to_npy(dual))
            _trace['ploss'].append(to_npy(ploss))
            _trace['kl'].append(to_npy(kl))
            _trace['ent'].append(to_npy(ent))

            if verbose:
                print('it=', it,
                      f'rwrd={to_npy(rwrd):{5}.{4}}',
                      f'dual={to_npy(dual):{5}.{4}}',
                      f'ploss={to_npy(ploss):{5}.{4}}',
                      f'kl={to_npy(kl):{5}.{4}}',
                      f'ent={to_npy(ent):{5}.{4}}')

        return _trace
