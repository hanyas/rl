import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

import torch
import torch.nn as nn

from itertools import islice
import random


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


class RandomFourierNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(RandomFourierNet, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.linear.weight = torch.nn.Parameter(1e-8 * torch.ones(output_dim, input_dim))

    def forward(self, x):
        return self.linear(x)


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

    def __init__(self, dim_state, dim_action,
                 cov, band, n_feat,
                 n_epochs=100, n_batch=64,
                 lr=1e-3):

        self.dim_state = dim_state
        self.dim_action = dim_action

        self.cov = cov
        self.band = band
        self.n_feat = n_feat
        self.basis = FourierFeatures(self.dim_state, self.n_feat, self.band)

        self.mu = RandomFourierNet(self.n_feat, self.dim_action)
        self.log_std = torch.tensor(self.dim_action * [np.log(np.sqrt(cov))],
                                    requires_grad=True, dtype=torch.float32)

        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.lr = lr
        self.opt = torch.optim.Adam([{'params': self.mu.parameters()},
                                     {'params': self.log_std}], lr=self.lr)

    @torch.no_grad()
    def features(self, x):
        return torch.as_tensor(self.basis.fit_transform(x), dtype=torch.float32)

    @torch.no_grad()
    def actions(self, x, stoch):
        feat = self.features(x)
        with torch.no_grad():
            mean = self.mu(feat)
            if stoch:
                return torch.normal(mean, torch.exp(self.log_std))
            else:
                return mean

    @torch.no_grad()
    def entropy(self):
        return 0.5 * np.log(torch.exp(self.log_std)**2 * 2.0 * np.pi * np.exp(1.0)).numpy().squeeze()

    def loss(self, u, feat, w):
        a = self.mu(feat)
        var = torch.exp(self.log_std) ** 2
        log_prob = - (u - a) ** 2 / (2.0 * var) - self.log_std
        l = - torch.mean(w * log_prob)
        return l

    def minimize(self, u, feat, w):
        loss = None
        for epoch in range(self.n_epochs):
            for batch in batches(self.n_batch, feat.shape[0]):
                loss = self.loss(u[batch], feat[batch], w[batch])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return loss


class Dual:

    def __init__(self, dim_state, epsi,
                 n_feat, band,
                 discount, lmbda,
                 n_epochs=100, n_batch=64,
                 lr=1e-3):

        self.dim_state = dim_state
        self.epsi = epsi

        self.band = band
        self.n_feat = n_feat
        self.basis = FourierFeatures(self.dim_state, self.n_feat, self.band)

        self.discount = discount
        self.lmbda = lmbda

        self.vfunc = RandomFourierNet(self.n_feat, 1)
        self.kappa = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.lr = lr
        self.opt = torch.optim.Adam([{'params': self.vfunc.parameters()},
                                    {'params': self.kappa}], lr=self.lr)

    @torch.no_grad()
    def features(self, x):
        return torch.as_tensor(self.basis.fit_transform(x), dtype=torch.float32)

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

        return (values + adv)


    def loss(self, feat, mfeat, targets):
        adv = targets - self.vfunc(feat)
        w = torch.exp(torch.clamp((adv - torch.max(adv)) / torch.exp(self.kappa), EXP_MIN, EXP_MAX))
        g = self.vfunc(mfeat) + torch.max(adv) +\
            torch.exp(self.kappa) * self.epsi + torch.exp(self.kappa) * torch.log(torch.mean(w))
        return g.squeeze()

    def minimize(self, feat, mfeat, targets):
        loss = None
        for epoch in range(self.n_epochs):
            for batch in batches(self.n_batch, feat.shape[0]):
                loss = self.loss(feat[batch], mfeat, targets[batch])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return loss


class ACREPS:

    def __init__(self, env,
                 n_samples, n_keep, n_rollouts,
                 kl_bound, discount, lmbda,
                 n_vfeat, n_pfeat,
                 cov0, band):

        self.env = env

        self.dim_state = self.env.observation_space.shape[0]
        self.dim_action = self.env.action_space.shape[0]

        self.n_samples = n_samples
        self.n_keep = n_keep
        self.n_rollouts = n_rollouts

        self.kl_bound = kl_bound
        self.discount = discount
        self.lmbda = lmbda

        self.n_vfeat = n_vfeat
        self.n_pfeat = n_pfeat

        self.band = band

        self.dual = Dual(self.dim_state, n_feat=self.n_vfeat, band=self.band,
                         discount=self.discount, lmbda=self.lmbda, epsi=self.kl_bound)

        self.ctl = Policy(self.dim_state, self.dim_action, cov=cov0, n_feat=self.n_pfeat,
                          band=self.band)

        self.action_limit = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.vfeatures = None
        self.mvfeatures = None

        self.targets = None

        self.advantage = None
        self.weights = None

        self.pfeatures = None

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
                    'r': np.empty((0, 1)),
                    'done': np.empty((0,), np.int64)}

            x = self.env.reset()

            done = False
            while not done:
                u = self.ctl.actions(x, stoch)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                roll['xn'] = np.vstack((roll['xn'], x))
                roll['r'] = np.vstack((roll['r'], r))
                roll['done'] = np.hstack((roll['done'], done))

                n = n + 1
                if n >= n_samples:
                    roll['done'][-1] = True
                    rollouts.append(roll)
                    data = merge(*rollouts)

                    data['x'] = torch.from_numpy(np.stack(data['x'])).float()
                    data['u'] = torch.from_numpy(np.stack(data['u'])).float()
                    data['xn'] = torch.from_numpy(np.stack(data['xn'])).float()
                    data['r'] = torch.from_numpy(np.stack(data['r'])).float()

                    return rollouts, data

            rollouts.append(roll)

    def evaluate(self, n_rollouts, render=False):
        rollouts = []

        for n in range(n_rollouts):
            roll = {'x': np.empty((0, self.dim_state)),
                    'u': np.empty((0, self.dim_action)),
                    'r': np.empty((0, 1))}

            x = self.env.reset()

            done = False
            while not done:
                u = self.ctl.actions(x, False)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                roll['r'] = np.vstack((roll['r'], r))

            rollouts.append(roll)

        data = merge(*rollouts)

        data['x'] = torch.from_numpy(np.stack(data['x'])).float()
        data['u'] = torch.from_numpy(np.stack(data['u'])).float()
        data['r'] = torch.from_numpy(np.stack(data['r'])).float()

        return rollouts, data

    def kl_samples(self, weights):
        w = torch.clamp(weights, 1e-75, np.inf)
        w = w / torch.mean(w, dim=0)
        return torch.mean(w * torch.log(w), dim=0).numpy().squeeze()

    def run(self, nb_iter=10, verbose=False):
        _trace = {'rwrd': [],
                  'dual': [], 'ploss': [], 'kl': [],
                  'ent': []}

        for it in range(nb_iter):
            # _, eval = self.evaluate(self.n_rollouts, self.n_steps)

            self.rollouts, self.data = self.sample(self.n_samples)

            self.vfeatures = self.dual.features(self.data['x'])
            self.mvfeatures = torch.mean(self.vfeatures, dim=0)

            self.targets = self.dual.gae(self.vfeatures, self.data)

            # reset parameters
            self.dual.vfunc = RandomFourierNet(self.n_vfeat, 1)
            self.kappa = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)

            dual = self.dual.minimize(self.vfeatures, self.mvfeatures,
                                      self.targets)

            self.advantage = self.targets - self.dual.values(self.vfeatures)
            self.weights = torch.exp(torch.clamp((self.advantage - torch.max(self.advantage)) /
                                                 self.dual.eta(), EXP_MIN, EXP_MAX))
            kl = self.kl_samples(self.weights)

            self.pfeatures = self.ctl.features(self.data['x'])
            ploss = self.ctl.minimize(self.data['u'], self.pfeatures, self.weights)

            ent = self.ctl.entropy()

            rwrd = torch.mean(self.data['r']).numpy()
            # rwrd = torch.mean(eval['r']).numpy()

            _trace['rwrd'].append(rwrd)
            _trace['dual'].append(dual)
            _trace['ploss'].append(ploss)
            _trace['klm'].append(kl)
            _trace['ent'].append(ent)

            if verbose:
                print('it=', it,
                      f'rwrd={rwrd:{5}.{4}}',
                      f'dual={dual:{5}.{4}}',
                      f'ploss={ploss:{5}.{4}}',
                      f'klm={kl:{5}.{4}}',
                      f'ent={ent:{5}.{4}}')

            if ent < -3e2:
                break

        return _trace
