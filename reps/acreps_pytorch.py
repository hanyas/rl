import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

import torch
import torch.nn as nn

from itertools import islice
import random


EXP_MAX = 700.0
EXP_MIN = -700.0


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

    def __init__(self, n_states, n_actions,
                 cov, band, n_feat,
                 n_epochs=100, n_batch=64,
                 lr=1e-3):

        self.n_states = n_states
        self.n_actions = n_actions

        self.cov = cov
        self.band = band
        self.n_feat = n_feat
        self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)

        self.mu = RandomFourierNet(self.n_feat, self.n_actions)
        self.log_std = torch.tensor(self.n_actions * [np.log(np.sqrt(cov))],
                                    requires_grad=True,
                                    dtype=torch.float32)

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

    def __init__(self, n_states, epsi,
                 n_feat, band,
                 discount, trace,
                 n_epochs=100, n_batch=64,
                 lr=1e-3):

        self.n_states = n_states
        self.epsi = epsi

        self.band = band
        self.n_feat = n_feat
        self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)

        self.discount = discount
        self.trace = trace

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
                         self.discount * self.trace * adv[k + 1]

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
                 n_samples, n_iter,
                 n_rollouts, n_steps, n_keep,
                 kl_bound, discount, trace,
                 n_vfeat, n_pfeat,
                 cov0, band):

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
        self.trace = trace

        self.n_vfeat = n_vfeat
        self.n_pfeat = n_pfeat

        self.band = band

        self.dual = Dual(self.n_states, n_feat=self.n_vfeat, band=self.band,
                         discount=self.discount, trace=self.trace, epsi=self.kl_bound)

        self.ctl = Policy(self.n_states, self.n_actions, cov=cov0, n_feat=self.n_pfeat,
                          band=self.band)

        self.action_limit = self.env.action_space.high

        self.data = {'x': np.empty((0, self.n_states)),
                     'u': np.empty((0, self.n_actions)),
                     'xn': np.empty((0, self.n_states)),
                     'r': np.empty((0, 1)),
                     'done': np.empty((0,), np.int64)}

        self.vfeatures = None
        self.mvfeatures = None

        self.targets = None

        self.advantage = None
        self.weights = None

        self.pfeatures = None

    def sample(self, n_samples, n_keep=0, stoch=True, render=False):
        if n_keep==0:
            data = {'x': np.empty((0, self.n_states)),
                    'u': np.empty((0, self.n_actions)),
                    'xn': np.empty((0, self.n_states)),
                    'r': np.empty((0, 1)),
                    'done': np.empty((0,), np.int64)}
        else:
            data = {
                    'x': self.data['x'][-n_keep:, :],
                    'u': self.data['u'][-n_keep:, :],
                    'xn': self.data['xn'][-n_keep:, :],
                    'r': self.data['r'][-n_keep:, :],
                    'done': self.data['done'][-n_keep:]}

        n = 0
        while True:
            x = self.env.reset()

            while True:
                u = self.ctl.actions(x, stoch)

                data['x'] = np.vstack((data['x'], x))
                data['u'] = np.vstack((data['u'], u))

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                data['xn'] = np.vstack((data['xn'], x))
                data['r'] = np.vstack((data['r'], r))
                data['done'] = np.hstack((data['done'], done))

                n = n + 1
                if n >= n_samples:
                    data['done'][-1] = True

                    data['x'] = torch.from_numpy(np.stack(data['x'])).float()
                    data['u'] = torch.from_numpy(np.stack(data['u'])).float()
                    data['xn'] = torch.from_numpy(np.stack(data['xn'])).float()
                    data['r'] = torch.from_numpy(np.stack(data['r'])).float()

                    return data

                if done:
                    break

    def evaluate(self, n_rollouts, n_steps, render=False):
        data = {'x': np.empty((0, self.n_states)),
                'u': np.empty((0, self.n_actions)),
                'r': np.empty((0, 1))}

        for n in range(n_rollouts):
            x = self.env.reset()

            for t in range(n_steps):
                u = self.ctl.actions(x, False)

                data['x'] = np.vstack((data['x'], x))
                data['u'] = np.vstack((data['u'], u))

                x, r, done, _ = self.env.step(
                    np.clip(u, - self.action_limit, self.action_limit))
                if render:
                    self.env.render()

                data['r'] = np.vstack((data['r'], r))

                if done:
                    break

        data['x'] = torch.from_numpy(np.stack(data['x'])).float()
        data['u'] = torch.from_numpy(np.stack(data['u'])).float()
        data['r'] = torch.from_numpy(np.stack(data['r'])).float()

        return data

    def kl_samples(self):
        w = torch.clamp(self.weights, 1e-75, np.inf)
        w = w / torch.mean(w, dim=0)
        return torch.mean(w * torch.log(w), dim=0).numpy().squeeze()

    @torch.no_grad()
    def entropy(self):
        return torch.log(torch.exp(self.ctl.log_std) * np.sqrt(2.0 * np.pi * np.exp(1.0))).numpy().squeeze()

    def run(self, n_iter):
        for it in range(n_iter):
            # eval = self.evaluate(self.n_rollouts, self.n_steps)

            self.data = self.sample(self.n_samples)

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
            kl = self.kl_samples()

            self.pfeatures = self.ctl.features(self.data['x'])
            ploss = self.ctl.minimize(self.data['u'], self.pfeatures, self.weights)

            ent = self.entropy()

            # rwrd = torch.sum(eval['r']).numpy() / self.n_rollouts
            rwrd = torch.sum(self.data['r']).numpy() / np.sum(self.data['done'])

            print('it=', it, f'dual={dual:{5}.{4}}', f'ploss={ploss:{5}.{4}}',
                  f'rwrd={rwrd:{5}.{4}}', f'kl={kl:{5}.{4}}', f'ent={ent:{5}.{4}}' )


if __name__ == "__main__":

    import gym
    import lab

    torch.set_num_threads(4)

    np.random.seed(0)
    env = gym.make('Pendulum-v0')
    env._max_episode_steps = 1000
    env.seed(0)

    torch.manual_seed(0)
    random.seed(0)

    acreps = ACREPS(env=env,
                    n_samples=5000, n_iter=25,
                    n_rollouts=25, n_steps=200, n_keep=0,
                    kl_bound=0.1, discount=0.98, trace=0.95,
                    n_vfeat=75, n_pfeat=75,
                    cov0=4.0, band=np.array([0.5, 0.5, 4.0]))

    acreps.run(acreps.n_iter)