"""
A PyTorch implementation of PPO that learns a swing-up controller.
"""

from itertools import islice
import random
import numpy as np
import torch


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(batch_size, data_size):
    """
    Helper function for doing SGD on mini-batches.

    This function returns a generator returning random sub-samples.

    Example:
        If data_size = 5 and batch_size = 2, then the output might be
        out = ((0, 3), (2, 1), (4,)).

    :param batch_size: number of samples in each mini-batch
    :param data_size: total number of samples
    :return: generator of lists of indices
    """
    idx_all = random.sample(range(data_size), data_size)
    idx_iter = iter(idx_all)
    yield from iter(lambda: list(islice(idx_iter, batch_size)), [])


def rollouts(env, policy, nb_trans):
    """
    Collect `nb_trans` transitions in environment `env` under policy `policy`.
    """
    def _rollout(env, policy):
        obs = env.reset()
        done = False
        while not done:
            act = policy(obs)
            nobs, rwd, done, _ = env.step(act)
            yield obs, act, rwd, done
            obs = nobs
    keys = ['obs', 'act', 'rwd', 'done']
    paths = {}
    for k in keys:
        paths[k] = []
    nb_paths = 0
    while len(paths['rwd']) < nb_trans:
        for trans_vect in _rollout(env, policy):
            for key, val in zip(keys, trans_vect):
                paths[key].append(val)
        nb_paths += 1

    paths['obs'] = torch.from_numpy(np.stack(paths['obs'])).float()
    paths['act'] = torch.from_numpy(np.stack(paths['act'])).float()
    paths['rwd'] = torch.from_numpy(np.stack(paths['rwd'])).float()
    paths['done'] = np.stack(paths['done'])
    paths['nb_paths'] = nb_paths
    return paths


def render(env, policy):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        act = policy(obs)
        obs, _, done, _ = env.step(act)


class Net(torch.nn.Module):
    """
    Fully-connected tanh-activated 2-hidden-layer feed-forward neural network.
    """
    def __init__(self, layer_sizes):
        super(Net, self).__init__()
        self.in_to_L1 = torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        self.L1_to_L2 = torch.nn.Linear(layer_sizes[1], layer_sizes[2])
        self.L2_to_out = torch.nn.Linear(layer_sizes[2], layer_sizes[3])

    def forward(self, x):
        x1 = torch.tanh(self.in_to_L1(x))
        x2 = torch.tanh(self.L1_to_L2(x1))
        return self.L2_to_out(x2)


class Policy:
    """
    Gaussian policy with diagonal covariance matrix.

    The mean is state-dependent and is represented by a neural network.
    The covariance matrix is state-independent but nevertheless adjustable.
    """
    def __init__(self, s_dim, a_dim, sig0, hl_size=(64, 64), e_clip=0.1,
                 lr=1e-3, nb_epochs=10, batch_size=64):
        self._nb_epochs = nb_epochs
        self._batch_size = batch_size
        self._e_clip = e_clip
        self._log_prob_const = 0.5 * np.log(2 * np.pi)
        self._ent_const = 0.5 * a_dim + self._log_prob_const
        self._mu = Net((s_dim, *hl_size, a_dim))
        self._log_scale = torch.tensor(a_dim * [np.log(sig0)],
                                       requires_grad=True,
                                       dtype=torch.float32)
        self._opt = torch.optim.Adam([{'params': self._mu.parameters()},
                                      {'params': self._log_scale}], lr=lr)

    @torch.no_grad()
    def __call__(self, obs: np.ndarray):
        loc = self._mu(torch.as_tensor(obs, dtype=torch.float32))
        scale = torch.exp(self._log_scale)
        return torch.normal(loc, scale).numpy()

    @torch.no_grad()
    def mu(self, obs: np.ndarray):
        return self._mu(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def entropy(self):
        return self._log_scale.sum() + self._ent_const

    def log_probs(self, obs: torch.Tensor, act: torch.Tensor):
        dist = 0.5 * ((act - self._mu(obs)) * torch.exp(-self._log_scale)) ** 2
        neg_log_prob = (dist + self._log_scale + self._log_prob_const)
        return - neg_log_prob.sum(dim=1, keepdim=True)

    def _loss(self, log_probs, log_probs_old, adv):
        prob_ratio = torch.exp(log_probs - log_probs_old)
        clip_pr = prob_ratio.clamp(1 - self._e_clip, 1 + self._e_clip)
        return -torch.mean(torch.min(prob_ratio * adv, clip_pr * adv))

    def fit(self, paths, adv):
        loss = None
        with torch.no_grad():
            log_probs_old = self.log_probs(paths['obs'], paths['act'])
        for epoch in range(self._nb_epochs):
            for batch in batches(self._batch_size, paths['done'].shape[0]):
                log_probs = self.log_probs(paths['obs'][batch],
                                           paths['act'][batch])
                loss = self._loss(log_probs, log_probs_old[batch], adv[batch])
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
        return loss


class Advantage:
    """
    Advantage is calculated using GAE. Value function is parameterized by a NN.
    """
    def __init__(self, s_dim, hl_size=(64, 64), gam=0.99, lam=0.95,
                 lr=1e-3, nb_epochs=10, batch_size=64):
        self._gam = gam
        self._lam = lam
        self._nb_epochs = nb_epochs
        self._batch_size = batch_size
        self._v = Net((s_dim, *hl_size, 1))
        self._loss = torch.nn.MSELoss()
        self._opt = torch.optim.Adam(self._v.parameters(), lr=lr)

    @torch.no_grad()
    def _gae(self, paths):
        v_pred = self._v(paths['obs'])
        adv = torch.empty_like(v_pred, dtype=torch.float32)
        for k in reversed(range(paths['done'].shape[0])):
            if paths['done'][k]:
                adv[k] = paths['rwd'][k] - v_pred[k]
            else:
                adv[k] = paths['rwd'][k] + self._gam * v_pred[k + 1] \
                         - v_pred[k] + self._gam * self._lam * adv[k + 1]
        v_targ = v_pred + adv
        return adv, v_targ

    def fit(self, paths):
        loss = None
        for epoch in range(self._nb_epochs):
            _, v_targ = self._gae(paths)
            for batch_idx in batches(self._batch_size, paths['done'].shape[0]):
                v_pred = self._v(paths['obs'][batch_idx])
                loss = self._loss(v_pred, v_targ[batch_idx])
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
        adv, _ = self._gae(paths)
        return adv, loss


if __name__ == '__main__':
    import time
    import gym
    import lab

    class GentlyTerminating(gym.Wrapper):
        """
        This env wrapper sends zero command to the robot when an episode is done.
        """

        def step(self, action):
            observation, reward, done, info = self.env.step(action)
            if done:
                self.env.step(np.zeros(self.env.action_space.shape))
            return observation, reward, done, info

        def reset(self):
            return self.env.reset()


    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.style.use('seaborn')

    torch.set_num_threads(1)

    seed = 2739011
    nb_iter = 100
    nb_trans = 5000
    pol_params = dict(
        sig0=5.0,
        hl_size=(8, 8),
        e_clip=0.1,
        lr=1e-3,
        nb_epochs=15,
        batch_size=64
    )
    adv_params = dict(
        hl_size=(8, 8),
        gam=0.98,
        lam=0.95,
        lr=1e-3,
        nb_epochs=15,
        batch_size=64
    )

    set_global_seed(seed)
    env = gym.make('Cartpole-v0')
    env._max_episode_steps = 500
    env = GentlyTerminating(env)

    pol = Policy(env.observation_space.shape[0],
                 env.action_space.shape[0],
                 **pol_params)
    adv = Advantage(env.observation_space.shape[0], **adv_params)
    env.seed(seed)

    def pol_iter(nb_iter, nb_trans):
        for i in range(nb_iter):
            t_start = time.perf_counter()
            paths = rollouts(env, pol, nb_trans)
            t_roll = time.perf_counter()
            A, v_loss = adv.fit(paths)
            t_A_fit = time.perf_counter()
            p_loss = pol.fit(paths, A)
            t_pol_fit = time.perf_counter()
            # yield paths
            t_end = time.perf_counter()
            print(f'{i}: dt = {t_end - t_start:{5}.{4}} '
                  f'roll = {t_roll - t_start:{5}.{4}} '
                  f'adv_fit = {t_A_fit - t_roll:{5}.{4}} '
                  f'pol_fit = {t_pol_fit - t_A_fit:{5}.{4}}')
            print(f'    v_loss = {v_loss:{5}.{4}} '
                  f'p_loss = {p_loss:{5}.{4}} '
                  f'nb_paths = {paths["nb_paths"]}')
            print(paths['rwd'].sum().item() / paths['nb_paths'])
            if (i + 1) % 20 == 0:
                render(env, lambda x: pol.mu(x))

        return paths

    pol_iter(75, 5000)
