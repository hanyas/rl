import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import autograd.numpy as np

import gym

from rl.hyreps.v1 import BaumWelch
from rl.hyreps.v1 import HyREPS
from rl.hyreps.v1 import cart_polar

import pickle
from joblib import Parallel, delayed


import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

import seaborn as sns


np.set_printoptions(precision=5, suppress=True)

sns.set_style("white")
sns.set_context("paper")

color_names = ["red", "windows blue", "medium green", "dusty purple", "orange",
               "amber", "clay", "pink", "greyish", "light cyan", "steel blue",
               "forest green", "pastel purple", "mint", "salmon", "dark brown"]

colors = []
for k in color_names:
    colors.append(mcd.XKCD_COLORS['xkcd:' + k].upper())


def backplot(plt, z):
    jumps = np.where(np.diff(z) != 0)[0]
    begin = np.r_[0, jumps + 1]
    end = np.r_[jumps, bw.n_steps - 1]

    for i in range(begin.shape[0]):
        plt.axvspan(begin[i], end[i] + 1, facecolor=colors[z[begin[i]]], alpha=1.0)

    plt.xlim(0, z.shape[0] - 1)


def sample(env, ctl, n_rollouts, n_steps, n_states, n_actions):
    x = np.zeros((n_rollouts, n_steps, n_states))
    u = np.zeros((n_rollouts, n_steps, n_actions))

    for n in range(n_rollouts):
        x[n, 0, :] = env.reset()

        for t in range(1, n_steps):
            u[n, t - 1, :] = ctl.actions(x[n, t - 1, :], True)
            x[n, t, :], _, _, _ = env.step(u[n, t - 1, :])
    return x, u


def baumWelchFunc(args):
    x, u, w, n_regions, priors, regs, rslds, update = args
    rollouts = x.shape[0]
    choice = np.random.choice(rollouts, size=int(0.8 * rollouts), replace=False)
    x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
    bw = BaumWelch(x, u, w, n_regions, priors, regs, rslds, update)
    lklhd = bw.run(n_iter=100, save=False)
    return bw, lklhd


if __name__ == "__main__":
    # np.random.seed(0)
    env = gym.make('Pendulum-v1')
    env._max_episode_steps = 5000
    # env.seed(0)

    n_rollouts, n_steps = 50, 250
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 5

    file = open('reps_pndlv1_ctl.pickle', 'rb')
    opt_ctl = pickle.load(file)
    file.close()
    opt_ctl.cov = 0.5 * np.eye(n_actions)

    x, u = sample(env, opt_ctl, n_rollouts, n_steps, n_states, n_actions)
    w = np.ones((n_rollouts, n_steps))

    dyn_prior = {'nu': n_states + 2, 'psi': 1e-2}
    ctl_prior = {'nu': n_actions + 2, 'psi': 0.5}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(np.float64).tiny, 1e-16, 1e-16, 1e-16])

    # do id
    n_jobs = 25
    args = [(x, u, w, n_regions, priors, regs, None, [True, True]) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))
    bw = bwl[np.argmax(lklhd)]

    # test hybrid policy
    hyreps = HyREPS(env, n_regions,
                    n_samples=5000, n_iter=5,
                    n_rollouts=n_rollouts, n_steps=n_steps, n_keep=0,
                    kl_bound=0.1, discount=0.99,
                    vreg=1e-12, preg=1e-12, cov0=8.0,
                    rslds=bw.rslds, priors=priors)

    # overwrite initialization
    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    _, eval = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=True)

    x = eval['x'].reshape((-1, hyreps.n_steps, hyreps.n_states))
    u = eval['u'].reshape((-1, hyreps.n_steps, hyreps.n_actions))

    args = [(x, u, w, n_regions, priors, regs, bw.rslds, [False, True]) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))
    bw = bwl[np.argmax(lklhd)]

    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    rollouts, _ = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=False)

    lgnd = ['Cos, Sin', 'Velocity', 'Angle', 'Reward', 'Action', 'Region']
    fig, axs = plt.subplots(nrows=2, ncols=3)
    for i, ax in enumerate(fig.axes):
        ax.set_title(lgnd[i])

    for roll in rollouts:
        angle = cart_polar(roll['x'])[0, :]
        axs[0, 0].plot(roll['x'][:, :-1])
        axs[0, 1].plot(roll['x'][:, -1])
        axs[0, 2].plot(angle)
        axs[1, 0].plot(roll['r'])
        axs[1, 1].plot(np.clip(roll['u'],
                           env.action_space.low, env.action_space.high))
        axs[1, 2].plot(roll['z'])
    plt.show()
