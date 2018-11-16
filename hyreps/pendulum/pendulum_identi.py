import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd

import seaborn as sns

import gym
import lab

from rl.reps.reps_numpy import Policy
from rl.reps.reps_numpy import FourierFeatures

from rl.hyreps import BaumWelch
from rl.hyreps.hyreps_v0 import HyREPS

from rl.hyreps import cart_polar

import pickle
import copy
from joblib import Parallel, delayed


np.set_printoptions(precision=10, suppress=True)

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
    choice = np.random.choice(50, size=40, replace=False)

    if len(args) == 6:
        x, u, w, n_regions, priors, regs = args
        x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
        bw = BaumWelch(x, u, w, n_regions, priors, regs)
    else:
        x, u, w, n_regions, priors, regs, rslds = args
        x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
        bw = BaumWelch(x, u, w, n_regions, priors, regs, rslds=rslds)

    lklhd = bw.run(n_iter=100, save=False)
    return bw, lklhd


if __name__ == "__main__":
    # np.random.seed(2409810241)
    env = gym.make('Pendulum-v1')
    env._max_episode_steps = 5000
    # env.seed(17023313094129695151)

    n_rollouts, n_steps = 50, 250
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 5

    file = open("reps_pendulum-v1_ctl.pickle", "rb")
    opt_ctl = pickle.load(file)
    file.close()
    opt_ctl.cov = 0.5 * np.eye(n_actions)

    x, u = sample(env, opt_ctl, n_rollouts, n_steps, n_states, n_actions)
    w = np.ones((n_rollouts, n_steps))

    dyn_prior = {"nu": 7, "psi": 1e-2}
    ctl_prior = {"nu": 3, "psi": 0.5}
    priors = [dyn_prior, ctl_prior]

    # msg, dyn, lgstc, ctl
    regs = np.array([np.finfo(float).tiny, 1e-16, 1e-12, 1e-12])

    # do system identification
    n_jobs = 1
    args = [(x, u, w, n_regions, priors, regs) for _ in range(n_jobs)]
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
    hyreps.rslds = copy.deepcopy(bw.rslds)
    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    _, eval = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=True)

    State = eval['x'].reshape((-1, hyreps.n_steps, hyreps.n_states))
    Action = eval['u'].reshape((-1, hyreps.n_steps, hyreps.n_actions))

    # retrieve policy again
    x, u = State, Action

    args = [(x, u, w, n_regions, priors, regs, bw.rslds) for _ in range(n_jobs)]
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

    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    _, eval = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=False)

    State = eval['x'].reshape((-1, hyreps.n_steps, hyreps.n_states))
    Angle = cart_polar(State)
    Region = eval['z'].reshape((-1, hyreps.n_steps,))
    Reward = eval['r'].reshape((-1, hyreps.n_steps,))
    Action = eval['u'].reshape((-1, hyreps.n_steps, hyreps.n_actions))

    plt.figure()
    sfig0 = plt.subplot(231)
    plt.title('Cos, Sin')
    sfig1 = plt.subplot(232)
    plt.title('Velocity')
    sfig2 = plt.subplot(233)
    plt.title('Angle')
    sfig3 = plt.subplot(234)
    plt.title('Reward')
    sfig4 = plt.subplot(235)
    plt.title('Action')
    sfig5 = plt.subplot(236)
    plt.title('Region')

    for rollout in range(hyreps.n_rollouts):
        sfig0.plot(State[rollout, :, :-1])
        sfig1.plot(State[rollout, :, -1])
        sfig2.plot(Angle[rollout, :, 0])
        sfig3.plot(Reward[rollout, :])
        sfig4.plot(np.clip(Action[rollout, :, :],
                           env.action_space.low, env.action_space.high))
        sfig5.plot(Region[rollout, :])
    plt.show()
