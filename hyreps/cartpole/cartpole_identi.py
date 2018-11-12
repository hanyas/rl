import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

import gym

from rl.hyreps import BaumWelch
from rl.hyreps.hyreps_v0 import HyREPS

import pickle
from joblib import Parallel, delayed


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

    if len(args) == 5:
        x, u, w, n_regions, priors = args
        x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
        bw = BaumWelch(x, u, w, n_regions, priors)
    else:
        x, u, w, n_regions, priors, rslds = args
        x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
        bw = BaumWelch(x, u, w, n_regions, priors, rslds=rslds)

    lklhd = bw.run(n_iter=100, save=False)
    return bw, lklhd


if __name__ == "__main__":
    np.random.seed(0)
    env = gym.make('Cartpole-v0')
    env._max_episode_steps = 5000
    env.seed(0)

    n_rollouts, n_steps = 50, 500
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 5

    file = open("acreps_cartpole_ctl.pickle", "rb")
    opt_ctl = pickle.load(file)
    file.close()

    x, u = sample(env, opt_ctl, n_rollouts, n_steps, n_states, n_actions)
    w = np.ones((n_rollouts, n_steps))

    dyn_prior = {"nu": 2 * n_states + 1, "psi": 1e-2}
    ctl_prior = {"nu": 2 * n_actions + 1, "psi": 1e1}
    priors = [dyn_prior, ctl_prior]

    # do system identification
    n_jobs = 25
    args = [(x, u, w, n_regions, priors) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=1)(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))

    bw = bwl[np.argmax(lklhd)]

    # test hybrid policy
    hyreps = HyREPS(env, n_regions,
                    n_samples=5000, n_iter=5,
                    n_rollouts=n_rollouts, n_steps=n_steps, n_keep=1000,
                    kl_bound=0.1, discount=0.99,
                    vreg=1e-16, preg=1e-12, cov0=4.0,
                    rslds=bw.rslds, priors=priors)

    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    eval = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=True)

    # retrieve policy again
    x = eval['x'].reshape((-1, hyreps.n_steps, hyreps.n_states))
    u = eval['u'].reshape((-1, hyreps.n_steps, hyreps.n_actions))

    args = [(x, u, w, n_regions, priors, bw.rslds) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=1)(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))

    bw = bwl[np.argmax(lklhd)]

    # test hybrid policy
    hyreps = HyREPS(env, n_regions,
                    n_samples=5000, n_iter=5,
                    n_rollouts=n_rollouts, n_steps=n_steps, n_keep=1000,
                    kl_bound=0.1, discount=0.99,
                    vreg=1e-16, preg=1e-12, cov0=4.0,
                    rslds=bw.rslds, priors=priors)

    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    eval = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=False, render=True)
