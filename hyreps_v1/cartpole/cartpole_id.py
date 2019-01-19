import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import autograd.numpy as np

import gym
import lab

from rl.reps.reps_numpy import Policy
from rl.reps.reps_numpy import FourierFeatures

from rl.hyreps import BaumWelch
from rl.hyreps import HyREPS

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
    x, u, w, n_regions, priors, regs, rslds = args
    rollouts = x.shape[0]
    choice = np.random.choice(rollouts, size=int(0.8 * rollouts), replace=False)
    x, u, w = x[choice, ...], u[choice, ...], w[choice, ...]
    bw = BaumWelch(x, u, w, n_regions, priors, regs, rslds)
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

    file = open('acreps_cartpole_ctl.pickle', 'rb')
    opt_ctl = pickle.load(file)
    file.close()

    x, u = sample(env, opt_ctl, n_rollouts, n_steps, n_states, n_actions)
    w = np.ones((n_rollouts, n_steps))

    dyn_prior = {'nu': n_states + 2, 'psi': 1e-2}
    ctl_prior = {'nu': n_actions + 2, 'psi': 1e1}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(np.float64).tiny, 1e-16, 1e-16, 1e-16])

    # do id
    n_jobs = 25
    args = [(x, u, w, n_regions, priors, regs, None) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))
    bw = bwl[np.argmax(lklhd)]

    # test hybrid policy
    hyreps = HyREPS(env, n_regions,
                    n_samples=5000, n_iter=5,
                    n_rollouts=n_rollouts, n_steps=n_steps, n_keep=0,
                    kl_bound=0.1, discount=0.99,
                    vreg=1e-16, preg=1e-12, cov0=4.0,
                    rslds=bw.rslds, priors=priors)

    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    _, eval = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=True)

    x = eval['x'].reshape((-1, hyreps.n_steps, hyreps.n_states))
    u = eval['u'].reshape((-1, hyreps.n_steps, hyreps.n_actions))

    args = [(x, u, w, n_regions, priors, regs, bw.rslds) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))
    bw = bwl[np.argmax(lklhd)]

    for n in range(n_regions):
        hyreps.ctl.K[n] = bw.rslds.linear_policy[n].K
        hyreps.ctl.cov[n] = bw.rslds.linear_policy[n].cov

    _, eval = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=False, render=True)
