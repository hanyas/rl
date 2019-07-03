import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import autograd.numpy as np

import gym

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

from rl.hyreps.v0 import BaumWelch


def sample(env, n_rollouts, n_steps, n_states, n_actions):
    x = np.zeros((n_rollouts, n_steps, n_states))
    u = np.zeros((n_rollouts, n_steps, n_actions))

    for n in range(n_rollouts):
        x[n, 0, :] = env.reset()

        for t in range(1, n_steps):
            u[n, t - 1, :] = np.random.normal(loc=0.0, scale=5.0)
            x[n, t, :], _, _, _ = env.step(u[n, t - 1, :])

        u[n, - 1, :] = np.random.normal(loc=0.0, scale=5.0)

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
    env = gym.make('Hybrid-v0')
    env._max_episode_steps = 5000
    # env.seed(0)

    n_rollouts, n_steps = 100, 100
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 2

    x, u = sample(env, n_rollouts, n_steps, n_states, n_actions)
    w = np.ones((n_rollouts, n_steps))

    dyn_prior = {'nu': n_states + 2, 'psi': 1e-4}
    ctl_prior = {'nu': n_actions + 2, 'psi': 0.1}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(np.float64).tiny, 1e-32, 1e-32, 1e-32])

    # do id
    n_jobs = 1
    args = [(x, u, w, n_regions, priors, regs, None, [True, False]) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Iteraions")
    ax.set_ylabel("Likelihood")
    ax.grid()

    lklhd_fin = []
    for i in range(len(lklhd)):
        lklhd_fin.append(lklhd[i][-1])
        plt.plot(lklhd[i], 'b-')
    plt.show()

    plt.xlim(left=0, right=100)
    plt.ylim(bottom=-1e4, top=1e5)
    ax.relim()

    bw = bwl[np.argmax(lklhd_fin)]
