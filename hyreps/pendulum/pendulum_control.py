import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import numpy as np

import gym
import lab

from rl.reps.reps_numpy import Policy
from rl.reps.reps_numpy import FourierFeatures

from rl.hyreps import BaumWelch

import pickle
from joblib import Parallel, delayed


np.set_printoptions(precision=10, suppress=True)



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
    regs = np.array([np.finfo(float).tiny, 1e-16, 1e-12, 1e-12])

    # do system identification
    n_jobs = 25
    args = [(x, u, w, n_regions, priors, regs) for _ in range(n_jobs)]
    results = Parallel(n_jobs=n_jobs, verbose=0)(map(delayed(baumWelchFunc), args))
    bwl, lklhd = list(map(list, zip(*results)))

    bw = bwl[np.argmax(lklhd)]


    # dyn_prior = {"nu": 7, "psi": 1e-2}
    # ctl_prior = {"nu": 3, "psi": 8.0}
    # priors = [dyn_prior, ctl_prior]
    # regs = np.array([np.finfo(float).tiny, 1e-16, 1e-6, 1e-6])
    #
    # env._max_episode_steps = 5000
    # from rl.hyreps.hyreps_v0 import HyREPS
    # hyreps = HyREPS(env, n_regions,
    #                 n_samples=5000, n_iter=10,
    #                 n_rollouts=25, n_steps=250, n_keep=0,
    #                 kl_bound=0.1, discount=0.99,
    #                 vreg=1e-16, preg=regs, cov0=8.0,
    #                 rslds=bw.rslds, priors=priors,
    #                 band=0.5 * np.array([0.5, 0.5, 4.0]), n_vfeat=75)


    dyn_prior = {"nu": 7, "psi": 1e-2}
    ctl_prior = {"nu": 3, "psi": 8.0}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(float).tiny, 1e-16, 1e-12, 1e-12])

    env._max_episode_steps = 5000
    from rl.hyreps.hyreps_v1 import HyREPS
    hyreps = HyREPS(env, n_regions,
                    n_samples=5000, n_iter=10,
                    n_rollouts=25, n_steps=250, n_keep=0,
                    kl_bound=0.05, discount=0.96,
                    vreg=1e-32, preg=regs, cov0=8.0,
                    rslds=bw.rslds, priors=priors,
                    degree=3)


    # dyn_prior = {"nu": 7, "psi": 1e-2}
    # ctl_prior = {"nu": 3, "psi": 16.0}
    # priors = [dyn_prior, ctl_prior]
    # regs = np.array([np.finfo(float).tiny, 1e-16, 1e-4, 1e-6])
    #
    # env._max_episode_steps = 500
    # from rl.hyreps.hyreps_v2 import HyREPS_AC
    # hyreps = HyREPS_AC(env, n_regions,
    #                    n_iter=10, n_rollouts=25,
    #                    n_steps=250, n_keep=0,
    #                    kl_bound=0.1, discount=0.99, trace=0.95,
    #                    vreg=1e-16, preg=regs, cov0=16.0,
    #                    rslds=bw.rslds, priors=priors,
    #                    band=np.array([0.5, 0.5, 4.0]), n_vfeat=75)

    hyreps.run(n_iter=15)
