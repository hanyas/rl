import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import numpy as np

import gym

from rl.hyreps.v1.em import BaumWelch
from rl.hyreps.v1.rslds import rSLDS

from rl.hyreps.v1.hyreps_v import HyREPS_V
from rl.hyreps.v1.hyreps import HyREPS
from rl.hyreps.v1.hyreps_ac import HyREPS_AC


def sample(env, ctl, n_rollouts, n_steps, n_states, n_actions):
    x = np.zeros((n_rollouts, n_steps, n_states))
    u = np.zeros((n_rollouts, n_steps, n_actions))

    for n in range(n_rollouts):
        x[n, 0, :] = env.reset()

        for t in range(1, n_steps):
            u[n, t - 1, :] = ctl.actions(x[n, t - 1, :], True)
            x[n, t, :], _, _, _ = env.step(u[n, t - 1, :])

        u[n, - 1, :] = ctl.actions(x[n, -1, :], True)

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
    # np.random.seed(1337)
    env = gym.make('Pendulum-v1')
    env._max_episode_steps = 5000
    # env.seed(1337)

    n_rollouts, n_steps = 50, 150
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 3

    dyn_prior = {'nu': n_states + 2, 'psi': 1e-2}
    ctl_prior = {'nu': n_actions + 2, 'psi': 0.5}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(float).tiny, 1e-16, 1e-16, 1e-16])

    # file = open('reps_pndlv1_ctl.pickle', 'rb')
    # opt_ctl = pickle.load(file)
    # file.close()
    # opt_ctl.cov = 0.5 * np.eye(n_actions)
    #
    # x, u = sample(env, opt_ctl, n_rollouts, n_steps, n_states, n_actions)
    # w = np.ones((n_rollouts, n_steps))
    #
    # # do system identification
    # n_jobs = 25
    # args = [(x, u, w, n_regions, priors, regs, None) for _ in range(n_jobs)]
    # results = Parallel(n_jobs=n_jobs, verbose=0)(map(delayed(baumWelchFunc), args))
    # bwl, lklhd = list(map(list, zip(*results)))
    # bw = bwl[np.argmax(lklhd)]

    pndl_rslds = rSLDS(n_states, n_actions, n_regions, priors)
    pndl_rslds.load("hybrd_pndlv1_rslds_imit_3z.pickle")

    # dyn_prior = {'nu': 5, 'psi': 1e-2}
    # ctl_prior = {'nu': 3, 'psi': 8.0}
    # priors = [dyn_prior, ctl_prior]
    # regs = np.array([np.finfo(float).tiny, 1e-16, 1e-16, 1e-16])
    #
    # env._max_episode_steps = 5000
    # hyreps = HyREPS(env, n_regions,
    #                 n_samples=5000, n_iter=10,
    #                 n_rollouts=25, n_steps=250, n_keep=0,
    #                 kl_bound=0.1, discount=0.98,
    #                 vreg=1e-6, preg=regs, cov0=16.0,
    #                 rslds=pndl_rslds, priors=priors,
    #                 band=np.array([0.5, 0.5, 4.0]), n_vfeat=75)


    # dyn_prior = {'nu': 5, 'psi': 1e-2}
    # ctl_prior = {'nu': 3, 'psi': 16.0}
    # priors = [dyn_prior, ctl_prior]
    # regs = np.array([np.finfo(float).tiny, 1e-16, 1e-16, 1e-16])
    #
    # env._max_episode_steps = 500
    # hyreps = HyREPS_AC(env, n_regions,
    #                    n_rollouts=25, n_steps=250, n_keep=0,
    #                    kl_bound=0.1, discount=0.98, trace=0.95,
    #                    vreg=1e-6, preg=regs, cov0=16.0,
    #                    rslds=pndl_rslds, priors=priors,
    #                    band=np.array([0.5, 0.5, 4.0]), n_vfeat=75)


    dyn_prior = {'nu': 5, 'psi': 1e-2}
    ctl_prior = {'nu': 3, 'psi': 8.0}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(float).tiny, 1e-16, 1e-16, 1e-16])

    env._max_episode_steps = 5000

    for i in range(10):
        hyreps = HyREPS_V(env, n_regions, n_samples=5000,
                          n_rollouts=50, n_steps=150, n_keep=0,
                          kl_bound=0.05, discount=0.99,
                          vreg=1e-6, preg=regs, cov0=8.0,
                          rslds=pndl_rslds, priors=priors,
                          degree=3)

        hyreps.run()
