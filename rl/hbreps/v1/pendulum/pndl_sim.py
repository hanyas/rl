import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import autograd.numpy as np

import gym

from rl.hyreps.v1.rslds import rSLDS
from rl.hyreps.v1.hyreps_v import HyREPS_V
from rl.hyreps.v1.util import cart_polar

import matplotlib.pyplot as plt

import copy


if __name__ == "__main__":
    # np.random.seed(0)
    env = gym.make('Pendulum-v1')
    env._max_episode_steps = 5000
    # env.seed(0)

    n_rollouts, n_steps = 100, 250
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 3

    dyn_prior = {'nu': n_states + 2, 'psi': 1e-2}
    ctl_prior = {'nu': n_actions + 2, 'psi': 0.5}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(np.float64).tiny, 1e-16, 1e-16, 1e-16])

    rslds = rSLDS(n_states, n_actions, n_regions, priors)
    rslds.load("hybrd_pndlv1_rslds_imit_3z.pickle")

    hyreps = HyREPS_V(env, n_regions, n_samples=5000,
                    n_rollouts=n_rollouts, n_steps=n_steps, n_keep=0,
                    kl_bound=0.1, discount=0.99,
                    vreg=1e-12, preg=1e-12, cov0=8.0,
                    rslds=rslds, priors=priors)

    # overwrite initialization
    for n in range(n_regions):
        hyreps.ctl.K[n] = copy.deepcopy(rslds.linear_ctls[n].K)
        hyreps.ctl.cov[n] = copy.deepcopy(rslds.linear_ctls[n].cov)

    rollouts, _ = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=False)

    # fig, axs = plt.subplots(nrows=1, ncols=1)
    # for n in range(0, 5):
        # angle = cart_polar(rollouts[n]['x'])[0, :]
        # axs.plot(angle)
        # axs.plot(rollouts[n]['x'][:150, -1])
        # axs.plot(rollouts[n]['z'][:1500])

    # fig, axs = plt.subplots(nrows=1, ncols=1)
    # for n in range(75):
    #     angle = cart_polar(rollouts[n]['x'])[0, :]
    #     vel = rollouts[n]['x'][:, -1]
    #     s = [2 for i in range(angle.shape[0])]
    #     axs.scatter(angle[:100], vel[:100], s=s, c='blue', marker='.')
    #
    # axs.set_frame_on(True)
    # axs.minorticks_on()
    #
    # axs.tick_params(which='both', direction='in',
    #                 bottom=True, labelbottom=True,
    #                 top=True, labeltop=False,
    #                 right=True, labelright=False,
    #                 left=True, labelleft=True)
    #
    # axs.tick_params(which='major', length=6)
    # axs.tick_params(which='minor', length=3)
    #
    # axs.autoscale(tight=True)
    #
    # plt.show()
    #
    # from matplotlib2tikz import save as tikz_save
    # tikz_save("hyreps_pndl_imit_phase.tex")