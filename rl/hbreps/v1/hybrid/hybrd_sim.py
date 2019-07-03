import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import autograd.numpy as np

import gym

from rl.hyreps.v1.hyreps_v import HyREPS_V

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # np.random.seed(0)
    env = gym.make('Hybrid-v1')
    env._max_episode_steps = 5000
    # env.seed(0)

    n_rollouts, n_steps = 25, 1000
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 2

    dyn_prior = {'nu': n_states + 2, 'psi': 1e-4}
    ctl_prior = {'nu': n_actions + 2, 'psi': 0.1}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(np.float64).tiny, 1e-32, 1e-32, 1e-32])

    hybrd_rslds = env.unwrapped.rslds

    hyreps = HyREPS_V(env, n_regions, n_samples=5000,
                    n_rollouts=n_rollouts, n_steps=n_steps, n_keep=0,
                    kl_bound=0.1, discount=0.99,
                    vreg=1e-12, preg=1e-12, cov0=8.0,
                    rslds=hybrd_rslds, priors=priors)

    # overwrite initialization
    for n in range(n_regions):
        hyreps.ctl.K[n] = hybrd_rslds.linear_ctls[n].K
        hyreps.ctl.cov[n] = hybrd_rslds.linear_ctls[n].cov

    rollouts, _ = hyreps.evaluate(hyreps.n_rollouts, hyreps.n_steps, stoch=False)


    fig, axs = plt.subplots(nrows=1, ncols=1)

    for n in range(0, 3):
        axs.plot(rollouts[n]['x'][:200, 0])
        # axs.plot(rollouts[n]['x'][:200, 1])
        # axs.plot(rollouts[n]['z'][:200])

    axs.set_frame_on(True)
    axs.minorticks_on()

    axs.tick_params(which='both', direction='in',
                    bottom=True, labelbottom=True,
                    top=True, labeltop=False,
                    right=True, labelright=False,
                    left=True, labelleft=True)

    axs.tick_params(which='major', length=6)
    axs.tick_params(which='minor', length=3)

    axs.autoscale(tight=True)

    plt.show()

    # from matplotlib2tikz import save as tikz_save
    # tikz_save("hyreps_msd_trajs_region.tex")