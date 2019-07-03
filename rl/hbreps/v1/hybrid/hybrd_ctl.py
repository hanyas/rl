import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import numpy as np

import gym
import lab

from rl.hyreps.v1.hyreps_v import HyREPS_V


np.set_printoptions(precision=10, suppress=True)


if __name__ == "__main__":
    # np.random.seed(1337)
    env = gym.make('Hybrid-v1')
    env._max_episode_steps = 5000
    # env.seed(1337)

    n_rollouts, n_steps = 50, 200
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    n_regions = 2

    dyn_prior = {'nu': n_states + 2, 'psi': 1e-4}
    ctl_prior = {'nu': n_actions + 2, 'psi': 8.0}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(float).tiny, 1e-16, 1e-16, 1e-16])

    hybrd_rslds = env.unwrapped.rslds

    env._max_episode_steps = 5000
    hyreps = HyREPS_V(env, n_regions, n_samples=5000,
                      n_rollouts=25, n_steps=200, n_keep=0,
                      kl_bound=0.05, discount=0.99,
                      vreg=1e-6, preg=regs, cov0=8.0,
                      rslds=hybrd_rslds, priors=priors,
                      degree=2)

    hyreps.run()
