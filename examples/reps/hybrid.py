import autograd.numpy as np
import gym

from rl.reps import REPS

env = gym.make('Hybrid-v0')
env._max_episode_steps = 5000

reps = REPS(env=env, nb_samples=2500, nb_keep=0,
            nb_rollouts=25, nb_steps=150,
            kl_bound=0.1, discount=0.98,
            vreg=1e-16, preg=1e-12, cov0=25.0,
            nb_vfeat=15, nb_pfeat=15,
            band=np.array([1., 1.]),
            mult=1.0)

reps.run(nb_iter=10, verbose=True)
