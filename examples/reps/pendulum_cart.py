import autograd.numpy as np
import gym

from rl.reps import REPS

# np.random.seed(1337)

env = gym.make('Pendulum-RL-v1')
env._max_episode_steps = 5000
env.unwrapped._dt = 0.05
env.unwrapped._sigma = 1e-8
# env.seed(1337)

reps = REPS(env=env, nb_samples=5000, nb_keep=0,
            nb_rollouts=25, nb_steps=250,
            kl_bound=0.1, discount=0.99,
            vreg=1e-12, preg=1e-12, cov0=25.0,
            nb_vfeat=75, nb_pfeat=75,
            band=np.array([0.5, 0.5, 4.0]),
            mult=0.5)

reps.run(nb_iter=15, verbose=True)

# evaluate
reps.ctl.cov = 0.1 * np.eye(reps.dm_act)
rollouts, _ = reps.evaluate(nb_rollouts=50, nb_steps=250, stoch=False)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=reps.dm_state + reps.dm_act, figsize=(12, 4))
for roll in rollouts:
    for k, col in enumerate(ax[:-1]):
        col.plot(roll['x'][:, k])
    ax[-1].plot(roll['u'])
plt.show()

# # save ctl
# import pickle
# reps.ctl.cov = np.eye(reps.dm_act)
# pickle.dump(reps.ctl, open("reps_pendulum_cart.pkl", "wb"))
