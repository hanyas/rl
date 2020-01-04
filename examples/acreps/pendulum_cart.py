import autograd.numpy as np
import gym

from rl.acreps import acREPS

# np.random.seed(1337)

env = gym.make('Pendulum-RL-v1')
env._max_episode_steps = 500
# env.seed(1337)

acreps = acREPS(env=env, nb_samples=5000, nb_keep=0,
                nb_rollouts=25, nb_steps=250,
                kl_bound=0.1, discount=0.98, lmbda=0.95,
                vreg=1e-12, preg=1e-12, cov0=16.0,
                nb_vfeat=75, nb_pfeat=75,
                s_band=np.array([0.5, 0.5, 4.0]),
                sa_band=np.array([0.5, 0.5, 4.0, 1.0]),
                mult=0.5)

acreps.run(nb_iter=10, verbose=True)

# evaluate
acreps.ctl.cov = 0.1 * np.eye(acreps.dm_act)
rollouts, _ = acreps.evaluate(nb_rollouts=25, nb_steps=250, stoch=True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=acreps.dm_state + acreps.dm_act, figsize=(12, 4))
for roll in rollouts:
    for k, col in enumerate(ax[:-1]):
        col.plot(roll['x'][:, k])
    ax[-1].plot(roll['u'])
plt.show()

# # save ctl
# import pickle
# acreps.ctl.cov = np.eye(acreps.dm_act)
# pickle.dump(acreps.ctl, open("acreps_pendulum_cart.pkl", "wb"))
