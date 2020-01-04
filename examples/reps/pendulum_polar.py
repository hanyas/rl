import autograd.numpy as np
import gym

from rl.reps import REPS

# np.random.seed(1337)

env = gym.make('Pendulum-RL-v0')
env._max_episode_steps = 10000
env.unwrapped._dt = 0.01
env.unwrapped._sigma = 1e-16
# env.seed(1337)

reps = REPS(env=env, nb_samples=10000, nb_keep=0,
            nb_rollouts=25, nb_steps=500,
            kl_bound=0.1, discount=0.995,
            vreg=1e-16, preg=1e-16, cov0=25.0,
            nb_vfeat=100, nb_pfeat=100,
            band=np.array([np.pi/2., 4.0]),
            mult=0.5)

reps.run(nb_iter=10, verbose=True)

# evaluate
reps.ctl.cov = 0.1 * np.eye(reps.dm_act)
rollouts, _ = reps.evaluate(nb_rollouts=50, nb_steps=500, stoch=False)

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
# pickle.dump(reps.ctl, open("reps_pendulum_polar.pkl", "wb"))
