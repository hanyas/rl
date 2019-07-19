import autograd.numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


rc('lines', **{'linewidth': 1})
rc('text', usetex=True)

def beautify(ax):
    ax.set_frame_on(True)
    ax.minorticks_on()

    ax.grid(True)
    ax.grid(linestyle=':')

    ax.tick_params(which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False,
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

    ax.autoscale(tight=True)
    ax.set_aspect('equal')

    if ax.get_legend():
        ax.legend(loc='best')

    return ax


def sample(env, nb_rollouts, nb_steps, dm_obs, dm_act):
    obs, act = [], []

    for n in range(nb_rollouts):
        _obs = np.empty((nb_steps, dm_obs))
        _act = np.empty((nb_steps, dm_act))

        x = env.reset()

        for t in range(nb_steps):
            u = np.random.uniform(-5., 5., size=(1,))

            _obs[t, :] = x
            _act[t, :] = u

            x, r, _, _ = env.step(u)

        obs.append(_obs)
        act.append(_act)

    return obs, act


if __name__ == "__main__":

    import os
    import pickle

    import gym
    import rl

    from sds import rARHMM

    env = gym.make('Pendulum-RL-v0')
    env._max_episode_steps = 5000

    nb_rollouts, nb_steps = 5, 2500
    dm_obs = env.observation_space.shape[0]
    dm_act = env.action_space.shape[0]
    nb_states = 5

    obs, act = sample(env, nb_rollouts, nb_steps, dm_obs, dm_act)

    rarhmm = rARHMM(nb_states=nb_states,
                    dm_obs=dm_obs,
                    dm_act=dm_act,
                    type='recurrent')

    rarhmm.initialize(obs, act)
    lls = rarhmm.em(obs=obs, act=act, nb_iter=100, prec=1e-4, verbose=True)

    # path = os.path.dirname(rl.__file__)
    # pickle.dump(rarhmm, open(path + '/envs/control/hybrid/models/hybrid_pendulum.p', 'wb'))
