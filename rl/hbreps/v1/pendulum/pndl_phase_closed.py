import autograd.numpy as np
import gym

import pickle

from rl.hyreps.v1.util import cart_polar

import matplotlib.pyplot as plt
from misc.beautify import beautify


if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    env._max_episode_steps = 5000

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    file = open('reps_pndlv1_ctl.pickle', 'rb')
    opt_ctl = pickle.load(file)
    file.close()

    xlim = (-np.pi, np.pi)
    ylim = (-8.0, 8.0)

    npts = 20

    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)

    X, Y = np.meshgrid(x, y)
    XY = np.stack((X, Y))

    XYn = np.zeros((2, npts, npts))

    env.reset()

    for i in range(npts):
        for j in range(npts):
            state = env.unwrapped.fake_reset(XY[:, i, j])
            u = opt_ctl.actions(state, False)
            state, _, _, _  = env.step(u)
            XYn[:, i, j] = cart_polar(state)

    dydt = XYn - XY

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    ax.streamplot(x, y, dydt[0, ...], dydt[1, ...], linewidth=1, color='r',
                  cmap=plt.cm.Reds, density=1.25, arrowstyle='->', arrowsize=1.5)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()

    # from matplotlib2tikz import save as tikz_save
    # tikz_save("pendulum_dyn_closed.tex")
