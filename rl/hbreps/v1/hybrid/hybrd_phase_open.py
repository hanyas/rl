import autograd.numpy as np

import gym
import lab

import matplotlib.pyplot as plt
from matplotlib import cm

from misc.beautify import beautify


if __name__ == "__main__":
    env = gym.make('Hybrid-v1')
    env._max_episode_steps = 5000

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    lim = (-5.0, 5.0)
    npts = 20

    x = np.linspace(*lim, npts)
    y = np.linspace(*lim, npts)

    X, Y = np.meshgrid(x, y)
    XY = np.stack((X, Y))

    XYn = np.zeros((2, npts, npts))

    env.reset()

    for i in range(npts):
        for j in range(npts):
            env.unwrapped.fake_reset(XY[:, i, j])
            XYn[:, i, j], _, _, _  = env.step(np.array([0.0]))

    dydt = XYn - XY

    fig = plt.figure(figsize=(5, 5), frameon=True)
    ax = fig.gca()

    color = 2.0 * np.log(np.hypot(dydt[0, ...], dydt[1, ...]))
    ax.streamplot(x, y, dydt[0, ...], dydt[1, ...], color=color, linewidth=1,
                  cmap=plt.cm.Blues, density=1.25, arrowstyle='->', arrowsize=1.5)

    ax = beautify(ax)
    ax.grid(False)

    ax.set_xlim(lim)
    ax.set_ylim(lim)

    plt.show()

    # from matplotlib2tikz import save as tikz_save
    # tikz_save("hybrid_dyn.tex")
