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


if __name__ == "__main__":

    import gym
    import rl

    env = gym.make('Hybrid-v0')
    env._max_episode_steps = 5000

    lim = (-10.0, 10.0)
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
            XYn[:, i, j], _, _, _ = env.step(np.array([0.0]))

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
