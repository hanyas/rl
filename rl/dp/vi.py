import numpy as np
from itertools import chain


class VI:
    def __init__(self, env):

        self.env = env

        self.sdim = self.env.sdim
        self.adim = self.env.adim

        self.states = self.env.states
        self.actions = self.env.actions

        self.dyn = self.env.dynamics
        self.r = self.env.reward
        self.p = self.env.prob

        self.V = None
        self.Q = None
        self.A = None

    def finhor(self, V, Q, horizon):
        V[-1, ...] = self.r(tuple(self.states))

        for h in range(horizon - 2, -1, -1):
            for state in chain.from_iterable(np.dstack(self.states)):
                for act in chain.from_iterable(np.dstack(self.actions)):
                    sn = tuple([h + 1]) + tuple(self.dyn(state, act))
                    sa = tuple([h]) + tuple(state) + tuple(act)
                    Q[sa] = self.r(tuple(state)) + V[sn]

            if self.p is not None:
                Q[h, ...] = np.einsum('hkm,ml->hkl', Q[h, ...], self.p())

            V[h, ...] = np.max(Q[h, ...], axis=-1)

        A = np.argmax(Q, axis=-1)
        return V, Q, A

    def infhor(self, V, Q, loop, discount):
        for l in range(loop):
            for state in chain.from_iterable(np.dstack(self.states)):
                for act in chain.from_iterable(np.dstack(self.actions)):
                    Q[tuple(state) + tuple(act)] = self.r(tuple(state)) +\
                                                   discount * V[tuple(self.dyn(state, act))]

            if self.p is not None:
                Q = np.einsum('hkm,ml->hkl', Q, self.p())

            V = np.max(Q, axis=-1)

        A = np.argmax(Q, axis=-1)
        return V, Q, A

    def run(self, type='fin', **kwargs):
        if type == 'fin':
            assert('horizon' in kwargs)
            horizon = kwargs['horizon']

            self.V = np.zeros((horizon, ) + self.sdim)
            self.Q = np.zeros((horizon - 1, ) + self.sdim + self.adim)

            self.V, self.Q, self.A = self.finhor(self.V, self.Q, horizon)

        if type == 'inf':
            assert('loop' in kwargs)
            loop = kwargs['loop']

            assert('discount' in kwargs)
            discount = kwargs['discount']

            self.V = np.zeros(self.sdim)
            self.Q = np.zeros(self.sdim + self.adim)

            self.V, self.Q, self.A = self.infhor(self.V, self.Q, loop, discount)


if __name__ == '__main__':
    from rl.dp.envs import Grid
    import matplotlib.pyplot as plt

    env = Grid()

    vi = VI(env)
    vi.run(type='fin', horizon=25, loop=25)

    ax = vi.env.world('Environment Finite Horizon')
    vi.env.policy(vi.A[0, ...], ax)
    plt.show()

    vi = VI(env)
    vi.run(type='inf', discount=0.9, loop=25)

    ax = vi.env.world('Environment Infinite Horizon')
    vi.env.policy(vi.A, ax)
    plt.show()
