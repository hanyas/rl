import numpy as np


class MC:

    def __init__(self, env, n_episodes, discount):
        self.env = env

        self.d_state = 16  # self.env.observation_space.shape[0]
        self.dm_act = 4  # self.env.action_space.shape[0]

        self.n_episodes = n_episodes
        self.discount = discount

        self.ctl = 1.0 / self.dm_act * np.ones((self.dm_act,))

        self.vfunc = np.zeros((self.d_state,))
        self.ret = [[] for _ in range(self.d_state)]

        self.rollouts = []

    def eval(self):
        rollouts = []

        for _ in range(self.n_episodes):
            roll = {'x': np.empty((0,), np.int64),
                    'xn': np.empty((0,), np.int64),
                    'u': np.empty((0,), np.int64),
                    'r': np.empty((0,))}

            x = self.env.reset()

            done = False
            while not done:
                u = np.random.choice(self.dm_act, p=self.ctl)

                roll['x'] = np.hstack((roll['x'], x))
                roll['u'] = np.hstack((roll['u'], u))

                x, r, done, _ = self.env.step(u)
                roll['xn'] = np.hstack((roll['xn'], x))
                roll['r'] = np.hstack((roll['r'], r))

            G = 0.0
            for t in range(len(roll['r']) - 1, -1, -1):
                G = self.discount * G + roll['r'][t]
                for s in range(self.d_state):
                    if s not in roll['x'][:t]:
                        self.ret[s].append(G)
                        self.vfunc[s] = np.mean(self.ret[s])

            rollouts.append(roll)

        return rollouts
