import numpy as np


class TD:

    def __init__(self, env, n_episodes, discount, alpha):
        self.env = env

        self.d_state = 16  # self.env.observation_space.shape[0]
        self.d_action = 4  # self.env.action_space.shape[0]

        self.n_episodes = n_episodes
        self.discount = discount

        self.ctl = 1.0 / self.d_action * np.ones((self.d_action,))

        self.alpha = alpha

        self.vfunc = np.random.randn(self.d_state, )
        self.td_error = []

        self.rollouts = []

    def eval(self):
        rollouts = []

        for n in range(self.n_episodes):
            roll = {'x': np.empty((0,), np.int64),
                    'xn': np.empty((0,), np.int64),
                    'u': np.empty((0,), np.int64),
                    'r': np.empty((0,))}

            x = self.env.reset()

            done = False
            while not done:
                u = np.random.choice(self.d_action, p=self.ctl)

                roll['x'] = np.hstack((roll['x'], x))
                roll['u'] = np.hstack((roll['u'], u))

                xn, r, done, _ = self.env.step(u)
                roll['xn'] = np.hstack((roll['xn'], xn))
                roll['r'] = np.hstack((roll['r'], r))

                if not done:
                    self.vfunc[x] += self.discount * (r + self.discount * self.vfunc[xn] - self.vfunc[x])
                    self.td_error = np.append(self.td_error, r + self.discount * self.vfunc[xn] - self.vfunc[x])
                if done:
                    self.vfunc[x] += self.alpha * (r - self.vfunc[x])
                    self.td_error = np.append(self.td_error, r - self.vfunc[x])

                x = xn

            print("it:", n, "td error:", self.td_error[-1])

            rollouts.append(roll)

        return rollouts


if __name__ == "__main__":
    import gym
    from matplotlib import pyplot as plt

    env = gym.make('FrozenLake-v0')

    td = TD(env, n_episodes=1000, discount=0.95, alpha=0.25)
    td.eval()

    print(td.vfunc)

    plt.plot(td.td_error)
    plt.show()
