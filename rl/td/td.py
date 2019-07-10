import numpy as np


class TD:

    def __init__(self, env, discount, alpha):
        self.env = env

        self.d_state = 16  # self.env.observation_space.shape[0]
        self.d_action = 4  # self.env.action_space.shape[0]

        self.discount = discount

        # random policy
        self.ctl = 1.0 / self.d_action * np.ones((self.d_action,))

        self.alpha = alpha

        self.vfunc = np.random.randn(self.d_state, )

        self.td_error = []
        self.rollouts = None

    def eval(self, n_samples):
        rollouts = []

        n_samp = 0
        n_eps = 0
        while True:
            roll = {'x': np.empty((0, ), np.int64),
                    'u': np.empty((0, ), np.int64),
                    'xn': np.empty((0, ), np.int64),
                    'done': np.empty((0,), np.int64),
                    'r': np.empty((0,))}

            # reset env
            x = self.env.reset()

            done = False
            while not done:
                u = np.random.choice(self.d_action, p=self.ctl)

                roll['x'] = np.hstack((roll['x'], x))
                roll['u'] = np.hstack((roll['u'], u))

                xn, r, done, _ = self.env.step(u)
                roll['xn'] = np.hstack((roll['xn'], xn))
                roll['done'] = np.hstack((roll['done'], done))
                roll['r'] = np.hstack((roll['r'], r))

                err = 0.0
                if not done:
                    err = r + self.discount * self.vfunc[xn] - self.vfunc[x]
                if done:
                    err = r - self.vfunc[x]

                self.vfunc[x] += self.alpha * err
                self.td_error = np.append(self.td_error, err)

                x = xn

                n_samp += 1
                if n_samp >= n_samples:
                    roll['done'][-1] = True
                    rollouts.append(roll)
                    return rollouts

            print("eps: {}, error: {}".format(n_eps, self.td_error[-1]))

            n_eps += 1
            rollouts.append(roll)


if __name__ == "__main__":
    import gym
    from matplotlib import pyplot as plt

    env = gym.make('FrozenLake-v0')

    td = TD(env, discount=0.95, alpha=0.25)
    td.eval(n_samples=10000)

    print(td.vfunc)

    plt.plot(td.td_error)
    plt.show()
