import numpy as np


class Policy:

    def __init__(self, d_state, d_action, **kwargs):
        self.d_state = d_state
        self.d_action = d_action

        self.type = kwargs.get('type', False)

        if 'beta' in kwargs:
            self.beta = kwargs.get('beta', False)
        if 'eps' in kwargs:
            self.eps = kwargs.get('eps', False)
        if 'weights' in kwargs:
            self.weights = kwargs.get('weights', False)

    def anneal(self, n=0):
        if self.type == 'softmax':
            self.beta = self.beta * 0.9995
        elif self.type == 'greedy':
            self.eps = 1.0 / (1.0 * (n + 1.0))
        else:
            pass

    def action(self, qfunc, x):
        if self.type == 'softmax':
            pmf = np.exp(np.clip(qfunc[x, :] / self.beta, -700, 700))
        elif self.type == 'geedy':
            pmf = self.eps / self.d_action * np.ones((self.d_action, ))
            idx = np.argmax(qfunc[x, :])
            pmf[idx] = 1.0 - np.sum(np.concatenate((pmf[:idx], pmf[idx + 1:])), axis=0)
        else:
            return np.random.choice(self.d_action, p=self.weights)

        pmf = pmf / np.sum(pmf)
        return np.argmax(np.random.multinomial(1, pmf))


class QLearning:

    def __init__(self, env, n_episodes, discount, alpha, **kwargs):
        self.env = env

        self.d_state = 16  # self.env.observation_space.shape[0]
        self.d_action = 4  # self.env.action_space.shape[0]

        self.n_episodes = n_episodes
        self.discount = discount

        self.ctl = Policy(self.d_state, self.d_action, **kwargs)

        self.alpha = alpha

        self.vfunc = np.zeros((self.d_state, ))
        self.qfunc = np.zeros((self.d_state, self.d_action))

        self.td_error = []

        self.rollouts = []

    def run(self):
        score = np.empty((0, 1))

        rollouts = []

        for n in range(self.n_episodes):
            roll = {'x': np.empty((0,), np.int64),
                    'xn': np.empty((0,), np.int64),
                    'u': np.empty((0,), np.int64),
                    'r': np.empty((0,))}

            # reset env
            x = self.env.reset()

            # anneal policy
            self.ctl.anneal(n=n)

            done = False
            while not done:
                u = self.ctl.action(self.qfunc, x)

                roll['x'] = np.hstack((roll['x'], x))
                roll['u'] = np.hstack((roll['u'], u))

                xn, r, done, _ = self.env.step(u)
                roll['xn'] = np.hstack((roll['xn'], xn))
                roll['r'] = np.hstack((roll['r'], r))

                err = 0.0
                if not done:
                    err = r + self.discount * np.max(self.qfunc[xn, :]) - self.qfunc[x, u]
                if done:
                    err = r - self.qfunc[x, u]

                self.qfunc[x, u] += self.alpha * err
                self.td_error = np.append(self.td_error, err)

                x = xn

                if len(score) < 100:
                    score = np.append(score, r)
                else:
                    score[n % 100] = r

            print("it: {} step: {} rwd:{} score:{}".format(n, len(roll['r']), r, np.mean(score, axis=0)))

            rollouts.append(roll)

        return rollouts


if __name__ == "__main__":
    import gym

    env = gym.make('FrozenLake-v0')

    qlearning = QLearning(env, n_episodes=10000, discount=0.95, alpha=0.1, type='softmax')
    qlearning.run()
