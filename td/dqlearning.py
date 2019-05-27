import numpy as np


def greedy(x, qfunc, eps, d_actions):
    pmf = eps / d_actions * np.ones((d_actions, ))
    idx = np.argmax(qfunc[x, :])
    pmf[idx] = 1.0 - np.sum(np.concatenate((pmf[:idx], pmf[idx + 1:])), axis=0)
    pmf = pmf / np.sum(pmf)
    return np.argmax(np.random.multinomial(1, pmf))


def softmax(x, qfunc, beta, d_actions):
    pmf = np.exp(np.clip(qfunc[x, :] / beta, -700, 700))
    pmf = pmf / np.sum(pmf)
    return np.argmax(np.random.multinomial(1, pmf))


class DQLearning:

    def __init__(self, env, n_episodes, discount, alpha, type):
        self.env = env

        self.d_state = 16  # self.env.observation_space.shape[0]
        self.d_action = 4  # self.env.action_space.shape[0]

        self.n_episodes = n_episodes
        self.discount = discount

        self.type = type
        self.eps = 0.1  # epsilon greedy
        self.beta = 0.98  # softmax
        self.ctl = 1.0 / self.d_action * np.ones((self.d_action, ))  # random

        self.alpha = alpha

        self.vfunc = np.zeros((2, self.d_state, ))
        self.qfunc = np.zeros((2, self.d_state, self.d_action))

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

            x = self.env.reset()

            if self.type == 'greedy':
                self.eps = 1.0 / (1.0 * (n + 1.0))
            elif self.type == 'softmax':
                self.beta = self.beta * 0.9995

            done = False
            while not done:
                qidx = 0 if np.random.uniform() < .5 else 1

                if self.type == 'rnd':
                    u = np.random.choice(self.d_action, p=self.ctl)
                elif self.type == 'greedy':
                    u = greedy(x, self.qfunc[qidx, ...], self.eps, self.d_action)
                elif self.type == 'softmax':
                    u = softmax(x, self.qfunc[qidx, ...], self.beta, self.d_action)

                roll['x'] = np.hstack((roll['x'], x))
                roll['u'] = np.hstack((roll['u'], u))

                xn, r, done, _ = self.env.step(u)
                roll['xn'] = np.hstack((roll['xn'], xn))
                roll['r'] = np.hstack((roll['r'], r))

                err = 0.0
                if not done:
                    amax = np.argmax(self.qfunc[qidx, xn, :])[np.newaxis]
                    err = r + self.discount * self.qfunc[1 - qidx, xn, np.random.choice(amax)] - self.qfunc[qidx, x, u]
                if done:
                    err = r - self.qfunc[qidx, x, u]

                self.qfunc[qidx, x, u] += self.alpha * err
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

    dqlearning = DQLearning(env, n_episodes=10000, discount=0.95, alpha=0.1, type='softmax')
    dqlearning.run()
