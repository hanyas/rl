import numpy as np


class Eligibity:

    def __init__(self, d_state, dm_act, type='accumulate'):
        self.d_state = d_state
        self.dm_act = dm_act

        self.type = type

        self.table = np.zeros((self.d_state, self.dm_act))

    def reset(self):
        self.table = np.zeros((self.d_state, self.dm_act))

    def update(self, x, u):
        if self.type == 'accumulate':
            self.table[x, u] += 1.0
        else:
            self.table[x, u] = 1.0


class Policy:

    def __init__(self, d_state, dm_act, pdict):
        self.d_state = d_state
        self.dm_act = dm_act

        self.type = pdict['type']

        if 'beta' in pdict:
            self.beta = pdict['beta']
        if 'eps' in pdict:
            self.eps = pdict['eps']
        if 'weights' in pdict:
            self.weights = pdict['weights']

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
            return np.random.choice(self.dm_act, p=pmf/np.sum(pmf))
        elif self.type == 'greedy':
            if self.eps >= np.random.rand():
                return np.random.choice(self.dm_act)
            else:
                return np.argmax(qfunc[x, :])
        else:
            return np.random.choice(self.dm_act, p=self.weights)


class SARSA:

    def __init__(self, env, discount, alpha, lmbda, pdict):
        self.env = env

        self.d_state = 16  # self.env.observation_space.shape[0]
        self.dm_act = 4  # self.env.action_space.shape[0]

        self.discount = discount

        self.ctl = Policy(self.d_state, self.dm_act, pdict)

        self.lmbda = lmbda
        self.traces = Eligibity(self.d_state, self.dm_act)

        self.alpha = alpha

        self.vfunc = np.zeros((self.d_state, ))
        self.qfunc = np.zeros((self.d_state, self.dm_act))

        self.td_error = []
        self.rollouts = None

    def run(self, nb_samples):
        score = np.empty((0, 1))

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
            u = self.ctl.action(self.qfunc, x)

            # anneal policy
            self.ctl.anneal(n=n_eps)

            # reset trace
            self.traces.reset()

            done = False
            while not done:
                roll['x'] = np.hstack((roll['x'], x))
                roll['u'] = np.hstack((roll['u'], u))

                xn, r, done, _ = self.env.step(u)
                roll['xn'] = np.hstack((roll['xn'], xn))
                roll['done'] = np.hstack((roll['done'], done))
                roll['r'] = np.hstack((roll['r'], r))

                un = self.ctl.action(self.qfunc, xn)

                self.traces.update(x, u)

                err = 0.0
                if not done:
                    err = r + self.discount * self.qfunc[xn, un] - self.qfunc[x, u]
                    self.qfunc += self.alpha * err * self.traces.table
                    self.traces.table *= self.discount * self.lmbda
                    x, u = xn, un
                if done:
                    err = r - self.qfunc[x, u]
                    self.qfunc += self.alpha * err * self.traces.table

                self.td_error = np.append(self.td_error, err)

                if len(score) < 100:
                    score = np.append(score, r)
                else:
                    score[n_eps % 100] = r

                n_samp += 1
                if n_samp >= nb_samples:
                    roll['done'][-1] = True
                    rollouts.append(roll)
                    return rollouts

            print("eps: {} step: {} rwd:{} score:{}".format(n_eps, len(roll['r']), r, np.mean(score, axis=0)))

            n_eps += 1
            rollouts.append(roll)
