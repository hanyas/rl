import numpy as np

import scipy as sc
from scipy import special

from sklearn.preprocessing import PolynomialFeatures


def merge(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]

    for key in d:
        d[key] = np.concatenate(d[key]).squeeze()

    return d


class FourierFeatures:

    def __init__(self, d_state, n_feat, band):
        self.d_state = d_state

        self.n_feat = n_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.d_state),
                                                  cov=np.diag(1.0 / band),
                                                  size=self.n_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.n_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        return phi


class RadialFeatures:

    def __init__(self, d_state, n_centers, ranges, width):
        self.d_state = d_state

        self.n_centers = n_centers

        self.ranges = ranges
        self.sigma = width
        self.invsigma = np.linalg.inv(self.sigma)

        self.n_feat = np.power(self.n_centers, self.d_state) + 1

        centers = np.zeros((self.d_state, self.n_centers))
        for n in range(self.d_state):
            lim = self.ranges[n]
            centers[n, :] = np.linspace(lim[0], lim[1], self.n_centers)

        mesh = np.meshgrid(*centers)
        self.centers = np.dstack(tuple(mesh)).reshape((-1, self.d_state))

    def fit_transform(self, x):
        phi = np.ones((x.shape[0], self.n_feat))

        for k in range(1, self.n_feat):
            dist = x - self.centers[k - 1]
            feat = np.einsum('...k,kh,...h->...', dist, self.invsigma, dist)
            phi[:, k] = np.exp(- 0.5 * feat)

        return phi


class Qfunction:

    def __init__(self, d_state, d_action, qdict):
        self.d_state = d_state
        self.d_action = d_action

        self.type = qdict['type']

        if self.type == 'fourier':
            self.n_feat = qdict['n_feat']
            self.band = qdict['band']
            self.basis = FourierFeatures(self.d_state,
                                         self.n_feat, self.band)

        elif self.type == 'rbf':
            self.n_centers = qdict['n_centers']
            self.ranges = qdict['ranges']
            self.width = qdict['width']
            self.basis = RadialFeatures(self.d_state, self.n_centers,
                                        self.ranges, self.width)
            self.n_feat = self.basis.n_feat

        elif self.type == 'poly':
            self.degree = qdict['degree']
            self.basis = PolynomialFeatures(self.degree)
            self.n_feat = int(sc.special.comb(self.degree + self.d_state,
                                              self.degree))

        self.omega = 1e-3 * np.random.randn(self.d_action * self.n_feat)

    def features(self, x):
        x = np.reshape(x, (-1, self.d_state))

        phi = np.zeros((self.d_action, x.shape[0], self.d_action * self.n_feat))
        for n in range(self.d_action):
            idx = np.ix_(range(n, n + 1),
                         range(x.shape[0]),
                         range(n * self.n_feat, n * self.n_feat + self.n_feat))
            phi[idx] = self.basis.fit_transform(x)[np.newaxis, :, :]
        return phi

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class Policy:

    def __init__(self, d_state, d_action, pdict):
        self.d_state = d_state
        self.d_action = d_action

        self.type = pdict['type']

        if 'beta' in pdict:
            self.beta = pdict['beta']
        if 'eps' in pdict:
            self.eps = pdict['eps']
        if 'weights' in pdict:
            self.weights = pdict['weights']

    def action(self, qvals, stoch=True):
        if stoch:
            if self.type == 'softmax':
                pmf = np.exp(np.clip(qvals / self.beta, -700, 700))
                return np.random.choice(self.d_action, p=pmf/np.sum(pmf))
            elif self.type == 'greedy':
                if self.eps >= np.random.rand():
                    return np.random.choice(self.d_action)
                else:
                    return np.argmax(qvals)
            else:
                return np.random.choice(self.d_action, p=self.weights)
        else:
            return np.argmax(qvals)


class LSPI:

    def __init__(self, env, n_samples, n_actions,
                 discount, lmbda, alpha, beta, qdict, pdict):

        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.d_action = n_actions  # self.env.action_space.shape[0]

        self.n_samples = n_samples
        self.discount = discount

        self.ctl = Policy(self.d_state, self.d_action, pdict)

        self.lmbda = lmbda

        # lstd regression
        self.alpha = alpha
        self.beta = beta

        self.qdict = qdict
        self.qfunc = Qfunction(self.d_state, self.d_action, self.qdict)

        self.n_feat = self.qfunc.n_feat

        self.rollouts = None

    def sample(self, n_samples, stoch=True):
        rollouts = []

        n = 0
        while True:
            roll = {'x': np.empty((0, self.d_state)),
                    'u': np.empty((0, ), np.int64),
                    'xn': np.empty((0, self.d_state)),
                    'done': np.empty((0,), np.int64),
                    'r': np.empty((0,))}

            x = self.env.reset()

            done = False
            while not done:
                u = self.ctl.action(self.qfunc.values(x), stoch)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.hstack((roll['u'], u))

                x, r, done, _ = self.env.step(u)
                roll['xn'] = np.vstack((roll['xn'], x))
                roll['done'] = np.hstack((roll['done'], done))
                roll['r'] = np.hstack((roll['r'], r))

                n = n + 1
                if n >= n_samples:
                    roll['done'][-1] = True
                    rollouts.append(roll)
                    return rollouts

            rollouts.append(roll)

    def lstd(self, rollouts, lmbda):
        _K = self.qfunc.n_feat * self.qfunc.d_action

        _A = np.zeros((_K, _K))
        _b = np.zeros((_K,))

        _I = np.eye(_K)

        _PHI = np.zeros((0, _K))

        for roll in rollouts:
            _t = 0
            _z = roll['phi'][_t, :]

            done = False
            while not done:
                done = roll['done'][_t]

                _PHI = np.vstack((_PHI, roll['phi'][_t, :]))

                _A += np.outer(_z, roll['phi'][_t, :] - (1 - done) * self.discount * roll['nphi'][_t, :])
                _b += _z * roll['r'][_t]

                if not done:
                    _z = lmbda * _z + roll['phi'][_t + 1, :]
                    _t = _t + 1

        _C = np.linalg.solve(_PHI.T.dot(_PHI) + self.alpha * _I, _PHI.T).T
        _X = _C.dot(_A + self.alpha * _I)
        _y = _C.dot(_b)

        return np.linalg.solve(_X.T.dot(_X) + self.beta * _I, _X.T.dot(_y))

    def run(self, delta):
        self.rollouts = self.sample(self.n_samples)

        for roll in self.rollouts:
            # state-action features
            phi = self.qfunc.features(roll['x'])

            # select corresponding action features
            idx = (roll['u'], np.asarray(range(len(roll['u']))))
            roll['phi'] = phi[idx]

        it = 0
        norm = np.inf
        while norm > delta:
            for roll in self.rollouts:
                # actions under max-q policy
                roll['un'] = np.argmax(self.qfunc.values(roll['xn']), axis=0)

                # next-state-action features
                nphi = self.qfunc.features(roll['xn'])

                # find and turn-off features of absorbing states
                absorbing = np.argwhere(roll['done']).flatten()
                nphi[:, absorbing, ...] *= 0.0

                idx = (roll['un'], np.asarray(range(len(roll['un']))))
                roll['nphi'] = nphi[idx]

            _omega = self.qfunc.omega.copy()
            self.qfunc.omega = self.lstd(self.rollouts, lmbda=self.lmbda)

            norm = np.linalg.norm(self.qfunc.omega - _omega)

            print('it=', it, f'conv={norm:{5}.{4}}')
            it += 1


if __name__ == "__main__":
    import gym

    import matplotlib.pyplot as plt

    np.set_printoptions(precision=5)

    # from gym.envs.registration import register
    # register(id='Lagoudakis-v0',
    #          entry_point='rl.lspi.cartpole:CartPole',
    #          max_episode_steps=1000)
    #
    # env = gym.make('Lagoudakis-v0')

    env = gym.make('CartPole-v0')
    env._max_episode_steps = 100

    lspi = LSPI(env=env, n_samples=10000, n_actions=2,
                discount=0.95, lmbda=.25,
                alpha=1e-12, beta=1e-8,
                qdict={'type': 'poly',
                       'degree': 2},
                pdict={'type': 'greedy',
                       'eps': 1.0},
                )

    lspi.run(delta=1e-3)

    # test deterministic policy
    rollouts = lspi.sample(n_samples=1000, stoch=False)
    data = merge(*rollouts)

    plt.subplot(411)
    plt.plot(data['x'][:, 2])
    plt.ylabel('Angle')

    plt.subplot(412)
    plt.plot(data['x'][:, 3])
    plt.ylabel('Velocity')

    plt.subplot(413)
    plt.plot(data['u'])
    plt.ylabel('Action')

    plt.subplot(414)
    plt.plot(1 - data['done'])
    plt.ylabel('Stable')
    plt.xlabel('Step')

    plt.show()
