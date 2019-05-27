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

    def action(self, qfunc, x):
        qvals = qfunc.values(x).flatten()

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


class LSPI:

    def __init__(self, env, n_samples, n_actions,
                 discount, alpha, beta, qdict, pdict):

        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.d_action = n_actions  # self.env.action_space.shape[0]

        self.n_samples = n_samples
        self.discount = discount

        self.ctl = Policy(self.d_state, self.d_action, pdict)

        # lstd regression
        self.alpha = alpha
        self.beta = beta

        self.qdict = qdict
        self.qfunc = Qfunction(self.d_state, self.d_action, self.qdict)

        self.n_feat = self.qfunc.n_feat

        self.data = None
        self.rollouts = None

    def sample(self, n_samples):
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
                u = self.ctl.action(self.qfunc, x)

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

                    data = merge(*rollouts)
                    return rollouts, data

            rollouts.append(roll)

    def lstd(self, data, phi, nphi, type='batch'):
        if type == 'batch':
            _A = np.einsum('nk,nh->kh', phi, (phi - self.discount * nphi))
            _b = np.einsum('nk,n->k', phi, data['r'])

            _I = np.eye(phi.shape[1])

            _C = np.linalg.solve(phi.T.dot(phi) + self.alpha * _I, phi.T).T
            _X = _C.dot(_A + self.alpha * _I)
            _y = _C.dot(_b)

            return np.linalg.solve(_X.T.dot(_X) + self.beta * _I, _X.T.dot(_y))

            # from sklearn.linear_model import Ridge
            # clf = Ridge(alpha=self.alpha, fit_intercept=False)
            # clf.fit(_A, _b)
            # return clf.coef_

        elif type == 'iter':
            _K = phi.shape[1]
            _N = phi.shape[0]

            _B = np.eye(_K) * self.alpha
            _b = np.zeros((_K, ))

            for n in range(_N):
                _denom = 1.0 + (phi[n, :] - self.discount * nphi[n, :]).T @ _B @ phi[n, :]
                _B -= _B @ np.outer(phi[n, :], (phi[n, :] - self.discount * nphi[n, :])) @ _B / _denom

                _b += phi[n, :] * data['r'][n]

            return _B @ _b

    def run(self, n_iter):
        self.rollouts, self.data = self.sample(self.n_samples)

        feat = self.qfunc.features(self.data['x'])
        phi = np.zeros((feat.shape[1], feat.shape[2]))

        for n in range(self.data['u'].shape[0]):
            u = self.data['u'][n]
            phi[n, :] = feat[u, n, :]

        nfeat = self.qfunc.features(self.data['xn'])
        nphi = np.zeros((nfeat.shape[1], nfeat.shape[2]))

        for it in range(n_iter):
            # actions under new policy
            self.data['un'] = np.zeros((self.n_samples, ), np.int64)

            for n in range(self.data['xn'].shape[0]):
                un = self.ctl.action(self.qfunc, self.data['xn'][n, :])
                self.data['un'][n] = un
                nphi[n, :] = nfeat[un, n, :]

            _omega = self.qfunc.omega.copy()
            self.qfunc.omega = self.lstd(self.data, phi, nphi)

            conv = np.mean(np.linalg.norm(self.qfunc.omega - _omega))

            print('it=', it, f'conv={conv:{5}.{4}}')

            if conv < 1e-3:
                break


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
    env._max_episode_steps = 10000

    lspi = LSPI(env=env, n_samples=10000, n_actions=2,
                discount=0.9, alpha=1e-8, beta=1e-12,
                qdict={'type': 'poly',
                       'degree': 2},
                pdict={'type': 'greedy',
                       'eps': 1.0},
                )

    lspi.run(10)

    # test deterministic policy
    lspi.ctl.eps = 0.0
    _, data = lspi.sample(n_samples=1000)

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
