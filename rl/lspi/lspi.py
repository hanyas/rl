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

    def __init__(self, d_state, nb_feat, band):
        self.d_state = d_state

        self.nb_feat = nb_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.d_state),
                                                  cov=np.diag(1.0 / band),
                                                  size=self.nb_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.nb_feat)

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

        self.nb_feat = np.power(self.n_centers, self.d_state) + 1

        centers = np.zeros((self.d_state, self.n_centers))
        for n in range(self.d_state):
            lim = self.ranges[n]
            centers[n, :] = np.linspace(lim[0], lim[1], self.n_centers)

        mesh = np.meshgrid(*centers)
        self.centers = np.dstack(tuple(mesh)).reshape((-1, self.d_state))

    def fit_transform(self, x):
        phi = np.ones((x.shape[0], self.nb_feat))

        for k in range(1, self.nb_feat):
            dist = x - self.centers[k - 1]
            feat = np.einsum('...k,kh,...h->...', dist, self.invsigma, dist)
            phi[:, k] = np.exp(- 0.5 * feat)

        return phi


class Qfunction:

    def __init__(self, d_state, dm_act, qdict):
        self.d_state = d_state
        self.dm_act = dm_act

        self.type = qdict['type']

        if self.type == 'fourier':
            self.nb_feat = qdict['nb_feat']
            self.band = qdict['band']
            self.basis = FourierFeatures(self.d_state,
                                         self.nb_feat, self.band)

        elif self.type == 'rbf':
            self.n_centers = qdict['n_centers']
            self.ranges = qdict['ranges']
            self.width = qdict['width']
            self.basis = RadialFeatures(self.d_state, self.n_centers,
                                        self.ranges, self.width)
            self.nb_feat = self.basis.nb_feat

        elif self.type == 'poly':
            self.degree = qdict['degree']
            self.basis = PolynomialFeatures(self.degree)
            self.nb_feat = int(sc.special.comb(self.degree + self.d_state,
                                              self.degree))

        self.omega = 1e-3 * np.random.randn(self.dm_act * self.nb_feat)

    def features(self, x):
        x = np.reshape(x, (-1, self.d_state))

        phi = np.zeros((self.dm_act, x.shape[0], self.dm_act * self.nb_feat))
        for n in range(self.dm_act):
            idx = np.ix_(range(n, n + 1),
                         range(x.shape[0]),
                         range(n * self.nb_feat, n * self.nb_feat + self.nb_feat))
            phi[idx] = self.basis.fit_transform(x)[np.newaxis, :, :]
        return phi

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


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

    def action(self, qvals, stoch=True):
        if stoch:
            if self.type == 'softmax':
                pmf = np.exp(np.clip(qvals / self.beta, -700, 700))
                return np.random.choice(self.dm_act, p=pmf/np.sum(pmf))
            elif self.type == 'greedy':
                if self.eps >= np.random.rand():
                    return np.random.choice(self.dm_act)
                else:
                    return np.argmax(qvals)
            else:
                return np.random.choice(self.dm_act, p=self.weights)
        else:
            return np.argmax(qvals)


class LSPI:

    def __init__(self, env, nb_samples, n_actions,
                 discount, lmbda, alpha, beta, qdict, pdict):

        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.dm_act = n_actions  # self.env.action_space.shape[0]

        self.nb_samples = nb_samples
        self.discount = discount

        self.ctl = Policy(self.d_state, self.dm_act, pdict)

        self.lmbda = lmbda

        # lstd regression
        self.alpha = alpha
        self.beta = beta

        self.qdict = qdict
        self.qfunc = Qfunction(self.d_state, self.dm_act, self.qdict)

        self.nb_feat = self.qfunc.nb_feat

        self.rollouts = None

    def sample(self, nb_samples, stoch=True):
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
                if n >= nb_samples:
                    roll['done'][-1] = True
                    rollouts.append(roll)
                    return rollouts

            rollouts.append(roll)

    def lstd(self, rollouts, lmbda):
        for roll in rollouts:
            # state-action features
            phi = self.qfunc.features(roll['x'])

            # select corresponding action features
            idx = (roll['u'], np.asarray(range(len(roll['u']))))
            roll['phi'] = phi[idx]

            # actions under max-q policy
            roll['un'] = np.argmax(self.qfunc.values(roll['xn']), axis=0)

            # next-state-action features
            nphi = self.qfunc.features(roll['xn'])

            # find and turn-off features of absorbing states
            absorbing = np.argwhere(roll['done']).flatten()
            nphi[:, absorbing, ...] *= 0.0

            idx = (roll['un'], np.asarray(range(len(roll['un']))))
            roll['nphi'] = nphi[idx]

        _K = self.qfunc.nb_feat * self.qfunc.dm_act

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

        return np.linalg.solve(_X.T.dot(_X) + self.beta * _I, _X.T.dot(_y)), rollouts

    def run(self, delta):
        self.rollouts = self.sample(self.nb_samples)

        it = 0
        norm = np.inf
        while norm > delta:
            _omega = self.qfunc.omega.copy()
            self.qfunc.omega, self.rollouts = self.lstd(self.rollouts, lmbda=self.lmbda)

            norm = np.linalg.norm(self.qfunc.omega - _omega)

            print('it=', it, f'conv={norm:{5}.{4}}')
            it += 1
