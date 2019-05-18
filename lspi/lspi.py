import numpy as np

import scipy as sc
from scipy import special
from scipy import stats

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


class FourierFeatures:

    def __init__(self, dim_state, n_feat, band):
        self.dim_state = dim_state

        self.n_feat = n_feat

        self.freq = np.random.multivariate_normal(mean=np.zeros(self.dim_state),
                                                  cov=np.diag(1.0 / band),
                                                  size=self.n_feat)
        self.shift = np.random.uniform(-np.pi, np.pi, size=self.n_feat)

    def fit_transform(self, x):
        phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
        return phi


class RadialFeatures:

    def __init__(self, dim_state, n_centers, ranges, width):
        self.dim_state = dim_state

        self.n_centers = n_centers

        self.ranges = ranges
        self.sigma = width
        self.invsigma = np.linalg.inv(self.sigma)

        self.n_feat = np.power(self.n_centers, self.dim_state) + 1

        centers = np.zeros((self.dim_state, self.n_centers))
        for n in range(self.dim_state):
            lim = self.ranges[n]
            centers[n, :] = np.linspace(lim[0], lim[1], self.n_centers)

        mesh = np.meshgrid(*centers)
        self.centers = np.dstack(tuple(mesh)).reshape((-1, self.dim_state))

    def fit_transform(self, x):
        phi = np.ones((x.shape[0], self.n_feat))

        for k in range(1, self.n_feat):
            dist = x - self.centers[k - 1]
            feat = np.einsum('...k,kh,...h->...', dist, self.invsigma, dist)
            phi[:, k] = np.exp(- 0.5 * feat)

        return phi


class Qfunction:

    def __init__(self, dim_state, dim_action, n_actions, qdict):
        self.dim_state = dim_state
        self.dim_action = dim_action

        self.n_actions = n_actions

        self.type = qdict['type']

        if self.type == 'Fourier':
            self.n_feat = qdict['n_feat']
            self.band = qdict['band']
            self.basis = FourierFeatures(self.dim_state,
                                         self.n_feat, self.band)

        elif self.type == 'RBF':
            self.n_centers = qdict['n_centers']
            self.ranges = qdict['ranges']
            self.width = qdict['width']
            self.basis = RadialFeatures(self.dim_state, self.n_centers,
                                        self.ranges, self.width)
            self.n_feat = self.basis.n_feat

        elif self.type == 'Poly':
            self.degree = qdict['degree']
            self.basis = PolynomialFeatures(self.degree)
            self.n_feat = int(sc.special.comb(self.degree + self.dim_state,
                                              self.degree))

        self.omega = 1e-3 * np.random.randn(self.n_actions * self.n_feat)

    def features(self, x):
        x = np.reshape(x, (-1, self.dim_state))

        phi = np.zeros((self.n_actions, x.shape[0], self.n_actions * self.n_feat))
        for n in range(self.n_actions):
            idx = np.ix_(range(n, n + 1),
                         range(x.shape[0]),
                         range(n * self.n_feat, n * self.n_feat + self.n_feat))
            phi[idx] = self.basis.fit_transform(x)[np.newaxis, :, :]
        return phi

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class LSPI:

    def __init__(self, env, n_samples, n_actions,
                 discount, eps, alpha, beta, qdict):

        self.env = env

        self.dim_state = self.env.observation_space.shape[0]
        self.dim_action = self.env.action_space.shape[0]

        self.n_samples = n_samples

        self.n_actions = n_actions
        self.actions = np.linspace(self.env.action_space.low,
                                   self.env.action_space.high,
                                   self.n_actions)[:, np.newaxis]

        self.discount = discount
        self.eps = eps

        self.alpha = alpha
        self.beta = beta

        self.qdict = qdict
        self.qfunc = Qfunction(self.dim_state, self.dim_action,
                               self.n_actions, self.qdict)

        self.n_feat = self.qfunc.n_feat

        self.data = {}

    def maxq(self, x):
        Q = self.qfunc.values(x)
        return self.actions[np.argmax(Q)]

    def act(self, x, eps):
        if eps >= np.random.rand():
            return np.random.choice(self.actions.flatten(),
                                    size=(self.dim_action, ))
        else:
            return self.maxq(x)

    def sample(self, n_samples, eps):
        data = {'x': np.empty((0, self.dim_state)),
                'u': np.empty((0, self.dim_action)),
                'xn': np.empty((0, self.dim_state)),
                'done': np.empty((0, )),
                'r': np.empty((0,))}

        n = 0
        while n < n_samples:
            x = self.env.reset()

            while True:
                u = self.act(x, eps)

                data['x'] = np.vstack((data['x'], x))
                data['u'] = np.vstack((data['u'], u))

                x, r, done, _ = self.env.step(u)

                data['xn'] = np.vstack((data['xn'], x))
                data['r'] = np.hstack((data['r'], r))
                data['done'] = np.hstack((data['done'], done))

                n = n + 1
                if n >= n_samples:
                    return data

                if done:
                    break

    def run(self, n_iter):
        self.data = self.sample(self.n_samples, 1.0)

        clf = Ridge(alpha=self.alpha, fit_intercept=False)

        PHI = self.qfunc.features(self.data['x'])
        phi = np.zeros((PHI.shape[1], PHI.shape[2]))

        for n in range(self.data['u'].shape[0]):
            u = self.data['u'][n, :]
            iu = np.where(u == self.actions)[0]
            phi[n, :] = PHI[iu, n, :]

        PHIn = self.qfunc.features(self.data['xn'])
        phin = np.zeros((PHIn.shape[1], PHIn.shape[2]))

        for it in range(n_iter):
            # actions under new policy
            self.data['un'] = np.empty((0, self.dim_action))

            for n in range(self.data['xn'].shape[0]):
                un = self.act(self.data['xn'][n, :], self.eps)
                self.data['un'] = np.vstack((self.data['un'], un))

                iun = np.where(un == self.actions)[0]
                phin[n, :] = PHIn[iun, n, :]

            _A = np.einsum('nk,nh->kh', phi, (phi - self.discount * phin))
            _b = np.einsum('nk,n->k', phi, self.data['r'])

            _I = np.eye(phi.shape[1])

            _C = np.linalg.solve(phi.T.dot(phi) + self.alpha * _I, phi.T).T
            _X = _C.dot(_A + self.alpha * _I)
            _y = _C.dot(_b)

            _omega = self.qfunc.omega.copy()
            self.qfunc.omega = np.linalg.solve(_X.T.dot(_X) + self.beta * _I,
                                                _X.T.dot(_y))

            # _omega = self.qfunc.omega.copy()
            # clf.fit(_A, _b)
            # self.qfunc.omega = clf.coef_

            conv = np.mean(np.linalg.norm(self.qfunc.omega - _omega))

            print('it=', it, f'conv={conv:{5}.{4}}')

            if conv < 1e-3:
                break


if __name__ == "__main__":
    import gym
    import lab

    import matplotlib.pyplot as plt

    np.set_printoptions(precision=5)

    env = gym.make('CartPole-v3')
    env._max_episode_steps = 10000

    # lspi = LSPI(env=env, n_samples=10000, n_actions=3,
    #             discount=0.95, reg=1e-32, eps=0.0,
    #             qdict={'type': 'Fourier',
    #                    'n_feat': 15,
    #                    'band': np.array([np.pi, 1.0])}
    #             )

    lspi = LSPI(env=env, n_samples=2500, n_actions=3,
                discount=0.95, eps=0.1,
                alpha=1e-8, beta=1e-6,
                qdict={'type': 'RBF', 'n_centers': 3,
                       'ranges': ((-np.pi / 4.0, np.pi / 4.0), (-1.0, 1.0)),
                       'width': np.eye(2)}
                )

    lspi.run(10)

    # test policy
    data = lspi.sample(n_samples=10000, eps=0.0)

    plt.subplot(411)
    plt.plot(data['x'][:, 0])
    plt.ylabel('Angle')

    plt.subplot(412)
    plt.plot(data['x'][:, 1])
    plt.ylabel('Velocity')

    plt.subplot(413)
    plt.plot(data['u'])
    plt.ylabel('Action')

    plt.subplot(414)
    plt.plot(data['r'])
    plt.ylabel('Reward')
    plt.xlabel('Step')

    plt.show()
