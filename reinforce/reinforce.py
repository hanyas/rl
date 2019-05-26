import numpy as np
import scipy as sc
from scipy import special

from sklearn.preprocessing import PolynomialFeatures


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


class Policy:

    def __init__(self, d_state, d_action, **kwargs):
        self.d_state = d_state
        self.d_action = d_action

        if 'band' in kwargs:
            self.band = kwargs.get('band', False)
            self.n_feat = kwargs.get('n_feat', False)
            self.basis = FourierFeatures(self.d_state, self.n_feat, self.band)
        else:
            self.degree = kwargs.get('degree', False)
            self.n_feat = int(sc.special.comb(self.degree + self.d_state, self.degree)) - 1
            self.basis = PolynomialFeatures(self.degree, include_bias=False)

        self.K = 1e-8 * np.random.randn(self.d_action, self.n_feat)
        self.cov = np.eye(d_action)

    def features(self, x):
        return self.basis.fit_transform(x.reshape(-1, self.d_state)).squeeze()

    def mean(self, x):
        feat = self.features(x)
        return np.einsum('...k,mk->...m', feat, self.K)

    def actions(self, x, stoch):
        mean = self.mean(x)
        if stoch:
            return np.random.multivariate_normal(mean, self.cov)
        else:
            return mean

    def grad(self, x, u):
        return np.einsum('nh,hh,nk->nk', u - self.mean(x), np.linalg.inv(self.cov), self.features(x))


class REINFORCE:

    def __init__(self, env, n_episodes, n_steps, discount, alpha):
        self.env = env

        self.d_state = self.env.observation_space.shape[0]
        self.d_action = self.env.action_space.shape[0]

        self.action_limit = self.env.action_space.high

        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.discount = discount

        self.alpha = alpha

        self.ctl = Policy(self.d_state, self.d_action, degree=1)
        self.ctl.cov = 0.01 * self.ctl.cov

        self.rollouts = []

    def sample(self, n_episodes, n_steps, stoch=True):
        rollouts = []

        for _ in range(n_episodes):
            roll = {'x': np.empty((0, self.d_state)),
                    'u': np.empty((0, self.d_action)),
                    'r': np.empty((0,))}

            x = self.env.reset()

            for _ in range(n_steps):
                u = self.ctl.actions(x, stoch)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))

                x, r, done, _ = self.env.step(np.clip(u, - self.action_limit, self.action_limit))
                roll['r'] = np.hstack((roll['r'], r))

            rollouts.append(roll)

        return rollouts

    def baseline(self, returns, gradient):
        _norm = np.zeros((self.ctl.n_feat, ))
        for g in gradient:
            _norm += np.sum(g, axis=0)**2

        _b = np.zeros((self.ctl.n_feat, ))
        for r, g in zip(returns, gradient):
            _b += np.sum(g, axis=0)**2 * r / _norm

        return _b

    def run(self):
        self.rollouts = self.sample(n_episodes=self.n_episodes, n_steps=self.n_steps)

        _disc = np.hstack((1.0, np.cumprod(self.discount * np.ones((self.n_steps,))[:-1])))

        _return = []
        for roll in self.rollouts:
            _return.append(np.sum(_disc * roll['r']))

        _grad = []
        for roll in self.rollouts:
            _g = self.ctl.grad(roll['x'], roll['u'])
            _grad.append(_g)

        _baseline = self.baseline(_return, _grad)

        _wgrad = np.zeros((self.ctl.n_feat, ))
        for r, g in zip(_return, _grad):
            _wgrad += np.sum(g, axis=0) * (r - _baseline) / len(_return)

        self.ctl.K += self.alpha * _wgrad

        return np.mean(_return)


if __name__ == "__main__":
    import gym
    import lab

    import matplotlib.pyplot as plt

    env = gym.make('LQR-v0')
    env._max_episode_steps = 100

    reinforce = REINFORCE(env, n_episodes=25, n_steps=100,
                          discount=0.995, alpha=1e-5)

    for it in range(15):
        ret = reinforce.run()
        print('it=', it, f'ret={ret:{5}.{4}}')

    rollouts = reinforce.sample(25, 100, stoch=False)

    fig = plt.figure()
    for r in rollouts:
        plt.plot(r['x'][:, 0])
    plt.show()
