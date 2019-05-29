import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize

from inf.mm import models, distributions

import copy
import operator


EXP_MAX = 700.0
EXP_MIN = -700.0


class Himmelblau:
    def __init__(self, d_action=2):
        self.d_action = d_action

    def eval(self, x):
        a = x[:, 0] * x[:, 0] + x[:, 1] - 11.0
        b = x[:, 0] + x[:, 1] * x[:, 1] - 7.0
        return -1.0 * (a * a + b * b)


class Policy:

    def __init__(self, d_action, n_comp):
        self.d_action = d_action
        self.n_comp = n_comp

        self.alpha_0 = self.n_comp
        self.hypparams = dict(mu_0=np.zeros((self.d_action, )),
                              sigma_0=1.0 * np.eye(2),
                              kappa_0=0.01,
                              nu_0=2 * self.d_action + 1)

        self.mixture = models.Mixture(alpha_0=self.alpha_0,
                                      weights=np.ones((self.n_comp, )) / self.n_comp,
                                      components=[distributions.Gaussian(**self.hypparams) for _ in range(self.n_comp)],
                                      prior='dirichlet')

    def action(self, n):
        samples, _, resp = self.mixture.generate(n, resp=True)
        return samples, resp

    def update(self, weights):
        allscores = []
        allmodels = []

        for superitr in range(3):
            # Gibbs sampling to wander around the posterior
            for _ in range(100):
                self.mixture.resample_model()
            # mean field to lock onto a mode
            scores = [self.mixture.meanfield_coordinate_descent_step(importance=weights) for _ in range(100)]

            allscores.append(scores)
            allmodels.append(copy.deepcopy(self.mixture))

        models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)], key=operator.itemgetter(1), reverse=True)

        plt.figure()
        models_and_scores[0][0].plot()
        plt.title('best model')
        plt.show()
        plt.pause(3)

        self.mixture = models_and_scores[0][0]

        # clear stuff
        self.mixture.clear_plot()
        self.mixture.clear_data()
        self.mixture.clear_caches()


class HiREPS:

    def __init__(self, func, n_episodes, n_comp, kl_bound, ent_bound):

        self.func = func
        self.d_action = self.func.d_action

        self.n_comp = n_comp

        self.n_episodes = n_episodes

        self.kl_bound = kl_bound
        self.ent_bound = ent_bound

        self.ctl = Policy(self.d_action, self.n_comp)

        self.data = None
        self.w = None
        self.eta = np.array([1.0])
        self.beta = np.array([100.0])

    def sample(self, n_episodes):
        x, p = self.ctl.action(n_episodes)
        data = {'x': x, 'p': p}
        data['r'] = self.func.eval(data['x'])
        return data

    def weights(self, r, eta):
        adv = r - np.max(r)
        w = np.exp(np.clip(adv / eta, EXP_MIN, EXP_MAX))
        return w, adv

    def dual(self, var, eps, delta, r, p):
        eta, beta = var[0], var[1]
        w, _ = self.weights(r, eta)
        resp = np.power(p.T, 1.0 + beta / eta)
        g = eta * eps + beta * delta + np.max(r) + eta * np.log(np.mean(np.sum(resp * w, axis=0)))
        return g

    def kl_samples(self, w):
        resp = np.power(self.data['p'].T, 1.0 + self.beta / self.eta)
        resp = np.clip(resp, 1e-75, np.inf)

        w = np.clip(w, 1e-75, np.inf)
        w = np.sum(resp * w, axis=0)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def ent_samples(self, w):
        resp = np.power(self.data['p'].T, 1.0 + self.beta / self.eta)
        resp = np.clip(resp, 1e-75, np.inf)

        w = np.clip(w, 1e-75, np.inf)
        aux = resp * w / np.mean(np.sum(resp * w, axis=0))
        return - np.mean(np.sum(aux * np.log(resp), axis=0))

    def run(self):
        self.data = self.sample(self.n_episodes)
        rwrd = np.mean(self.data['r'])

        res = sc.optimize.minimize(self.dual, np.array([1.0, 10.0]),
                                   method='SLSQP',
                                   jac=grad(self.dual),
                                   args=(
                                       self.kl_bound,
                                       self.ent_bound,
                                       self.data['r'],
                                       self.data['p']),
                                   bounds=((1e-8, 1e8), (1e-8, 1e8), ))

        self.eta = res.x[0]
        self.beta = res.x[1]

        self.w, _ = self.weights(self.data['r'], self.eta)

        kls = self.kl_samples(self.w)
        ent = self.ent_samples(self.w)

        self.ctl.update(self.w)

        return rwrd, kls, ent


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hireps = HiREPS(func=Himmelblau(), n_comp=5,
                    n_episodes=2500, kl_bound=0.1, ent_bound=1.0)

    for it in range(250):
        rwrd, kls, ent = hireps.run()

        print('it=', it, f'rwrd={rwrd:{5}.{4}}',
              f'kls={kls:{5}.{4}}', f'ent={ent:{5}.{4}}')
