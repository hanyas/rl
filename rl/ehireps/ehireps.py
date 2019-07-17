import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize

from mimo import models, distributions

import copy
import operator

EXP_MAX = 700.0
EXP_MIN = -700.0


class Policy:

    def __init__(self, d_action, n_comp):
        self.d_action = d_action
        self.n_comp = n_comp

        gating_hypparams = dict(K=self.n_comp, alphas=np.ones((self.n_comp,)))
        gating_prior = distributions.Dirichlet(**gating_hypparams)

        components_hypparams = dict(mu=np.zeros((self.d_action, )),
                                    kappa=0.01,
                                    psi=np.eye(self.d_action),
                                    nu=self.d_action + 2)

        components_prior = distributions.NormalInverseWishart(**components_hypparams)

        self.mixture = models.Mixture(gating=distributions.BayesianCategoricalWithDirichlet(gating_prior),
                                      components=[distributions.BayesianGaussian(components_prior) for _ in range(self.n_comp)])

    def action(self, n):
        samples, _, resp = self.mixture.generate(n, resp=True)
        return samples, resp

    def update(self, weights):
        allscores = []
        allmodels = []

        for superitr in range(3):
            # Gibbs sampling to wander around the posterior
            for _ in range(100):
                self.mixture.resample_model(importance=[weights])
            # mean field to lock onto a mode
            scores = [self.mixture.meanfield_coordinate_descent_step(importance=[weights]) for _ in range(100)]

            allscores.append(scores)
            allmodels.append(copy.deepcopy(self.mixture))

        models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)], key=operator.itemgetter(1), reverse=True)

        self.mixture = models_and_scores[0][0]

        # clear stuff
        self.mixture.clear_plot()
        self.mixture.clear_data()
        self.mixture.clear_caches()


class eHiREPS:

    def __init__(self, func, n_episodes,
                 n_comp, kl_bound):

        self.func = func
        self.d_action = self.func.d_action

        self.n_comp = n_comp

        self.n_episodes = n_episodes

        self.kl_bound = kl_bound

        self.ctl = Policy(self.d_action, self.n_comp)

        self.data = None
        self.w = None

        self.eta = np.array([1.0])

    def sample(self, n_episodes):
        x, p = self.ctl.action(n_episodes)
        data = {'x': x, 'p': p}
        data['r'] = self.func.eval(data['x'])
        return data

    def weights(self, r, eta):
        adv = r - np.max(r)
        w = np.exp(np.clip(adv / eta, EXP_MIN, EXP_MAX))
        return w, adv

    def dual(self, eta, eps, r, p):
        w, _ = self.weights(r, eta)
        g = eta * eps + np.max(r)\
            + eta * np.log(np.mean(np.sum(p.T * w, axis=0)))
        return g

    def kl_samples(self, w):
        w = np.clip(w, 1e-75, np.inf)
        w = np.sum(self.data['p'].T * w, axis=0)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self, nb_iter=1, verbose=False):
        _trace = {'rwrd': [],
                  'kls': []}

        for it in range(nb_iter):
            self.data = self.sample(self.n_episodes)
            rwrd = np.mean(self.data['r'])

            res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                       method='SLSQP',
                                       jac=grad(self.dual),
                                       args=(
                                           self.kl_bound,
                                           self.data['r'],
                                           self.data['p']),
                                       bounds=((1e-8, 1e8), ))

            self.eta = res.x

            self.w, _ = self.weights(self.data['r'], self.eta)

            kls = self.kl_samples(self.w)

            self.ctl.update(self.w)

            _trace['rwrd'].append(rwrd)
            _trace['kls'].append(kls)

            if verbose:
                print('it=', it,
                      f'rwrd={rwrd:{5}.{4}}',
                      f'kls={kls:{5}.{4}}')

        return _trace
