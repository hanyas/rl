import autograd.numpy as np
import scipy as sc
from scipy import optimize
from scipy.special import logit, logsumexp

from sklearn.metrics import mean_squared_error
from sklearn import linear_model

from rl.hyreps.v1.rslds import rSLDS
from rl.hyreps.v1.util import normalize

import copy

import matplotlib.pyplot as plt


class BaumWelch:

    def __init__(self, x, u, w,
                 n_regions, priors,
                 regs, rslds, update):

        self.n_rollouts = x.shape[0]
        self.n_steps = x.shape[1]

        self.n_states = x.shape[2]
        self.n_actions = u.shape[2]
        self.n_regions = n_regions

        self.priors = priors
        self.dyn_prior = priors[0]
        self.ctl_prior = priors[1]

        self.x = x
        self.u = u
        self.w = w

        self.msg_reg = regs[0]
        self.dyn_reg = regs[1]
        self.lgstc_reg = regs[2]
        self.ctl_reg = regs[3]

        self.update_dyn = update[0]
        self.update_ctl = update[1]

        self.rslds = rSLDS(self.n_states, self.n_actions, self.n_regions, self.priors)

        if self.update_dyn and self.update_ctl:
            pass
        elif ((not self.update_dyn) and self.update_ctl):
            self.rslds.linear_models = copy.deepcopy(rslds.linear_models)
            self.rslds.logistic_model = copy.deepcopy(rslds.logistic_model)

        self.alpha = np.zeros((self.n_rollouts, self.n_steps, self.n_regions))
        self.beta = np.zeros((self.n_rollouts, self.n_steps, self.n_regions))
        self.gamma = np.zeros((self.n_rollouts, self.n_steps, self.n_regions))
        self.zeta = np.zeros((self.n_rollouts, self.n_steps - 1, self.n_regions, self.n_regions))
        self.theta = np.zeros((self.n_rollouts, self.n_steps - 1, 1), np.int64)

        self.lik = np.zeros((self.n_rollouts, self.n_steps, 1))

        self.lgstc = None
        self.dyn = np.zeros((self.n_rollouts, self.n_steps, self.n_regions))
        self.ctl = np.zeros((self.n_rollouts, self.n_steps, self.n_regions))
        self.feat = self.rslds.logistic_model.features(x)

    def lognorm(self, var):
        x = np.log(var + np.finfo(float).tiny)
        return x - logsumexp(x, 1)[:, np.newaxis, :]

    def forward(self, x, dyn, lgstc, ctl):
        n_rollouts = x.shape[0]
        n_steps = x.shape[1]

        alpha = np.empty((n_rollouts, n_steps, self.n_regions), dtype=np.float128)
        norm = np.empty((n_rollouts, n_steps))

        p = np.ones(self.n_regions) / self.n_regions
        alpha[:, 0, :] = np.einsum('nml,m->nl', lgstc[:, 0, :, :], p)
        # alpha[:, 0, :], norm[:, 0] = normalize(alpha[:, 0, :] + self.msg_reg, dim=(1,))

        alpha[:, 0, :] = alpha[:, 0, :] + self.msg_reg
        norm[:, 0] = np.sum(alpha[:, 0, :], axis=-1)
        alpha[:, 0, :] = alpha[:, 0, :] / norm[:, 0, np.newaxis]

        for t in range(1, n_steps):
            alpha[:, t, :] = np.einsum('nml,nm->nl', lgstc[:, t, :, :],
                                       alpha[:, t - 1, :]) * dyn[:, t, :] * ctl[:, t, :]
            # alpha[:, t, :], norm[:, t] = normalize(alpha[:, t, :] + self.msg_reg, dim=(1, ))

            alpha[:, t, :] = alpha[:, t, :] + self.msg_reg
            norm[:, t] = np.sum(alpha[:, t, :], axis=-1)
            alpha[:, t, :] = alpha[:, t, :] / norm[:, t, np.newaxis]

        return alpha, norm

    def backward(self, x, scale, dyn, lgstc, ctl):
        n_rollouts = x.shape[0]
        n_steps = x.shape[1]

        beta = np.empty((n_rollouts, n_steps, self.n_regions), dtype=np.float128)

        beta[:, -1, :] = np.ones((n_rollouts, self.n_regions)) / scale[:, -1, np.newaxis]

        for t in range(n_steps - 2, -1, -1):
            beta[:, t, :] = np.einsum('nml,nl->nm', lgstc[:, t + 1, :, :],
                                      (beta[:, t + 1, :] * dyn[:, t + 1, :] * ctl[:, t + 1, :]))
            beta[:, t, :] = beta[:, t, :] / scale[:, t, np.newaxis]

        return beta

    def marginal(self, x, alpha, beta, dyn, lgstc, ctl):
        n_rollouts = x.shape[0]
        n_steps = x.shape[1]

        zeta = np.empty((n_rollouts, n_steps - 1, self.n_regions, self.n_regions), dtype=np.float128)
        norm = np.empty((n_rollouts, n_steps - 1))

        for t in range(n_steps - 1):
            aux = np.einsum('nm,nl->nml', alpha[:, t, :],
                            dyn[:, t + 1, :] * ctl[:, t + 1, :] * beta[:, t + 1, :])
            zeta[:, t, :, :] = lgstc[:, t + 1, :, :] * aux
            # zeta[:, t, :, :], _ = normalize(zeta[:, t, :, :] + self.msg_reg, dim=(2, 1))

            zeta[:, t, :, :] = zeta[:, t, :, :] + self.msg_reg
            norm[:, t] = np.sum(zeta[:, t, :, :], axis=(2, 1))
            zeta[:, t, :, :] = zeta[:, t, :, :] / norm[:, t, np.newaxis, np.newaxis]

        return zeta

    def forward_backward(self, alpha, beta):
        # gamma, _ = normalize(alpha * beta + self.msg_reg, dim=(2, ))

        gamma = (alpha * beta + self.msg_reg) / np.sum(alpha * beta + self.msg_reg,
                                                   axis=2, keepdims=True)
        return gamma

    def predict(self, xt, ut):
        n_rollouts, n_steps = xt.shape[0], xt.shape[1]

        alpha = np.zeros((n_rollouts, n_steps, self.n_regions))
        z = np.zeros((n_rollouts, n_steps), np.int64)
        x = np.zeros((n_rollouts, n_steps, self.n_states))
        u = np.zeros((n_rollouts, n_steps, self.n_actions))

        x[:, 0] = xt[:, 0]

        p = np.ones(self.n_regions) / self.n_regions
        z[:, 0], alpha[:, 0] = self.rslds.filter(x[:, 0], p)

        for t in range(1, n_steps):
            u[:, t - 1] = self.rslds.act(x[:, t - 1], alpha[:, t - 1])
            z[:, t], x[:, t], alpha[:, t] =\
                self.rslds.step(z[:, t - 1], xt[:, t - 1], ut[:, t - 1], alpha[:, t - 1])

        u[:, -1] = self.rslds.act(xt[:, -1], alpha[:, -1])

        state_err = mean_squared_error(np.reshape(xt, (-1, self.n_states)),
                                       np.reshape(x, (-1, self.n_states)))

        action_err = mean_squared_error(np.reshape(ut, (-1, self.n_actions)),
                                        np.reshape(u, (-1, self.n_actions)))

        return z, x, alpha, state_err, u, action_err

    def expectation(self):
        for i in range(self.n_regions):
            self.dyn[:, 0, i] = self.rslds.init_state.pdf(self.x[:, 0, :])
            self.rslds.linear_models[i].update()
            self.dyn[:, 1:, i] = self.rslds.linear_models[i].prob(self.x, self.u)
            if self.update_ctl:
                self.rslds.linear_ctls[i].update()
                self.ctl[:, :, i] = self.rslds.linear_ctls[i].prob(self.x, self.u)
            else:
                self.ctl[:, :, i] = np.ones(self.u.shape[:-1])

        self.lgstc = self.rslds.logistic_model.transitions(self.x)

        self.alpha, self.lik = self.forward(self.x, self.dyn, self.lgstc, self.ctl)
        self.beta = self.backward(self.x, self.lik, self.dyn, self.lgstc, self.ctl)
        self.gamma = self.forward_backward(self.alpha, self.beta)
        self.zeta = self.marginal(self.x, self.alpha, self.beta, self.dyn, self.lgstc, self.ctl)

        self.alpha = self.alpha.astype(np.float64)
        self.beta = self.beta.astype(np.float64)
        self.gamma = self.gamma.astype(np.float64)
        self.zeta = self.zeta.astype(np.float64)
        self.lik = self.lik.astype(np.float64)

        liklhd = np.sum(np.log(self.lik), axis=(0, 1))

        return liklhd

    def logistic_opt(self, feat, zeta, theta, p, i):
        idx = np.where(theta[:, :-1] == i)

        input, output = feat[idx], zeta[idx]

        output = output / np.sum(output, axis=-1, keepdims=True)
        output = output[:, i, :]

        res = sc.optimize.least_squares(self.rslds.logistic_model.logistic_err,
                                        x0=p[i, :],
                                        args=(input, output),
                                        method='trf',
                                        jac=self.rslds.logistic_model.dlogistic_err)
        return res.x


    def run(self, n_iter, save=False, plot=False, verbose=False):
        if plot:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel("Iteraions")
            ax.set_ylabel("Likelihood")
            ax.grid()

            xdata = np.linspace(0, n_iter - 1, n_iter)
            ydata = np.zeros((n_iter,))
            line, = ax.plot(xdata, ydata, 'b-')

        last_liklhd = - np.inf
        train_err = 1e50

        rdg_dyn = linear_model.Ridge(alpha=self.dyn_reg, fit_intercept=False, tol=1e-8, solver="auto", max_iter=None)
        rdg_lgstc = linear_model.Ridge(alpha=self.lgstc_reg, fit_intercept=False, tol=1e-8, solver="auto", max_iter=None)
        rdg_ctl = linear_model.Ridge(alpha=self.ctl_reg, fit_intercept=False, tol=1e-8, solver="auto", max_iter=None)

        liklhd = []

        for it in range(n_iter):
            liklhd.append(self.expectation())

            self.gamma = self.gamma * self.w[..., np.newaxis]
            self.zeta = self.zeta * self.w[:, 1:, np.newaxis, np.newaxis]
            self.theta = np.argmax(self.gamma, axis=-1)

            if liklhd[-1] < last_liklhd:
                liklhd[-1] = last_liklhd
                break
            else:
                last_liklhd = liklhd[-1]

            # initial state distribution
            self.rslds.init_state.mean = np.mean(self.x[:, 0, :], axis=0)
            self.rslds.init_state.cov = np.cov(m=self.x[:, 0, :], rowvar=False, bias=False)

            # dynamics matrices
            tmpx = np.einsum('ntk,nth->ntkh', self.x[:, :-1, :], self.x[:, :-1, :])
            tmpu = np.einsum('ntk,nth->ntkh', self.u[:, :-1, :], self.u[:, :-1, :])

            if self.update_dyn:
                for i in range(self.n_regions):
                    aux = np.einsum('ntk,nth->ntkh', self.x[:, 1:, :] - np.einsum('kh,nth->ntk', self.rslds.linear_models[i].B, self.u[:, :-1, :]) - self.rslds.linear_models[i].C, self.x[:, :-1, :])

                    # self.rslds.linear_models[i].A = np.dot(np.einsum('nt,ntkh->kh', self.gamma[:, 1:, i], aux), np.linalg.pinv(np.einsum('nt,ntkh->kh', self.gamma[:, 1:, i], tmpx)))
                    rdg_dyn.fit(np.einsum('nt,ntkh->kh', self.gamma[:, 1:, i], tmpx), np.einsum('nt,ntkh->kh', self.gamma[:, 1:, i], aux))
                    self.rslds.linear_models[i].A = rdg_dyn.coef_

                    aux = np.einsum('ntk,nth->ntkh', self.x[:, 1:, :] - np.einsum('kh,nth->ntk', self.rslds.linear_models[i].A, self.x[:, :-1, :]) - self.rslds.linear_models[i].C, self.u[:, :-1, :])

                    # self.rslds.linear_models[i].B = np.dot(np.einsum('nt,ntkh->kh', self.gamma[:, 1:, i], aux), np.linalg.pinv(np.einsum('nt,ntkh->kh', self.gamma[:, 1:, i], tmpu)))
                    rdg_dyn.fit(np.einsum('nt,ntkh->hk', self.gamma[:, 1:, i], tmpu), np.einsum('nt,ntkh->hk', self.gamma[:, 1:, i], aux))
                    self.rslds.linear_models[i].B = rdg_dyn.coef_

                    aux = self.x[:, 1:, :] - np.einsum('kh,nth->ntk', self.rslds.linear_models[i].A, self.x[:, :-1, :]) - np.einsum('kh,nth->ntk', self.rslds.linear_models[i].B, self.u[:, :-1, :])
                    self.rslds.linear_models[i].C = np.einsum('nt,ntk->k', self.gamma[:, 1:, i], aux) / np.sum(self.gamma[:, 1:, i], axis=(0, 1))

                    # Z = np.sum(self.gamma[:, 1:, i], axis=(0,1))
                    Z = (np.square(np.sum(self.gamma[:, 1:, i], axis=(0, 1))) - np.sum(np.square(self.gamma[:, 1:, i]), axis=(0, 1))) / np.sum(self.gamma[:, 1:, i], axis=(0, 1))

                    aux = self.x[:, 1:, :] - np.einsum('kh,nth->ntk', self.rslds.linear_models[i].A, self.x[:, :-1, :]) - np.einsum('kh,nth->ntk', self.rslds.linear_models[i].B, self.u[:, :-1, :]) - self.rslds.linear_models[i].C
                    self.rslds.linear_models[i].cov = (np.einsum('nt,ntk,nth->kh', self.gamma[:, 1:, i], aux, aux)) / (Z + 1e-64)
                    self.rslds.linear_models[i].cov = (self.dyn_prior["psi"] + (self.n_rollouts + self.n_steps) * self.rslds.linear_models[i].cov) / (self.dyn_prior["nu"] + (self.n_rollouts + self.n_steps) + self.n_states + 1)

                # switching logistic functions
                for i in range(self.n_regions):
                    feat = self.feat[:, 1:, :].reshape((-1, self.rslds.logistic_model.n_feat))
                    zeta = self.zeta.reshape((-1, self.n_regions, self.n_regions))

                    for i in range(self.n_regions):
                        fopt, _ = normalize(zeta[:, i, :] + self.msg_reg, dim=(1, ))
                        fopt = np.clip(fopt, 0.0001, 0.9999)
                        lgt = logit(fopt)

                        rdg_lgstc.fit(feat, lgt)
                        self.rslds.logistic_model.par[i, :] = np.reshape(rdg_lgstc.coef_, (self.rslds.logistic_model.n_feat * self.n_regions), order='C')

                # for i in range(self.n_regions):
                #     fopt, _ = normalize(zeta[:, i, :] + self.msg_reg, dim=(1, ))
                #     labels = np.tile(np.arange(self.n_regions), fopt.shape[0])
                #     inputs = np.repeat(feat, self.n_regions, 0)
                #     weights = fopt.reshape(-1, )
                #
                #     rdg_lgstc = linear_model.LogisticRegression(
                #         solver='lbfgs', fit_intercept=False,
                #         C=1.0 / self.lgstc_reg,
                #         multi_class='multinomial', max_iter=100, n_jobs=-1)
                #
                #     rdg_lgstc.fit(inputs, labels, weights)
                #     self.rslds.logistic_model.par[i, :] = np.reshape(rdg_lgstc.coef_, (self.rslds.logistic_model.n_feat * self.n_regions), order='C')

                # for i in range(self.n_regions):
                #     self.rslds.logistic_model.par[i, :] = self.logistic_opt(self.feat, self.zeta, self.theta,
                #                                                             self.rslds.logistic_model.par, i)

            if self.update_ctl:
                # controller matrices
                state = np.concatenate((np.ones((self.n_rollouts, self.n_steps, 1)), self.x), axis=-1)
                tmps = np.einsum('ntk,nth->ntkh', state, state)

                for i in range(self.n_regions):
                    aux = np.einsum('ntk,nth->ntkh', self.u, state)

                    # self.rslds.linear_ctls[i].K = np.dot(np.einsum('nt,ntkh->kh', self.gamma[..., i], aux), np.linalg.inv(np.einsum('nt,ntkh->kh', self.gamma[..., i], tmps)))
                    rdg_ctl.fit(np.einsum('nt,ntkh->kh', self.gamma[..., i], tmps), np.einsum('nt,ntkh->kh', self.gamma[..., i], aux).squeeze())
                    self.rslds.linear_ctls[i].K = rdg_ctl.coef_.reshape(self.n_actions, -1)

                    aux = self.u - np.einsum('kh,nth->ntk', self.rslds.linear_ctls[i].K, state)

                    # Z = np.sum(self.gamma[:, :, i], axis=(0,1))
                    Z = (np.square(np.sum(self.gamma[..., i], axis=(0,1))) - np.sum(np.square(self.gamma[..., i]), axis=(0,1))) / np.sum(self.gamma[..., i], axis=(0,1))

                    self.rslds.linear_ctls[i].cov = (np.einsum('nt,ntk,nth->kh', self.gamma[..., i], aux, aux)) / (Z + 1e-64)
                    self.rslds.linear_ctls[i].cov = (self.ctl_prior["psi"] + (self.n_rollouts + self.n_steps) * self.rslds.linear_ctls[i].cov) / (self.ctl_prior["nu"] + (self.n_rollouts + self.n_steps) + self.n_actions + 1)

            if verbose:
                if np.remainder(it, 10) == 0:
                    _, _, _, train_err, _, _ = self.predict(self.x, self.u)

                print("Iteration: ", it, " Likelihood: ", "%.3f" % liklhd[-1],
                      " Error: ", "%.8f" % train_err)

            if plot:
                ydata[it] = liklhd[-1]
                line.set_xdata(xdata[:it + 1])
                line.set_ydata(ydata[:it + 1])

                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                plt.pause(0.001)

            if save:
                self.rslds.save("")

        return liklhd


if __name__ == "__main__":
    n_rollouts, n_steps = 50, 100
    n_states, n_actions = 2, 1
    n_regions = 2

    x = np.random.randn(n_rollouts, n_steps, n_states)
    u = np.random.randn(n_rollouts, n_steps, n_actions)
    w = np.ones((n_rollouts, n_steps))

    dyn_prior = {"nu": n_states + 1, "psi": 1e-2}
    ctl_prior = {"nu": n_actions + 1, "psi": 1.0}
    priors = [dyn_prior, ctl_prior]
    regs = np.array([np.finfo(np.float64).tiny, 1e-16, 1e-12, 1e-16])

    bw = BaumWelch(x, u, w, n_regions, priors, regs, None, [True, False])
    bw.run(10, save=False)
