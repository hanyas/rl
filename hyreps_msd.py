import numpy as np
import numexpr as ne
import scipy as sc
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns

np.set_printoptions(precision=5)

sns.set_style("white")
sns.set_context("paper")

color_names = ["red",
               "windows blue",
               "medium green",
               "dusty purple",
               "orange",
               "amber",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "mint",
               "salmon",
               "dark brown"]

colors = []
for k in color_names:
	colors.append(mcd.XKCD_COLORS['xkcd:'+k].upper())

np.random.seed(1337)

REG = 1e-75
EXP_MAX = 1000.0
EXP_MIN = -1000.0


def lgstc_feat(x, u):
	xu = np.dstack((x, u))
	data = np.reshape(xu, (-1, xu.shape[-1]), order='C')
	poly = PolynomialFeatures(1)
	feat = np.reshape(poly.fit_transform(data), (xu.shape[0], xu.shape[1], -1), order='C')
	return feat


def logistic(p, feat):
	n_features = feat.shape[-1]
	if p.ndim == 1:
		par = np.reshape(p, (-1, n_features), order='C')
		a = np.einsum('tk,lk->tl', feat, par)
	if p.ndim == 2:
		n_steps = p.shape[0]
		par = np.reshape(p, (n_steps, -1, n_features), order='C')
		a = np.einsum('tk,tlk->tl', feat, par)

	a = np.clip(a, EXP_MIN, EXP_MAX)
	expa = ne.evaluate('exp(a)')
	l = expa / np.sum(expa, axis=1, keepdims=True)
	return l


class LinearGaussian:

	def __init__(self, n_states, n_actions):
		self.cov = sc.stats.invwishart.rvs(n_states + 1, np.eye(n_states))
		self.A = sc.stats.matrix_normal.rvs(mean=None, rowcov=self.cov, colcov=self.cov)
		self.B = sc.stats.matrix_normal.rvs(mean=None, rowcov=self.cov, colcov=self.cov)[:, [0]]
		self.C = sc.stats.matrix_normal.rvs(mean=None, rowcov=self.cov, colcov=self.cov)[:, 0]
		self.perc = np.linalg.inv(self.cov)
		self.const = 1.0 / np.sqrt(np.linalg.det(2.0 * np.pi * self.cov + 1e-16 * np.eye(n_states)))

	def update(self):
		self.perc = np.linalg.inv(self.cov + 1e-16 * np.eye(self.cov.shape[0]))
		self.const = 1.0 / np.sqrt(np.linalg.det(2.0 * np.pi * (self.cov + 1e-16 * np.eye(self.cov.shape[0]))))

	def prob(self, x, u):
		err = x[1:, :] - np.einsum('kh,th->tk', self.A, x[:-1, :]) - np.einsum('kh,th->tk', self.B, u[:-1, :]) - self.C
		return self.const * np.exp(-0.5 * np.einsum('tk,kh,th->t', err, self.perc, err))


class MultiLogistic:

	def __init__(self, n_regions, n_features):
		self.n_regions = n_regions
		self.n_features = n_features
		self.par = np.random.randn(n_regions, n_regions * n_features)

	def transitions(self, feat):
		n_steps = feat.shape[0]
		trans = np.zeros((n_steps, self.n_regions, self.n_regions))
		for i in range(self.n_regions):
			trans[:, i, :] = logistic(self.par[i, :], feat)

		return trans


class Policy:

	def __init__(self, n_states, n_actions, n_regions):
		self.n_states = n_states
		self.n_actions = n_actions
		self.n_regions = n_regions

		self.K = np.random.randn(n_regions, n_actions, n_states + 1)
		self.cov = np.zeros((n_regions, n_actions, n_actions))
		for n in range(n_regions):
			self.cov[n, :, :] = np.eye(n_actions)

	def actions(self, z, x, stoch):
		mean = np.dot(self.K[z, :, :], x)
		if stoch:
			return np.random.normal(mean, np.sqrt(self.cov[z, :, :])).flatten()
		else:
			return mean


class SLDS:
	def __init__(self, n_states, n_actions, n_regions, n_feat):
		self.n_states = n_states
		self.n_actions = n_actions
		self.n_regions = n_regions
		self.n_feat = n_feat

		self.init_region = sc.stats.multinomial(1, np.ones(n_regions) / n_regions)
		self.init_state = sc.stats.multivariate_normal(mean=np.random.randn(n_states), cov=np.eye(n_states))

		self.linear_models = [LinearGaussian(n_states, n_actions) for _ in range(n_regions)]
		self.logistic_model = MultiLogistic(n_regions, n_feat)

	def load(self, const, cov, dt):
		for i in range(self.n_regions):
			mass, spring, damper = const[i, 0], const[i, 1], const[i, 2]
			self.linear_models[i].A = dt * np.array([[0.0, 1.0], [- spring / mass, - damper / mass]]) + np.eye(self.n_states)
			self.linear_models[i].B = dt * np.array([[0.0], [1.0 / mass]])
			self.linear_models[i].C = np.array([0.0, 0.0])
			self.linear_models[i].cov = cov * np.eye(self.n_states)

	def forward(self, x, u):
		n_steps = x.shape[0]

		# pad action
		u = np.vstack((u, np.zeros((1, u.shape[1]))))

		pdfs = np.zeros((n_steps - 1, self.n_regions))
		for i in range(self.n_regions):
			pdfs[:, i] = self.linear_models[i].prob(x, u)

		feat = lgstc_feat(x, u)
		trans = self.logistic_model.transitions(feat)

		alpha = np.zeros((n_steps, self.n_regions))
		norm = np.zeros((n_steps))

		alpha[0, :] = self.init_region.p

		alpha[0, :] = alpha[0, :] + REG
		norm[0] = np.sum(alpha[0, :], axis=-1, keepdims=False)
		alpha[0, :] = alpha[0, :] / norm[0, None]

		for t in range(1, n_steps):
			alpha[t, :] = np.einsum('ml,m->l', trans[t - 1, :, :], alpha[t - 1, :]) * pdfs[t - 1, :]

			alpha[t, :] = alpha[t, :] + REG
			norm[t] = np.sum(alpha[t, :], axis=-1, keepdims=False)
			alpha[t, :] = alpha[t, :] / norm[t, None]

		return alpha


class Environment:

	def __init__(self, n_states, n_actions,
					action_limit, state_limit,
					goal, state_weight, action_weight,
	                slds):

		self.n_states = n_states
		self.n_actions = n_actions

		self.action_min = - action_limit
		self.action_max = action_limit

		self.state_min = - state_limit
		self.state_max = state_limit

		self.goal = goal
		self.state_weight = state_weight
		self.action_weight = action_weight

		self.z_intern = np.zeros((1, ), np.int64)
		self.x_intern = np.zeros((2, ))

		self.slds = slds

		slds.init_region = 0
		slds.init_state = sc.stats.multivariate_normal(mean=np.array([0.0, 0.0]), cov=1.0 * np.eye(n_states))

		slds.logistic_model.par[0, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
		slds.logistic_model.par[1, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])

		# slds.logistic_model.par[0, :] = np.array([1.0, 0.0, 0.2, 3.0, 2.0, 2.0])
		# slds.logistic_model.par[1, :] = np.array([1.0, 0.0, 0.2, 3.0, 2.0, 2.0])

	def reset(self):
		# self.z_intern = np.argmax(self.slds.init_region.rvs())

		self.z_intern = self.slds.init_region
		self.x_intern = self.slds.init_state.rvs()
		return self.z_intern, self.x_intern

	def step(self, u):
		# uc = np.clip(u, self.action_min, self.action_max)
		uc = u.copy()

		r = self.reward(self.z_intern, self.x_intern, uc)

		xin = np.broadcast_to(self.x_intern, (1, self.n_states))
		uin = np.broadcast_to(uc, (1, self.n_actions))

		feat = lgstc_feat(xin, uin)
		lgstc = logistic(self.slds.logistic_model.par[self.z_intern, :], feat)

		self.z_intern = np.argmax(lgstc.flatten())

		self.x_intern = np.dot(self.slds.linear_models[self.z_intern].A, self.x_intern) +\
		                np.dot(self.slds.linear_models[self.z_intern].B, uc) + self.slds.linear_models[self.z_intern].C

		self.x_intern = np.clip(self.x_intern, self.state_min, self.state_max)
		self.x_intern = np.random.multivariate_normal(self.x_intern, self.slds.linear_models[self.z_intern].cov)

		return self.z_intern, self.x_intern, r

	def reward(self, z, x, u):
		diff = x - self.goal
		if z==0:
			r = - np.dot(diff, self.state_weight).dot(diff) - np.dot(u, self.action_weight).dot(u)
		else:
			r = - np.dot(diff, self.state_weight).dot(diff) - np.dot(u, 0.1 * self.action_weight).dot(u)
		return r


def dual_eta(eta, omega, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_eta(eta, omega, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	deta = epsilon + np.log(np.mean(w, axis=0, keepdims=True)) - np.sum(w * delta, axis=0, keepdims=True) / (eta * np.sum(w, axis=0, keepdims=True))
	return deta


def dual_omega(omega, eta, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	g = np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_omega(omega, eta, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	domega = (1.0 - gamma) * np.mean(ivfeatures, axis=0, keepdims=False) + np.sum(w[:, None] * (gamma * qfeatures - vfeatures), axis=0, keepdims=False) / np.sum(w, axis=0, keepdims=False)
	return domega


def hess_omega(omega, eta, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	w = w / np.sum(w, axis=0, keepdims=True)

	ui = gamma * qfeatures - vfeatures
	uj = np.sum(w[:, None] * ui, axis=0, keepdims=True)
	tmp = ui - uj

	homega = 1./eta * np.einsum('n,nk,nh->kh', w, tmp, tmp)
	return homega


def linear_features(x):
	poly = PolynomialFeatures(1)
	if x.ndim > 1:
		return poly.fit_transform(x)
	else:
		return poly.fit_transform(x.reshape(1, -1)).flatten()


class HyREPS:

	def __init__(self, n_states, n_actions, n_regions,
	             n_rollouts, n_steps, n_filter,
	             n_iter, n_keep,
	             kl_bound, discount,
	             action_limit, state_limit):

		self.n_states = n_states
		self.n_actions = n_actions
		self.n_regions = n_regions
		self.n_rollouts = n_rollouts
		self.n_steps = n_steps
		self.n_filter = n_filter
		self.n_iter = n_iter
		self.n_keep = n_keep
		self.kl_bound = kl_bound
		self.discount = discount

		self.n_sim_rollouts = n_rollouts
		self.n_sim_steps = n_steps

		self.n_lgstc_features = n_states + n_actions + 1
		self.n_vfeatures = int(2 *n_states + sc.special.comb(n_states, 2) + 1)

		self.goal = np.array([1.0, 0.0])
		self.state_weight = np.diag([1e0, 1e-1])
		self.action_weight = np.diag([1e-3])

		self.slds = SLDS(n_states, n_actions, n_regions, self.n_lgstc_features)

		# mass, spring, damper
		const = np.array([[1.0, 5.0, 1.0], [2.0, 5.0, 1.0]])
		self.slds.load(const, 1e-8, 5e-2)

		for i in range(self.n_regions):
			self.slds.linear_models[i].update()

		self.env = Environment(n_states, n_actions, action_limit=action_limit, state_limit=state_limit,
								goal=self.goal, state_weight=self.state_weight, action_weight=self.action_weight,
								slds=self.slds)

		self.ctl = Policy(self.n_states, n_actions, n_regions)
		self.ctl.cov = (action_limit)**2 * self.ctl.cov

		self.data = {'zi': np.empty((0, ), np.int64),
					 'xi': np.empty((0, n_states)),
		             'ui': np.empty((0, n_actions)),
		             'z': np.empty((0,), np.int64),
		             'x': np.empty((0, n_states)),
		             'u': np.empty((0, n_actions)),
		             'zn': np.empty((0,), np.int64),
		             'xn': np.empty((0, n_states)),
		             'un': np.empty((0, n_actions)),
		             'r': np.empty((0,))}

		self.vfeatures = None
		self.ivfeatures = None
		self.qfeatures = None

		self.omega = np.random.randn(n_regions * self.n_vfeatures)
		self.eta = np.array([0.01])

	def sample(self, n_rollouts, n_steps, n_filter, n_keep, reset=True, stoch=True):
		if n_keep==0:
			data = {'zi': np.empty((0, ), np.int64),
					 'xi': np.empty((0, self.n_states)),
		             'ui': np.empty((0, self.n_actions)),
		             'z': np.empty((0,), np.int64),
		             'x': np.empty((0, self.n_states)),
		             'u': np.empty((0, self.n_actions)),
		             'zn': np.empty((0,), np.int64),
		             'xn': np.empty((0, self.n_states)),
		             'un': np.empty((0, self.n_actions)),
		             'r': np.empty((0,))}

			n_samples = n_rollouts * n_steps

		else:
			data = {'zi': self.data['zi'],
			        'xi': self.data['xi'],
			        'ui': self.data['ui'],
			        'z': self.data['z'][-n_keep * n_steps:],
			        'x': self.data['x'][-n_keep * n_steps:, :],
			        'u': self.data['u'][-n_keep * n_steps:, :],
			        'zn': self.data['zn'][-n_keep * n_steps:],
			        'xn': self.data['xn'][-n_keep * n_steps:, :],
			        'un': self.data['un'][-n_keep * n_steps:, :],
			        'r': self.data['r'][-n_keep * n_steps:]}

			n_samples = (n_rollouts - n_keep) * n_steps


		n = 0
		while n < n_samples:
			z_aux, x_aux = self.env.reset()

			data['xi'] = np.vstack((data['xi'], x_aux))
			data['zi'] = np.hstack((data['zi'], z_aux))

			u_aux = self.ctl.actions(z_aux, self.get_pfeatures(x_aux), stoch)
			data['ui'] = np.vstack((data['ui'], u_aux))

			for t in range(n_steps):
				init = np.random.binomial(1, 1.0 - self.discount)
				if reset and init:
					break
				else:
					data['x'] = np.vstack((data['x'], x_aux))
					data['z'] = np.hstack((data['z'], z_aux))
					data['u'] = np.vstack((data['u'], u_aux))

					z_aux, x_aux, r_aux = self.env.step(u_aux)
					data['xn'] = np.vstack((data['xn'], x_aux))
					data['zn'] = np.hstack((data['zn'], z_aux))
					data['r'] = np.hstack((data['r'], r_aux))

					u_aux = self.ctl.actions(z_aux, self.get_pfeatures(x_aux), stoch)
					data['un'] = np.vstack((data['un'], u_aux))

					n = n + 1

		return data

	def evaluate(self, n_rollouts, n_steps):
		return self.sample(n_rollouts, n_steps, 0, 0, False, False)

	def get_vfeatures(self, z, x):
		vfeat = np.zeros((x.shape[0], self.n_regions * self.n_vfeatures))

		poly = PolynomialFeatures(2)
		for n in range(self.n_regions):
			i = np.where(z == n)[0]
			if i.size > 0:
				idx = np.ix_(i, range(n * self.n_vfeatures, n * self.n_vfeatures + self.n_vfeatures))
				vfeat[idx] = poly.fit_transform(x[i])

		return vfeat

	def get_qfeatures(self, zn, z, xn, x, u):
		qfeat = np.zeros((x.shape[0], self.n_regions * self.n_vfeatures))

		pred = lgstc_feat(x, u)

		poly = PolynomialFeatures(2)
		for n in range(self.n_regions):
			mu = np.einsum('kh,th->tk', self.slds.linear_models[n].A, x) \
				      + np.einsum('kh,th->tk', self.slds.linear_models[n].B, u) + self.slds.linear_models[n].C

			ind = np.triu_indices(n=self.n_states, k=0, m=self.n_states)
			vec = self.slds.linear_models[n].cov[ind]
			vec = np.hstack((np.zeros(self.n_states + 1), vec))

			feat = poly.fit_transform(mu) + np.tile(vec, (mu.shape[0], 1))

			prob = logistic(self.slds.logistic_model.par[z, :], pred)
			qfeat = np.tile(feat, (1, self.n_regions)) * np.repeat(prob, self.n_vfeatures, axis=1)

		return qfeat

	def get_pfeatures(self, x):
		return linear_features(x)

	def kl_divergence(self):
		adv = self.data['r'] + self.discount * np.dot(self.qfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(1.0 / self.eta * delta, EXP_MIN, EXP_MAX))
		w = w[w >= 1e-75]
		w = w / np.mean(w, axis=0, keepdims=True)
		return np.mean(w * np.log(w), axis=0, keepdims=True)

	def update_policy(self):
		poly = PolynomialFeatures(1)

		adv = self.data['r'] + self.discount * np.dot(self.qfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))

		for n in range(self.n_regions):
			i = np.where(self.data['z'] == n)[0]
			if i.size > 0:
				psi = poly.fit_transform(self.data['x'][i])

				from sklearn.linear_model import Ridge
				clf = Ridge(alpha=0.0001, fit_intercept=False, solver='sparse_cg', max_iter=5000, tol=1e-4)
				clf.fit(psi, self.data['u'][i], sample_weight=w[i])
				self.ctl.K[n, :] = clf.coef_

				# wm = np.diagflat(w[i].flatten())
				# aux = np.linalg.inv(psi.T @ wm @ psi + 1e-12 * np.eye(3)) @ psi.T @ wm @ self.data['u'][i]
				# self.ctl.K[n, :] = aux.T

				Z = (np.square(np.sum(w[i], axis=0, keepdims=True)) - np.sum(np.square(w[i]), axis=0, keepdims=True)) / np.sum(w[i], axis=0, keepdims=True)
				tmp = self.data['u'][i] - psi @ self.ctl.K[n, :].T
				self.ctl.cov[n, :, :] = np.einsum('t,tk,th->kh', w[i], tmp, tmp) / (Z + 1e-12)


hyreps = HyREPS(n_states=2, n_actions=1, n_regions=2,
				n_rollouts=50, n_steps=25, n_filter=5,
				n_iter=10, n_keep=0,
				kl_bound=1.0, discount=0.98,
				action_limit=25.0, state_limit=np.array([np.inf, 20.0]))

for it in range(hyreps.n_iter):
	eval = hyreps.evaluate(hyreps.n_sim_rollouts, hyreps.n_sim_steps)

	hyreps.data = hyreps.sample(hyreps.n_rollouts, hyreps.n_steps, hyreps.n_filter, hyreps.n_keep)

	hyreps.ivfeatures = hyreps.get_vfeatures(hyreps.data['zi'], hyreps.data['xi'])
	hyreps.vfeatures = hyreps.get_vfeatures(hyreps.data['z'], hyreps.data['x'])
	hyreps.qfeatures = hyreps.get_qfeatures(hyreps.data['zn'], hyreps.data['z'], hyreps.data['xn'], hyreps.data['x'], hyreps.data['u'])

	for _ in range(500):
		res = sc.optimize.minimize(dual_eta, hyreps.eta, method='L-BFGS-B', jac=grad_eta,
									args=(hyreps.omega, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.data['r']), bounds=((1e-8, 1e8),))
		# print(res)

		# check = sc.optimize.check_grad(dual_eta, grad_eta, res.x, hyreps.omega, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.data['r'])
		# print('Eta Error', check)

		hyreps.eta = res.x

		res = sc.optimize.minimize(dual_omega, hyreps.omega, method='SLSQP', jac=grad_omega,
									args=(hyreps.eta, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.data['r']))

		# res = sc.optimize.minimize(dual_omega, hyreps.omega, method='BFGS', jac=grad_omega,
		# 							args=(hyreps.eta, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.data['r']),
		# 							options={'maxiter': 1000, 'gtol': 1e-6, 'norm': np.inf})

		# res = sc.optimize.minimize(dual_omega, hyreps.omega, method='trust-exact', jac=grad_omega, hess=hess_omega,
		# 							args=(hyreps.eta, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.data['r']))

		# print(res)

		# check = sc.optimize.check_grad(dual_omega, grad_omega, res.x, hyreps.eta, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.data['r'])
		# print('Omega Error', check)

		hyreps.omega = res.x

	kl_div = hyreps.kl_divergence()

	hyreps.update_policy()

	print('Iteration:', it, 'Reward:', np.mean(eval['r']), 'KL:', kl_div, 'Cov:', *hyreps.ctl.cov)

eval = hyreps.evaluate(hyreps.n_sim_rollouts, hyreps.n_sim_steps)

State = eval['x'].reshape((-1, hyreps.n_sim_steps, hyreps.n_states))
Region = eval['z'].reshape((-1, hyreps.n_sim_steps,))
Reward = eval['r'].reshape((-1, hyreps.n_sim_steps,))

fig = plt.figure()
sfig0 = plt.subplot(221)
sfig1 = plt.subplot(222)
sfig2 = plt.subplot(223)
sfig3 = plt.subplot(224)

for rollout in range(hyreps.n_sim_rollouts):
	sfig0.plot(State[rollout, :, 0])
	sfig1.plot(State[rollout, :, 1])
	sfig2.plot(Region[rollout, :])
	sfig3.plot(Reward[rollout, :])
plt.show()

fig = plt.figure()
for rollout in range(hyreps.n_sim_rollouts):
	plt.plot(State[rollout, :, 0], State[rollout, :, 1], )
plt.show()
