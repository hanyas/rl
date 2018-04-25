import numpy as np
import numexpr as ne
import scipy as sc
from scipy import integrate

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

np.random.seed(123)

REG = 1e-75


def lgstc_feat(x, u):
	# xu = np.hstack((x, u))
	# data = np.reshape(xu, (-1, xu.shape[-1]), order='C')
	# poly = PolynomialFeatures(1)

	xu = x
	data = np.reshape(xu, (-1, xu.shape[-1]), order='C')
	poly = PolynomialFeatures(2, interaction_only=True)

	feat = np.reshape(poly.fit_transform(data), (xu.shape[0], -1), order='C')
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

	a = np.clip(a, -420.0, 420.0)
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

		self.K = np.random.randn(n_regions, n_actions, n_states)
		self.kff = np.random.randn(n_regions, n_actions)
		self.cov = np.zeros((n_regions, n_actions, n_actions))
		for n in range(n_regions):
			self.cov[n, :, :] = 100.0 * np.eye(n_actions)

	def actions(self, x, z):
		mean = np.dot(self.K[z, :, :], x) + self.kff[z, :]
		return np.random.normal(mean, np.sqrt(self.cov[z, :, :])).flatten()

	def weighted_actions(self, x, p):
		a = np.zeros((p.shape[0]))
		for n in range(a.shape[0]):
			mean = np.dot(self.K[n, :, :], x) + self.kff[n, :]
			a[n] = p[n] * np.random.normal(mean, np.sqrt(self.cov[n, :, :])).flatten()

		return a.sum()


class Dynamics:

	def __init__(self, n_states, n_actions, n_regions, n_features, action_limit):
		self.n_states = n_states
		self.n_actions = n_actions
		self.n_regions = n_regions
		self.n_features = n_features

		self.init_region = sc.stats.multinomial(1, np.ones(n_regions) / n_regions)
		self.init_state = sc.stats.multivariate_normal(mean=np.random.randn(n_states), cov=np.eye(n_states))
		self.linear_models = [LinearGaussian(n_states, n_actions) for _ in range(n_regions)]
		self.logistic_model = MultiLogistic(n_regions, n_features)

		self.action_min = - action_limit
		self.action_max = action_limit

	def simulate(self, x, u):
		def dynamics(x, t, u, g, m, l, k):
			return [x[1], 3 * g * np.sin(x[0]) / l + 3 * u / (m * l ** 2) - 3 * k * x[1] / (m * l ** 2)]

		uc = np.clip(u, self.action_min, self.action_max)

		dt = 0.05
		g = 9.81
		l = 0.5
		m = 10.0
		k = 0.25

		xn = sc.integrate.odeint(dynamics, x, np.array([0.0, dt]), args=(uc, g, m, l, k))[1, :]

		# clip velocity
		xn[1] = np.clip(xn[1], -20.0, 20.0)

		return xn, uc

	def load(self, stamp, path):
		file = open(path + "bw_init_region_" + stamp + ".pickle", "rb")
		self.init_region.p = pickle.load(file)
		file.close()

		file = open(path + "bw_init_state_" + stamp + ".pickle", "rb")
		init_state = pickle.load(file)
		self.init_state.mean = init_state[0]
		self.init_state.cov = init_state[1]
		file.close()

		file = open(path + "bw_linear_models_" + stamp + ".pickle", "rb")
		self.linear_models = pickle.load(file)
		for i in range(self.n_regions):
			self.linear_models[i].update()
		file.close()

		file = open(path + "bw_logistic_model_" + stamp + ".pickle", "rb")
		self.logistic_model.par = pickle.load(file)
		file.close()

	def forward(self, x, u):
		n_steps = x.shape[0]

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

		return alpha, norm


def reward(x, u, Wx, Wu, g):
	# transform state for swing up
	x[:, :, 0] = np.mod(x[:, :, 0] + 0.5 * np.pi, 2 * np.pi) - 0.5 * np.pi

	r = - np.einsum('ntk,kh,nth->nt', x - g, Wx, x - g) - np.einsum('ntk,kh,nth->nt', u, Wu, u)
	return r


def dual_eta(eta, omega, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_eta(eta, omega, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	deta = epsilon + np.log(np.mean(w, axis=0, keepdims=True)) - np.sum(w * delta, axis=0, keepdims=True) / (eta * np.sum(w, axis=0, keepdims=True))
	return deta


def dual_omega(omega, eta, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	g = np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_omega(omega, eta, epsilon, gamma, qfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(qfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	domega = (1.0 - gamma) * np.mean(ivfeatures, axis=0, keepdims=False) + np.sum(w[:, None] * (gamma * qfeatures - vfeatures), axis=0, keepdims=False) / np.sum(w, axis=0, keepdims=False)
	return domega


class HyREPS:

	def __init__(self, n_states, n_actions, n_regions, n_rollouts, n_steps, n_replace, n_iter, kl_bound, discount, action_limit):
		self.n_states = n_states
		self.n_actions = n_actions
		self.n_regions = n_regions
		self.n_rollouts = n_rollouts
		self.n_steps = n_steps
		self.n_replace = n_replace
		self.n_iter = n_iter
		self.kl_bound = kl_bound
		self.discount = discount

		self.n_sim_rollouts = 1 * n_rollouts
		self.n_sim_steps = n_steps

		self.n_lgstc_features = 4 #n_states + n_actions + 1
		self.n_vfeatures = int(2 * n_states + sc.special.comb(n_states, 2) + 1)

		self.dyn = Dynamics(n_states, n_actions, n_regions, n_features=self.n_lgstc_features, action_limit=action_limit)
		self.dyn.load("15:17:32", "data/pendulum/")

		self.ctl = Policy(n_states, n_actions, n_regions)

		self.Z = np.zeros((n_rollouts, n_steps), np.int64)
		self.X = np.zeros((n_rollouts, n_steps, n_states))
		self.X_tmp = np.zeros((n_rollouts, n_steps, n_states))
		self.U = np.zeros((n_rollouts, n_steps, n_actions))
		self.U_sat = np.zeros((n_rollouts, n_steps, n_actions))
		self.R = np.zeros((n_rollouts, n_steps, n_actions))

		self.zi = np.zeros((n_rollouts), np.int64)
		self.xi = np.zeros((n_rollouts, n_states))
		self.ui = np.zeros((n_rollouts, n_actions))
		self.ui_sat = np.zeros((n_rollouts, n_actions))

		self.z = np.zeros((n_rollouts * (n_steps - 1)), np.int64)
		self.x = np.zeros((n_rollouts * (n_steps - 1), n_states))
		self.u = np.zeros((n_rollouts * (n_steps - 1), n_actions))
		self.u_sat = np.zeros((n_rollouts * (n_steps - 1), n_actions))

		self.zn = np.zeros((n_rollouts * (n_steps - 1)), np.int64)
		self.xn = np.zeros((n_rollouts * (n_steps - 1), n_states))
		self.un = np.zeros((n_rollouts * (n_steps - 1), n_actions))
		self.un_sat = np.zeros((n_rollouts * (n_steps - 1), n_actions))

		self.vfeatures = np.zeros((self.x.shape[0], self.n_regions * self.n_vfeatures))
		self.ivfeatures = np.zeros((self.xi.shape[0], self.n_regions * self.n_vfeatures))
		self.qfeatures = np.zeros((self.x.shape[0], self.n_regions * self.n_vfeatures))

		self.goal = np.array([0.0, 0.0])
		self.reward = np.zeros((n_rollouts * (n_steps - 1)))

		self.state_weight = np.diag([1e1, 1e-1])
		self.action_weight = np.diag([1e-3])

		self.omega = np.random.randn(n_regions * self.n_vfeatures)
		self.eta = np.array([1e3])

	def sample(self, n_total_rollouts, n_steps, n_replace, init, Z, X, X_tmp, U, U_sat):
		if init == True:
			n_rollouts = n_total_rollouts
		else:
			n_rollouts = n_replace

		for n in range(n_rollouts):
			# tmp = np.hstack((np.random.uniform(0.01, 2 * np.pi - 0.01), 0.0))
			x_tmp = np.hstack((np.random.uniform(np.pi - np.pi/6, np.pi + np.pi/6), 0.0))

			x_aux = x_tmp.copy()
			x_aux[0] = np.mod(x_aux[0], 2 * np.pi)

			z_aux = np.argmax(self.dyn.init_region.rvs())

			u_aux = self.ctl.actions(x_aux, z_aux)
			u_sat_aux = np.clip(u_aux, self.dyn.action_min, self.dyn.action_max)

			# roll arrays
			Z = np.roll(Z, 1, axis=0)
			X = np.roll(X, 1, axis=0)
			X_tmp = np.roll(X_tmp, 1, axis=0)
			U = np.roll(U, 1, axis=0)
			U_sat = np.roll(U_sat, 1, axis=0)

			# replace first rollout and init first time step
			Z[0, 0] = z_aux
			X[0, 0, :] = x_aux
			X_tmp[0, 0, :] = x_tmp
			U[0, 0, :] = u_aux
			U_sat[0, 0, :] = u_sat_aux

			for t in range(1, n_steps):
				X_tmp[0, t, :], _ = self.dyn.simulate(X_tmp[0, t - 1, :], U_sat[0, t - 1, :])

				# transform angle
				X[0, t, :] = X_tmp[0, t, :].copy()
				X[0, t, 0] = np.mod(X[0, t, 0], 2 * np.pi)

				p, _ = self.dyn.forward(X[0, :t + 1, :], U_sat[0, :t + 1, :])
				Z[0, t] = np.argmax(p[-1, :])

				U[0, t, :] = self.ctl.actions(X[0, t, :], Z[0, t])
				U_sat[0, t, :] = np.clip(U[0, t, :], self.dyn.action_min, self.dyn.action_max)


		R = reward(X_tmp, U_sat, self.state_weight, self.action_weight, self.goal)
		r = R[:, :-1].flatten(order='C')

		zi = np.reshape(Z[:, 0], (-1,), order='C')
		xi = np.reshape(X[:, 0, :], (-1, self.n_states), order='C')
		ui = np.reshape(U[:, 0, :], (-1, self.n_actions), order='C')
		ui_sat = np.reshape(U_sat[:, 0, :], (-1, self.n_actions), order='C')

		z = np.reshape(Z[:, :-1], (-1,), order='C')
		x = np.reshape(X[:, :-1, :], (-1, self.n_states), order='C')
		u = np.reshape(U[:, :-1, :], (-1, self.n_actions), order='C')
		u_sat = np.reshape(U_sat[:, :-1, :], (-1, self.n_actions), order='C')

		zn = np.reshape(Z[:, 1:], (-1,), order='C')
		xn = np.reshape(X[:, 1:, :], (-1, self.n_states), order='C')
		un = np.reshape(U[:, 1:, :], (-1, self.n_actions), order='C')
		un_sat = np.reshape(U_sat[:, 1:, :], (-1, self.n_actions), order='C')

		return zn, xn, un, un_sat, z, x, u, u_sat, r, zi, xi, ui, ui_sat, Z, X, X_tmp, U, U_sat, R

	def evaluate(self, n_rollouts, n_steps):
		Z = np.zeros((n_rollouts, n_steps), np.int64)
		X = np.zeros((n_rollouts, n_steps, self.n_states))
		X_tmp = np.zeros((n_rollouts, n_steps, self.n_states))
		U = np.zeros((n_rollouts, n_steps, self.n_actions))
		U_sat = np.zeros((n_rollouts, n_steps, self.n_actions))

		zn, xn, un, un_sat,\
		z, x, u, u_sat, r,\
		zi, xi, ui, ui_sat,\
		Z, X, X_tmp, U, U_sat, R = self.sample(n_rollouts, n_steps, n_rollouts, True, Z, X, X_tmp, U, U_sat)

		return Z, X, U, U_sat, R

	def get_vfeatures(self, init=0):
		poly = PolynomialFeatures(2)

		if init == 0:
			x = self.x
			z = self.z
		else:
			x = self.xi
			z = self.zi

		phi = np.zeros((x.shape[0], self.n_regions * self.n_vfeatures))

		for n in range(self.n_regions):
			i = np.where(z == n)[0]
			if i.size != 0:
				idx = np.ix_(i, range(n * self.n_vfeatures, n * self.n_vfeatures + self.n_vfeatures))
				phi[idx] = poly.fit_transform(x[i])

		return phi

	def get_qfeatures(self):
		poly = PolynomialFeatures(2)
		pred = lgstc_feat(self.x, self.u_sat)

		self.qfeatures = np.zeros((self.x.shape[0], self.n_regions * self.n_vfeatures))

		for n in range(self.n_regions):
			i = np.where(self.zn == n)[0]
			if i.size != 0:
				lgstc = logistic(self.dyn.logistic_model.par[self.z[i], :], pred[i])
				aux = np.einsum('kh,th->tk', self.dyn.linear_models[n].A, self.x[i]) \
				      + np.einsum('kh,th->tk', self.dyn.linear_models[n].B, self.u_sat[i]) + self.dyn.linear_models[n].C

				ind = np.triu_indices(n=self.n_states, k=0, m=self.n_states)
				vec = self.dyn.linear_models[n].cov[ind]
				vec = np.hstack((np.zeros(self.n_states + 1), vec))

				feat = poly.fit_transform(aux) + np.tile(vec, (aux.shape[0], 1))

				self.qfeatures[i, :] = np.tile(feat, (1, self.n_regions)) * np.repeat(lgstc, self.n_vfeatures, axis=1)

	def kl_divergence(self):
		adv = self.reward + self.discount * np.dot(self.qfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv
		w = np.exp(np.clip(1.0 / self.eta * delta, -420.0, 420.0))
		w = w[w >= 1e-45]
		w = w / np.mean(w, axis=0, keepdims=True)
		return np.mean(w * np.log(w), axis=0, keepdims=True)

	def update_policy(self):
		poly = PolynomialFeatures(1)

		adv = self.reward + self.discount * np.dot(self.qfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv
		w = np.exp(np.clip(delta / self.eta, -420.0, 420.0))

		for n in range(self.n_regions):
			i = np.where(self.z == n)[0]
			if i.size != 0:
				psi = poly.fit_transform(self.x[i])
				wm = np.diagflat(w[i].flatten())

				aux = np.linalg.inv((psi.T @ wm @ psi) + 1e-12 * np.eye(self.n_states + 1)) @ psi.T @ wm @ self.u_sat[i]
				self.ctl.kff[n, :self.n_actions] = aux[:self.n_actions]
				self.ctl.K[n, :] = aux[self.n_actions:].T

				Z = (np.square(np.sum(w[i], axis=0, keepdims=True)) - np.sum(np.square(w[i]), axis=0, keepdims=True)) / np.sum(w[i], axis=0, keepdims=True)
				tmp = self.u_sat[i] - psi @ aux
				self.ctl.cov[n, :, :] = np.einsum('t,tk,th->kh', w[i], tmp, tmp) / Z


hyreps = HyREPS(n_states=2, n_actions=1, n_regions=3,
				n_rollouts=50, n_steps=50, n_replace=10,
				n_iter=25, kl_bound=0.250, discount=0.98, action_limit=30.0)

for it in range(hyreps.n_iter):
	hyreps.zn, hyreps.xn, hyreps.un, hyreps.un_sat, \
	hyreps.z, hyreps.x, hyreps.u, hyreps.u_sat, hyreps.reward, \
	hyreps.zi, hyreps.xi, hyreps.ui, hyreps.ui_sat,\
	hyreps.Z, hyreps.X, hyreps.X_tmp, hyreps.U, hyreps.U_sat, hyreps.R = hyreps.sample(hyreps.n_rollouts, hyreps.n_steps, hyreps.n_replace,
																it==0, hyreps.Z, hyreps.X, hyreps.X_tmp, hyreps.U, hyreps.U_sat)

	hyreps.vfeatures = hyreps.get_vfeatures()
	hyreps.ivfeatures = hyreps.get_vfeatures(init=1)
	hyreps.get_qfeatures()

	kl_div = 99.0
	inner = 0
	while (np.fabs(kl_div - hyreps.kl_bound) >= 0.1 * hyreps.kl_bound):
		res = sc.optimize.minimize(dual_eta, 1000.0 * np.ones((1,)), method='L-BFGS-B', jac=grad_eta,
			args=(hyreps.omega, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.reward), bounds=((1e-8, 1e8),))
		# print(res)

		# check = sc.optimize.check_grad(dual_eta, grad_eta, res.x, hyreps.omega, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.reward)
		# print('Eta Error', check)

		hyreps.eta = res.x

		res = sc.optimize.minimize(dual_omega, hyreps.omega, method='BFGS', jac=grad_omega,
			args=(hyreps.eta, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.reward),
			options={'gtol': 1e-4, 'norm': -np.inf})
		# print(res)

		# check = sc.optimize.check_grad(dual_omega, grad_omega, res.x, hyreps.eta, hyreps.kl_bound, hyreps.discount, hyreps.qfeatures, hyreps.vfeatures, hyreps.ivfeatures, hyreps.reward)
		# print('Omega Error', check)

		hyreps.omega = res.x

		kl_div = hyreps.kl_divergence()
		# print(kl_div)
		inner = inner + 1
		if inner > 50:
			break

	hyreps.update_policy()

	# evaluation
	# Z_eval, X_eval, U_eval, U_sat_eval, RWRD_eval = hyreps.evaluate(hyreps.n_sim_rollouts, hyreps.n_sim_steps)

	print('Iteration:', it, 'Reward:', np.mean(hyreps.reward), 'KL:', kl_div, 'Cov', *hyreps.ctl.cov)

for n in range(1):
	rollout = np.random.randint(0, hyreps.n_rollouts)
	plt.subplot(321)
	plt.plot(hyreps.X[rollout, :, 0])
	plt.subplot(322)
	plt.plot(hyreps.X[rollout, :, 1])
	plt.subplot(323)
	plt.plot(hyreps.R[rollout, :])
	plt.subplot(325)
	plt.plot(hyreps.U_sat[rollout, :, :])
	plt.subplot(326)
	plt.plot(hyreps.Z[rollout, :])
plt.show()
