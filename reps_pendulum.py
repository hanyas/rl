import numpy as np
import scipy as sc
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns

# import os
# os.environ['DISPLAY'] = 'russell:11.0'


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

np.random.seed(99)

EXP_MAX = 700.0
EXP_MIN = -700.0


class PolicyML:

	def __init__(self, n_features, n_actions):
		self.n_features = n_features
		self.n_actions = n_actions

		self.K = np.random.randn(n_actions, n_features)
		self.cov = 3600.0 * np.eye(n_actions)

	def actions(self, x):
		mean = np.dot(self.K, x)
		return np.random.normal(mean, np.sqrt(self.cov)).flatten()


class PolicyBayes:

	def __init__(self, n_features, n_actions):
		self.n_features = n_features
		self.n_actions = n_actions
		self.alpha = 1e-4
		self.beta = 1e-1

		self.K = np.zeros((n_actions, n_features))
		self.Sn = np.linalg.inv(self.alpha * np.eye(n_features))

	def actions(self, x):
		mean = self.beta * np.einsum('k,kh,mh->m', x, self.Sn, self.K)
		cov = np.einsum('k,km,m->', x, self.Sn, x) + 1./self.beta
		return np.random.normal(mean, np.sqrt(cov)).flatten()


class Dynamics:

	def __init__(self, n_states, n_actions, action_limit, state_limit):
		self.n_states = n_states
		self.n_actions = n_actions

		self.action_min = - action_limit
		self.action_max = action_limit

		self.state_min = - state_limit
		self.state_max = state_limit

		self.x_intern = np.zeros((2, ))

	def reset(self):
		# self.x_intern = np.array([np.random.uniform(0.01, 2 * np.pi - 0.01), 0.0])
		self.x_intern = np.array([np.random.uniform(np.pi - np.pi/3.0, np.pi + np.pi/3.0), 0.0])
		self.x_intern = np.random.multivariate_normal(self.x_intern, 1e-8 * np.eye(2))

		xt = transform_angle(self.x_intern)
		xc = pol2cart(xt)
		return xc

	def step(self, u):
		uc = np.clip(u, self.action_min, self.action_max)

		dt, g = 0.05, 9.81
		# l, m, k = 0.5, 10.0, 0.25
		l, m, k = 1.0, 0.5, 0.1

		self.x_intern = sc.integrate.odeint(pendulum_ode, self.x_intern, np.array([0.0, dt]), args=(uc, g, m, l, k))[1, :]
		self.x_intern[0] = np.remainder(self.x_intern[0], 2 * np.pi)
		self.x_intern = np.clip(self.x_intern, self.state_min, self.state_max)
		self.x_intern = np.random.multivariate_normal(self.x_intern, 1e-8 * np.eye(2))

		xt = transform_angle(self.x_intern)
		xc = pol2cart(xt)
		return xc


def pendulum_ode(x, t, u, g, m, l, k):
	return [x[1], 3 * g * np.sin(x[0]) / l + 3 * u / (m * l ** 2) - 3 * k * x[1] / (m * l ** 2)]
	# return [x[1], g * np.sin(x[0]) / l + u / (m * l ** 2) - k * x[1] / (m * l ** 2)]


def transform_angle(state):
	transformed = state.copy()
	transformed[0] = np.remainder(transformed[0] + np.pi, 2.0 * np.pi) - np.pi
	return transformed


def pol2cart(state):
	# [cos, sin, -sin * thd, cos * thd]
	return np.array([np.cos(state[0]), np.sin(state[0]), - np.sin(state[0]) * state[1], np.cos(state[0]) * state[1]])


def cart2pol(state):
	ang = np.arctan2(state[:, :, 1], state[:, :, 0])
	vel = - state[:, :, 1] * state[:, :, 2] + state[:, :, 0] * state[:, :, 3]
	polar = np.stack((ang, vel), axis=2)
	return polar


def reward(x, u, Wx, Wu, g):
	xp = cart2pol(x)
	r = - np.einsum('ntk,kh,nth->nt', xp - g, Wx, xp - g) - np.einsum('ntk,kh,nth->nt', u, Wu, u)
	return r


def dual_eta(eta, omega, epsilon, gamma, nvfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(nvfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_eta(eta, omega, epsilon, gamma, nvfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(nvfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	deta = epsilon + np.log(np.mean(w, axis=0, keepdims=True)) - np.sum(w * delta, axis=0, keepdims=True) / (eta * np.sum(w, axis=0, keepdims=True))
	return deta


def dual_omega(omega, eta, epsilon, gamma, nvfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(nvfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	g = np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_omega(omega, eta, epsilon, gamma, nvfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(nvfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	domega = (1.0 - gamma) * np.mean(ivfeatures, axis=0, keepdims=False) + np.sum(w[:, None] * (gamma * nvfeatures - vfeatures), axis=0, keepdims=False) / np.sum(w, axis=0, keepdims=False)
	return domega


def hess_omega(omega, eta, epsilon, gamma, nvfeatures, vfeatures, ivfeatures, r):
	adv = r + gamma * np.dot(nvfeatures, omega) - np.dot(vfeatures, omega) + (1.0 - gamma) * np.mean(np.dot(ivfeatures, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
	w = w / np.sum(w, axis=0, keepdims=True)

	ui = gamma * nvfeatures - vfeatures
	uj = np.sum(w[:, None] * ui, axis=0, keepdims=True)
	tmp = ui - uj

	homega = 1./eta * np.einsum('n,nk,nh->kh', w, tmp, tmp)
	return homega


def fourier_features(x, n_feat, freq, shift):
	if x.ndim > 1:
		phi = np.zeros((x.shape[0], n_feat))
		for i in range(n_feat):
			phi[:, i] = np.sin(np.einsum('k,nk->n', freq[i, :], x) / 2.0 + shift[i])
	else:
		phi = np.zeros((n_feat, ))
		for i in range(n_feat):
			phi[i] = np.sin(freq[i, :] @ x / 2.0 + shift[i])

	return phi


class REPS:

	def __init__(self, n_states, n_actions,
	             n_rollouts, n_steps, n_replace,
	             n_iter, kl_bound, discount,
	             action_limit, state_limit,
	             n_vfeat, n_pfeat):

		self.n_states = n_states
		self.n_actions = n_actions
		self.n_rollouts = n_rollouts
		self.n_steps = n_steps
		self.n_replace = n_replace
		self.n_iter = n_iter
		self.kl_bound = kl_bound
		self.discount = discount

		self.n_sim_rollouts = n_rollouts
		self.n_sim_steps = n_steps

		self.n_vfeatures = n_vfeat
		self.n_pfeatures = n_pfeat

		self.vfeat_freq = np.random.randn(self.n_vfeatures, self.n_states)
		self.vfeat_shift = np.random.uniform(-np.pi, np.pi, size=self.n_vfeatures)

		self.pfeat_freq = self.vfeat_freq
		self.pfeat_shift = self.vfeat_shift

		# self.pfeat_freq = np.random.randn(self.n_pfeatures, self.n_states)
		# self.pfeat_shift = np.random.uniform(-np.pi, np.pi, size=self.n_pfeatures)

		self.dyn = Dynamics(n_states, n_actions, action_limit=action_limit, state_limit=state_limit)

		self.ctl = PolicyML(self.n_pfeatures, n_actions)
		# self.ctl = PolicyBayes(self.n_pfeatures, n_actions)

		self.X = np.zeros((n_rollouts, n_steps, n_states))
		self.U = np.zeros((n_rollouts, n_steps, n_actions))
		self.U_sat = np.zeros((n_rollouts, n_steps, n_actions))
		self.R = np.zeros((n_rollouts, n_steps, n_actions))

		self.xi = np.zeros((n_rollouts, n_states))
		self.ui = np.zeros((n_rollouts, n_actions))
		self.ui_sat = np.zeros((n_rollouts, n_actions))

		self.x = np.zeros((n_rollouts * (n_steps - 1), n_states))
		self.u = np.zeros((n_rollouts * (n_steps - 1), n_actions))
		self.u_sat = np.zeros((n_rollouts * (n_steps - 1), n_actions))

		self.xn = np.zeros((n_rollouts * (n_steps - 1), n_states))
		self.un = np.zeros((n_rollouts * (n_steps - 1), n_actions))
		self.un_sat = np.zeros((n_rollouts * (n_steps - 1), n_actions))

		self.vfeatures = np.zeros((self.x.shape[0], self.n_vfeatures))
		self.ivfeatures = np.zeros((self.xi.shape[0], self.n_vfeatures))
		self.nvfeatures = np.zeros((self.xn.shape[0], self.n_vfeatures))

		self.goal = np.array([0.0, 0.0])

		self.reward = np.zeros((n_rollouts * (n_steps - 1)))

		self.state_weight = np.diag([1e1, 1e-1])
		self.action_weight = np.diag([1e-3])

		self.omega = np.random.randn(self.n_vfeatures)
		self.eta = np.array([1e3])

	def sample(self, n_total_rollouts, n_steps, n_replace, init, X, U, U_sat, reset=True):
		if init == True:
			n_rollouts = n_total_rollouts
		else:
			n_rollouts = n_replace

		for n in range(n_rollouts):
			x_aux = self.dyn.reset()
			u_aux = self.ctl.actions(self.get_pfeatures(x_aux))
			u_sat_aux = np.clip(u_aux, self.dyn.action_min, self.dyn.action_max)

			# roll arrays
			X = np.roll(X, 1, axis=0)
			U = np.roll(U, 1, axis=0)
			U_sat = np.roll(U_sat, 1, axis=0)

			# replace first rollout and init first time step
			X[0, 0, :] = x_aux
			U[0, 0, :] = u_aux
			U_sat[0, 0, :] = u_sat_aux

			for t in range(1, n_steps):
				prob = np.random.binomial(1, 1.0 - self.discount)
				if reset and prob:
					X[0, t, :] = self.dyn.reset()
				else:
					X[0, t, :] = self.dyn.step(U_sat[0, t - 1, :])

				U[0, t, :] = self.ctl.actions(self.get_pfeatures(X[0, t, :]))
				U_sat[0, t, :] = np.clip(U[0, t, :], self.dyn.action_min, self.dyn.action_max)

		R = reward(X, U_sat, self.state_weight, self.action_weight, self.goal)
		r = R[:, :-1].flatten(order='C')

		xi = np.reshape(X[:, 0, :], (-1, self.n_states), order='C')
		ui = np.reshape(U[:, 0, :], (-1, self.n_actions), order='C')
		ui_sat = np.reshape(U_sat[:, 0, :], (-1, self.n_actions), order='C')

		x = np.reshape(X[:, :-1, :], (-1, self.n_states), order='C')
		u = np.reshape(U[:, :-1, :], (-1, self.n_actions), order='C')
		u_sat = np.reshape(U_sat[:, :-1, :], (-1, self.n_actions), order='C')

		xn = np.reshape(X[:, 1:, :], (-1, self.n_states), order='C')
		un = np.reshape(U[:, 1:, :], (-1, self.n_actions), order='C')
		un_sat = np.reshape(U_sat[:, 1:, :], (-1, self.n_actions), order='C')

		return xn, un, un_sat, x, u, u_sat, r, xi, ui, ui_sat, X, U, U_sat, R

	def evaluate(self, n_rollouts, n_steps):
		X = np.zeros((n_rollouts, n_steps, self.n_states))
		U = np.zeros((n_rollouts, n_steps, self.n_actions))
		U_sat = np.zeros((n_rollouts, n_steps, self.n_actions))

		return self.sample(n_rollouts, n_steps, n_rollouts, True, X, U, U_sat, False)

	def get_vfeatures(self, x):
		# poly = PolynomialFeatures(2)
		# return poly.fit_transform(x)
		return fourier_features(x, self.n_vfeatures, self.vfeat_freq, self.vfeat_shift)

	def get_pfeatures(self, x):
		# poly = PolynomialFeatures(1)
		# return poly.fit_transform(x)
		return fourier_features(x, self.n_pfeatures, self.pfeat_freq, self.pfeat_shift)

	def kl_divergence(self):
		adv = self.reward + self.discount * np.dot(self.nvfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(1.0 / self.eta * delta, EXP_MIN, EXP_MAX))
		w = w[w >= 1e-45]
		w = w / np.mean(w, axis=0, keepdims=True)
		return np.mean(w * np.log(w), axis=0, keepdims=True)

	def ml_policy(self):
		adv = self.reward + self.discount * np.dot(self.nvfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))

		psi = self.get_pfeatures(self.x)
		wm = np.diagflat(w.flatten())

		aux = np.linalg.inv((psi.T @ wm @ psi) + 1e-16 * np.eye(self.n_pfeatures)) @ psi.T @ wm @ self.u_sat
		self.ctl.K = aux.T

		Z = (np.square(np.sum(w, axis=0, keepdims=True)) - np.sum(np.square(w), axis=0, keepdims=True)) / np.sum(w, axis=0, keepdims=True)
		tmp = self.u_sat - psi @ self.ctl.K.T
		self.ctl.cov = np.einsum('t,tk,th->kh', w, tmp, tmp) / (Z + 1e-16)

	def bayes_policy(self):
		adv = self.reward + self.discount * np.dot(self.nvfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))

		psi = self.get_pfeatures(self.x)
		wm = np.diagflat(w.flatten())

		self.ctl.Sn = np.linalg.inv(self.ctl.beta * psi.T @ wm @ psi +  self.ctl.alpha * np.eye(self.n_pfeatures))
		self.ctl.K = (psi.T @ wm @ self.u_sat).T

# reps = REPS(n_states=4, n_actions=1,
# 			n_rollouts=25, n_steps=100, n_replace=25,
# 			n_iter=15, kl_bound=0.5, discount=0.985,
# 			action_limit=30.0, state_limit=np.array([np.inf, 20.0]),
# 			n_vfeat=100, n_pfeat=100)

reps = REPS(n_states=4, n_actions=1,
			n_rollouts=25, n_steps=100, n_replace=25,
			n_iter=15, kl_bound=0.5, discount=0.98,
			action_limit=3.5, state_limit=np.array([np.inf, 20.0]),
			n_vfeat=100, n_pfeat=100)

for it in range(reps.n_iter):
	reps.xn, reps.un, reps.un_sat, \
	reps.x, reps.u, reps.u_sat, reps.reward, \
	reps.xi, reps.ui, reps.ui_sat,\
	reps.X, reps.U, reps.U_sat, reps.R = reps.sample(reps.n_rollouts, reps.n_steps, reps.n_replace,
														it==0, reps.X, reps.U, reps.U_sat)

	reps.ivfeatures = reps.get_vfeatures(reps.xi)
	reps.vfeatures = reps.get_vfeatures(reps.x)
	reps.nvfeatures = reps.get_vfeatures(reps.xn)

	kl_div = 99.0
	inner = 0
	while (np.fabs(kl_div - reps.kl_bound) >= 0.1 * reps.kl_bound):
		res = sc.optimize.minimize(dual_eta, reps.eta, method='L-BFGS-B', jac=grad_eta,
									args=(reps.omega, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.reward), bounds=((1e-8, 1e8),))
		# print(res)

		# check = sc.optimize.check_grad(dual_eta, grad_eta, res.x, reps.omega, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.reward)
		# print('Eta Error', check)

		reps.eta = res.x

		# res = sc.optimize.minimize(dual_omega, reps.omega, method='BFGS', jac=grad_omega,
		# 	args=(reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.reward))

		res = sc.optimize.minimize(dual_omega, reps.omega, method='trust-exact', jac=grad_omega, hess=hess_omega,
									args=(reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.reward))

		# print(res)

		# check = sc.optimize.check_grad(dual_omega, grad_omega, res.x, reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.reward)
		# print('Omega Error', check)

		reps.omega = res.x

		kl_div = reps.kl_divergence()
		# print(kl_div)
		inner = inner + 1
		if inner > 50:
			break

	reps.ml_policy()
	# reps.bayes_policy()

	# evaluation
	xn, un, un_sat, \
	x, u, u_sat, r, \
	xi, ui, ui_sat, \
	X, U, U_sat, R = reps.evaluate(reps.n_sim_rollouts, reps.n_sim_steps)

	print('Iteration:', it, 'Reward:', np.mean(r), 'KL:', kl_div, 'Cov:', *reps.ctl.cov)
	# print('Iteration:', it, 'Reward:', np.mean(r), 'KL:', kl_div)

for rollout in range(reps.n_sim_rollouts):
	Ang = cart2pol(X)
	plt.subplot(221)
	plt.title('Cos, Sin')
	plt.plot(X[rollout, :, 0:2])
	plt.subplot(222)
	plt.title('Angle')
	plt.plot(Ang[rollout, :, 0])
	plt.subplot(223)
	plt.title('Reward')
	plt.plot(R[rollout, :])
	plt.subplot(224)
	plt.title('Action')
	plt.plot(U_sat[rollout, :, :])
plt.show()
