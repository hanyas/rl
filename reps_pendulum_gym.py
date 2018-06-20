import numpy as np
import scipy as sc
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns

import gym
env = gym.make('Pendulum-v0')

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

EXP_MAX = 700.0
EXP_MIN = -700.0


class MLPolicy:

	def __init__(self, n_features, n_actions):
		self.n_features = n_features
		self.n_actions = n_actions

		self.K = np.random.randn(n_actions, n_features)
		self.cov = np.eye(n_actions)

	def actions(self, x, stoch):
		mean = np.dot(self.K, x)
		if stoch:
			return np.random.normal(mean, np.sqrt(self.cov)).flatten()
		else:
			return mean


class BayesPolicy:

	def __init__(self, n_features, n_actions):
		self.n_features = n_features
		self.n_actions = n_actions
		self.alpha = 1e-3
		self.beta = 1e-1

		self.K = np.zeros((n_actions, n_features))
		self.Sn = np.linalg.inv(self.alpha * np.eye(n_features))

	def actions(self, x, stoch):
		mean = self.beta * np.einsum('k,kh,mh->m', x, self.Sn, self.K)
		if stoch:
			cov = np.einsum('k,km,m->', x, self.Sn, x) + 1./self.beta
			return np.random.normal(mean, np.sqrt(cov)).flatten()
		else:
			return mean


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
			phi[:, i] = np.sin(np.einsum('k,nk->n', freq[i, :], x) / 2.5 + shift[i])
	else:
		phi = np.zeros((n_feat, ))
		for i in range(n_feat):
			phi[i] = np.sin(freq[i, :] @ x / 2.5 + shift[i])

	return phi


def cart2pol(state):
	ang = np.arctan2(state[:, :, 1], state[:, :, 0])
	vel = state[:, :, 2]
	polar = np.stack((ang, vel), axis=2)
	return polar


class REPS:

	def __init__(self, n_states, n_actions,
	             n_rollouts, n_steps,
	             n_iter, n_keep,
	             kl_bound, discount,
	             n_vfeat, n_pfeat):

		self.n_states = n_states
		self.n_actions = n_actions
		self.n_rollouts = n_rollouts
		self.n_steps = n_steps
		self.n_iter = n_iter
		self.n_keep = n_keep
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

		self.env = gym.make('Pendulum-v0')
		self.action_limit = 2.0

		self.ctl = MLPolicy(self.n_pfeatures, n_actions)
		self.ctl.cov = (2.0 * self.action_limit)**2

		# self.ctl = BayesPolicy(self.n_pfeatures, n_actions)

		self.data = {'xi': np.empty((0, n_states)),
		             'ui': np.empty((0, n_actions)),
		             'x': np.empty((0, n_states)),
		             'u': np.empty((0, n_actions)),
		             'xn': np.empty((0, n_states)),
		             'un': np.empty((0, n_actions)),
		             'r': np.empty((0,))}

		self.vfeatures = None
		self.ivfeatures = None
		self.nvfeatures = None

		self.omega = np.random.randn(self.n_vfeatures)
		self.eta = np.array([1e3])

	def sample(self, n_rollouts, n_steps, n_keep, reset=True, stoch=True):
		if n_keep==0:
			data = { 'xi': np.empty((0, self.n_states)),
		             'x': np.empty((0, self.n_states)),
		             'u': np.empty((0, self.n_actions)),
		             'xn': np.empty((0, self.n_states)),
		             'r': np.empty((0,))}

			n_samples = n_rollouts * n_steps

		else:
			data = { 'xi': self.data['xi'],
		             'x': self.data['x'][-n_keep * n_steps:, :],
		             'u': self.data['u'][-n_keep * n_steps:, :],
		             'xn': self.data['xn'][-n_keep * n_steps:, :],
		             'r': self.data['r'][-n_keep * n_steps:]}

			n_samples = (n_rollouts - n_keep) * n_steps

		n = 0
		while n < n_samples:
			x_aux = self.env.reset()
			data['xi'] = np.vstack((data['xi'], x_aux))

			u_aux = self.ctl.actions(self.get_pfeatures(x_aux), stoch)

			for t in range(n_steps):
				init = np.random.binomial(1, 1.0 - self.discount)
				if reset and init:
					break
				else:
					data['x'] = np.vstack((data['x'], x_aux))
					data['u'] = np.vstack((data['u'], u_aux))

					x_aux, r_aux, _, _ = self.env.step(u_aux)
					data['xn'] = np.vstack((data['xn'], x_aux))
					data['r'] = np.hstack((data['r'], r_aux))

					u_aux = self.ctl.actions(self.get_pfeatures(x_aux), stoch)

					n = n + 1

		return data

	def evaluate(self, n_rollouts, n_steps):
		return self.sample(n_rollouts, n_steps, 0, False, False)

	def get_vfeatures(self, x):
		return fourier_features(x, self.n_vfeatures, self.vfeat_freq, self.vfeat_shift)

	def get_pfeatures(self, x):
		return fourier_features(x, self.n_pfeatures, self.pfeat_freq, self.pfeat_shift)

	def kl_divergence(self):
		adv = self.data['r'] + self.discount * np.dot(self.nvfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(1.0 / self.eta * delta, EXP_MIN, EXP_MAX))
		w = w[w >= 1e-45]
		w = w / np.mean(w, axis=0, keepdims=True)
		return np.mean(w * np.log(w), axis=0, keepdims=True)

	def ml_policy(self):
		adv = self.data['r'] + self.discount * np.dot(self.nvfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))

		psi = self.get_pfeatures(self.data['x'])

		from sklearn.linear_model import Ridge
		clf = Ridge(alpha=0.0001, fit_intercept=False, solver='sparse_cg', max_iter=2500, tol=1e-4)
		clf.fit(psi, self.data['u'], sample_weight=w)
		self.ctl.K = clf.coef_

		# wm = np.diagflat(w.flatten())
		# aux = np.linalg.inv((psi.T @ wm @ psi) + 1e-16 * np.eye(self.n_pfeatures)) @ psi.T @ wm @ self.data['u']
		# self.ctl.K = aux.T

		Z = (np.square(np.sum(w, axis=0, keepdims=True)) - np.sum(np.square(w), axis=0, keepdims=True)) / np.sum(w, axis=0, keepdims=True)
		tmp = self.data['u'] - psi @ self.ctl.K.T
		self.ctl.cov = np.einsum('t,tk,th->kh', w, tmp, tmp) / (Z + 1e-24)

	def bayes_policy(self):
		adv = self.data['r'] + self.discount * np.dot(self.nvfeatures, self.omega) - np.dot(self.vfeatures, self.omega) \
		      + (1.0 - self.discount) * np.mean(np.dot(self.ivfeatures, self.omega), axis=0, keepdims=True)
		delta = adv - np.max(adv)
		w = np.exp(np.clip(delta / self.eta, EXP_MIN, EXP_MAX))

		psi = self.get_pfeatures(self.data['x'])
		D = np.diagflat(1. / w.flatten())
		invD = np.linalg.inv(D)

		self.ctl.Sn = np.linalg.inv(self.ctl.beta * psi.T @ invD @ psi +  self.ctl.alpha * np.eye(self.n_pfeatures))
		self.ctl.K = (psi.T @ invD @ self.data['u']).T

	def show_value(self, fig, ax, res=50):
		p = np.linspace(-np.pi, np.pi, res)
		v = np.linspace(-20.0, 20.0, res)

		pp, vv = np.meshgrid(p, v, sparse=False, indexing='ij')

		cc = np.empty((res, res, self.n_states))
		ff = np.empty((res, res))

		# plot value function
		for i in range(res):
			for j in range(res):
				state = np.hstack((pp[i,j], vv[i, j]))
				cc[i, j, :] = self.env.pol2cart(state)
				ff[i, j] = self.get_vfeatures(cc[i, j, :]) @ self.omega

		ax.imshow(ff, extent=[-np.pi, np.pi, -20.0, 20.0], interpolation='bicubic', aspect='auto')
		fig.canvas.draw()
		plt.pause(0.5)
		plt.show()


reps = REPS(n_states=3, n_actions=1,
			n_rollouts=30, n_steps=100,
			n_iter=10, n_keep=5,
			kl_bound=0.1, discount=0.98,
			n_vfeat=75, n_pfeat=75)

# value function plot
# plt.ion()
# fig = plt.figure(13)
# ax = fig.add_subplot(111)

for it in range(reps.n_iter):
	eval = reps.evaluate(reps.n_sim_rollouts, reps.n_sim_steps)

	reps.data = reps.sample(reps.n_rollouts, reps.n_steps, reps.n_keep)

	reps.ivfeatures = reps.get_vfeatures(reps.data['xi'])
	reps.vfeatures = reps.get_vfeatures(reps.data['x'])
	reps.nvfeatures = reps.get_vfeatures(reps.data['xn'])

	for _ in range(100):
		res = sc.optimize.minimize(dual_eta, reps.eta, method='L-BFGS-B', jac=grad_eta,
									args=(reps.omega, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r']), bounds=((1e-8, 1e8),))
		# print(res)

		# check = sc.optimize.check_grad(dual_eta, grad_eta, res.x, reps.omega, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r'])
		# print('Eta Error', check)

		reps.eta = res.x

		# res = sc.optimize.minimize(dual_omega, reps.omega, method='BFGS', jac=grad_omega,
		# 							args=(reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r']))

		res = sc.optimize.minimize(dual_omega, reps.omega, method='trust-exact', jac=grad_omega, hess=hess_omega,
									args=(reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r']))

		# print(res)

		# check = sc.optimize.check_grad(dual_omega, grad_omega, res.x, reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r'])
		# print('Omega Error', check)

		reps.omega = res.x

	kl_div = reps.kl_divergence()

	reps.ml_policy()
	# reps.bayes_policy()

	print('Iteration:', it, 'Reward:', np.mean(eval['r']), 'KL:', kl_div, 'Cov:', *reps.ctl.cov)
	# print('Iteration:', it, 'Reward:', np.mean(eval['r']), 'KL:', kl_div)

	# plot value function
	# reps.show_value(fig, ax, 50)


eval = reps.evaluate(reps.n_sim_rollouts, reps.n_sim_steps)

State = eval['xn'].reshape((-1, reps.n_sim_steps, reps.n_states))
Angle = cart2pol(State)
Reward = eval['r'].reshape((-1, reps.n_sim_steps,))
Action = np.clip(eval['u'].reshape((-1, reps.n_sim_steps, reps.n_actions)), -reps.action_limit, reps.action_limit)

plt.figure()
sfig0 = plt.subplot(221)
plt.title('Cos, Sin')
sfig1 = plt.subplot(222)
plt.title('Angle')
sfig2 = plt.subplot(223)
plt.title('Reward')
sfig3 = plt.subplot(224)
plt.title('Action')

for rollout in range(reps.n_sim_rollouts):
	sfig0.plot(State[rollout, :, :-1])
	sfig1.plot(Angle[rollout, :, 0])
	sfig2.plot(Reward[rollout, :])
	sfig3.plot(Action[rollout, :])
plt.show()
