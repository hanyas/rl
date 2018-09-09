import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np

import scipy as sc
from scipy import optimize
from scipy import special

from sklearn.preprocessing import PolynomialFeatures

import gym
import lab

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

# np.random.seed(1337)

EXP_MAX = 700.0
EXP_MIN = -700.0


class FourierFeatures:

	def __init__(self, n_states, n_feat, band):
		self.n_states = n_states
		self.n_feat = n_feat

		self.freq = np.random.multivariate_normal(mean=np.zeros(self.n_states), cov=np.diag(1.0 / band), size=self.n_feat)
		self.shift = np.random.uniform(-np.pi, np.pi, size=self.n_feat)

	def fit_transform(self, x):
		phi = np.sin(np.einsum('nk,...k->...n', self.freq, x) + self.shift)
		return phi


class Policy:

	def __init__(self, n_states, n_actions, **kwargs):
		self.n_states = n_states
		self.n_actions = n_actions

		if 'band' in kwargs:
			self.band = kwargs.get('band', False)
			self.n_feat = kwargs.get('n_feat', False)
			self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)
		else:
			self.degree = kwargs.get('degree', False)
			self.n_feat = int(sc.special.comb(self.degree + self.n_states, self.degree))
			self.basis = PolynomialFeatures(self.degree)

		self.K = 0.0 * np.random.randn(self.n_actions, self.n_feat)
		self.cov = np.eye(n_actions)

	def features(self, x):
		return self.basis.fit_transform(x.reshape(-1, self.n_states)).squeeze()

	def actions(self, x, stoch):
		feat = self.features(x)
		mean = np.dot(self.K, feat)
		if stoch:
			return np.random.normal(mean, np.sqrt(self.cov)).flatten()
		else:
			return mean


class Vfunction:

	def __init__(self, n_states, **kwargs):
		self.n_states = n_states

		if 'band' in kwargs:
			self.band = kwargs.get('band', False)
			self.n_feat = kwargs.get('n_feat', False)
			self.basis = FourierFeatures(self.n_states, self.n_feat, self.band)
		else:
			self.degree = kwargs.get('degree', False)
			self.n_feat = int(sc.special.comb(self.degree + self.n_states, self.degree))
			self.basis = PolynomialFeatures(self.degree)

	def features(self, x):
		return self.basis.fit_transform(x)


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


class REPS:

	def __init__(self, n_states, n_actions,
	             n_samples, n_iter, n_keep,
	             kl_bound, discount,
	             n_vfeat, n_pfeat,
	             n_rollouts, n_steps):

		self.n_states = n_states
		self.n_actions = n_actions
		self.n_samples = n_samples
		self.n_iter = n_iter
		self.n_keep = n_keep
		self.kl_bound = kl_bound
		self.discount = discount

		self.n_rollouts = n_rollouts
		self.n_steps = n_steps

		self.n_vfeat = n_vfeat
		self.n_pfeat = n_pfeat

		self.env = gym.make('Pendulum-v0')
		self.render = False

		self.action_limit = self.env.action_space.high

		self.ctl = Policy(self.n_states, self.n_actions, n_feat=self.n_pfeat, band=np.array([2.5, 2.5, 10.0, 12.0]))
		self.ctl.cov = (2.0 * self.action_limit)**2

		self.vfunc = Vfunction(self.n_states, n_feat=self.n_vfeat, band=np.array([2.5, 2.5, 10.0, 12.0]))

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

		self.omega = 0.0 * np.random.randn(self.n_vfeat)
		self.eta = np.array([1.0])

	def sample(self, n_samples, n_keep, reset=True, stoch=True):
		if n_keep==0:
			data = { 'xi': np.empty((0, self.n_states)),
		             'x': np.empty((0, self.n_states)),
		             'u': np.empty((0, self.n_actions)),
		             'xn': np.empty((0, self.n_states)),
		             'r': np.empty((0,))}
		else:
			data = { 'xi': self.data['xi'],
		             'x': self.data['x'][-n_keep:, :],
		             'u': self.data['u'][-n_keep:, :],
		             'xn': self.data['xn'][-n_keep:, :],
		             'r': self.data['r'][-n_keep:]}

		n = 0
		while True:
			x_aux = self.env.reset()
			data['xi'] = np.vstack((data['xi'], x_aux))

			u_aux = self.ctl.actions(x_aux, stoch)

			while True:
				coin = np.random.binomial(1, 1.0 - self.discount)
				if reset and coin:
					break
				else:
					data['x'] = np.vstack((data['x'], x_aux))
					data['u'] = np.vstack((data['u'], u_aux))

					x_aux, r_aux, done, _ = self.env.step(np.clip(u_aux, - self.action_limit, self.action_limit))
					if self.render:
						self.env.render()

					data['xn'] = np.vstack((data['xn'], x_aux))
					data['r'] = np.hstack((data['r'], r_aux))

					n = n + 1
					if n >= n_samples:
						return data

					if done:
						break
					else:
						u_aux = self.ctl.actions(x_aux, stoch)

	def evaluate(self, n_rollouts, n_steps):
		data = { 'xi': np.empty((0, self.n_states)),
	             'x': np.empty((0, self.n_states)),
	             'u': np.empty((0, self.n_actions)),
	             'xn': np.empty((0, self.n_states)),
	             'r': np.empty((0,))}

		for n in range(n_rollouts):
			x_aux = self.env.reset()
			data['xi'] = np.vstack((data['xi'], x_aux))

			u_aux = self.ctl.actions(x_aux, False)

			for t in range(n_steps):
				data['x'] = np.vstack((data['x'], x_aux))
				data['u'] = np.vstack((data['u'], u_aux))

				x_aux, r_aux, done, _ = self.env.step(np.clip(u_aux, - self.action_limit, self.action_limit))
				if self.render:
					self.env.render()

				data['xn'] = np.vstack((data['xn'], x_aux))
				data['r'] = np.hstack((data['r'], r_aux))

				if done:
					break
				else:
					u_aux = self.ctl.actions(x_aux, False)

		return data

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

		psi = self.ctl.features(self.data['x'])

		from sklearn.linear_model import Ridge
		clf = Ridge(alpha=0.0001, fit_intercept=False, solver='sparse_cg', max_iter=2500, tol=1e-4)
		clf.fit(psi, self.data['u'], sample_weight=w)
		self.ctl.K = clf.coef_

		Z = (np.square(np.sum(w, axis=0, keepdims=True)) - np.sum(np.square(w), axis=0, keepdims=True)) / np.sum(w, axis=0, keepdims=True)
		tmp = self.data['u'] - psi @ self.ctl.K.T
		self.ctl.cov = np.einsum('t,tk,th->kh', w, tmp, tmp) / (Z + 1e-24)


reps = REPS(n_states=4, n_actions=1,
			n_samples=3000, n_iter=10, n_keep=600,
			kl_bound=0.1, discount=0.99,
			n_vfeat=250, n_pfeat=250,
			n_rollouts=10, n_steps=1000)

for it in range(reps.n_iter):
	eval = reps.evaluate(reps.n_rollouts, reps.n_steps)

	reps.data = reps.sample(reps.n_samples, reps.n_keep)

	reps.ivfeatures = reps.vfunc.features(reps.data['xi'])
	reps.vfeatures = reps.vfunc.features(reps.data['x'])
	reps.nvfeatures = reps.vfunc.features(reps.data['xn'])

	for _ in range(250):
		res = sc.optimize.minimize(dual_eta, reps.eta, method='L-BFGS-B', jac=grad_eta,
									args=(reps.omega, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r']), bounds=((1e-8, 1e8),))
		# print(res)

		# check = sc.optimize.check_grad(dual_eta, grad_eta, res.x, reps.omega, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r'])
		# print('Eta Error', check)

		reps.eta = res.x

		res = sc.optimize.minimize(dual_omega, reps.omega, method='BFGS', jac=grad_omega,
									args=(reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r']))

		# res = sc.optimize.minimize(dual_omega, reps.omega, method='trust-exact', jac=grad_omega, hess=hess_omega,
		# 							args=(reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r']))

		# print(res)

		# check = sc.optimize.check_grad(dual_omega, grad_omega, res.x, reps.eta, reps.kl_bound, reps.discount, reps.nvfeatures, reps.vfeatures, reps.ivfeatures, reps.data['r'])
		# print('Omega Error', check)

		reps.omega = res.x

	kl_div = reps.kl_divergence()

	reps.ml_policy()

	print('Iteration:', it, 'Reward:', np.sum(eval['r']) / reps.n_rollouts, 'KL:', kl_div, 'Cov:', *reps.ctl.cov)
