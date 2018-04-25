import numpy as np
import scipy as sc
from scipy.optimize import minimize
from scipy.optimize import check_grad
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

class Pi:
	def __init__(self):
		self.theta = np.random.uniform(size=2)
		self.sigma = 10.0 * np.ones((1,))


class Mu:
	def __init__(self):
		self.m = np.zeros((1,))
		self.sigma = 1.0 * np.ones((1,))


pi = Pi()
mu = Mu()

# V function parameters (x^2 x 1)
omegai = np.random.randn(3)

etai = 10.0 * np.ones((1,))
epsilon = 0.5
gamma = 0.99

N = 10
L = 50
C = 50

dt = 0.01

Q = 1e1
R = 1e-3

def dynamics(x, u):
	# x' = x + u
	return np.random.normal(x + dt * (x + u), 0.01, size=x.shape)


def reward(x, u):
	return - np.square(x - 0.0) * Q - np.square(u) * R


def vfeatures(x):
	return np.column_stack([np.square(x), x, np.ones(x.shape)])


def vfunction(x, omega):
	phi = vfeatures(x)
	return phi @ omega


def actions(pi, x):
	return np.random.normal(np.column_stack([x, np.ones(x.shape)]) @ pi.theta, pi.sigma, size=x.shape)


def states(mu, N):
	# return np.random.normal(mu.m, mu.sigma, (N,))
	return np.random.uniform(-1.0, 1.0, (N,))


def dual_eta(eta, omega, epsilon, xn, x, xi, r):
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_eta(eta, omega, epsilon, xn, x, xi, r):
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	deta = epsilon + np.log(np.mean(w, axis=0)) - np.sum(w * delta, axis=0, keepdims=True) / (eta * np.sum(w, axis=0, keepdims=True))
	return deta

def hess_eta(eta, omega, epsilon, xn, x, xi, r):
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	w = w / np.sum(w, axis=0, keepdims=True)

	ui = -delta
	uj = np.sum(w * (-delta), axis=0, keepdims=True)
	tmp = ui - uj

	heta = 1./eta * np.sum(w * tmp * tmp, axis=0, keepdims=True)
	return heta[:, None]

def dual_omega(omega, eta, xn, x, xi, r):
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	g = np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_omega(omega, eta, xn, x, xi, r):
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	domega = (1.0 - gamma) * np.mean(vfeatures(xi), axis=0, keepdims=False) + np.sum(w[:, None] * (gamma * vfeatures(xn) - vfeatures(x)), axis=0, keepdims=False) / np.sum(w, axis=0, keepdims=True)
	return domega


def hess_omega(omega, eta, xn, x, xi, r):
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	w = w / np.sum(w, axis=0, keepdims=True)

	ui = gamma * vfeatures(xn) - vfeatures(x)
	uj = np.sum(w[:, None] * ui, axis=0, keepdims=True)
	tmp = ui - uj

	homega = 1./eta * np.einsum('n,nk,nh->kh', w, tmp, tmp)

	return homega


def dual(var, epsilon, xn, x, xi, r):
	eta = var[0:1]
	omega = var[1:]
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad(var, epsilon, xn, x, xi, r):
	eta = var[0:1]
	omega = var[1:]
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	deta = epsilon + np.log(np.mean(w, axis=0, keepdims=True)) - np.sum(w * delta, axis=0, keepdims=True) / (eta * np.sum(w, axis=0, keepdims=True))
	domega = (1.0 - gamma) * np.mean(vfeatures(xi), axis=0, keepdims=False) + np.sum(w[:, None] * (gamma * vfeatures(xn) - vfeatures(x)), axis=0, keepdims=False) / np.sum(w, axis=0, keepdims=True)
	dg = np.concatenate([deta, domega])
	return dg


def sample(pi, mu, N, L, reset=True):
	X = np.zeros((N, L))
	U = np.zeros((N, L))

	xi = states(mu, N)
	ui = actions(pi, xi)

	X[:, 0] = xi
	U[:, 0] = ui

	if reset:
		for l in range(1, L):
			p = np.random.binomial(1, 1.0 - gamma, size=1)
			if p == 1:
				X[:, l] = states(mu, N)
			else:
				X[:, l] = dynamics(X[:, l - 1], U[:, l - 1])

			U[:, l] = actions(pi, X[:, l])
	else:
		for l in range(1, L):
			X[:, l] = dynamics(X[:, l - 1], U[:, l - 1])
			U[:, l] = actions(pi, X[:, l])

	x = X[:, :-1].flatten(order='C')
	u = U[:, :-1].flatten(order='C')

	r = reward(x, u)

	xn = X[:, 1:].flatten(order='C')
	un = U[:, 1:].flatten(order='C')

	return xn, x, xi, un, u, ui, r


def kl_div(eta, omega, xn, x, xi, r):
	adv = r + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -np.inf, np.inf))
	w = w[w >= 1e-45]
	w = w / np.mean(w, axis=0, keepdims=True)
	return np.mean(w * np.log(w), axis=0, keepdims=True)


for _ in range(0, C):
	xn, x, xi, un, u, ui, r = sample(pi, mu, N, L, True)

	# bnds = ((1e-8, None), (None, None), (None, None), (None, None))
	# var = np.concatenate([etai, omegai])
	# res = sc.optimize.minimize(dual, var, method='L-BFGS-B', jac=grad, args=(epsilon, xn, x, xi, r), bounds=bnds)
	# # print(res)
	# etai = res.x[0:1]
	# omegai = res.x[1:]
	# # print('Eta', etai)
	# # print('Omega', omegai)

	for _ in range(0, 25):
		res = sc.optimize.minimize(dual_eta, etai, method='L-BFGS-B', jac=grad_eta, args=(omegai, epsilon, xn, x, xi, r), bounds=((1e-8, 1e8),))
		# res = sc.optimize.minimize(dual_eta, etai, method='trust-exact', jac=grad_eta, hess=hess_eta, args=(omegai, epsilon, xn, x, xi, r), bounds=((1e-8, 1e8),))
		# print(res)
		etai = res.x
		# check = sc.optimize.check_grad(dual_eta, grad_eta, etai, omegai, epsilon, xn, x, xi, r)
		# print('Eta Error', check)

		# res = sc.optimize.minimize(dual_omega, omegai, method='BFGS', jac=grad_omega, args=(etai, xn, x, xi, r))
		res = sc.optimize.minimize(dual_omega, omegai, method='trust-exact', jac=grad_omega, hess=hess_omega, args=(etai, xn, x, xi, r))
		# print(res)
		omegai = res.x
		# check = sc.optimize.check_grad(dual_omega, grad_omega, omegai, etai, xn, x, xi, r)
		# print('Omega Error', check)

	# print('Eta', etai)
	# print('Omega', omegai)

	kl = kl_div(etai, omegai, xn, x, xi, r)

	# policy max-likelihood update
	adv = r + gamma * vfunction(xn, omegai) - vfunction(x, omegai) + (1.0 - gamma) * np.mean(vfunction(xi, omegai), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / etai * delta, -np.inf, np.inf))

	psi = np.column_stack([x, np.ones(x.shape)])

	wm = np.diagflat(w.flatten())
	pi.theta = np.linalg.inv(psi.T @ wm @ psi) @ psi.T @ wm @ u

	Z = (np.square(np.sum(w, axis=0, keepdims=True)) - np.sum(np.square(w), axis=0, keepdims=True)) / np.sum(w, axis=0, keepdims=True)
	pi.sigma = np.sqrt(np.sum(w * np.square(u - psi @ pi.theta) / Z, axis=0, keepdims=True))

	# evaluation
	xn, x, xi, un, u, ui, r = sample(pi, mu, N, L * 2, False)
	print('Reward', np.mean(r, axis=0, keepdims=True), 'KL', kl)

print('theta', pi.theta.T, 'sigma', pi.sigma)
print('omega', omegai.T)

for n in range(10):
	xn, _, _, _, _, _, _ = sample(pi, mu, 1, L * 2, False)
	plt.plot(xn)
plt.show()
