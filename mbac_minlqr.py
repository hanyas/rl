import numpy as np
import scipy as sc
from scipy.optimize import minimize
from scipy.optimize import check_grad
import matplotlib.pyplot as plt

np.random.seed(99)

# init distribtion
m = 0.0 * np.ones((1,))
sigma = 1.0 * np.ones((1,))

# policy parameters
thetai = np.random.uniform(1.0, 5.0)

# state action value function parameters (x^2)
omegai = np.random.randn(1,)

etai = 10.0 * np.ones((1,))
epsilon = 0.1
gamma = 0.98

N = 250
L = 50
C = 25

A = 1.01
B = 0.01

Q = 1e1
R = 1e0


def dynamics(x, u):
	return A * x + B * u


def reward(x, u):
	return - np.square(x - 0.0) * Q - np.square(u) * R


def dreward_du(x, u):
	return -2 * u * R * x


def features(x):
	return np.square(x)


def vfunction(x, omega):
	phi = features(x)
	return phi * omega


def dfeatures_ds(x):
	return 2 * x


def dvfunction_ds(x, omega):
	dphi = dfeatures_ds(x)
	return dphi * omega


def actions(theta, x):
	return x * theta


def states(m, sigma, N):
	return np.random.normal(m, sigma, (N,))
	# return np.random.uniform(-1.0, 1.0, (N,))


def dual_theta(theta, omega, eta, x, xi):
	xn, _, _, _, u, _ = simulate(x, theta, x.shape[0], 2)
	adv = reward(x, u) + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -420.0, 420.0))
	g = -1.0 * (np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True)))
	return g


def grad_theta(theta, omega, eta, x, xi):
	xn, _, _, _, u, _ = simulate(x, theta, x.shape[0], 2)
	adv = reward(x, u) + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -420.0, 420.0))
	# specific gradient for this problem
	dadv = dreward_du(x, u) + gamma * dvfunction_ds(xn, omega) * B * x
	dtheta = np.sum(w * dadv, axis=0, keepdims=True) / np.sum(w, axis=0, keepdims=True)
	dg = -1.0 * dtheta
	return dg


def dual_omega(omega, theta, eta, x, xi):
	xn, _, _, _, u, _ = simulate(x, theta, x.shape[0], 2)
	adv = reward(x, u) + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -420.0, 420.0))
	g = np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_omega(omega, theta, eta, x, xi):
	xn, _, _, _, u, _ = simulate(x, theta, x.shape[0], 2)
	adv = reward(x, u) + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -420.0, 420.0))
	domega = (1.0 - gamma) * np.mean(features(xi), axis=0, keepdims=False) + np.sum(w * (gamma * features(xn) - features(x)), axis=0, keepdims=False) / np.sum(w, axis=0, keepdims=True)
	dg = domega
	return dg


def dual_eta(eta, theta, omega, epsilon, x, xi):
	xn, _, _, _, u, _ = simulate(x, theta, x.shape[0], 2)
	adv = reward(x, u) + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -420.0, 420.0))
	g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_eta(eta, theta, omega, epsilon, x, xi):
	xn, _, _, _, u, _ = simulate(x, theta, x.shape[0], 2)
	adv = reward(x, u) + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -420.0, 420.0))
	deta = epsilon + np.log(np.mean(w, axis=0, keepdims=True)) - np.sum(w * delta, axis=0, keepdims=True) / (eta * np.sum(w, axis=0, keepdims=True))
	dg = deta
	return dg


def sample(m, sigma, N):
	xi = states(m, sigma, N)
	return xi


def simulate(xi, theta, N, L):
	X = np.zeros((N, L))
	U = np.zeros((N, L))

	ui = actions(theta, xi)

	X[:, 0] = xi
	U[:, 0] = ui

	for l in range(1, L):
		p = np.random.binomial(1, gamma, size=1)
		if p == 1:
			X[:, l] = dynamics(X[:, l - 1], U[:, l - 1])
		else:
			X[:, l] = sample(m, sigma, N)

		U[:, l] = actions(theta, X[:, l])

	x = X[:, :-1].flatten(order='C')
	u = U[:, :-1].flatten(order='C')

	xn = X[:, 1:].flatten(order='C')
	un = U[:, 1:].flatten(order='C')

	return xn, x, xi, un, u, ui


def kl_div(eta, omega, theta, x, xi):
	xn, _, _, _, u, _ = simulate(x, theta, x.shape[0], 2)
	adv = reward(x, u) + gamma * vfunction(xn, omega) - vfunction(x, omega) + (1.0 - gamma) * np.mean(vfunction(xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / eta * delta, -420.0, 420.0))
	w = w[w >= 1e-35]
	w = w / np.mean(w, axis=0, keepdims=True)
	return np.mean(w * np.log(w), axis=0, keepdims=True)


for _ in range(0, C):
	xi = sample(m, sigma, N)
	xn, x, xi, un, _, _ = simulate(xi, thetai, xi.shape[0], L)
	rn = reward(xn, un)
	print('Reward', np.mean(rn, axis=0, keepdims=True))

	res = sc.optimize.minimize(dual_theta, thetai, method='BFGS', jac=grad_theta, args=(omegai, etai, x, xi))
	# print(res)
	thetai = res.x
	# check = sc.optimize.check_grad(dual_theta, grad_theta, thetai, omegai, etai, x, xi)
	# print('Theta Error', check)

	# res = sc.optimize.minimize(dual_eta, 1.0, method='L-BFGS-B', jac=grad_eta, args=(thetai, omegai, epsilon, x, xi),
	# 							bounds=((1e-8, 1e8),), options={'disp': False, 'iprint': 1, 'maxiter': 500, 'ftol': 1e-6})
	# # print(res)
	# etai = res.x

	res = sc.optimize.minimize(dual_omega, omegai, method='BFGS', jac=grad_omega, args=(thetai, etai, x, xi))
	# print(res)
	omegai = res.x
	# check = sc.optimize.check_grad(dual_omega, grad_omega, omegai, thetai, etai, x, xi)
	# print('Omega Error', check)

print('Theta', thetai)
print('Omega', omegai)
print('Eta', etai)
