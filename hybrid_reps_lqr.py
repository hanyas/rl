import numpy as np
import scipy as sc
import scipy.stats
from scipy.optimize import minimize
from scipy.optimize import check_grad
import matplotlib.pyplot as plt


np.random.seed(99)

class Pi:
	def __init__(self):
		self.theta = np.random.uniform(size=4)
		self.sigma = 10.0 * np.ones((2,))


class Mu:
	def __init__(self):
		self.m = np.zeros((1, 1))
		self.sigma = 1.0 * np.ones((1,))
		self.p = 0.5


class Decoder:
	def __init__(self):
		self.m = np.array([0.0, 0.0])
		self.sigma = np.array([0.1, 0.1])


pi = Pi()
mu = Mu()
dc = Decoder()

# value function with [x^2 x 1]
omegai = np.random.randn(6)

etai = 100.0 * np.ones((1,))
epsilon = 0.1
gamma = 0.99

N = 10
L = 50
C = 25

Q = np.array([1e2, 2e2])
R = np.array([2e-1, 1e-1])

A = np.array([1.01, 1.05])
B = np.array([0.01, 0.03])
S = np.array([0.1, 0.1])


def decoder(z, x):
	p = sc.stats.logistic.cdf(x, dc.m[z], dc.sigma[z])
	return 1 - p, p


def dynamics(z, x, u):
	_, p = decoder(z, x)
	zn = np.random.binomial(1, p)
	xn = np.zeros(x.shape)

	i = np.where(zn == 0)[0]
	xn[i] = np.random.normal(A[0] * x[i] + B[0] * u[i], S[0], size=x[i].shape)
	i = np.where(zn == 1)[0]
	xn[i] = np.random.normal(A[1] * x[i] + B[1] * u[i], S[1], size=x[i].shape)

	return zn, xn


def reward(z, x, u):
	r = np.zeros(x.shape)

	i = np.where(z == 0)[0]
	r[i] = - np.square(x[i] - 0.0) * Q[0] - np.square(u[i]) * R[0]
	i = np.where(z == 1)[0]
	r[i] = - np.square(x[i] - 0.0) * Q[1] - np.square(u[i]) * R[1]

	return r


def vfeatures(z, x):
	phi = np.zeros((x.shape[0], 6))

	i = np.where(z == 0)[0]
	phi[i] = np.column_stack([np.square(x[i]), x[i], np.ones(x[i].shape), np.zeros((x[i].shape[0], 3))])
	i = np.where(z == 1)[0]
	phi[i] = np.column_stack([np.zeros((x[i].shape[0], 3)), np.square(x[i]), x[i], np.ones(x[i].shape)])

	return phi


def vfunction(z, x, omega):
	phi = vfeatures(z, x)
	return phi @ omega


def qfeatures(zn, z, xn, x, u):
	phi = np.zeros((x.shape[0], 6))

	i = np.where(zn == 0)[0]
	_, p = decoder(z[i], x[i])
	Q = np.square(x[i] * A[0] + u[i] * B[0]) + np.square(S[0])
	q = x[i] * A[0] + u[i] * B[0]
	q0 = np.ones(x[i].shape)
	stack = np.column_stack([Q, q, q0])
	phi[i] = np.column_stack([(1.0 - p)[:, None] * stack, p[:, None] * stack])

	i = np.where(zn == 1)[0]
	_, p = decoder(z[i], x[i])
	Q = np.square(x[i] * A[1] + u[i] * B[1]) + np.square(S[1])
	q = x[i] * A[1] + u[i] * B[1]
	q0 = np.ones(x[i].shape)
	stack = np.column_stack([Q, q, q0])
	phi[i] = np.column_stack([(1.0 - p)[:, None] * stack, p[:, None] * stack])

	# phi = np.zeros((xn.shape[0], 6))
	#
	# i = np.where(zn == 0)[0]
	# phi[i] = np.column_stack([np.square(xn[i]), xn[i], np.ones(xn[i].shape), np.zeros((xn[i].shape[0], 3))])
	# i = np.where(zn == 1)[0]
	# phi[i] = np.column_stack([np.zeros((xn[i].shape[0], 3)), np.square(xn[i]), xn[i], np.ones(xn[i].shape)])

	return phi


def qfunction(zn, z, xn, x, u, omega):
	phi = qfeatures(zn, z, xn, x, u)
	return phi @ omega


def actions(pi, z, x):
	u = np.zeros(x.shape)

	i = np.where(z == 0)[0]
	psi = np.column_stack([x[i], np.ones(x[i].shape)])
	u[i] = np.random.normal(psi @ pi.theta[0:2], pi.sigma[0], size=x[i].shape)
	i = np.where(z == 1)[0]
	psi = np.column_stack([x[i], np.ones(x[i].shape)])
	u[i] = np.random.normal(psi @ pi.theta[2:4], pi.sigma[1], size=x[i].shape)

	return u


def states(mu, N):
	# x = np.random.normal(mu.m, mu.sigma, (N,))
	x = np.random.uniform(-1.0, 1.0, (N,))
	z = np.random.binomial(1, mu.p, (N,))
	return z, x


def sample(pi, mu, N, L, reset=True):
	Z = np.zeros((N, L), np.int64)
	X = np.zeros((N, L))
	U = np.zeros((N, L))

	zi, xi = states(mu, N)
	ui = actions(pi, zi, xi)

	Z[:, 0] = zi
	X[:, 0] = xi
	U[:, 0] = ui

	if reset:
		for l in range(1, L):
			p = np.random.binomial(1, 1.0 - gamma, size=1)
			if p == 1:
				Z[:, l], X[:, l] = states(mu, N)
			else:
				Z[:, l], X[:, l] = dynamics(Z[:, l - 1], X[:, l - 1], U[:, l - 1])

			U[:, l] = actions(pi, Z[:, l], X[:, l])
	else:
		for l in range(1, L):
			Z[:, l], X[:, l] = dynamics(Z[:, l - 1], X[:, l - 1], U[:, l - 1])
			U[:, l] = actions(pi, Z[:, l], X[:, l])

	z = Z[:, :-1].flatten(order='C')
	x = X[:, :-1].flatten(order='C')
	u = U[:, :-1].flatten(order='C')
	r = reward(z, x, u)

	zn = Z[:, 1:].flatten(order='C')
	xn = X[:, 1:].flatten(order='C')
	un = U[:, 1:].flatten(order='C')

	return zn, xn, un, z, x, u, zi, xi, ui, r


def kl_div(eta, omega, zn, z, xn, x, u, zi, xi, r):
	adv = r + gamma * qfunction(zn, z, xn, x, u, omega) - vfunction(z, x, omega) + (1.0 - gamma) * np.mean(vfunction(zi, xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	w = w[w >= 1e-60]
	w = w / np.mean(w, axis=0, keepdims=True)
	return np.mean(w * np.log(w), axis=0, keepdims=True)


def dual_eta(eta, omega, epsilon, zn, z, xn, x, u, zi, xi, r):
	adv = r + gamma * qfunction(zn, z, xn, x, u, omega) - vfunction(z, x, omega) + (1.0 - gamma) * np.mean(vfunction(zi, xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	g = eta * epsilon + np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_eta(eta, omega, epsilon, zn, z, xn, x, u, zi, xi, r):
	adv = r + gamma * qfunction(zn, z, xn, x, u, omega) - vfunction(z, x, omega) + (1.0 - gamma) * np.mean(vfunction(zi, xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	deta = epsilon + np.log(np.mean(w, axis=0, keepdims=False)) - np.sum(w * delta, axis=0, keepdims=True) / (eta * np.sum(w, axis=0, keepdims=True))
	dg = deta
	return dg


def dual_omega(omega, eta, zn, z, xn, x, u, zi, xi, r):
	adv = r + gamma * qfunction(zn, z, xn, x, u, omega) - vfunction(z, x, omega) + (1.0 - gamma) * np.mean(vfunction(zi, xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	g = np.max(adv) + eta * np.log(np.mean(w, axis=0, keepdims=True))
	return g


def grad_omega(omega, eta, zn, z, xn, x, u, zi, xi, r):
	adv = r + gamma * qfunction(zn, z, xn, x, u, omega) - vfunction(z, x, omega) + (1.0 - gamma) * np.mean(vfunction(zi, xi, omega), axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(delta / eta, -420.0, 420.0))
	domega = (1.0 - gamma) * np.mean(vfeatures(zi, xi), axis=0, keepdims=False) + np.sum(w[:, None] * (gamma * qfeatures(zn, z, xn,  x, u) - vfeatures(z, x)), axis=0, keepdims=False) / np.sum(w, axis=0, keepdims=True)
	dg = domega
	return dg


for _ in range(0, C):
	zn, xn, un, z, x, u, zi, xi, ui, r = sample(pi, mu, N, L)

	for _ in range(0, 50):
		res = sc.optimize.minimize(dual_eta, 100.0 * np.ones((1,)), method='L-BFGS-B', jac=grad_eta, args=(omegai, epsilon, zn, z, xn, x, u, zi, xi, r),
			bounds=((1e-8, 1e8),))
		# print(res)
		check = sc.optimize.check_grad(dual_eta, grad_eta, res.x, omegai, epsilon, zn, z, xn, x, u, zi, xi, r)
		# print('Eta Error', check)

		etai = res.x
		# print(etai)

		res = sc.optimize.minimize(dual_omega, omegai, method='BFGS', jac=grad_omega, args=(etai, zn, z, xn, x, u, zi, xi, r))
		# print(res)
		check = sc.optimize.check_grad(dual_omega, grad_omega, res.x, etai, zn, z, xn, x, u, zi, xi, r)
		# print('Omega Error', check)

		omegai = res.x
		# print(omegai)

	kl = kl_div(etai, omegai, zn, z, xn, x, u, zi, xi, r)

	# policy max-likelihood update
	adv = r + gamma * qfunction(zn, z, xn, x, u, omegai) - vfunction(z, x, omegai) + (1.0 - gamma) * np.mean(vfunction(zi, xi, omegai),
		axis=0, keepdims=True)
	delta = adv - np.max(adv)
	w = np.exp(np.clip(1.0 / etai * delta, -420.0, 420.0))

	i = np.where(z == 0)[0]
	psi = np.column_stack([x[i], np.ones(x[i].shape)])

	wm = np.diagflat(w[i].flatten())
	pi.theta[0:2] = np.linalg.inv(psi.T @ wm @ psi) @ psi.T @ wm @ u[i]

	Z = (np.square(np.sum(w[i], axis=0, keepdims=True)) - np.sum(np.square(w[i]), axis=0, keepdims=True)) / np.sum(w[i], axis=0, keepdims=True)
	pi.sigma[0] = np.sqrt(np.sum(w[i] * np.square(u[i] - psi @ pi.theta[0:2]) / Z, axis=0, keepdims=True))

	i = np.where(z == 1)[0]
	psi = np.column_stack([x[i], np.ones(x[i].shape)])

	wm = np.diagflat(w[i].flatten())
	pi.theta[2:4] = np.linalg.inv(psi.T @ wm @ psi) @ psi.T @ wm @ u[i]

	Z = (np.square(np.sum(w[i], axis=0, keepdims=True)) - np.sum(np.square(w[i]), axis=0, keepdims=True)) / np.sum(w[i], axis=0, keepdims=True)
	pi.sigma[1] = np.sqrt(np.sum(w[i] * np.square(u[i] - psi @ pi.theta[2:4]) / Z, axis=0, keepdims=True))

	zn, xn, un, z, x, u, zi, xi, ui, r = sample(pi, mu, N, L * 2, False)
	print('Reward', np.mean(r, axis=0, keepdims=True), 'KL', kl, 'Sigma', pi.sigma)

print('Theta', pi.theta, 'Sigma', pi.sigma)
print('Omega', omegai)
print('Eta', etai)

for n in range(10):
	zn, xn, _, z, x, _, _, _, _, _ = sample(pi, mu, 1, L * 2, False)
	plt.plot(x)
plt.show()

# plt.figure()
# zn, xn, _, z, x, _, _, _, _, _ = sample(pi, mu, 10, L * 2, False)
# i = np.where(z == 0)[0]
# plt.scatter(x[i], vfunction(z[i], x[i], omegai), color='r')
# i = np.where(z == 1)[0]
# plt.scatter(x[i], vfunction(z[i], x[i], omegai), color='b')
# plt.show()

# # check gradient fidelity
# zn, xn, un, z, x, u, zi, xi, ui, r = sample(pi, mu, N, L)
# check = sc.optimize.check_grad(dual_eta, grad_eta, etai, omegai, epsilon, zn, z, x, u, zi, xi, r)
# print('Eta Error', check)
#
# check = sc.optimize.check_grad(dual_omega, grad_omega, omegai, etai, zn, z, x, u, zi, xi, r)
# print('Omega Error', check)

# # plot dual function w.r.t eta
# zn, xn, un, z, x, u, zi, xi, ui, r = sample(pi, mu, N, L)
# t = np.linspace(0.0001, 10000, 1000)
# Z = np.zeros((t.shape[0], 1))
#
# for i in range(0, t.shape[0]):
# 	Z[i, :] = dual_eta(t[i], omegai, epsilon, zn, z, x, u, zi, xi, r)
#
# plt.plot(t, Z)
# plt.show()

# # plot dual function w.r.t omega
# zn, xn, un, z, x, u, zi, xi, ui, r = sample(pi, mu, N, L)
# t = np.linspace(-50.0, 50.0, 100)
# Z = np.zeros((t.shape[0], 1))
#
# for i in range(0, t.shape[0]):
# 	omegai[0] = t[i]
# 	Z[i, :] = dual_omega(omegai, etai, z, x, u, zi, xi, r)
#
# plt.plot(t, Z)
# plt.show()
#
#
# # plot omega gradient
# zn, xn, un, z, x, u, zi, xi, ui, r = sample(pi, mu, N, L)
# t = np.linspace(-50.0, 50.0, 100)
# Z = np.zeros((t.shape[0], 6))
#
# for i in range(0, t.shape[0]):
# 	omegai[0] = t[i]
# 	Z[i, :] = grad_omega(omegai, etai, z, x, u, zi, xi, r)
#
# plt.plot(t, Z[:, 0])
# plt.show()
