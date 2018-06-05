import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(1337)

thetai = np.random.randn(2,) + 1.0
sigmai = 0.1

N = 500
L = 100
C = 250
gamma = 0.999

dt = 0.01

Q = 1e2
H = 1e-3

poly = PolynomialFeatures(1)

def dynamics(x, u):
	# x' = x + u
	return x + dt * (x + u)


def reward(x, u):
	return - np.square(x - 0.0) * Q - np.square(u) * H


def actions(theta, sigma, x):
	feat = poly.fit_transform(x.reshape(-1, 1))
	return np.random.normal(feat @ theta, sigma)


def simulate(theta, sigma, xi, N, L, gamma):
	X = np.zeros((N, L))
	U = np.zeros((N, L))
	R = np.zeros((N, L))

	ui = actions(theta, sigma, xi)

	X[:, 0] = xi
	U[:, 0] = ui
	R[:, 0] = gamma * reward(X[:, 0], U[:, 0])

	for l in range(1, L):
		X[:, l] = dynamics(X[:, l - 1], U[:, l - 1])
		U[:, l] = actions(theta, sigma, X[:, l])
		R[:, l] = np.power(gamma, l + 1) * reward(X[:, l], U[:, l])

	return X, U, R


for _ in range(0, C):
	xi = np.random.uniform(-1.0, 1.0, (N,))
	X, U, R = simulate(thetai, sigmai, xi, N, L, gamma)
	print('Reward', np.mean(R.sum(axis=1)))
	R = R[:, :, None]

	feat = poly.fit_transform(X.reshape(N * L, 1))
	feat = np.reshape(feat, (N, L, 2))

	dlog = np.einsum('nl,nlk->nlk', U - np.einsum('nlk,k->nl', feat, thetai), feat) / np.square(sigmai)
	dlog = np.cumsum(dlog, axis=1)

	b = np.sum(np.square(dlog) * R, axis=0, keepdims=False) / np.sum(np.square(dlog), axis=(0, 1), keepdims=False)
	b = np.broadcast_to(b, (N, L, 2))

	dj = np.sum(dlog * (R - b), axis=(0, 1), keepdims=False)
	thetai = thetai + 0.001 * dj / (N * L)

print(thetai)

plt.plot(X[::10, :].T)
plt.show()
