import numpy as np
from numpy import matlib


# np.random.seed(99)

thetai = 10.0 * np.random.rand(2, 1)

N = 100
L = 10
C = 250

dt = 0.01

Q = 1e1
R = 1e-1


def dynamics(x, u):
	# x' = x + u
	return x + dt * (x + u)


def reward(x, u):
	return - np.square(x - 0.0) * Q - np.square(u) * R


def actions(theta, x):
	return np.column_stack([x, np.ones(x.shape)]) @ theta


def simulate(theta, xi, N, L):
	X = np.zeros((N, L))
	U = np.zeros((N, L))

	ui = actions(theta, xi)

	X[:, [0]] = xi
	U[:, [0]] = ui

	for l in range(1, L):
		X[:, [l]] = dynamics(X[:, [l - 1]], U[:, [l - 1]])
		U[:, [l]] = actions(theta, X[:, [l]])

	R = reward(X, U)
	r = np.mean(R, axis=1, keepdims=True)

	return r

for _ in range(0, C):
	xi = np.random.uniform(-1.0, 1.0, (N, 1))
	r = simulate(thetai, xi, N, L)
	print('Reward', np.mean(r, axis=0, keepdims=True))

	thetan = thetai + 0.1 * np.random.randn(2, N)

	rn = np.zeros((N, 1))
	for i in range(0, N):
		rn[[i], :] = simulate(thetan[:, [i]], xi[[i], :], 1, L)

	dtheta = thetan - thetai
	dr = rn - r
	dj = np.linalg.inv(dtheta @ dtheta.T) @ dtheta @ dr

	thetai = thetai + 0.1 * dj

print(thetai)
