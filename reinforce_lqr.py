import numpy as np


# np.random.seed(99)

thetai = np.random.randn(2,) + 1.0
sigmai = 0.1

N = 100
L = 10
C = 250
gamma = 0.99

dt = 0.01

Q = 1e1
H = 1e-1


def dynamics(x, u):
	# x' = x + u
	return x + dt * (x + u)


def reward(x, u):
	return - np.square(x - 0.0) * Q - np.square(u) * H


def actions(theta, sigma, x):
	mean = np.column_stack((x, np.ones(x.shape[0])))
	return np.random.normal(mean @ theta, sigma)


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
	r = np.sum(R, axis=1, keepdims=False)
	print('Reward', np.mean(r, axis=0, keepdims=False))

	dlog = np.zeros((N, 2))
	for n in range(0, N):
		for l in range(0, L):
			psi = np.hstack([X[n, l], 1.0])
			dlog[n, :] = dlog[n, :] + (U[n, l] - psi @ thetai) * psi / np.square(sigmai)

	b = np.sum(np.square(dlog) * r[:, None], axis=0, keepdims=False) / np.sum(np.square(dlog), axis=0, keepdims=False)
	dj = np.mean(dlog * (r[:, None] - b), axis=0, keepdims=False)
	thetai = thetai + 0.01 * dj / L

print(thetai)
