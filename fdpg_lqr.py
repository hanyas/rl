import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


np.random.seed(1337)

thetai = np.random.randn(2,) + 1.0

N = 100
L = 100
C = 100

gamma = 0.999

dt = 0.01

Q = 1e2
H = 1e-3

poly = PolynomialFeatures(1)

def dynamics(x, u):
	# x' = x + u
	return x + dt * (x + u)


def reward(x, u):
	return - np.square(x) * Q - np.square(u) * H


def actions(theta, x):
	feat = poly.fit_transform(x.reshape(-1, 1))
	return feat @ theta


def simulate(theta, xi, N, L):
	X = np.zeros((N, L))
	U = np.zeros((N, L))
	R = np.zeros((N, L))

	ui = actions(theta, xi)

	X[:, 0] = xi
	U[:, 0] = ui
	R[:, 0] = gamma * reward(X[:, 0], U[:, 0])

	for l in range(1, L):
		X[:, l] = dynamics(X[:, l - 1], U[:, l - 1])
		U[:, l] = actions(theta, X[:, l])
		R[:, l] = np.power(gamma, l + 1) * reward(X[:, l], U[:, l])

	return X, U, R


for _ in range(0, C):
	xi = np.random.uniform(-1.0, 1.0, (N,))

	X, U, R = simulate(thetai, xi, N, L)
	r = np.mean(R, axis=1, keepdims=False)
	print('Reward', np.mean(r, axis=0, keepdims=True))

	thetan = thetai + 1.0 * np.random.randn(N, 2)

	Rn = np.zeros((N, L))
	for i in range(N):
		_, _, Rn[i, :] = simulate(thetan[i, :], xi[i], 1, L)

	rn = np.mean(Rn, axis=1, keepdims=False)

	dtheta = thetan - thetai
	dr = rn - r
	dj = np.linalg.inv(dtheta.T @ dtheta + 1e-8 * np.eye(2)) @ dtheta.T @ dr

	thetai = thetai + 0.05 * dj / N

print(thetai)
plt.plot(X[::5, :].T)
plt.show()
