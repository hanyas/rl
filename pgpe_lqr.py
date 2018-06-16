import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(1337)

N = 500
L = 100
C = 100
gamma = 0.999

dt = 0.01

Q = 1e2
H = 1e-3


def dynamics(x, u):
	# x' = x + u
	return x + dt * (x + u)


def reward(x, u):
	return - np.square(x) * Q - np.square(u) * H


def actions(theta, x):
	return x * theta


def simulate(mui, sigmai, xi, N, L, gamma):
	X = np.zeros((N, L))
	U = np.zeros((N, L))
	R = np.zeros((N, L))
	Theta  = np.zeros((N,))

	for n in range(N):
		Theta[n] = np.random.randn() * sigmai + mui

		ui = actions(Theta[n], xi[n])

		X[n, 0] = xi[n]
		U[n, 0] = ui
		R[n, 0] = gamma * reward(X[n, 0], U[n, 0])

		for l in range(1, L):
			X[n, l] = dynamics(X[n, l - 1], U[n, l - 1])
			U[n, l] = actions(Theta[n], X[n, l])
			R[n, l] = np.power(gamma, l + 1) * reward(X[n, l], U[n, l])

	return X, U, R, Theta


mui = 0.0
sigmai = 2.0

for _ in range(C):
	xi = np.random.uniform(-1.0, 1.0, (N,))

	X, U, R, Theta = simulate(mui, sigmai, xi, N, L, gamma)
	print('Reward', np.mean(R.sum(axis=1)))

	T = Theta - mui
	S = (np.square(T) - sigmai**2) / sigmai

	b = np.mean(R, axis=(0, 1))
	r = np.mean(R, axis=1) - b

	mui = mui + 1e-4 * T @ r / N
	sigmai = sigmai + 1e-5 * S @ r / N


plt.plot(X[::10, :].T)
plt.show()
