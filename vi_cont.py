import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1337)

# state transitions [A, B]
M = np.array([1.01, 0.01])

# reward matrices
Q = np.array([1e1, ])
R = np.array([1e-3, ])

xmax = 1.0
umax = 1.0

Nx = int(2 * xmax * 100)
Nu = int(2 * umax * 100)

gamma = 0.98
iter = 5000


def next_state(x, u, M, X, xmax):
	xn = M[0] * x + M[1] * u
	xn = np.clip(xn, - xmax, xmax)
	idx = np.digitize(xn, X, right=True)
	xn = X[idx]
	return xn


def reward(x, u, Q, R):
	r = - np.square(x - 0.0) * Q - np.square(u) * R
	return r


def maximize_action(Q, V, R, Vmask, Qmask, gamma):
	Q[Qmask] = V[Vmask]

	Q = R + gamma * Q

	V = np.amin(Q, axis=1)
	ctl = np.argmin(Q, axis=1)

	return Q, V, ctl


def value_iteration(Q, V, R, Vmask, Qmask, gamma, iter):
	ctl = np.zeros(V.shape, np.int64)

	for k in range(0, iter):
		Q, V, ctl = maximize_action(Q, V, R, Vmask, Qmask, gamma)

	return Q, V, ctl


# state action bins
X = np.linspace(-xmax, xmax, Nx)
U = np.linspace(-umax, umax, Nu)

# state action grid
xg, ug = np.meshgrid(X, U, indexing="ij")

# reshape grid
xr = np.reshape(xg, (Nx * Nu, 1), order='C')
ur = np.reshape(ug, (Nx * Nu, 1), order='C')

# reward of all grid points
R = reward(xr, ur, Q, R)
R = np.reshape(R, (Nx, Nu), order='C')

# next state of all grid points
xn = next_state(xr, ur, M, X, xmax)

# index of state-aciton pairs
xi = np.digitize(xr, X, right=True)
ui = np.digitize(ur, U, right=True)
xni = np.digitize(xn, X, right=True)

Vmask = xni
Qmask = [xi, ui]

V = np.zeros((Nx, ))
Q = np.zeros((Nx, Nu))

Q, V, ctl = value_iteration(Q, V, R, Vmask, Qmask, gamma, iter)

plt.imshow(Q, extent=[-xmax, xmax, -umax, umax])
plt.show()
