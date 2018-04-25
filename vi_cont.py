import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(99)

# state transitions [A, B]
M = np.array([[1.1, 0.1], [1.01, 0.01]])

# reward matrices
Q = np.array([1e1, 2e1])
R = np.array([2e0, 1e0])

xmax = 1.0
umax = 1.0

Nz = 2
Nx = int(2 * xmax * 100)
Nu = int(2 * umax * 100)

gamma = 0.96
iter = 5000


def next_state(z, x, u, M, X, xmax):
	zn = np.zeros(z.shape, np.int64)

	i = np.where(z == 0)[0]
	j = np.where(x > 0.0)[0]
	zn[np.intersect1d(i, j)] = 1
	zn[np.setdiff1d(i, j)] = 0

	i = np.where(z == 1)[0]
	j = np.where(x <= 0.0)[0]
	zn[np.intersect1d(i, j)] = 0
	zn[np.setdiff1d(i, j)] = 1

	xn = M[zn, 0] * x + M[zn, 1] * u
	xn = np.clip(xn, - xmax, xmax)

	idx = np.digitize(xn[:, 0], X[:, 0], right=True)
	xn = X[idx]

	return zn, xn


def reward(z, x, u, Q, R):
	r = np.zeros(x.shape)

	i = np.where(z == 0)[0]
	r[i] = - np.square(x[i] - 0.0) * Q[0] - np.square(u[i]) * R[0]
	i = np.where(z == 1)[0]
	r[i] = - np.square(x[i] - 0.0) * Q[1] - np.square(u[i]) * R[1]

	return r


def maximize_action(Q, V, R, Vmask, Qmask, gamma):
	Q[Qmask] = V[Vmask]

	Q = R + gamma * Q

	V = np.amin(Q, axis=2)
	ctl = np.argmin(Q, axis=2)

	return Q, V, ctl


def value_iteration(Q, V, R, Vmask, Qmask, gamma, iter):
	ctl = np.zeros(V.shape, np.int64)

	for k in range(0, iter):
		Q, V, ctl = maximize_action(Q, V, R, Vmask, Qmask, gamma)

	return Q, V, ctl


# state action bins
X = np.linspace(-xmax, xmax, Nx)[:, None]
U = np.linspace(-umax, umax, Nu)[:, None]
Z = np.array([[0], [1]], np.int64)

# state action grid
zg, xg, ug = np.meshgrid(Z, X, U, indexing="ij")

# reshape grid
zr = np.reshape(zg, (Nz * Nx * Nu, 1), order='C')
xr = np.reshape(xg, (Nz * Nx * Nu, 1), order='C')
ur = np.reshape(ug, (Nz * Nx * Nu, 1), order='C')

# reward of all grid points
R = reward(zr, xr, ur, Q, R)
R = np.reshape(R, (Nz, Nx, Nu), order='C')

# next state of all grid points
zn, xn = next_state(zr, xr, ur, M, X, xmax)

# index of state-aciton pairs
zi = np.digitize(zr[:, 0], Z[:, 0], right=True)
xi = np.digitize(xr[:, 0], X[:, 0], right=True)
ui = np.digitize(ur[:, 0], U[:, 0], right=True)

zni = np.digitize(zn[:, 0], Z[:, 0], right=True)
xni = np.digitize(xn[:, 0], X[:, 0], right=True)

Vmask = [zni, xni]
Qmask = [zi, xi, ui]

Q = np.zeros((Nz, Nx, Nu))
V = np.zeros((Nz, Nx))

Q, V, ctl = value_iteration(Q, V, R, Vmask, Qmask, gamma, iter)

plt.contourf(xg[0, :, :], ug[0, :, :], Q[0, :, :], alpha=0.5, cmap="plasma")
plt.show()
plt.contourf(xg[1, :, :], ug[1, :, :], Q[1, :, :], alpha=0.5, cmap="plasma")
plt.show()
