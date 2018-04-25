import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(99)

gamma = 0.95
iter = 100


def reward():
	O = -1e5  # Dangerous places to avoid
	D = 35  # Dirt
	W = -100  # Water
	C = -3000  # Cat
	T = 1000  # Toy
	reward = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
					   [0, 0, 0, 0, D, O, 0, 0, D, 0],
					   [0, D, 0, 0, 0, O, 0, 0, O, 0],
					   [O, O, O, O, 0, O, 0, O, O, O],
					   [D, 0, 0, D, 0, O, T, D, 0, 0],
					   [0, O, D, D, 0, O, W, 0, 0, 0],
					   [W, O, 0, O, 0, O, D, O, O, 0],
					   [W, 0, 0, O, D, 0, 0, O, D, 0],
					   [0, 0, 0, D, C, O, 0, 0, D, 0]])
	return reward


def show_grid(grid_world, tlt):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.set_title(tlt)
	ax.set_xticks(np.arange(0.5, 10.5, 1))
	ax.set_yticks(np.arange(0.5, 9.5, 1))
	ax.grid(color='b', linestyle='-', linewidth=1)
	ax.imshow(grid_world, interpolation='nearest', cmap='copper')
	return ax


def show_policy(policy, ax):
	for x in range(policy.shape[0]):
		for y in range(policy.shape[1]):
			if policy[x, y] == 0:
				ax.annotate('$\downarrow$', xy=(y, x), horizontalalignment='center')
			elif policy[x, y] == 1:
				ax.annotate(r'$\rightarrow$', xy=(y, x), horizontalalignment='center')
			elif policy[x, y] == 2:
				ax.annotate(r'$\uparrow$', xy=(y, x), horizontalalignment='center')
			elif policy[x, y] == 3:
				ax.annotate('$\leftarrow$', xy=(y, x), horizontalalignment='center')
			elif policy[x, y] == 4:
				ax.annotate('$\perp$', xy=(y, x), horizontalalignment='center')


def maximize_action(V, R, gamma):
	n = R.shape[0]
	m = R.shape[1]
	Q = np.zeros((n, m, 5))

	V = gamma * V

	Q[:, :, 0] = np.vstack((V[1:, :], V[-1:, :])) + R
	Q[:, :, 1] = np.hstack((V[:, 1:], V[:, -1:])) + R
	Q[:, :, 2] = np.vstack((V[:1, :], V[0:-1, :])) + R
	Q[:, :, 3] = np.hstack((V[:, :1], V[:, 0:-1])) + R
	Q[:, :, 4] = V + R

	V = np.amax(Q, axis=2)
	ctl = np.argmax(Q, axis=2)

	return V, ctl


def value_iteration(V, R, gamma, iter):
	ctl = np.zeros(V.shape, np.int64)

	for k in range(0, iter):
		V, ctl = maximize_action(V, R, gamma)

	return V, ctl


R = reward()
V = np.zeros(R.shape)

V, ctl = value_iteration(V, R, gamma, iter)

ax = show_grid(R, "")
show_policy(ctl, ax)
plt.show()
