import numpy as np
import scipy as sc
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns

from sklearn.linear_model import Ridge

np.set_printoptions(precision=5)

sns.set_style("white")
sns.set_context("paper")

color_names = ["red",
               "windows blue",
               "medium green",
               "dusty purple",
               "orange",
               "amber",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "mint",
               "salmon",
               "dark brown"]

colors = []
for k in color_names:
	colors.append(mcd.XKCD_COLORS['xkcd:'+k].upper())

# np.random.seed(99)

dt = 0.1

g = 9.81
l = 0.5
m = 2.0
M = 8.0

d_state = 2
d_action = 1

n_actions = 11
actions = np.linspace(-50.0, 50.0, n_actions)

n_feat = 75

freq = np.random.randn(n_feat, d_state + d_action)
shift = np.random.uniform(-np.pi, np.pi, size=n_feat)
band = 10.0

weights = np.random.randn(n_feat)


def fourier(x, a):
	xa = np.hstack((x, a))
	phi = np.sin(np.einsum('mk,nk->nm', freq, xa) / band + np.tile(shift, (xa.shape[0], 1)))
	return phi


def policy(x, weights, epsilon):
	n_samples = x.shape[0]
	act = np.empty((n_samples, d_action))

	for i in range(n_samples):
		if epsilon >= np.random.rand():
			act[i, :] = np.random.choice(actions, size=(d_action))
		else:
			act[i, :] = controller(x[i, :], weights)
	return act


def controller(x, weights):
	aa = np.reshape(actions, (actions.shape[0], d_action))
	xx = np.tile(x, (n_actions, 1))

	phi = fourier(xx, aa)
	q = np.dot(phi, weights)

	act = actions[np.argmax(q)]
	return act


def dynamics(x, t, u, g, m, M, l):
	a = 1.0 / (m + M)
	return [x[1], (g * np.sin(x[0]) - 0.5 * a * m * l * x[1]**2 * np.sin(2 * x[0]) - a * np.cos(x[0]) * u) / (4.0 * l / 3.0 - a * m * l * np.cos(x[0])**2)]


def reward(x, u):
	if np.fabs(x[0]) < 0.5 * np.pi:
		r = 0.0 - 2e-4 * np.square(u)
	else:
		r = -1.0 - 2e-4 * np.square(u)
	return r


def step(x, u):
	un = u + np.random.uniform(-1.0, 1.0)
	xn = sc.integrate.odeint(dynamics, x, np.array([0.0, dt]), args=(un, g, m, M, l))[1, :]
	xn[1] = np.remainder(xn[1] + np.pi, 2.0 * np.pi) - np.pi
	return xn


def sample(N, T, weights, gamma, epsilon=1.0):
	x = np.empty((0, d_state))
	u = np.empty((0, d_action))
	r = np.empty((0, ))
	s = np.empty((N, ))

	for n in range(N):
		xi = np.hstack((np.random.uniform(-1.0, 1.0), 0.0))
		x = np.append(x, [xi], axis=0)

		for t in range(T):
			ut = policy(x[[-1], :], weights, epsilon)
			u = np.append(u, ut, axis=0)

			rt = np.power(gamma, t) * reward(x[-1, :], u[-1, :])
			r = np.append(r, rt, axis=0)

			xt = step(x[-1, :], u[-1, :])
			x = np.append(x, [xt], axis=0)

			if np.fabs(xt[0]) > 0.5 * np.pi:
				break

		ut = policy(x[[-1], :], weights, epsilon)
		u = np.append(u, ut, axis=0)

		rt = np.power(gamma, t + 1) * reward(x[-1, :], u[-1, :])
		r = np.append(r, rt, axis=0)

		s[n] = t

	xn = x[1:, :]
	r = r[1:, ]
	x = x[:-1, :]
	u = u[:-1, :]

	return x, u, xn, r, s


N = 1000
T = 100
gamma = 0.95

clf = Ridge(alpha=1e-12, fit_intercept=False)

x, u, xn, r, s = sample(N, T, weights, gamma, 1.0)

for i in range(25):
	phi = fourier(x, u)
	phin = fourier(xn, policy(xn, weights, 0.0))

	A = phi.T @ (phi - gamma * phin)
	b = phi.T @ r

	old_weights = weights.copy()

	clf.fit(A, b)
	weights = clf.coef_.copy()

	conv = np.mean(np.linalg.norm(weights - old_weights))
	print('Conv: ', conv)

	if conv < 1e-3:
		break

x, u, xn, r, s = sample(100, T, weights, gamma, 0.0)
print('Steps: ', np.mean(s))

plt.subplot(411)
plt.title('position')
plt.plot(x[:100, 0])
plt.subplot(412)
plt.title('velocity')
plt.plot(x[:100, 1])
plt.subplot(413)
plt.title('action')
plt.plot(u[:100, 0])
plt.subplot(414)
plt.title('reward')
plt.plot(r[:100])
plt.show()
