import numpy as np
import scipy as sc
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns


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

np.random.seed(99)


dt = 0.1

g = 9.81
l = 0.5
m = 2.0
M = 8.0

T = 3000
N = 1000

gamma = 0.95

actions = np.array([-50.0, 0.0, 50.0])

def dynamics(x, t, u, g, m, M, l):
	a = 1.0 / (m + M)
	return [x[1], (g * np.sin(x[0]) - 0.5 * a * m * l * x[1]**2 * np.sin(2 * x[0]) - a * np.cos(x[0]) * u) / (4.0 * l / 3.0 - a * m * l * np.cos(x[0])**2)]


def reward(x):
	if np.abs(x[0]) < np.pi / 2.0:
		r = 0.0
	else:
		r = -1.0
	return r


def step(x, u):
	un = u + np.random.uniform(-10.0, 10.0)
	xn = sc.integrate.odeint(dynamics, x, np.array([0.0, dt]), args=(un, g, m, M, l))[1, :]
	xn[1] = np.remainder(xn[1] + np.pi, 2.0 * np.pi) - np.pi
	return xn


def sample(N, T):
	x = np.empty((0, 2))
	u = np.empty((0, 1))
	r = np.empty((0, ))

	for n in range(N):
		xi = np.hstack((np.random.uniform(-0.1, 0.1), 0.0))
		x = np.append(x, [xi], axis=0)

		for t in range(T):
			ut = np.random.choice(actions, size=(1, ))
			u = np.append(u, [ut], axis=0)

			rt = np.power(gamma, t) * reward(x[-1, :])
			r = np.append(r, [rt])

			xt = step(x[-1, :], u[-1, :])
			if np.fabs(xt[0]) > 0.5 * np.pi:
				break
			else:
				x = np.append(x, [xt], axis=0)

	return x, u, r


x, u, r = sample(N, T)