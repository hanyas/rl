import numpy as np
import scipy as sc
from scipy.optimize import minimize
from scipy.optimize import check_grad
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures

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


np.random.seed(1337)


class Pi:
	def __init__(self, n_states):
		self.theta = np.random.randn(n_states)
		self.cov = 10.0 * np.eye(n_states)


class Quad:
	def __init__(self, n_states):
		self.M = np.zeros((n_states, n_states))
		self.m = np.zeros((n_states,))
		self.m0 = np.zeros((1, ))


def func(x):
	# quadratic function
	# Q = np.eye(2)
	# x = x - 3.0
	# return - np.einsum('nk,kh,nh->n', x, Q, x)

	# rosenbrock
	# r = np.empty((x.shape[0],))
	# for i in range(x.shape[0]):
	# 	r[i] = - ((1.0 - x[i, 0])**2 + 100.0 * (x[i, 1] - x[i, 0]**2)**2)
	# return r

	# # rastrigin
	r = np.empty((x.shape[0],))
	for i in range(x.shape[0]):
		r[i] = - (10.0 * 2 + np.sum(x[i, :]**2 - 10.0 * np.cos(2.0 * np.pi * x[i, :]), axis=-1))
	return r


def sample(pi, n):
	x = np.random.multivariate_normal(mean=pi.theta, cov=pi.cov, size=(n,))
	r = func(x)
	return x, r


def fit_quad(x, r):
	n_states = x.shape[-1]

	model = Quad(n_states)

	poly = PolynomialFeatures(2)
	feat = poly.fit_transform(x)

	par = np.linalg.inv(feat.T @ feat + 1e-8 * np.eye(poly.n_output_features_)) @ feat.T @ r

	i_upper = np.triu_indices(n_states, 0)
	model.M[i_upper] = par[1 + n_states:]

	i_lower = np.tril_indices(n_states, -1)
	model.M[i_lower] = model.M.T[i_lower]

	model.m = par[1: 1 + n_states]
	model.m0 = par[0]

	# check for negative definitness
	w, v = np.linalg.eig(model.M)
	w[w >= 0.0] = -1e-6
	model.M = v @ np.diag(w) @ v.T

	# refit quadratic
	aux = r - np.einsum('nk,kh,nh->n', x, model.M, x)
	poly = PolynomialFeatures(1)
	feat = poly.fit_transform(x)

	par = np.linalg.inv(feat.T @ feat + 1e-8 * np.eye(poly.n_output_features_)) @ feat.T @ aux

	model.m = par[1:]
	model.m0 = par[0]

	return model


def policy_update(q, model, eta, omega):
	M, m, m0 = model.M, model.m, model.m0

	b = q.theta
	Q = q.cov

	invQ = np.linalg.inv(Q)
	F = np.linalg.inv(eta * invQ - 2.0 * M)
	f = eta * invQ @ b + m

	pi = Pi(q.theta.shape[0])

	pi.theta = F @ f
	pi.cov = F * (eta + omega) + np.eye(n_states) * 1e-24

	return pi


def kl_divergence(pi, q):
	kl = 0.5 * (np.trace(np.linalg.inv(q.cov) @ pi.cov) + (q.theta - pi.theta).T @ np.linalg.inv(q.cov) @ (q.theta - pi.theta)
	            - pi.theta.shape[0] + np.log(np.linalg.det(q.cov) / np.linalg.det(pi.cov)))
	return kl


def entropy(q):
	dim = q.theta.shape[0]
	_, lgdt = np.linalg.slogdet(q.cov)
	return 0.5 *  (dim * np.log(2.0 * np.pi * np.e) + lgdt)


def dual(var, eps, beta, q, model):
	eta = var[0]
	omega = var[1]

	M, m, m0 = model.M, model.m, model.m0

	b = q.theta
	Q = q.cov

	invQ = np.linalg.inv(Q)

	F = np.linalg.inv(eta * invQ - 2.0 * M)
	f = eta * invQ @ b + m

	_, q_lgdt = np.linalg.slogdet(2.0 * np.pi * Q)
	_, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * F)

	g = eta * eps - omega * beta + 0.5 * (f.T @ F @ f - eta * b.T @ invQ @ b
	                                      - eta * q_lgdt
	                                      + (eta + omega) * f_lgdt)
	return g

def grad(var, eps, beta, q, model):
	eta = var[0]
	omega = var[1]

	M, m, m0 = model.M, model.m, model.m0

	b = q.theta
	Q = q.cov

	n_states = Q.shape[0]

	invQ = np.linalg.inv(Q)

	F = np.linalg.inv(eta * invQ - 2.0 * M)
	f = eta * invQ @ b + m

	_, q_lgdt = np.linalg.slogdet(2.0 * np.pi * Q)
	_, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * F)

	dF_deta = - F.T @ invQ @ F
	df_deta = invQ @ b

	deta = eps + 0.5 * (2.0 * f.T @ F @ df_deta + f.T @ dF_deta @ f
	                    - b.T @ invQ @ b - q_lgdt
	                    + f_lgdt + n_states - (eta + omega) * np.trace(F @ invQ))

	domega = - beta + 0.5 * (f_lgdt + n_states)

	return np.hstack((deta, domega))


n_states = 2
n_samples = 100

q = Pi(n_states)

eta = 1000.0
omega = 1000.0

eps = 0.1
gamma = 0.99

iter = 1500

for i in range(iter):
	# compute entropy bound
	beta = gamma * (entropy(q) + 75) - 75

	# sample controller
	x, r = sample(q, n_samples)

	# fit quadratic model
	model = fit_quad(x, r)

	# optimize dual
	var = np.stack((0.5, 0.5))
	bnds = ((1e-8, 1e8), (1e-8, 1e8))

	res = sc.optimize.minimize(dual, np.array([0.5, 0.5]), method='L-BFGS-B', jac=grad, args=(eps, beta, q, model), bounds=bnds)
	eta = res.x[0]
	omega = res.x[1]

	# # gradient checks
	# step = np.sqrt(np.finfo(float).eps) * np.ones((2,))
	# approx = sc.optimize.approx_fprime(var, dual, step, eps, beta, q, model)
	# gradient = grad(var, eps, beta, q, model)
	# check = sc.optimize.check_grad(dual, grad, var, eps, beta, q, model)
	# print('Gradient Error', check)

	# update policy
	pi = policy_update(q, model, eta, omega)

	# check kl
	kl = kl_divergence(pi, q)

	q = pi

	print('Iter', i, 'Reward', np.mean(r), 'KL', kl)
