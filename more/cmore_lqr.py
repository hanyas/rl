import numpy as np
import scipy as sc
import scipy.optimize as minimize
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns

from sklearn.linear_model import Ridge
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
	colors.append(mcd.XKCD_COLORS['xkcd:' + k].upper())

np.random.seed(1337)


class Pi:
	def __init__(self, n_action, n_feat):
		self.K = np.random.randn(n_action, n_feat)
		self.cov = 100.0 * np.eye(n_action)


class Quad:
	def __init__(self, n_action, n_feat):
		self.Ra = np.zeros((n_action, n_action))
		self.Rca = np.zeros((n_feat, n_action))


def func(x, c, Q):
	diff = x - c
	return - np.einsum('nk,kh,nh->n', diff, Q, diff)


def sample(pi, Q, n_action, n_cntxt, n_feat, n_samples):
	c = np.random.uniform(-5.0, 5.0, size=(n_samples, n_cntxt))

	poly = PolynomialFeatures(1)
	feat = poly.fit_transform(c)

	x = np.zeros((n_samples, n_action))

	mu = np.einsum('kl,nl->nk', pi.K, feat)

	for i in range(n_samples):
		x[i, :] = np.random.multivariate_normal(mean=mu[i, :], cov=pi.cov)

	r = func(x, c, Q)

	return x, c, r


def kl_divergence(pi, q, phi, n_action, n_samples):
	mo = np.einsum('km,nm->nk', q.K, phi)
	mu = np.einsum('km,nm->nk', pi.K, phi)

	kl = np.zeros((n_samples,))

	for i in range(n_samples):
		kl[i] = 0.5 * (np.trace(np.linalg.inv(q.cov) @ pi.cov) + (mo[i, :] - mu[i, :]).T @ np.linalg.inv(q.cov) @ (mo[i, :] - mu[i, :])
		               - n_action + np.log(np.linalg.det(q.cov) / np.linalg.det(pi.cov)))
	return kl.mean()


def entropy(q, n_action):
	_, lgdt = np.linalg.slogdet(q.cov)
	return 0.5 * (n_action * np.log(2.0 * np.pi * np.e) + lgdt)


def fit_quad(x, phi, r, n_action, n_cntxt, n_feat, n_samples):
	model = Quad(n_action, n_feat)

	n_feat_action = n_action * n_action
	n_feat_cross = n_action * n_feat

	feat = np.zeros((n_samples, n_feat_action + n_feat_cross))

	aux = - 0.5 * np.einsum('nk,nh->nkh', x, x)
	aux = np.reshape(aux, (n_samples, -1), order='C')

	feat[:, :n_feat_action] = aux

	aux = np.einsum('nk,nm->nkm', x, phi)
	aux = np.reshape(aux, (n_samples, -1), order='C')

	feat[:, n_feat_action:] = aux

	# par = np.linalg.inv(feat.T @ feat + 1e-8 * np.eye(n_feat_action + n_feat_cross)) @ feat.T @ r

	clf = Ridge(alpha=0.0001, fit_intercept=False)
	clf.fit(feat, r)
	par = clf.coef_

	model.Ra = np.reshape(par[:n_feat_action], (n_action, n_action), order='C')
	model.Rca = np.reshape(par[n_feat_action:], (n_feat, n_action), order='C')

	# symmetrize
	model.Ra = 0.5 * (model.Ra + model.Ra.T)

	# check for positive definitness
	w, v = np.linalg.eig(model.Ra)

	if np.any(w <= 0.0):
		w[w <= 0.0] = 1e-6
		model.Ra = v @ np.diag(w) @ v.T

		model.Ra = 0.5 * (model.Ra + model.Ra.T)

		# refit quadratic
		tmp_r = r + 0.5 * np.einsum('nk,kh,nh->n', x, model.Ra, x)

		feat = np.einsum('nk,nm->nkm', x, phi)
		feat = np.reshape(feat, (n_samples, -1), order='C')

		# par = np.linalg.inv(feat.T @ feat + 1e-8 * np.eye(n_feat_cross)) @ feat.T @ tmp_r

		clf = Ridge(alpha=0.0001, fit_intercept=False)
		clf.fit(feat, tmp_r)
		par = clf.coef_

		model.Rca = np.reshape(par, (n_feat, n_action), order='C')

	return model


def dual(var, eps, beta, q, model, phi):
	eta = var[0]
	omega = var[1]

	Ra, Rca = model.Ra, model.Rca

	prec = np.linalg.inv(q.cov)

	Haa = eta * prec + Ra
	Hca = eta * q.K.T @ prec + Rca
	Hcc = eta * q.K.T @ prec @ q.K

	# Haa = 0.5 * (Haa + Haa.T)

	invHaa = np.linalg.inv(Haa)

	M = Hca @ invHaa @ Hca.T - Hcc

	_, q_lgdt = np.linalg.slogdet(2.0 * np.pi * q.cov)
	_, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * invHaa)

	g = eta * eps - omega * beta + 0.5 * (eta + omega) * f_lgdt - 0.5 * eta * q_lgdt \
									+ 0.5 * np.einsum('nk,kh,nh->', phi, M , phi) / n_samples

	return g


def grad(var, eps, beta, q, model, phi):
	eta = var[0]
	omega = var[1]

	Ra, Rca = model.Ra, model.Rca

	prec = np.linalg.inv(q.cov)

	Haa = eta * prec + Ra
	Hca = eta * q.K.T @ prec + Rca
	Hcc = eta * q.K.T @ prec @ q.K

	# Haa = 0.5 * (Haa + Haa.T)

	invHaa = np.linalg.inv(Haa)

	_, q_lgdt = np.linalg.slogdet(2.0 * np.pi * q.cov)
	_, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * invHaa)

	dHaa_deta = - invHaa.T @ prec @ invHaa
	dHca_deta = q.K.T @ prec
	dHcc_deta = q.K.T @ prec @ q.K

	M = 2.0 * dHca_deta @ invHaa @ Hca.T + Hca @ dHaa_deta @ Hca.T - dHcc_deta

	n_action = q.cov.shape[0]

	deta = eps + 0.5 * (f_lgdt - q_lgdt + n_action - (eta + omega) * np.trace(invHaa @ prec))\
				+ 0.5 * np.einsum('nk,kh,nh->', phi, M, phi) / n_samples

	domega = - beta + 0.5 * (f_lgdt + n_action)

	return np.hstack([deta, domega])


def policy_update(q, model, eta, omega, n_action, n_feat, n_samples):
	Ra, Rca = model.Ra, model.Rca

	prec = np.linalg.inv(q.cov)

	Haa = eta * prec + Ra
	Hca = eta * q.K.T @ prec + Rca
	Hcc = eta * q.K.T @ prec @ q.K

	# Haa = 0.5 * (Haa + Haa.T)

	invHaa = np.linalg.inv(Haa)

	pi = Pi(n_action, n_feat)

	pi.K = invHaa @ Hca.T
	pi.cov = invHaa * (eta + omega) + np.eye(n_action) * 1e-16

	# pi.cov = 0.5 * (pi.cov + pi.cov.T)

	return pi


n_action = 2
n_cntxt = 2
n_feat = n_cntxt + 1

# generate pos-def matrix
Q = np.random.randn(n_action, n_action)
Q = 0.5 * (Q + Q.T)
Q = Q @ Q.T

n_samples = 100

q = Pi(n_action, n_feat)

eps = 0.05
gamma = 0.99

iter = 100

returns = np.zeros((iter, ))

for i in range(iter):
	# compute entropy bound
	beta = gamma * (entropy(q, n_action) + 75) - 75

	# sample controller
	x, c, r = sample(q, Q, n_action, n_cntxt, n_feat, n_samples)

	poly = PolynomialFeatures(1)
	phi = poly.fit_transform(c)

	# fit quadratic model
	model = fit_quad(x, phi, r, n_action, n_cntxt, n_feat, n_samples)

	# optimize dual
	bnds = ((1e-8, 1e8), (1e-8, 1e8))

	res = sc.optimize.minimize(dual, np.array([0.1, 0.1]), method='SLSQP', jac=grad,
								args=(eps, beta, q, model, phi), bounds=bnds)
	eta = res.x[0]
	omega = res.x[1]

	# # gradient checks
	# step = np.sqrt(np.finfo(float).eps) * np.ones((2,))
	# approx = sc.optimize.approx_fprime(var, dual, step, eps, beta, q, model, phi)
	# gradient = grad(var, eps, beta, q, model, phi)
	# check = sc.optimize.check_grad(dual, grad, var, eps, beta, q, model, phi)
	# print('Gradient Error', check)

	# update policy
	pi = policy_update(q, model, eta, omega, n_action, n_feat, n_samples)

	# check kl
	kl = kl_divergence(pi, q, phi, n_action, n_samples)

	q = pi

	print('Iter', i, 'Reward', np.mean(r), 'KL', kl, 'Ent', entropy(pi, n_action), 'eta', eta, 'omega', omega)
	returns[i] = np.mean(r)


plt.plot(returns)
plt.show()