import autograd.numpy as np


def cart_polar(state):
    ang = np.arctan2(state[..., 1], state[..., 0])
    vel = state[..., 2]
    polar = np.stack((ang, vel))
    return polar


def solve_linear(a, b, x1):
    return ((a[1] - b[1]) * x1 + (a[0] - b[0])) / (a[2] - b[2])


def merge_dicts(*dicts):
    d = {}
    for dict in dicts:
        for key in dict:
            try:
                d[key].append(dict[key])
            except KeyError:
                d[key] = [dict[key]]

    for key in d:
        d[key] = np.concatenate(d[key]).squeeze()

    return d


def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[ ~ np.isfinite(c)] = 0.0  # -inf inf NaN
    return c


def normalize(x, dim):
    norm = np.sum(x, axis=dim, keepdims=True)
    c = div0(x, norm)
    return c, norm.flatten()
