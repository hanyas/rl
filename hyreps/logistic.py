import numpy as np
import numexpr as ne

from rl.hyreps.logistic_cy import dlogistic


EXP_MAX = 700.0
EXP_MIN = -700.0


def logistic(p, feat):
    n_features = feat.shape[-1]
    par = np.reshape(p, (-1, n_features), order='C')
    a = np.einsum('...k,lk->...l', feat, par, optimize=False)
    a = np.clip(a, EXP_MIN, EXP_MAX)
    expa = ne.evaluate('exp(a)')
    l = expa / np.sum(expa, axis=-1, keepdims=True)
    return l


def logistic_err(p, feat, fx):
    err = np.sum(np.square(logistic(p, feat) - fx), axis=0)
    return err


def dlogistic_err(p, feat, fx):
    lgstc = logistic(p, feat)
    feat_noview = feat.copy()  # cython memoryview issue
    derr = 2.0 * np.einsum('tl,tlp->lp', lgstc - fx,
                           dlogistic(lgstc, feat_noview))
    return derr


def logit_err(p, feat, fx):
    n_features = feat.shape[-1]
    par = np.reshape(p, (-1, n_features), order='C')
    y = np.einsum('nk,lk->nl', feat, par)
    return np.einsum('nk,nh->', y - fx, y - fx) + 1e-32 * np.linalg.norm(par)**2
