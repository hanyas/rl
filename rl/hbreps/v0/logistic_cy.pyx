import cython
cimport cython

from cython.parallel import prange, parallel

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)

cpdef double[:, :, :] dlogistic(double[:, :] lgstc, double[:, :] feat):
	cdef int i, j, n, l

	cdef int n_samples = lgstc.shape[0]
	cdef int n_regions = lgstc.shape[1]
	cdef int n_features = feat.shape[1]

	cdef double[:, :, :] jac
	jac = np.empty((n_samples, n_regions, n_regions * n_features))

	with nogil, parallel():
		for n in range(n_samples):
			for i in range(n_regions):
				for j in range(n_regions):
					for l in range(n_features):
						if i == j:
							jac[n, i, j * n_features + l] = feat[n, l] * (lgstc[n, i] - lgstc[n, i] * lgstc[n, i])
						else:
							jac[n, i, j * n_features + l] = - feat[n, l] * lgstc[n, i] * lgstc[n, j]

	return jac


