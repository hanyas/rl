# python setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "logistic_cy",
        ["logistic_cy.pyx"],
        # extra_compile_args=['-ffast-math', '-O3'],
        extra_compile_args=['-ffast-math', '-O3', '-fopenmp'],
        extra_link_args=['-fopenmp'],
		libraries=["m"],
    )
]

setup(
    name='logistic_cy.pyx',
	include_dirs=[np.get_include()],
    ext_modules=cythonize(ext_modules),
)