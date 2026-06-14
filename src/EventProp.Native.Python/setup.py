"""Build script for eventprop Cython bindings."""

import os
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# Find the C library
c_lib_dir = os.path.join(os.path.dirname(__file__), '..', 'EventProp.C', 'build')
c_lib_path = os.path.join(c_lib_dir, 'libeventprop_c_api.dylib')

if not os.path.exists(c_lib_path):
    c_lib_path = os.path.join(c_lib_dir, 'libeventprop_c_api.so')

include_dirs = [
    os.path.join(os.path.dirname(__file__), '..', 'EventProp.C'),
    np.get_include(),
]
library_dirs = [c_lib_dir] if os.path.exists(c_lib_dir) else []
libraries = ['eventprop_c_api'] if library_dirs else []

# Extra link args for macOS dylib
extra_link_args = []
if c_lib_path.endswith('.dylib'):
    extra_link_args = ['-Wl,-rpath,' + c_lib_dir]

ext_modules = [
    Extension(
        "eventprop_c._native",
        sources=["eventprop_c/_native.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="eventprop",
    version="0.1.0",
    description="EventProp Cython bindings — zero-copy access to C library",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}),
    python_requires=">=3.10",
    install_requires=["numpy"],
)
