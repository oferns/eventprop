#!/usr/bin/env python3
"""Build Cython extension for EventProp C library bindings."""

import os, subprocess
from setuptools import Extension, setup
from Cython.Build import cythonize

# Find C library headers and binary
c_dir = os.path.join(os.path.dirname(__file__), '..', 'EventProp.C')
build_dir = os.path.join(c_dir, 'build')

# Platform-specific library suffix
suffix = '.dylib' if os.uname().sysname == 'Darwin' else '.so'
lib_name = f'libeventprop_c_api{suffix}'

# Copy C library into package dir so Cython can find it at runtime
lib_src = os.path.join(build_dir, lib_name)
pkg_dir = os.path.join(os.path.dirname(__file__), 'eventprop_c')
if os.path.exists(lib_src):
    import shutil
    shutil.copy2(lib_src, os.path.join(pkg_dir, lib_name))

import numpy as np
ext = Extension(
    'eventprop_c._native',
    sources=[os.path.join('eventprop_c', '_native.pyx')],
    include_dirs=[c_dir, np.get_include()],  # for eventprop.h + numpy/arrayobject.h
    library_dirs=[build_dir],       # for libeventprop_c_api
    libraries=['eventprop_c_api'],  # -leventprop_c_api
    extra_compile_args=['-O3', '-mcpu=apple-m3'] if os.uname().machine == 'arm64' else ['-O3'],
)

setup(
    ext_modules=cythonize([ext], language_level=3, annotate=False),
    include_package_data=True,
    script_args=['build_ext', '--inplace'],
)
