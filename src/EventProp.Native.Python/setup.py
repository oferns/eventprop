"""Build script for eventprop_c Python C-API extension."""

import os
import sys
import glob
import numpy as np
from setuptools import setup, Extension

# Find the C library build directory
c_build_base = os.path.join(os.path.dirname(__file__), '..', 'EventProp.C', 'build')

# Platform-specific library discovery
if sys.platform == 'win32':
    lib_candidates = glob.glob(os.path.join(c_build_base, '*', 'eventprop_c_api.lib'))
    c_lib_dir = os.path.dirname(lib_candidates[0]) if lib_candidates else c_build_base
    libraries = ['eventprop_c_api'] if os.path.exists(c_lib_dir) else []
    extra_link_args = []
    # _native.c includes eventprop_batch.h, which includes <stdatomic.h> in C mode.
    # MSVC defines __STDC_NO_ATOMICS__ (so the header #errors "C atomic support is
    # not enabled") even with /std:c11, on every toolset tested up to 14.52 / VS 2026;
    # the switch that enables them is /experimental:c11atomics. EventProp.C applies the
    # same switch to the library itself.
    extra_compile_args = ['/experimental:c11atomics']
elif sys.platform == 'darwin':
    c_lib_dir = c_build_base
    libraries = ['eventprop_c_api']
    # Build native arch only (not universal2) to match C library
    import platform
    arch = platform.machine()
    extra_link_args = ['-arch', arch, '-Wl,-rpath,' + c_lib_dir]
    extra_compile_args = ['-arch', arch]
else:
    c_lib_dir = c_build_base
    libraries = ['eventprop_c_api']
    extra_link_args = []
    extra_compile_args = []

ext = Extension(
    "eventprop_c._native",
    sources=["eventprop_c/_native.c"],
    include_dirs=[
        os.path.join(os.path.dirname(__file__), '..', 'EventProp.C'),
        np.get_include(),
    ],
    library_dirs=[c_lib_dir] if libraries else [],
    libraries=libraries,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="eventprop_c",
    version="0.1.0",
    description="EventProp Python C-API bindings",
    ext_modules=[ext],
    python_requires=">=3.10",
    install_requires=["numpy"],
)
