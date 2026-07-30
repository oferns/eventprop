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
elif sys.platform == 'darwin':
    c_lib_dir = c_build_base
    libraries = ['eventprop_c_api']
    # Build native arch only (not universal2) to match C library
    import platform
    arch = platform.machine()
    extra_link_args = ['-arch', arch, '-Wl,-rpath,' + c_lib_dir]
else:
    c_lib_dir = c_build_base
    libraries = ['eventprop_c_api']
    extra_link_args = []

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
    extra_compile_args=['-arch', platform.machine()] if sys.platform == 'darwin' else [],
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
