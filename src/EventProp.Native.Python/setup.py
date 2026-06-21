"""Build script for eventprop Cython bindings."""

import os
import subprocess
import sys
import numpy as np
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


class build_ext(_build_ext):
    """Custom build_ext that fixes macOS dylib install_name after linking."""

    def run(self):
        super().run()
        # On macOS, the .so records @rpath/libeventprop_c_api.dylib from the
        # dylib's own install_name. Fix it to use @loader_path so the package
        # works when copied/moved (dylib is placed alongside the .so).
        if sys.platform == 'darwin':
            for ext in self.extensions:
                so_path = self.get_ext_fullpath(ext.name)
                if os.path.exists(so_path) and '.cpython-' in so_path:
                    subprocess.check_call([
                        'install_name_tool', '-change',
                        '@rpath/libeventprop_c_api.dylib',
                        '@loader_path/libeventprop_c_api.dylib',
                        so_path,
                    ])

# Find the C library — check platform-specific locations
import glob as _glob
_c_build_base = os.path.join(os.path.dirname(__file__), '..', 'EventProp.C', 'build')
# Windows: .lib/.dll in build/Release or build/Debug
_win_libs = _glob.glob(os.path.join(_c_build_base, '*', 'eventprop_c_api.lib')) if sys.platform == 'win32' else []
c_lib_dir = os.path.dirname(_win_libs[0]) if _win_libs else _c_build_base
c_lib_path = os.path.join(c_lib_dir, 'libeventprop_c_api.dylib')

if not os.path.exists(c_lib_path):
    c_lib_path = os.path.join(c_lib_dir, 'libeventprop_c_api.so')

include_dirs = [
    os.path.join(os.path.dirname(__file__), '..', 'EventProp.C'),
    np.get_include(),
]
library_dirs = [c_lib_dir] if os.path.exists(c_lib_dir) else []
libraries = ['eventprop_c_api'] if library_dirs else []

# Extra link args for macOS dylib — use @loader_path so the .so finds
# libeventprop_c_api.dylib in its own directory (copied by build step).
extra_link_args = []
if c_lib_path.endswith('.dylib'):
    extra_link_args = ['-Wl,-rpath,@loader_path']

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
    name="eventprop_c",
    version="0.1.0",
    description="EventProp Cython bindings — zero-copy access to C library",
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}),
    python_requires=">=3.10",
    install_requires=["numpy"],
)
