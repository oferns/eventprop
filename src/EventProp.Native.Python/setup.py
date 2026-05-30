"""Build script for eventprop_cffi._native C extension.

Pure C — calls into EventProp.C library via libeventprop_c_api.dylib.
No pybind11, no C++. Just C calling into the C library.
"""

import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
c_lib_dir = os.path.join(here, "..", "EventProp.C")

c_ext = Extension(
    name="eventprop_cffi._native",
    sources=["src/eventprop_cffi.c"],
    include_dirs=[os.path.relpath(c_lib_dir, here), np.get_include()],
    library_dirs=[os.path.relpath(os.path.join(c_lib_dir, "build"), here)],
    libraries=["eventprop_c_api"],
    extra_compile_args=["-std=c11", "-O3", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
    extra_link_args=["-lm"],  # math library
)

class _build_ext(build_ext):
    """Custom build_ext that avoids universal2 builds and fixes rpath."""

    def finalize_options(self):
        super().finalize_options()
        # Prevent universal2 build which fails when C lib is single-arch
        if self.plat_name and 'universal2' in self.plat_name:
            self.plat_name = None
        # Override compiler/linker to use arm64-only (C lib is arm64-only)
        import platform, distutils.sysconfig as sc
        if platform.machine() == "arm64":
            cfg = sc.get_config_vars()
            # Remove all -arch x86_64 from CFLAGS/LDSHARED
            for key in ('CFLAGS', 'LDSHARED', 'BLDSHARED'):
                if key in cfg:
                    parts = cfg[key].split()
                    # Remove -arch x86_64 pairs
                    filtered = []
                    i = 0
                    while i < len(parts):
                        if parts[i] == '-arch' and i + 1 < len(parts) and 'x86_64' in parts[i+1]:
                            i += 2
                        else:
                            filtered.append(parts[i])
                            i += 1
                    cfg[key] = ' '.join(filtered)

    def run(self):
        super().run()
        for ext in self.extensions:
            if ext.name == "eventprop_cffi._native":
                for output_file in self.get_outputs():
                    if output_file.endswith(".so"):
                        subprocess.run(
                            [
                                "install_name_tool",
                                "-change",
                                "libeventprop_c_api.dylib",
                                os.path.abspath(os.path.join(c_lib_dir, "build", "libeventprop_c_api.dylib")),
                                output_file,
                            ],
                            check=True,
                        )
                        break


setup(
    ext_modules=[c_ext],
    cmdclass={"build_ext": _build_ext},
)
