"""Build script for eventprop._native C extension."""

import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


here = os.path.dirname(os.path.abspath(__file__))
c_lib_dir = os.path.join(here, "..", "EventProp.C")

cpp_ext = Extension(
    name="eventprop._native",
    sources=[os.path.join(here, "src", "eventprop_cpp.cpp")],
    include_dirs=[pybind11.get_include(), c_lib_dir],
    library_dirs=[c_lib_dir],
    libraries=["eventprop_c_api"],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3"],
)


class _build_ext(build_ext):
    """Custom build_ext that fixes rpath after building."""

    def run(self):
        super().run()
        for ext in self.extensions:
            if ext.name == "eventprop._native":
                for output_file in self.get_outputs():
                    if output_file.endswith(".so"):
                        # Use absolute path to C library (works for local development)
                        subprocess.run(
                            [
                                "install_name_tool",
                                "-change",
                                "libeventprop_c_api.dylib",
                                os.path.abspath(os.path.join(c_lib_dir, "libeventprop_c_api.dylib")),
                                output_file,
                            ],
                            check=True,
                        )
                        break


setup(
    ext_modules=[cpp_ext],
    cmdclass={"build_ext": _build_ext},
)
