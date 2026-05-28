import os
import sys
import setuptools

# Find pybind11 include directory
import pybind11
pybind11_include = pybind11.get_include()

# Paths: C library is at ../EventProp.C relative to this setup.py
here = os.path.dirname(os.path.abspath(__file__))
c_lib_dir = os.path.join(here, "..", "EventProp.C")

cpp_ext = setuptools.Extension(
    name="eventprop.eventprop_cpp._native",  # → eventprop/eventprop_cpp/_native.so
    sources=[os.path.join(here, "src", "eventprop_cpp.cpp")],
    include_dirs=[pybind11_include, c_lib_dir],  # eventprop.h lives here
    library_dirs=[c_lib_dir],
    libraries=["eventprop_c_api"],
    runtime_library_dirs=[os.path.abspath(c_lib_dir)],
    language="c++",
    extra_compile_args=["-std=c++17", "-O3"],
    extra_link_args=[f"-Wl,-rpath,{os.path.abspath(c_lib_dir)}"] if sys.platform == "darwin" else [],
)

setuptools.setup(
    name="eventprop-native",
    version="0.1",
    author="EventProp team",
    packages=["eventprop"],
    python_requires=">=3.8",
    install_requires=["numpy", "scipy", "dask"],
    ext_modules=[cpp_ext],
)
