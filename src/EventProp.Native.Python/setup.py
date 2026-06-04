"""Build script for eventprop ctypes bindings.

Pure Python — no C extension to build. Uses ctypes to load libeventprop_c_api.
No numpy, no pybind11, no C++. Just Python ctypes calling into the C library.
"""

from setuptools import setup, find_packages

setup(
    name="eventprop",
    version="0.1.0",
    description="EventProp ctypes bindings — zero-copy access to C library",
    packages=find_packages(),
    python_requires=">=3.10",
    # No external dependencies — pure ctypes
    install_requires=[],
)
