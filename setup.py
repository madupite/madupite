import os
import subprocess
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy
from Cython.Build import cythonize

"""
.
├── CMakeLists.txt
├── MANIFEST.in
├── example
│   └── idm.py
├── include
│   ├── JsonWriter.h
│   ├── MDP.h
│   └── json.h
├── madupite
│   ├── __init__.py
│   ├── madupite.pyx
│   └── libmadupite.so
├── pyproject.toml
├── setup.py
├── src
│   ├── MDP
│   │   ├── MDP_algorithm.cpp
│   │   ├── MDP_change.cpp
│   │   └── MDP_setup.cpp
│   └── utils.cpp
"""

# cython code: src/madupite_wrapper.pyx
# libmadupite.so: madupite/libmadupite.so


# Define the extension module
extensions = [
    Extension(
        name='madupite.madupite',  # Name of the module
        sources=['madupite/madupite.pyx'],  # Source file
        include_dirs=[numpy.get_include(), './include'],  # Include directories for C++ headers
        libraries=['madupite'],  # Name of the library (without 'lib' prefix and '.so' suffix)
        library_dirs=['./madupite'],  # Directory where the library is located
        runtime_library_dirs=['$ORIGIN'],  # Search path for runtime libraries
        extra_compile_args=['-std=c++11'],  # Additional flags for the compiler
        # extra_link_args=['-Wl,-rpath,$ORIGIN'],  # Additional flags for the linker
        language='c++',  # Specify the language
    )
]

# Setup function
setup(
    name='madupite',
    version='5.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python wrapper for the Madupite library',
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3"),  # Cythonize the extension
    include_package_data=True,
    zip_safe=False,
)