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
│   └── libmadupite.so
├── pyproject.toml
├── setup.py
├── src
│   ├── MDP
│   │   ├── MDP_algorithm.cpp
│   │   ├── MDP_change.cpp
│   │   └── MDP_setup.cpp
│   ├── madupite_wrapper.cpp
│   ├── madupite_wrapper.pyx
│   └── utils.cpp
"""

# cython code: src/madupite_wrapper.pyx
# libmadupite.so: madupite/libmadupite.so


extension = Extension(
        "madupite.madupite",  # Name of the module
        ["madupite/madupite.pyx"],  # Source file
        include_dirs=[numpy.get_include(), "include"],  # Include directories for header files
        library_dirs=["madupite"],  # Directory where the libmadupite.so file is located
        libraries=["madupite"],  # The name of the library to link against, without the 'lib' prefix and '.so' suffix
        runtime_library_dirs=["$ORIGIN"],  # Search path for runtime libraries
        extra_link_args=["-Wl,-rpath,$ORIGIN"],
        extra_objects=["madupite/libmadupite.so"],  
        language="c++",  
    )

setup(
    name="madupite",
    author="Philip & Robin",
    packages=find_packages(),
    package_dir={"madupite": "madupite"},
    package_data={"madupite": ["libmadupite.so"], "src": ["wrapper.pyx"]},
    ext_modules=cythonize(extension, language_level="3"),  # Use cythonize on the extension modules
    include_package_data=True,
    zip_safe=False,
)