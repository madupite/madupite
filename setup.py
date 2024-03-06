import os
import numpy
import subprocess
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

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
# cython code: src/madupite.pyx
# libmadupite.so: madupite/libmadupite.so


class build_ext(_build_ext):
    def run(self):
        # Define the directory where this setup.py is located
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(setup_dir, 'build')

        # Ensure the build directory exists
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        # Run cmake and make commands
        subprocess.check_call(['cmake', setup_dir, '-B' + build_dir])
        subprocess.check_call(['make', '-C', build_dir, 'madupite'])

        # Copy the built library to the expected location
        lib_path = os.path.join(build_dir, 'lib', 'libmadupite.so')
        dest_path = os.path.join(setup_dir, 'madupite', 'libmadupite.so')
        subprocess.check_call(['cp', lib_path, dest_path])

        # Proceed with the standard build process
        super().run()


# Define the extension module
extensions = [
    Extension(
        name='madupite.madupite',
        sources=['madupite/madupite.pyx'],
        include_dirs=[numpy.get_include(), 'include'],
        libraries=['madupite'],  # Name of the library (without 'lib' prefix and '.so' suffix)
        library_dirs=['madupite'],  # Directory where the library is located
        runtime_library_dirs=['$ORIGIN'],  # Search path for runtime libraries
        extra_compile_args=['-std=c++11'],  # Additional flags for the compiler
        extra_link_args=['-Wl,-rpath,$ORIGIN'],  # Additional flags for the linker (else the library is not found at runtime)
        language='c++',
    )
]


# Setup function
setup(
    name='madupite',
    packages=find_packages(),
    ext_modules=cythonize(extensions, language_level="3"),
    include_package_data=True,
    zip_safe=False,
    cmdclass={'build_ext': build_ext},
)