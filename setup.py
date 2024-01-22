import shutil
import subprocess
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as build_ext

import os
import numpy


# Check if mpi is installed
def check_mpi_installed():
    try:
        subprocess.check_output(["mpicc", "--version"])
    except (subprocess.CalledProcessError, OSError):
        raise RuntimeError(
            "MPI is not installed. Please install MPI before running this setup script."
        )


# Check if cmake is installed
def check_cmake_installed():
    try:
        subprocess.check_output(["cmake", "--version"])
    except (subprocess.CalledProcessError, OSError):
        raise RuntimeError(
            "CMake is not installed. Please install CMake before running this setup script."
        )


# build extension class for compiling madupite with cmake. Before
# wrapping with Cython
class CMakeBuildExt(build_ext):
    def run(self) -> None:
        check_cmake_installed()
        check_mpi_installed()
        self.build_cmake()
        super().run()

    def build_cmake(self):
        # create directory for cmake/make build
        build_dir = Path(self.build_temp) / "cmake_build"
        build_dir.mkdir(parents=True, exist_ok=True)

        # Run cmake with verbose makefile generation
        cmake_command = ["cmake", "../../..", "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"]
        print(f"Running CMake: {' '.join(cmake_command)}")
        cmake_process = subprocess.Popen(
            cmake_command, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in cmake_process.stdout:
            print(line.decode(), end='')

        # Check for errors in cmake
        cmake_process.wait()
        if cmake_process.returncode != 0:
            raise subprocess.CalledProcessError(cmake_process.returncode, cmake_command)

        # Run make with verbosity
        make_command = ["make", "install", "VERBOSE=1"]
        print(f"Running Make: {' '.join(make_command)}")
        make_process = subprocess.Popen(
            make_command, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        for line in make_process.stdout:
            print(line.decode(), end='')

        # Check for errors in make
        make_process.wait()
        if make_process.returncode != 0:
            raise subprocess.CalledProcessError(make_process.returncode, make_command)

        # copy libmadupite.so to build_lib, s.t. setuptools finds and packages it
        shared_object_file = build_dir / "libmadupite.so"
        shutil.copy(shared_object_file, Path(self.build_lib))


# nlohmann json library
# nlohmann_json_include_path = "build/_deps/json-src/include/nlohmann/json.hpp"
nlohmann_json_include_path = os.path.join(os.getcwd(), "build/_deps/json-src/include/")


# class for the cython wrapping
cython_ext = Extension(
    name="madupite",
    sources=["MDP/MDP.pyx"],
    include_dirs=["MDP", nlohmann_json_include_path, numpy.get_include(), "utils"],
    language="c++",
    extra_compile_args=["-std=c++17"],
    extra_objects=["libmadupite.so"],
    extra_link_args=["-Wl,-rpath,$ORIGIN"],
    # cython_directives={"embedsignature": True},
    # libraries=["petsc", "mpi"]
)

if __name__ == "__main__":
    setup(
        name="madupite",
        ext_modules=cythonize(cython_ext, language_level="3"),
        cmdclass={"build_ext": CMakeBuildExt},
        package_data={"": ["*.so"]},
        include_package_data=True,
        zip_safe=False,
        packages=["MDP", "utils"],
    )


