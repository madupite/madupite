import shutil
import subprocess
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext as build_ext

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

        try:
            cmake_command = ["cmake", "../../.."]
            cmake_process = subprocess.Popen(
                cmake_command,
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            cmake_output, cmake_error = cmake_process.communicate()
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: Command {e.cmd} failed with exit status {e.returncode}, but continuing."
            )

        # if cmake_output:
        #     print(cmake_output.decode())

        # if cmake_error:
        #     print(cmake_error.decode())
        #     raise subprocess.CalledProcessError(cmake_process.returncode, cmake_command)

        make_command = ["make"]
        make_process = subprocess.Popen(
            make_command, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        make_output, make_error = make_process.communicate()
        if make_output:
            print(make_output.decode())

        if make_error:
            print(make_error.decode())
            raise subprocess.CalledProcessError(make_process.returncode, make_command)

        # copy libmadupite.so to build_lib, s.t. setuptools finds and packages it
        print("---------------------------")
        shared_object_file = build_dir / "libmadupite.so"
        # print(shared_object_file, Path(self.build_temp), Path(self.build_temp).parent)
        # shutil.copy(shared_object_file, Path(self.build_temp).parent)
        shared_object_file = build_dir / "libmadupite.so"
        shutil.copy(shared_object_file, Path(self.build_lib))


# nlohmann json library
# nlohmann_json_include_path = "build/_deps/json-src/include/nlohmann/json.hpp"
# nlohmann_json_include_path = os.path.join(os.getcwd(), "build/_deps/json-src/include/")


# class for the cython wrapping
cython_ext = Extension(
    name="madupite",
    sources=["MDP/MDP.pyx"],
    include_dirs=["MDP", numpy.get_include(), "utils"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_objects=["build/libmadupite.so"],
    extra_link_args=["-Wl,-rpath,$ORIGIN"],
    # compiler_directives={"embedsignature": True},
)

if __name__ == "__main__":
    setup(
        name="madupite",
        ext_modules=cythonize(cython_ext),
        packages=find_packages(include=["MDP", "MDP.*"]),  # Include only MDP package
        cmdclass={"build_ext": CMakeBuildExt},
        package_data={"": ["*.so"]},
        include_package_data=True,
        zip_safe=False,
    )
