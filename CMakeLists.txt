cmake_minimum_required(VERSION 3.15...3.26)

project(madupite LANGUAGES CXX)

if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# PETSc configuration
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)

# MPI configuration
find_package(MPI REQUIRED)

# GoogleTest configuration
# find_package(GTest REQUIRED)

# Generate compile_commands.json for use with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# nanobind configuration
# execute_process(
#     COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
#     OUTPUT_STRIP_TRAILING_WHITESPACE
#     OUTPUT_VARIABLE NB_DIR
# )
# list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
# find_package(nanobind CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Build the Python extension directly from source files
nanobind_add_module(
    _madupite_impl
    STABLE_ABI
    NB_STATIC
    src/pymadupite.cpp
    src/MDP/madupite_matrix.cpp
    src/MDP/madupite_vector.cpp
    src/MDP/MDP_algorithm.cpp
    src/MDP/MDP_matrix.cpp
    src/MDP/MDP_setup.cpp
    include/json.h
    include/JsonWriter.h
    include/madupite_errors.h
    include/madupite_layout.h
    include/madupite_matrix.h
    include/madupite_vector.h
    include/MDP_matrix.h
    include/MDP.h
)

target_link_libraries(_madupite_impl
    PRIVATE
    PkgConfig::PETSC
    MPI::MPI_CXX
)

# target_include_directories(_madupite_impl
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )

# Install the Python module
install(TARGETS _madupite_impl LIBRARY DESTINATION madupite)

# Install include files
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
#     DESTINATION ${CMAKE_INSTALL_PREFIX}/include
#     FILES_MATCHING PATTERN "*.h"
# )
