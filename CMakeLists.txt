cmake_minimum_required(VERSION 3.15...3.27)
project(madupite VERSION 1.0)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Generate compile_commands.json for use with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# PETSc configuration
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)

# MPI configuration
find_package(MPI REQUIRED)

# GoogleTest configuration
find_package(GTest REQUIRED)

# Python configuration
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind configuration
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Ensure the madupite directory exists
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/madupite)

# Build the library
add_subdirectory(src)

# Build the examples
add_subdirectory(examples)

# Build the Python extension
add_subdirectory(python)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/madupite DESTINATION .)
