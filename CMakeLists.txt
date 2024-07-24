cmake_minimum_required(VERSION 3.10)
project(madupite VERSION 1.0)

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

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Build the library
add_subdirectory(src)

# Build the examples
add_subdirectory(examples)