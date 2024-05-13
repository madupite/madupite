cmake_minimum_required(VERSION 3.10)
project(madupite VERSION 1.0)

# Generate compile_commands.json for use with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Turn on for verbose make output
set(CMAKE_VERBOSE_MAKEFILE OFF)

if (CMAKE_BUILD_TYPE MATCHES Coverage)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -g -O0" CACHE STRING "Flags used by the compiler during coverage builds." FORCE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -g -O0" CACHE STRING "Flags used by the compiler during coverage builds." FORCE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage" CACHE STRING "Flags used by the linker during coverage builds." FORCE)
    add_compile_options("--coverage")
    add_link_options("--coverage")
endif()

if (CMAKE_BUILD_TYPE MATCHES Coverage)
    add_definitions(-O0)  # Ensure no optimizations for coverage
endif()



# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI package
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

# PETSc configuration
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH $ENV{PETSC_ARCH})
set(PETSC ${PETSC_DIR}/${PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
include_directories(${PETSC_DIR}/include ${PETSC_DIR}/${PETSC_ARCH}/include include)

# Linking against MPI
add_library(madupite SHARED
    src/MDP/MDP_algorithm.cpp
    src/MDP/MDP_change.cpp
    src/MDP/MDP_setup.cpp
)
target_link_libraries(madupite
    PkgConfig::PETSC
    ${MPI_CXX_LIBRARIES}
)

add_executable(example example/example.cpp)
target_link_libraries(example
    madupite
    PkgConfig::PETSC
    ${MPI_CXX_LIBRARIES}
)

add_executable(ci-test test/ci_test.cpp)
target_link_libraries(ci-test
    madupite
    PkgConfig::PETSC
    ${MPI_CXX_LIBRARIES}
)

set_target_properties(example ci-test PROPERTIES
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib"
)
