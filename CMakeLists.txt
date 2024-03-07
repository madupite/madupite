cmake_minimum_required(VERSION 3.10)
project(madupite VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# PETSc configuration
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
include_directories($ENV{PETSC_DIR}/include)
include_directories($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)

# Figure out compilers (MPI wrappers) from PETSc configuration
execute_process ( COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE C_COMPILER)
execute_process ( COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_COMPILER)
SET(CMAKE_C_COMPILER ${C_COMPILER})
SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
message(STATUS "Compilers inferred from PETSc:")
message(STATUS "  C: ${CMAKE_C_COMPILER}")
message(STATUS "  C++: ${CMAKE_CXX_COMPILER}")

# Boost configuration
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Add include directories
include_directories(include)

# Define the shared library that compiles from the specified sources
add_library(madupite SHARED
    src/MDP/MDP_algorithm.cpp
    src/MDP/MDP_change.cpp
    src/MDP/MDP_setup.cpp
)

# Link the madupite library with PETSc, MPI, and Boost libraries
target_link_libraries(madupite
    PkgConfig::PETSC
    ${Boost_LIBRARIES}
)

# Specify where the compiled shared library should be placed
set_target_properties(madupite PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Add an executable target for example/example.cpp
add_executable(example example/example.cpp)

# Link the "example" target with the madupite library, PETSc, MPI, and Boost libraries
target_link_libraries(example
    madupite
    PkgConfig::PETSC
    ${Boost_LIBRARIES}
)

# Ensure the "example" target can find the madupite library at runtime
set_target_properties(example PROPERTIES
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib"
)
