cmake_minimum_required(VERSION 3.10)
project(madupite VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -ffast-math -fopenmp -v")

# PETSc configuration
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED  IMPORTED_TARGET PETSc)
include_directories($ENV{PETSC_DIR}/include)
include_directories($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)

# MPI configuration
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

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
    ${MPI_LIBRARIES}
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
    ${MPI_LIBRARIES}
    ${Boost_LIBRARIES}
)

# If the MPI compiler is used, specify that it should compile the "example" target
if(MPI_COMPILE_FLAGS)
  set_target_properties(example PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

# If the MPI linker flags are needed, specify them for the "example" target
if(MPI_LINK_FLAGS)
  set_target_properties(example PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

# Ensure the "example" target can find the madupite library at runtime
set_target_properties(example PROPERTIES
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib"
)