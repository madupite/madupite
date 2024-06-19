cmake_minimum_required(VERSION 3.10)
project(madupite VERSION 1.0)

# Generate compile_commands.json for use with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Turn on for verbose make output
set(CMAKE_VERBOSE_MAKEFILE OFF)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# PETSc configuration
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH $ENV{PETSC_ARCH})
set(PETSC ${PETSC_DIR}/${PETSC_ARCH})

set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
include_directories(${PETSC_DIR}/include)
include_directories(${PETSC_DIR}/${PETSC_ARCH}/include)

# Figure out compilers (MPI wrappers) from PETSc configuration
# We currenly handle only the debug build type and rely on PETSc to be built without --with-debugging=no
set(CMAKE_BUILD_TYPE Debug)
execute_process ( COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CMAKE_C_COMPILER)
execute_process ( COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CMAKE_CXX_COMPILER)
execute_process ( COMMAND pkg-config PETSc --variable=cflags_extra COMMAND tr -d '\n' OUTPUT_VARIABLE CMAKE_C_FLAGS_DEBUG)
execute_process ( COMMAND pkg-config PETSc --variable=cxxflags_extra COMMAND tr -d '\n' OUTPUT_VARIABLE CMAKE_CXX_FLAGS_DEBUG)
# Workaround for the -fvisibility=hidden flag that is added by PETSc
# - we currently need to have symbols exported by default
string(REPLACE " -fvisibility=hidden" "" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
string(REPLACE " -fvisibility=hidden" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "Compilers inferred from PETSc:")
message(STATUS "  C: ${CMAKE_C_COMPILER}  ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "  C++: ${CMAKE_CXX_COMPILER}  ${CMAKE_CXX_FLAGS_DEBUG}")

# Add include directories
include_directories(include)

# Define the shared library that compiles from the specified sources
add_library(madupite SHARED
    src/MDP/MDP_algorithm.cpp
    src/MDP/MDP_setup.cpp
)

# Link the madupite library with PETSc
target_link_libraries(madupite
    PkgConfig::PETSC
    # ${Boost_LIBRARIES}
)

# Specify where the compiled shared library should be placed
set_target_properties(madupite PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Add an executable target for example/example.cpp
add_executable(example example/example.cpp)

# Link the "example" target with the madupite library and PETSc
target_link_libraries(example
    madupite
    PkgConfig::PETSC
    # ${Boost_LIBRARIES}
)

# CI Test
add_executable(ci-test test/ci_test.cpp)
target_link_libraries(ci-test
    madupite
    PkgConfig::PETSC
)

# Ensure the "example" target can find the madupite library at runtime
set_target_properties(example PROPERTIES
    INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib"
)
