# CMake version requirement and project details
cmake_minimum_required(VERSION 3.24)
project(distributed_inexact_policy_iteration)

# C++ version specification
set(CMAKE_CXX_STANDARD 17)

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
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
include_directories(${MPI_INCLUDE_PATH})

# nlohmann json configuration
include(FetchContent)
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.2
        )
FetchContent_MakeAvailable(json)
set_target_properties(nlohmann_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)

# Project files
include_directories(MDP/)
add_executable(distributed_inexact_policy_iteration
        main.cpp
        utils/Logger.cpp
        MDP/MDP_algorithm.cpp
        MDP/MDP_setup.cpp
        )

# Linking libraries to the executable
target_link_libraries(distributed_inexact_policy_iteration PkgConfig::PETSC)
target_link_libraries(distributed_inexact_policy_iteration ${MPI_LIBRARIES})
target_link_libraries(distributed_inexact_policy_iteration nlohmann_json::nlohmann_json)
