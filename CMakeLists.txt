# CMake version requirement and project details
cmake_minimum_required(VERSION 3.10)
project(distributed_inexact_policy_iteration)

# C++ version specification
set(CMAKE_CXX_STANDARD 17)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -march=native -funroll-loops -ffast-math -fopenmp -v")

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
#set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
include_directories(${MPI_INCLUDE_PATH})

# Boost configuration
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# # nlohmann json configuration
# include(FetchContent)
# FetchContent_Declare(json
#         GIT_REPOSITORY https://github.com/nlohmann/json.git
#         GIT_TAG        v3.11.2
#         )
# FetchContent_MakeAvailable(json)
# set_target_properties(nlohmann_json PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>)

# Project files
include_directories(MDP/)

# Add a new library target
add_library(madupite SHARED
MDP/MDP_algorithm.cpp
MDP/MDP_setup.cpp
utils/Logger.cpp
)
# Specify the output directory for the shared library
set_target_properties(madupite PROPERTIES
LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
# Link the necessary libraries to your new library
target_link_libraries(madupite PkgConfig::PETSC)
target_link_libraries(madupite ${MPI_LIBRARIES})
# target_link_libraries(madupite nlohmann_json::nlohmann_json)


# add_executable(distributed_inexact_policy_iteration
#         main.cpp
#         utils/Logger.cpp
#         MDP/MDP_algorithm.cpp
#         MDP/MDP_setup.cpp
#         )
# # Linking libraries to the executable
# target_link_libraries(distributed_inexact_policy_iteration PkgConfig::PETSC)
# target_link_libraries(distributed_inexact_policy_iteration ${MPI_LIBRARIES})
# target_link_libraries(distributed_inexact_policy_iteration nlohmann_json::nlohmann_json)

# # Growth Model
# include_directories(GrowthModel/)
# add_executable(growth_model
#         GrowthModel/main.cpp
#         GrowthModel/GrowthModel.cpp
#         MDP/MDP_algorithm.cpp
#         MDP/MDP_setup.cpp
#         utils/Logger.cpp
#         )

# # Linking libraries to the second executable
# target_link_libraries(growth_model PkgConfig::PETSC)
# target_link_libraries(growth_model ${MPI_LIBRARIES})
# target_link_libraries(growth_model nlohmann_json::nlohmann_json)

# # Infectious Disease Model
# include_directories(InfectiousDiseaseModel/)
# add_executable(infectious_disease_model
#         InfectiousDiseaseModel/main.cpp
#         InfectiousDiseaseModel/InfectiousDiseaseModel.cpp
#         MDP/MDP_algorithm.cpp
#         MDP/MDP_setup.cpp
#         utils/Logger.cpp
#         )

# # Linking libraries to the third executable
# target_link_libraries(infectious_disease_model PkgConfig::PETSC)
# target_link_libraries(infectious_disease_model ${MPI_LIBRARIES})
# target_link_libraries(infectious_disease_model nlohmann_json::nlohmann_json)
# target_link_libraries(infectious_disease_model PRIVATE ${Boost_LIBRARIES})