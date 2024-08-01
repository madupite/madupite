cmake_minimum_required(VERSION 3.15...3.26)

project(madupite LANGUAGES CXX)

# Enable CMake's testing framework
enable_testing()

# Define a CMake option to control test building
option(BUILD_TESTS "Build the unit tests" OFF)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PetSc)

find_package(MPI REQUIRED)

# Generate compile_commands.json for use with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Automatically collect source and header files
file(GLOB MDP_SOURCES "src/MDP/*.cpp" "include/*.h")

# Check if the project is being built with pip install .
if (SKBUILD)
  find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

  find_package(nanobind CONFIG REQUIRED)

  # Build the Python extension using nanobind
  nanobind_add_module(
    _madupite_impl
    STABLE_ABI
    NB_STATIC
    src/pymadupite.cpp
    ${MDP_SOURCES}
  )

  target_link_libraries(_madupite_impl
    PRIVATE
    PkgConfig::PETSC
    MPI::MPI_CXX
  )

  # Install the Python module
  install(TARGETS _madupite_impl LIBRARY DESTINATION madupite)

else()
  add_library(madupite
    ${MDP_SOURCES}
  )

  # Link necessary libraries
  target_link_libraries(madupite
    PRIVATE
    PkgConfig::PETSC
    MPI::MPI_CXX
  )

  # Specify include directories
  target_include_directories(madupite
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  # Install the C++ library
  install(TARGETS madupite
    LIBRARY DESTINATION lib
  )

  # Install header files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
  )

  file(GLOB EXAMPLE_SOURCES "examples/**/*.cpp")
  foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME}
      PRIVATE
      madupite
    )
    install(TARGETS ${EXAMPLE_NAME}
      RUNTIME DESTINATION bin
    )
  endforeach()
endif()

# Build and add unit tests if enabled
if(BUILD_TESTS)
  # Download and configure GoogleTest
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
  )
  FetchContent_MakeAvailable(googletest)

  # Ensure GoogleTest uses the same C++ standard
  set_target_properties(gtest gtest_main PROPERTIES CXX_STANDARD 20)

  # Add unit test executable
  add_executable(unit_tests tests/unit_tests.cpp)

  # Link the GoogleTest libraries and your project library
  target_link_libraries(unit_tests
    PRIVATE
    madupite
    gtest
    gtest_main
    PkgConfig::PETSC
    MPI::MPI_CXX
  )

  # Add the test to CMake's testing framework
  add_test(NAME MadupiteUnitTests COMMAND unit_tests)

  # Optionally, install the test executable
  install(TARGETS unit_tests RUNTIME DESTINATION bin)
endif()
