# src/CMakeLists.txt

add_library(madupite SHARED
    MDP/madupite_matrix.cpp
    MDP/madupite_vector.cpp
    MDP/MDP_algorithm.cpp
    MDP/MDP_matrix.cpp
    MDP/MDP_setup.cpp
)

target_link_libraries(madupite
    PUBLIC
    PkgConfig::PETSC
    MPI::MPI_CXX
)

target_include_directories(madupite
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(madupite PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    POSITION_INDEPENDENT_CODE ON
)
