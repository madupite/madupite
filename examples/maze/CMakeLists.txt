# examples/maze/CMakeLists.txt

add_executable(maze maze.cpp)

target_link_libraries(maze
    PRIVATE
    madupite
    PkgConfig::PETSC
    MPI::MPI_CXX
)

set_target_properties(maze PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)