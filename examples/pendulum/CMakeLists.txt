# examples/pendulum/CMakeLists.txt

add_executable(pendulum pendulum.cpp)

target_link_libraries(pendulum
    PRIVATE
    madupite
    PkgConfig::PETSC
    MPI::MPI_CXX
)

set_target_properties(pendulum PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)
