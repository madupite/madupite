# examples/double-integrator/CMakeLists.txt

add_executable(double-integrator double-integrator.cpp)

target_link_libraries(double-integrator
    PRIVATE
    madupite
    PkgConfig::PETSC
    MPI::MPI_CXX
)

set_target_properties(double-integrator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)
