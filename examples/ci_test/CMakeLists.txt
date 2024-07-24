# examples/ci_test/CMakeLists.txt

add_executable(ci_test ci_test.cpp)

target_link_libraries(ci_test
    PRIVATE
    madupite
    PkgConfig::PETSC
    MPI::MPI_CXX
)

set_target_properties(ci_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)
