# examples/unit_tests/CMakeLists.txt

# Create the unit_tests executable
add_executable(unit_tests tests.cpp)

# Link against madupite library, PETSc, MPI, and GTest
target_link_libraries(unit_tests
    PRIVATE
    madupite
    PkgConfig::PETSC
    MPI::MPI_CXX
    GTest::GTest
    GTest::Main
)

# Set the output directory for the executable
set_target_properties(unit_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)
