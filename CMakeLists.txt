cmake_minimum_required(VERSION 3.24)

# petsc part from PETSC_DIR/share/petsc/CMakeLists.txt
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set root of location to find PETSc's pkg-config
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)
# Remove the lines below if you do not wish to have PETSc determine the compilers
execute_process ( COMMAND pkg-config PETSc --variable=ccompiler COMMAND tr -d '\n' OUTPUT_VARIABLE C_COMPILER)
SET(CMAKE_C_COMPILER ${C_COMPILER})
execute_process ( COMMAND pkg-config PETSc --variable=cxxcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE CXX_COMPILER)
if (CXX_COMPILER)
    SET(CMAKE_CXX_COMPILER ${CXX_COMPILER})
endif (CXX_COMPILER)
execute_process ( COMMAND pkg-config PETSc --variable=fcompiler COMMAND tr -d '\n' OUTPUT_VARIABLE FORTRAN_COMPILER)
if (FORTRAN_COMPILER)
    SET(CMAKE_Fortran_COMPILER ${FORTRAN_COMPILER})
    enable_language(Fortran)
endif (FORTRAN_COMPILER)


set(CMAKE_CXX_STANDARD 17)

project(distributed_inexact_policy_iteration)

# for PETSc
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED  IMPORTED_TARGET PETSc)
INCLUDE_DIRECTORIES($ENV{PETSC_DIR}/include)
INCLUDE_DIRECTORIES($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)



# Add the executable, link libraries
add_executable(distributed_inexact_policy_iteration main.cpp PETScFunctions.h)
target_link_libraries(distributed_inexact_policy_iteration PkgConfig::PETSC)
#target_link_libraries(distributed_inexact_policy_iteration ${PETSC_LIBRARIES})
