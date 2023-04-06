//
// Created by robin on 02.04.23.
//

#include <petscmat.h>
#include <petscvec.h>
#include <iostream>

int main(int argc, char** argv)
{
    // Initialize PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

    // Create a 3x3 matrix A with values 1 to 9 row-wise
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 3, 3);
    MatSetFromOptions(A);
    MatSetUp(A);
    PetscInt rowStart, rowEnd;
    MatGetOwnershipRange(A, &rowStart, &rowEnd);
    for (PetscInt i = rowStart; i < rowEnd; i++) {
        PetscInt ncols = 3;
        PetscInt cols[3] = {0, 1, 2};
        PetscScalar vals[3] = {i*3+1, i*3+2, i*3+3};
        MatSetValues(A, 1, &i, ncols, cols, vals, INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Create a vector v with values 10, 20, 30
    Vec v;
    VecCreate(PETSC_COMM_WORLD, &v);
    VecSetSizes(v, PETSC_DECIDE, 3);
    VecSetFromOptions(v);
    VecSetValue(v, 0, 10.0, INSERT_VALUES);
    VecSetValue(v, 1, 20.0, INSERT_VALUES);
    VecSetValue(v, 2, 30.0, INSERT_VALUES);
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);

    // Perform the matrix-vector multiplication A * v
    Vec result;
    VecDuplicate(v, &result);
    MatMult(A, v, result);

    // Print the original matrix, vector, and result
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    VecView(v, PETSC_VIEWER_STDOUT_WORLD);
    VecView(result, PETSC_VIEWER_STDOUT_WORLD);

    // Destroy PETSc objects and finalize PETSc
    VecDestroy(&v);
    VecDestroy(&result);
    MatDestroy(&A);
    PetscFinalize();

    return 0;
}
