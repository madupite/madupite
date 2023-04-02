//
// Created by robin on 02.04.23.
//

#include <petscmat.h>
#include <petscvec.h>
#include <petsc.h>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    // chat gpt example

    // Define matrix and vector sizes
    const PetscInt n = 3;

    // Create matrix A
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetFromOptions(A);
    MatSetUp(A);

    // Fill matrix A with values 1 to 9 rowwise
    PetscScalar *vals;
    MatGetArray(A, &vals);
    for (PetscInt i = 0; i < n; i++) {
        for (PetscInt j = 0; j < n; j++) {
            vals[i*n + j] = i*n + j + 1;
        }
    }
    MatRestoreArray(A, &vals);

    // Create vector v
    Vec v;
    VecCreate(PETSC_COMM_WORLD, &v);
    VecSetSizes(v, PETSC_DECIDE, n);
    VecSetFromOptions(v);
    VecSetUp(v);

    // Fill vector v with values 10, 20, 30
    PetscScalar *vvals;
    VecGetArray(v, &vvals);
    vvals[0] = 10;
    vvals[1] = 20;
    vvals[2] = 30;
    VecRestoreArray(v, &vvals);

    // Print matrix A, vector v, and the result A*v
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    VecView(v, PETSC_VIEWER_STDOUT_WORLD);
    Vec w;
    VecDuplicate(v, &w);
    MatMult(A, v, w);
    VecView(w, PETSC_VIEWER_STDOUT_WORLD);

    // Destroy objects and finalize PETSc
    VecDestroy(&v);
    VecDestroy(&w);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}
