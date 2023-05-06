//
// Created by robin on 06.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_PETSCFUNCTIONS_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_PETSCFUNCTIONS_H

#include <petscmat.h>

// Makes GMRES return the residual norm at each iteration
PetscErrorCode my_convergence_test(KSP ksp, PetscInt n, PetscReal rnorm, KSPConvergedReason *reason, void *ctx)
{
    PetscPrintf(PETSC_COMM_WORLD, "Iteration %D: Residual norm = %g\n", n, rnorm);
    return KSP_CONVERGED_ITERATING;
}

void generateDenseMatrix(Mat& A, unsigned int rows, unsigned int cols, double* values) {
    // create a dense matrix
    MatCreateSeqDense(PETSC_COMM_WORLD, rows, cols, nullptr, &A);

    auto rowIndices = new PetscInt[rows];
    auto colIndices = new PetscInt[cols];
    for(unsigned int i = 0; i < rows; i++) {
        rowIndices[i] = i;
    }
    for(unsigned int i = 0; i < cols; i++) {
        colIndices[i] = i;
    }

    // set values
    MatSetValues(A, rows, rowIndices, cols, colIndices, values, INSERT_VALUES);
    // assemble matrix
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

// create a dense vector, fill it with values, and assemble it
void fillVector(Vec& v, unsigned int size, double* values) {
    VecCreateSeq(PETSC_COMM_WORLD, size, &v);
    auto indices = new PetscInt[size];
    for(unsigned int i = 0; i < size; i++) {
        indices[i] = i;
    }

    VecSetValues(v, size, indices, values, INSERT_VALUES);
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
}




#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_PETSCFUNCTIONS_H
