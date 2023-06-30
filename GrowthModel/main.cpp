//
// Created by robin on 29.06.23.
//

#include "GrowthModel.h"
#include <petsc.h>

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    GrowthModel gm;

    /* ======== INIT PAPER EXAMPLE ======== */
    // create z_ = [0.726, 1.377]
    VecCreateSeq(PETSC_COMM_SELF, gm.numZ_, &gm.z_);
    PetscReal z_vals[2] = {0.726, 1.377};
    PetscInt z_indices[2] = {0, 1};
    VecSetValues(gm.z_, 2, z_indices, z_vals, INSERT_VALUES);
    // create P_z_ = [0.975, 0.025; 0.025, 0.975]
    MatCreateSeqAIJ(PETSC_COMM_SELF, gm.numZ_, gm.numZ_, 2, nullptr, &gm.P_z_);
    PetscReal P_z_vals[4] = {0.975, 0.025, 0.025, 0.975};
    PetscInt P_z_indices[2] = {0, 1};
    MatSetValues(gm.P_z_, 2, P_z_indices, 2, P_z_indices, P_z_vals, INSERT_VALUES);

    VecAssemblyBegin(gm.z_);
    MatAssemblyBegin(gm.P_z_, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(gm.z_);
    MatAssemblyEnd(gm.P_z_, MAT_FINAL_ASSEMBLY);
    /* ==================================== */

    gm.generateKInterval();
    PetscPrintf(PETSC_COMM_WORLD, "Capital stocks (k):\n");
    VecView(gm.k_, PETSC_VIEWER_STDOUT_SELF);

    gm.calculateAvailableResources();
    PetscPrintf(PETSC_COMM_WORLD, "Available resources (B):\n");
    VecView(gm.B_, PETSC_VIEWER_STDOUT_SELF);

    gm.calculateFeasibleActions();
    PetscPrintf(PETSC_COMM_WORLD, "Feasible actions (A):\n");
    ISView(gm.A_, PETSC_VIEWER_STDOUT_SELF);

    Vec V0;
    VecCreateSeq(PETSC_COMM_SELF, gm.numStates_, &V0);
    VecSet(V0, 0.0);

    Vec optimalCost;
    IS optimalPolicy;
    gm.inexactPolicyIteration(V0, optimalPolicy, optimalCost);

    VecView(optimalCost, PETSC_VIEWER_STDOUT_SELF);
    ISView(optimalPolicy, PETSC_VIEWER_STDOUT_SELF);

    VecDestroy(&V0);
    VecDestroy(&optimalCost);
    ISDestroy(&optimalPolicy);

    PetscFinalize();
}