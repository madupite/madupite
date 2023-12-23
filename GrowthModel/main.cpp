//
// Created by robin on 29.06.23.
//

// Run: cd build; ./growth_model -options_file ../GrowthModel/petsc_options.txt 

#include "GrowthModel.h"
#include <petsc.h>
#include "../utils/Timer.h"

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);
    GrowthModel gm;
    gm.setValuesFromOptions();

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

    Timer t;

    t.start();
    gm.generateKInterval();
    gm.writeVec(gm.k_, "GM/k.out");
    t.stop("Generating k took: ");

    t.start();
    gm.constructTransitionProbabilitiesRewards();
    t.stop("Construction of transition probabilities and rewards took: ");
    LOG("localNumStates: " + std::to_string(gm.localNumStates_) + ", numStates: " + std::to_string(gm.numStates_));

    // Vec V0;
    // VecCreateMPI(PETSC_COMM_WORLD, gm.localNumStates_, gm.numStates_, &V0);
    // VecSet(V0, 1.0);

    // Vec optimalCost;
    // IS optimalPolicy;
    t.start();
    // gm.benchmarkIPI(V0, optimalPolicy, optimalCost);
    gm.inexactPolicyIteration();
    t.stop("iPI took: ");

    // t.start();
    // gm.writeVec(optimalCost, gm.file_cost_);
    // gm.writeIS(optimalPolicy, gm.file_policy_);
    // t.stop("Writing took: ");

    //VecView(optimalCost, PETSC_VIEWER_STDOUT_WORLD);
    //ISView(optimalPolicy, PETSC_VIEWER_STDOUT_WORLD);

    // VecDestroy(&V0);
    // VecDestroy(&optimalCost);
    // ISDestroy(&optimalPolicy);

    gm.~GrowthModel();
    PetscFinalize();
}