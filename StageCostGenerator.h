//
// Created by robin on 20.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_STAGECOSTGENERATOR_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_STAGECOSTGENERATOR_H

#include <petscvec.h>
#include <random>

void generateStageCosts(Mat& stageCosts, PetscInt numStates, PetscInt numActions, PetscInt seed, PetscScalar perturbFactor) {
    MatCreateSeqDense(PETSC_COMM_WORLD, numStates, numActions, PETSC_NULL, &stageCosts);
    auto rows = new PetscInt[1];
    auto cols = new PetscInt[numActions];
    auto values = new PetscScalar[numActions];

    std::mt19937_64 gen(seed);
    PetscInt minCost = 0.0, maxCost = 10.0;
    std::uniform_real_distribution<PetscScalar> dis(minCost, maxCost);

    for(PetscInt i = 0; i < numStates; i++) {
        for(PetscInt j = 0; j < numActions; j++) {
            if(i == 0) cols[j] = j; // only set cols once
            auto val = dis(gen);
            if(dis(gen) < perturbFactor) {
                val *= 10;
            }
            values[j] = val;
        }
        *rows = {i};
        MatSetValues(stageCosts, 1, rows, numActions, cols, values, INSERT_VALUES);
    }

    //VecSetValues(stageCosts, numStates * numActions, indices, values, INSERT_VALUES)
    MatAssemblyBegin(stageCosts, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stageCosts, MAT_FINAL_ASSEMBLY);

    delete[] rows;
    delete[] cols;
    delete[] values;
}

#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_STAGECOSTGENERATOR_H
