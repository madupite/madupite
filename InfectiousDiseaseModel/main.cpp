//
// Created by robin on 05.07.23.
//

#include <boost/math/distributions/binomial.hpp>
#include "InfectiousDiseaseModel.h"
#include "../utils/Timer.h"

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    Timer t;

    InfectiousDiseaseModel idm;

    idm.setValuesFromOptions();

    t.start();
    idm.generateStageCosts();
    t.stop("Generating stage costs took: ");
    t.start();
    idm.generateTransitionProbabilities();
    t.stop("Generating transition probabilities took: ");

    Vec V0;
    VecCreateMPI(PETSC_COMM_WORLD, idm.localNumStates_, idm.numStates_, &V0);
    VecSet(V0, 1.0);

    Vec optimalCost;
    IS optimalPolicy;
    t.start();
    idm.inexactPolicyIteration(V0, optimalPolicy, optimalCost);
    t.stop("Inexact policy iteration took: ");

    PetscPrintf(PETSC_COMM_WORLD, "Optimal cost:\n");
    VecView(optimalCost, PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Optimal policy:\n");
    ISView(optimalPolicy, PETSC_VIEWER_STDOUT_WORLD);

    idm.writeVec(optimalCost, idm.file_cost_);
    idm.writeIS(optimalPolicy, idm.file_policy_);

    idm.~InfectiousDiseaseModel();
    PetscFinalize();
    return 0;
}