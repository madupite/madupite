//
// Created by robin on 28.06.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_GROWTHMODEL_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_GROWTHMODEL_H

#include "../MDP/MDP.h"

class GrowthModel {
public:
    GrowthModel();
    PetscErrorCode generateKInterval();
    PetscErrorCode calculateAvailableResources();
    PetscErrorCode calculateFeasibleActions();
    inline PetscInt ij2s(PetscInt i, PetscInt j) const { return i * numZ_ + j; }

    PetscInt numK_;
    PetscInt numZ_;
    Mat P_z_;
    IS A_; // nk * nz (feasible actions)
    Vec B_; // nk * nz (available resources)
    Vec z_;
    Vec k_;
    PetscReal riskAversionParameter_;
    PetscReal rho_; // f(k) = k^rho (implied production function)
    PetscInt numStates_;
    PetscInt numActions_;
    PetscReal discountFactor_;

};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_GROWTHMODEL_H
