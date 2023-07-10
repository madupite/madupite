//
// Created by robin on 28.06.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_GROWTHMODEL_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_GROWTHMODEL_H

#include "../MDP/MDP.h"
#include "../utils/Logger.h"

class GrowthModel : public MDP {
public:
    GrowthModel();
    ~GrowthModel();
    PetscErrorCode setValuesFromOptions() override;
    PetscErrorCode generateKInterval();
    PetscErrorCode constructTransitionProbabilitiesRewards();

    inline PetscInt ij2s(PetscInt i, PetscInt j) const { return i * numZ_ + j; }
    inline std::pair<PetscInt, PetscInt> s2ij(PetscInt s) const { return std::make_pair(s / numZ_, s % numZ_); }

    PetscInt numK_;
    PetscInt numZ_;
    PetscInt localNumK_;
    Mat P_z_;
    IS A_; // nk * nz (feasible actions)
    Vec B_; // nk * nz (available resources)
    Vec z_;
    Vec k_;
    PetscReal riskAversionParameter_;
    const PetscReal rho_ = 0.33; // f(k) = k^rho (implied production function)
};

#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_GROWTHMODEL_H
