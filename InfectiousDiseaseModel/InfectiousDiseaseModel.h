//
// Created by robin on 06.07.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_INFECTIOUSDISEASEMODEL_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_INFECTIOUSDISEASEMODEL_H

#include "../MDP/MDP.h"


class InfectiousDiseaseModel : public MDP {
public:
    InfectiousDiseaseModel();
    ~InfectiousDiseaseModel();
    PetscErrorCode setValuesFromOptions() override;
    PetscErrorCode generateStageCosts();

    std::pair<PetscInt, PetscInt> a2ij(PetscInt a) const; // 1d action index to 2d action index
    PetscReal g(PetscInt state, PetscInt action) const; // stage cost
    PetscReal q(PetscInt state, PetscInt action) const; // toal infection probability
    inline PetscReal ch(PetscInt state) const; // cost of hospitalization

    static const PetscInt numA1_ = 5;
    static const PetscInt numA2_ = 4;
    PetscInt populationSize_;
    PetscReal r_[numA1_];
    PetscInt lambda_[numA2_];
    PetscReal cf_a1_[numA1_];
    PetscReal cf_a2_[numA2_];
    PetscReal cq_a1_[numA1_];
    PetscReal cq_a2_[numA2_];
    PetscReal weights_[3];


};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_INFECTIOUSDISEASEMODEL_H
