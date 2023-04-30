//
// Created by robin on 27.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <petscvec.h>
#include <petscmat.h>
#include <vector>

class MDP {
public:

    MDP(PetscInt numStates, PetscInt numActions, PetscReal discountFactor);
    ~MDP();

    PetscErrorCode extractGreedyPolicy(Vec &V, PetscInt *policy);
    PetscErrorCode constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts);
    PetscErrorCode iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, PetscReal alpha);
    std::vector<PetscInt> inexactPolicyIteration(Vec &V0, PetscInt maxIter, PetscReal alpha);

    PetscErrorCode loadFromBinaryFile(std::string filename_P, std::string filename_g);

    const PetscInt    numStates_;
    const PetscInt    numActions_;
    const PetscReal   discountFactor_;

    Mat transitionProbabilityTensor_;   // transition probability tensor
    Mat stageCostMatrix_;               // stage cost matrix

};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
