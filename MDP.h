//
// Created by robin on 27.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <vector>

class MDP {
public:

    MDP(PetscInt numStates, PetscInt numActions, PetscReal discountFactor);
    ~MDP();

   enum GreedyPolicyType {V1, V2, V3};

    PetscErrorCode extractGreedyPolicy(Vec &V, PetscInt *policy, GreedyPolicyType type);
    PetscErrorCode constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts);
    PetscErrorCode iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, PetscReal alpha);
    std::vector<PetscInt> inexactPolicyIteration(Vec &V0, PetscInt maxIter, PetscReal alpha);

    //PetscErrorCode cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx); // Test if residual norm is smaller than alpha * r0_norm
    PetscErrorCode computeResidualNorm(Mat J, Vec V, Vec g, PetscReal *rnorm); // TODO: compute ||g - J*V||_\infty

    PetscErrorCode loadFromBinaryFile(std::string filename_P, std::string filename_g, std::string filename_nnz);

    const PetscInt    numStates_;
    const PetscInt    numActions_;
    const PetscReal   discountFactor_;
    PetscInt    localNumStates_; // number of states owned by this rank
    PetscInt rank_; // rank of this process
    PetscInt size_; // number of processes
    PetscInt localRowStart_; // global row index of first row owned by this rank
    PetscInt localRowEnd_; // global row index of last row owned by this rank


    Mat transitionProbabilityTensor_;   // transition probability tensor
    Mat stageCostMatrix_;               // stage cost matrix
    Vec nnz_;                           // number of non-zeros in each row of transitionProbabilityTensor_
};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
