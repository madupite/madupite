//
// Created by robin on 27.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <vector>
#include "utils/JsonWriter.h"


// declarations

// custom cvg test for KSP
PetscErrorCode cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx);

struct KSPContext {
    PetscInt maxIter;       // input
    PetscReal threshold;    // input
    PetscInt kspIterations; // output
};

class MDP {
public:

    MDP(PetscInt numStates, PetscInt numActions, PetscReal discountFactor);
    ~MDP();

    PetscErrorCode extractGreedyPolicy(Vec &V, PetscInt *policy);
    PetscErrorCode constructFromPolicy(PetscInt   *policy, Mat &transitionProbabilities, Vec &stageCosts);
    PetscErrorCode constructFromPolicy(PetscInt actionInd, Mat &transitionProbabilities, Vec &stageCosts);
    PetscErrorCode iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, KSPContext &ctx);
    PetscErrorCode createJacobian(Mat &jacobian, const Mat &transitionProbabilities);
    PetscErrorCode inexactPolicyIteration(Vec &V0, const PetscInt maxIter, PetscReal alpha, IS &policy, Vec &optimalCost);

    //PetscErrorCode cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx); // Test if residual norm is smaller than alpha * r0_norm
    PetscErrorCode computeResidualNorm(Mat J, Vec V, Vec g, PetscReal *rnorm); // TODO: compute ||g - J*V||_\infty

    PetscErrorCode loadFromBinaryFile(std::string filename_P, std::string filename_g, std::string filename_nnz);


    const PetscInt    numStates_;       // global
    const PetscInt    numActions_;      // global
    const PetscReal   discountFactor_;

    PetscInt localNumStates_;           // number of states owned by this rank
    PetscInt rank_;                     // rank of this process
    PetscInt size_;                     // number of processes
    PetscInt P_start_, P_end_;          // local row range of transitionProbabilityTensor_
    PetscInt g_start_, g_end_;          // local row range of stageCostMatrix_

    Mat transitionProbabilityTensor_;   // transition probability tensor
    Mat stageCostMatrix_;               // stage cost matrix
    Vec nnz_;                           // number of non-zeros in each row of transitionProbabilityTensor_

    JsonWriter *jsonWriter_;
};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
