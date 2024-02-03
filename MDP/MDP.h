//
// Created by robin on 27.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>
#include <vector>
#include "../utils/JsonWriter.h"

struct KSPContext {
    PetscInt maxIter;       // input
    PetscReal threshold;    // input
    PetscInt kspIterations; // output
};

struct JacobianContext {
    Mat P_pi;
    PetscReal discountFactor;
};

class MDP {
public:

    MDP();
    ~MDP();
    virtual PetscErrorCode setValuesFromOptions();
    virtual PetscErrorCode setOption(const char *option, const char *value);

    virtual PetscErrorCode extractGreedyPolicy(const Vec &V, PetscInt *policy, PetscReal &residualNorm);
    virtual PetscErrorCode constructFromPolicy(const PetscInt   *policy, Mat &transitionProbabilities, Vec &stageCosts);
    virtual PetscErrorCode iterativePolicyEvaluation(const Mat &jacobian, const Vec &stageCosts, Vec &V, KSPContext &ctx);
    virtual PetscErrorCode createJacobian(Mat &jacobian, const Mat &transitionProbabilities, JacobianContext &ctx);
    virtual PetscErrorCode inexactPolicyIteration();
    // virtual PetscErrorCode benchmarkIPI(const Vec &V0, IS &policy, Vec &optimalCost);

    static PetscErrorCode cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx); // Test if residual norm is smaller than alpha * r0_norm
    static void jacobianMultiplication(Mat mat, Vec x, Vec y);          // defines matrix vector product for jacobian shell
    static void jacobianMultiplicationTranspose(Mat mat, Vec x, Vec y); // defines tranposed matrix vector product for jacobian shell (needed for some KSP methods)

    virtual PetscErrorCode loadFromBinaryFile(); // TODO split into P and g
    virtual PetscErrorCode generateCostMatrix(double (*g)(PetscInt, PetscInt));
    // virtual PetscErrorCode generateTransitionProbabilityTensor(const PetscReal (*P)(PetscInt, PetscInt, PetscInt)); // P(s, a, s')
    // virtual PetscErrorCode generateTransitionProbabilityTensor(std::tuple<PetscInt, PetscInt*, PetscReal*> (*P)(PetscInt, PetscInt)); // (P(s, a) -> (nnz, action indices, probabilities) for more efficient allocation
    // virtual PetscErrorCode generateTransitionProbabilityTensor(const PetscInt *colIndices, const PetscReal *values, PetscInt nnz);
    virtual PetscErrorCode generateTransitionProbabilityTensor(double (*P)(PetscInt, PetscInt, PetscInt), PetscInt d_nz, const PetscInt *d_nnz, PetscInt o_nz, const PetscInt *o_nnz);

    PetscErrorCode writeVec  (const Vec  &vec, const PetscChar *filename);
    PetscErrorCode writeIS(const IS &is, const PetscChar *filename);

    // user specified options
    enum mode {MINCOST, MAXREWARD};
    mode        mode_;
    PetscInt    numStates_;       // global
    PetscInt    numActions_;      // global
    PetscReal   discountFactor_;
    PetscInt    maxIter_PI_;
    PetscInt    maxIter_KSP_;
    PetscInt    numPIRuns_;
    PetscReal   rtol_KSP_;
    PetscReal   atol_PI_;
    PetscChar   file_P_     [PETSC_MAX_PATH_LEN]; // input
    PetscChar   file_g_     [PETSC_MAX_PATH_LEN]; // input
    PetscChar   file_policy_[PETSC_MAX_PATH_LEN]; // output
    PetscChar   file_cost_  [PETSC_MAX_PATH_LEN]; // output
    PetscChar   file_stats_ [PETSC_MAX_PATH_LEN]; // output

    // derived parameters
    PetscInt localNumStates_;           // number of states owned by this rank
    PetscInt rank_;                     // rank of this process
    PetscInt size_;                     // number of processes
    PetscInt P_start_, P_end_;          // local row range of transitionProbabilityTensor_
    PetscInt g_start_, g_end_;          // local row range of stageCostMatrix_

    Mat transitionProbabilityTensor_;   // transition probability tensor
    Mat stageCostMatrix_;               // stage cost matrix (also rewards possible)
    Mat costMatrix_;                    // cost matrix used in extractGreedyPolicy
    Vec costVector_;                    // cost vector used in extractGreedyPolicy

    JsonWriter *jsonWriter_;
};

#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
