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
    PetscErrorCode calculateAvailableResources();
    PetscErrorCode calculateFeasibleActions();
    PetscErrorCode constructTransitionProbabilitiesRewards();
    //PetscErrorCode constructFromPolicy(PetscInt   *policy, Mat &transitionProbabilities, Vec &stageCosts);
    //PetscErrorCode constructFromPolicy(PetscInt actionInd, Mat &transitionProbabilities, Vec &stageCosts);
    //PetscErrorCode extractGreedyPolicy(Vec &V, PetscInt *policy, PetscReal &residualNorm);
    //PetscErrorCode iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, KSPContext &ctx);
    //PetscErrorCode createJacobian(Mat &jacobian, const Mat &transitionProbabilities, JacobianContext &ctx);
    //PetscErrorCode inexactPolicyIteration(Vec &V0, IS &policy, Vec &optimalCost);
    //static PetscErrorCode cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx); // Test if residual norm is smaller than alpha * r0_norm
    //static void jacobianMultiplication(Mat mat, Vec x, Vec y); // defines matrix vector product for jacobian shell
    inline PetscInt ij2s(PetscInt i, PetscInt j) const { return i * numZ_ + j; }

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
    /*PetscInt numStates_;
    PetscInt numActions_;
    PetscReal discountFactor_;
    PetscInt    maxIter_PI_;
    PetscInt    maxIter_KSP_;
    PetscInt    numPIRuns_;
    PetscReal   rtol_KSP_;
    PetscReal   atol_PI_;
    PetscInt localNumStates_;
    JsonWriter *jsonWriter_;*/

};


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_GROWTHMODEL_H
