#include <algorithm>
#include <iostream> // TODO: replace with some logger
#include <memory>
// #include "../utils/Logger.h"

#include "MDP.h"

// reshape costVector into costMatrix
void MDP::reshapeCostVectorToCostMatrix(const Vec costVector, Mat costMatrix)
{
    IS               rows, cols;
    const PetscReal* vals;
    PetscCallThrow(VecGetArrayRead(costVector, &vals));
    PetscCallThrow(ISCreateStride(Madupite::getCommWorld(), localNumStates_, g_start_, 1, &rows));
    PetscCallThrow(ISCreateStride(Madupite::getCommWorld(), numActions_, 0, 1, &cols));
    PetscCallThrow(MatSetValuesIS(costMatrix, rows, cols, vals, INSERT_VALUES));
    PetscCallThrow(VecRestoreArrayRead(costVector, &vals));
    PetscCallThrow(MatAssemblyBegin(costMatrix, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(costMatrix, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(ISDestroy(&rows));
    PetscCallThrow(ISDestroy(&cols));
}

// find $\argmin_{\pi} \{ g^\pi + \gamma P^\pi V \}$
// PRE: policy is a array of size localNumStates_ and must be allocated.
// POST: returns infinity norm of Bellman residual and writes optimal policy into policy
PetscReal MDP::getGreedyPolicyAndResidualNorm(Mat costMatrix, const Vec V, const std::unique_ptr<PetscInt[]>& policy)
{
    // Output
    PetscReal residualNorm;

    // find minimum for each row and compute Bellman residual norm
    Vec residual;
    PetscCallThrow(VecCreateMPI(Madupite::getCommWorld(), localNumStates_, numStates_, &residual));

    PetscInt         nnz;
    const PetscInt*  colIdx;
    const PetscReal* vals;

    // TODO VH: MatGetRow/MatRestoreRow might be inefficient for a dense matrix
    if (mode_ == mode::MINCOST) {
        for (PetscInt rowInd = 0; rowInd < localNumStates_; ++rowInd) {
            PetscCallThrow(MatGetRow(costMatrix, rowInd + g_start_, &nnz, &colIdx, &vals));
            const PetscReal* min = std::min_element(vals, vals + nnz);
            policy[rowInd]       = colIdx[min - vals];
            PetscCallThrow(VecSetValue(residual, rowInd + g_start_, *min, INSERT_VALUES));
            PetscCallThrow(MatRestoreRow(costMatrix, rowInd + g_start_, &nnz, &colIdx, &vals));
        }
    } else if (mode_ == mode::MAXREWARD) {
        for (PetscInt rowInd = 0; rowInd < localNumStates_; ++rowInd) {
            PetscCallThrow(MatGetRow(costMatrix, rowInd + g_start_, &nnz, &colIdx, &vals));
            const PetscReal* max = std::max_element(vals, vals + nnz);
            policy[rowInd]       = colIdx[max - vals];
            PetscCallThrow(VecSetValue(residual, rowInd + g_start_, *max, INSERT_VALUES));
            PetscCallThrow(MatRestoreRow(costMatrix, rowInd + g_start_, &nnz, &colIdx, &vals));
        }
    }

    PetscCallThrow(VecAssemblyBegin(residual));
    PetscCallThrow(VecAssemblyEnd(residual));

    PetscCallThrow(VecAXPY(residual, -1.0, V));
    PetscCallThrow(VecNorm(residual, NORM_INFINITY, &residualNorm));

    PetscCallThrow(VecDestroy(&residual));
    return residualNorm;
}

// Create P_pi from policy pi. P_pi is allocated by this function and must be destroyed using MatDestroy()
Mat MDP::getTransitionProbabilities(const std::unique_ptr<PetscInt[]>& policy)
{
    // Outputs
    Mat transitionProbabilities;

    // allocate memory for values
    auto rowIndArr = std::make_unique<PetscInt[]>(localNumStates_);

    // compute global row indices for P
    PetscInt p_start = transitionProbabilityTensor_.rowLayout().start();
    PetscInt actionInd;
    for (PetscInt localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
        actionInd                = policy[localStateInd];
        rowIndArr[localStateInd] = p_start + localStateInd * numActions_ + actionInd;
    }

    // generate index sets
    IS rowInd;
    PetscCallThrow(ISCreateGeneral(Madupite::getCommWorld(), localNumStates_, rowIndArr.get(), PETSC_USE_POINTER, &rowInd));

    // LOG("Creating transitionProbabilities submatrix");
    // TODO: can MatGetSubMatrix be used here?
    PetscCallThrow(MatCreateSubMatrix(transitionProbabilityTensor_.petsc(), rowInd, NULL, MAT_INITIAL_MATRIX, &transitionProbabilities));
    // TODO: Is this necessary?
    PetscCallThrow(MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY));

    PetscCallThrow(ISDestroy(&rowInd));
    return transitionProbabilities;
}

// Create g_pi from policy pi. g_pi is allocated by this function and must be destroyed using VecDestroy().
Vec MDP::getStageCosts(const std::unique_ptr<PetscInt[]>& policy)
{
    // Output
    Vec stageCosts;

    // allocate memory for values
    auto g_pi_values = std::make_unique<PetscScalar[]>(localNumStates_);

    // get values for g_pi
    PetscInt g_srcRow, actionInd;
    for (PetscInt localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
        actionInd = policy[localStateInd];
        g_srcRow  = g_start_ + localStateInd;
        PetscCallThrow(MatGetValue(stageCostMatrix_.petsc(), g_srcRow, actionInd, &g_pi_values[localStateInd]));
    }

    IS ind;
    PetscCallThrow(ISCreateStride(Madupite::getCommWorld(), localNumStates_, g_start_, 1, &ind));

    // LOG("Creating stageCosts vector");
    const PetscInt* indArr;
    PetscCallThrow(VecCreateMPI(Madupite::getCommWorld(), localNumStates_, numStates_, &stageCosts));
    PetscCallThrow(ISGetIndices(ind, &indArr));
    PetscCallThrow(VecSetValues(stageCosts, localNumStates_, indArr, g_pi_values.get(), INSERT_VALUES));
    PetscCallThrow(ISRestoreIndices(ind, &indArr));
    PetscCallThrow(VecAssemblyBegin(stageCosts));
    PetscCallThrow(VecAssemblyEnd(stageCosts));

    PetscCallThrow(ISDestroy(&ind));
    return stageCosts;
}

// returns number of inner iterations performed. V is the initial guess and will be overwritten with the solution.
PetscInt MDP::iterativePolicyEvaluation(const Mat jacobian, const Vec stageCosts, PetscInt maxIter, PetscReal threshold, Vec V)
{
    // TODO: Perhaps KSP could be reused
    // TODO: Then also the JSON outpur could be done somewhere else

    // KSP setup
    KSP ksp;
    PetscCallThrow(KSPCreate(Madupite::getCommWorld(), &ksp));
    PetscCallThrow(KSPSetOperators(ksp, jacobian, jacobian));

    // PC setup
    PC pc;
    PetscCallThrow(KSPGetPC(ksp, &pc));
    PetscCallThrow(PCSetFromOptions(pc));

    PetscCallThrow(KSPSetFromOptions(ksp));
    PetscCallThrow(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    // Use L2 norm of residual to compare. This works since ||x||_2 >= ||x||_inf. Much faster than infinity norm.
    PetscCallThrow(KSPSetTolerances(ksp, 1e-40, threshold, PETSC_DEFAULT, maxIter));
    // ierr = KSPSetConvergenceTest(ksp, &MDP::cvgTest, &ctx, NULL); CHKERRQ(ierr); // custom convergence test using infinity norm

    // Inner loop itself
    PetscCallThrow(KSPSolve(ksp, stageCosts, V));

    // Save KSP info to JSON
    KSPType ksptype;
    PCType  pctype;
    PetscCallThrow(KSPGetType(ksp, &ksptype));
    // must be written here since KSPType is only available after KSPSolve. Other data should be written in MDP::writeJSONmetadata
    PetscCallThrow(PCGetType(pc, &pctype));
    jsonWriter_->add_data("KSPType", ksptype);
    jsonWriter_->add_data("PCType", pctype);

    // Output iteration count
    PetscInt kspIterations;
    PetscCallThrow(KSPGetIterationNumber(ksp, &kspIterations));
    PetscCallThrow(KSPDestroy(&ksp));
    return kspIterations;
}

struct MDPJacobian {
    Mat       P_pi;
    PetscReal discountFactor;

    static PetscErrorCode mult(Mat mat, Vec x, Vec y)
    {
        MDPJacobian* ctx;
        PetscCall(MatShellGetContext(mat, (void**)&ctx));
        // (I - gamma * P_pi) * x == -gamma * P_pi * x + x
        PetscCall(MatMult(ctx->P_pi, x, y));
        PetscCall(VecScale(y, -ctx->discountFactor));
        PetscCall(VecAXPY(y, 1.0, x));
        return 0;
    }

    static PetscErrorCode multTranspose(Mat mat, Vec x, Vec y)
    {
        MDPJacobian* ctx;
        PetscCall(MatShellGetContext(mat, (void**)&ctx));
        // (I - gamma * P_pi)^T * x == -gamma * P_pi^T * x + x; since transposition distributes over subtraction
        PetscCall(MatMultTranspose(ctx->P_pi, x, y));
        PetscCall(VecScale(y, -ctx->discountFactor));
        PetscCall(VecAXPY(y, 1.0, x));
        return 0;
    }
};

// creates MPIAIJ matrix and computes jacobian = I - gamma * P_pi. Must be destroyed using MatDestroy()
Mat MDP::createJacobian(const Mat transitionProbabilities, PetscReal discountFactor)
{
    Mat jacobian;
    // TODO: Could I use s smart pointer here?
    auto ctx = new MDPJacobian(transitionProbabilities, discountFactor);
    PetscCallThrow(MatCreateShell(Madupite::getCommWorld(), localNumStates_, localNumStates_, numStates_, numStates_, ctx, &jacobian));
    PetscCallThrow(MatShellSetContextDestroy(jacobian, [](void* ctx) -> PetscErrorCode {
        delete (MDPJacobian*)ctx;
        return 0;
    }));
    PetscCallThrow(MatShellSetOperation(jacobian, MATOP_MULT, (void (*)(void))MDPJacobian::mult));
    PetscCallThrow(MatShellSetOperation(jacobian, MATOP_MULT_TRANSPOSE, (void (*)(void))MDPJacobian::multTranspose));
    PetscCallThrow(MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY));
    return jacobian;
}

void MDP::solve()
{
    // make sure MDP is set up
    setUp();

    // if(rank_ == 0) LOG("Entering solve");
    jsonWriter_->add_solver_run();
    writeJSONmetadata();

    // init guess V0
    Vec V0;
    PetscCallThrow(VecCreateMPI(Madupite::getCommWorld(), localNumStates_, numStates_, &V0));
    PetscCallThrow(VecSet(V0, 1.0));

    // allocate cost matrix used in extractGreedyPolicy (n x m; DENSE)
    Mat costMatrix;
    PetscCallThrow(MatCreate(Madupite::getCommWorld(), &costMatrix));
    PetscCallThrow(MatSetType(costMatrix, MATDENSE));
    PetscCallThrow(MatSetSizes(costMatrix, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetUp(costMatrix));

    // allocated cost vector used in extractGreedyPolicy (n; DENSE)
    Vec costVector;
    PetscCallThrow(MatCreateVecs(transitionProbabilityTensor_.petsc(), nullptr, &costVector));

    Vec V;
    PetscCallThrow(VecDuplicate(V0, &V));
    PetscCallThrow(VecCopy(V0, V));

    PetscLogDouble startTime, endTime, startiPI, endiPI;

    auto policyValues = std::make_unique<PetscInt[]>(localNumStates_);

    PetscCallThrow(PetscTime(&startiPI));
    PetscInt PI_iteration = 0;
    for (; PI_iteration < maxIter_PI_; ++PI_iteration) { // outer loop
        PetscCallThrow(PetscTime(&startTime));
        // costVector = discountFactor * P * V
        PetscCallThrow(MatMult(transitionProbabilityTensor_.petsc(), V, costVector));
        PetscCallThrow(VecScale(costVector, discountFactor_));

        // reshape costVector to costMatrix
        reshapeCostVectorToCostMatrix(costVector, costMatrix);

        // add g to costMatrix
        PetscCallThrow(MatAXPY(costMatrix, 1.0, stageCostMatrix_.petsc(), SAME_NONZERO_PATTERN));

        // extract greedy policy
        auto residualNorm = getGreedyPolicyAndResidualNorm(costMatrix, V, policyValues);

        if (residualNorm < atol_PI_) {
            PetscCallThrow(PetscTime(&endTime));
            jsonWriter_->add_iteration_data(PI_iteration, 0, (endTime - startTime) * 1000, residualNorm);
            // if(rank_ == 0) LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm));
            break;
        }

        // compute transition probabilities and stage costs
        // TODO split into two functions
        auto transitionProbabilities = getTransitionProbabilities(policyValues);
        auto stageCosts              = getStageCosts(policyValues);

        // create jacobian
        auto jacobian = createJacobian(transitionProbabilities, discountFactor_);

        // inner loop: solve linear system
        auto threshold     = residualNorm * alpha_;
        auto kspIterations = iterativePolicyEvaluation(jacobian, stageCosts, maxIter_KSP_, threshold, V);

        PetscCallThrow(MatDestroy(&transitionProbabilities));
        PetscCallThrow(MatDestroy(&jacobian)); // avoid memory leak
        PetscCallThrow(VecDestroy(&stageCosts));

        PetscCallThrow(PetscTime(&endTime));
        jsonWriter_->add_iteration_data(PI_iteration, kspIterations, (endTime - startTime) * 1000, residualNorm);
        // if(rank_ == 0) LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm));
    }
    PetscCallThrow(PetscTime(&endiPI));
    // LOG("Inexact Policy Iteration took: " + std::to_string((endiPI - startiPI) * 1000) + " ms");

    if (PI_iteration >= maxIter_PI_) {
        // LOG("Warning: maximum number of PI iterations reached. Solution might not be optimal.");
        std::cout << "Warning: maximum number of PI iterations reached. Solution might not be optimal." << std::endl;
        jsonWriter_->add_data("remark", "Warning: maximum number of iPI iterations reached. Solution might not be optimal.");
    }

    jsonWriter_->write_to_file(file_stats_);

    // output results
    IS optimalPolicy;
    // PetscCallThrow(PetscTime(&startTime));
    PetscCallThrow(ISCreateGeneral(Madupite::getCommWorld(), localNumStates_, policyValues.get(), PETSC_USE_POINTER, &optimalPolicy));
    writeVec(V, file_cost_);
    writeIS(optimalPolicy, file_policy_);

    // write optimal transition probabilities and stage costs
    if (file_optimal_transition_probabilities_[0] != '\0') {
        auto optimalTransitionProbabilities = getTransitionProbabilities(policyValues);
        writeMat(optimalTransitionProbabilities, file_optimal_transition_probabilities_);
        PetscCallThrow(MatDestroy(&optimalTransitionProbabilities));
    }
    if (file_optimal_stage_costs_[0] != '\0') {
        auto optimalStageCosts = getStageCosts(policyValues);
        writeVec(optimalStageCosts, file_optimal_stage_costs_);
        PetscCallThrow(VecDestroy(&optimalStageCosts));
    }

    // PetscCallThrow(PetscTime(&endTime));
    // PetscLogDouble duration = (endTime - startTime) * 1000;
    // if (rank_ == 0) {
    //     LOG("Saving results took: " + std::to_string(duration) + " ms");
    // }

    PetscCallThrow(ISDestroy(&optimalPolicy));
    PetscCallThrow(VecDestroy(&V0));
    PetscCallThrow(VecDestroy(&V));
    PetscCallThrow(VecDestroy(&costVector));
    PetscCallThrow(MatDestroy(&costMatrix));
}