//
// Created by robin on 28.06.23.
//

#include "GrowthModel.h"


GrowthModel::GrowthModel() {
    numK_ = 20;
    numZ_ = 2;
    discountFactor_ = 0.98;
    rho_ = 0.33;
    riskAversionParameter_ = 0.5;
    P_z_ = nullptr;
    z_ = nullptr;
    numStates_ = numK_ * numZ_;
    numActions_ = numK_;
    jsonWriter_ = new JsonWriter(0);
    localNumStates_ = numStates_;
    maxIter_PI_ = 100;
    maxIter_KSP_ = 10000;
    numPIRuns_ = 1;
    rtol_KSP_ = 1e-12;
    atol_PI_ = 1e-10;
}

PetscErrorCode GrowthModel::generateKInterval() {
    PetscReal z_vals[2];
    PetscInt z_indices[2] = {0, numZ_ - 1};
    VecGetValues(z_, 2, z_indices, z_vals);
    VecCreateSeq(PETSC_COMM_SELF, numK_, &k_);
    PetscReal *k_vals;
    PetscMalloc1(numK_, &k_vals);

    PetscReal k_star_z1 = std::pow(discountFactor_ * rho_ * z_vals[0] / (1 - discountFactor_), 1 / (1 - rho_));
    PetscReal k_star_z2 = std::pow(discountFactor_ * rho_ * z_vals[1] / (1 - discountFactor_), 1 / (1 - rho_));
    PetscReal k_min = k_star_z1 - 0.1 * (k_star_z2 - k_star_z1);
    PetscReal k_max = k_star_z2 + 0.1 * (k_star_z2 - k_star_z1);

    PetscMalloc1(numK_, &k_vals);
    PetscReal k_incr = (k_max - k_min) / (numK_ - 1);
    for(PetscInt i = 0; i < numK_; ++i) {
        k_vals[i] = k_min + i * k_incr;
    }

    IS indices;
    ISCreateStride(PETSC_COMM_SELF, numK_, 0, 1, &indices);
    const PetscInt *indices_arr;
    ISGetIndices(indices, &indices_arr);
    VecSetValues(k_, numK_, indices_arr, k_vals, INSERT_VALUES);
    ISRestoreIndices(indices, &indices_arr);
    ISDestroy(&indices);

    VecAssemblyBegin(k_);
    VecAssemblyEnd(k_);

    PetscFree(k_vals);

    return 0;
}

PetscErrorCode GrowthModel::calculateAvailableResources() {
    PetscReal *B_vals;
    PetscMalloc1(numK_ * numZ_, &B_vals);
    VecCreateSeq(PETSC_COMM_SELF, numK_ * numZ_, &B_);

    const PetscReal *k_vals, *z_vals;
    VecGetArrayRead(k_, &k_vals);
    VecGetArrayRead(z_, &z_vals);
    for(PetscInt i = 0; i < numK_; ++i) {
        for(PetscInt j = 0; j < numZ_; ++j) {
            // B[i,j] = z[j] * (k[i]^rho) + k[i]
            B_vals[ij2s(i, j)] = z_vals[j] * std::pow(k_vals[i], rho_) + k_vals[i];
        }
    }
    VecRestoreArrayRead(k_, &k_vals);
    VecRestoreArrayRead(z_, &z_vals);

    IS indices;
    ISCreateStride(PETSC_COMM_SELF, numK_ * numZ_, 0, 1, &indices);
    const PetscInt *indices_arr;
    ISGetIndices(indices, &indices_arr);
    VecSetValues(B_, numK_ * numZ_, indices_arr, B_vals, INSERT_VALUES);
    ISRestoreIndices(indices, &indices_arr);
    ISDestroy(&indices);

    VecAssemblyBegin(B_);
    VecAssemblyEnd(B_);

    PetscFree(B_vals);

    return 0;
}



PetscErrorCode GrowthModel::calculateFeasibleActions() {
    // find max_a {a | 0 <= a < nk, B[i,j] - k[a] >= 0}
    PetscInt *A_vals;
    PetscMalloc1(numK_ * numZ_, &A_vals);
    const PetscReal *k_vals, *B_vals;
    VecGetArrayRead(k_, &k_vals);
    VecGetArrayRead(B_, &B_vals);

    for(PetscInt i = 0; i < numK_; ++i) {
        for(PetscInt j = 0; j < numZ_; ++j) {
            PetscInt a = 0;
            while(a < numK_ && B_vals[ij2s(i, j)] - k_vals[a] > 0) {
                ++a;
            }
            A_vals[ij2s(i, j)] = std::max(a - 1, 0);
        }
    }

    VecRestoreArrayRead(k_, &k_vals);
    VecRestoreArrayRead(B_, &B_vals);
    ISCreateGeneral(PETSC_COMM_SELF, numK_ * numZ_, A_vals, PETSC_COPY_VALUES, &A_);
    PetscFree(A_vals);

    return 0;
}

PetscErrorCode GrowthModel::constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts) {
    MatCreate(PETSC_COMM_SELF, &transitionProbabilities);
    MatSetType(transitionProbabilities, MATSEQAIJ);
    MatSetSizes(transitionProbabilities, numStates_, numStates_, PETSC_DECIDE, PETSC_DECIDE);
    VecCreate(PETSC_COMM_SELF, &stageCosts);
    VecSetType(stageCosts, VECSEQ);
    VecSetSizes(stageCosts, numStates_, PETSC_DECIDE);
    MatSeqAIJSetPreallocation(transitionProbabilities, numZ_, NULL); // for MPIAIJ: preallocate min(numZ_, localCols) for diag and numZ_ for off-diag to be safe
    PetscReal *z_vals;
    PetscMalloc1(numZ_, &z_vals);
    PetscInt *z_indices_src, *z_indices_dest;
    PetscMalloc1(numZ_, &z_indices_src);
    PetscMalloc1(numZ_, &z_indices_dest);
    std::iota(z_indices_src, z_indices_src + numZ_, 0);
    PetscReal *reward_vals;
    PetscMalloc1(numStates_, &reward_vals);
    const PetscInt *feasibleActions;
    ISGetIndices(A_, &feasibleActions);
    const PetscReal *availableResources, *k_vals;
    VecGetArrayRead(B_, &availableResources);
    VecGetArrayRead(k_, &k_vals);

    for(PetscInt i = 0; i < numK_; ++i) {
        for(PetscInt j = 0; j < numZ_; ++j) {
            PetscInt s = ij2s(i, j);
            if(policy[s] <= feasibleActions[s]) {
                MatGetValues(P_z_, 1, &j, numZ_, z_indices_src, z_vals);
                std::iota(z_indices_dest, z_indices_dest + numZ_, policy[s] * numZ_);
                MatSetValues(transitionProbabilities, 1, &s, numZ_, z_indices_dest, z_vals, INSERT_VALUES);
                reward_vals[s] = std::pow(availableResources[s] - k_vals[policy[s]], riskAversionParameter_) / riskAversionParameter_;
            }
            else {
                reward_vals[s] = -1e10;
            }
        }
    }

    VecRestoreArrayRead(k_, &k_vals);
    VecRestoreArrayRead(B_, &availableResources);
    ISRestoreIndices(A_, &feasibleActions);

    PetscInt *stageCostIndices;
    PetscMalloc1(numStates_, &stageCostIndices);
    std::iota(stageCostIndices, stageCostIndices + numStates_, 0);
    VecSetValues(stageCosts, numStates_, stageCostIndices, reward_vals, INSERT_VALUES);
    PetscFree(stageCostIndices);

    PetscFree(z_indices_src);
    PetscFree(z_indices_dest);
    PetscFree(z_vals);
    PetscFree(reward_vals);

    MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(stageCosts);
    MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(stageCosts);

    return 0;
}

PetscErrorCode GrowthModel::constructFromPolicy(PetscInt actionInd, Mat &transitionProbabilities, Vec &stageCosts) {
    MatCreate(PETSC_COMM_SELF, &transitionProbabilities);
    MatSetType(transitionProbabilities, MATSEQAIJ);
    MatSetSizes(transitionProbabilities, numStates_, numStates_, PETSC_DECIDE, PETSC_DECIDE);
    VecCreate(PETSC_COMM_SELF, &stageCosts);
    VecSetType(stageCosts, VECSEQ);
    VecSetSizes(stageCosts, numStates_, PETSC_DECIDE);
    MatSeqAIJSetPreallocation(transitionProbabilities, numZ_, NULL); // for MPIAIJ: preallocate min(numZ_, localCols) for diag and numZ_ for off-diag to be safe
    PetscReal *z_vals;
    PetscMalloc1(numZ_, &z_vals);
    PetscInt *z_indices_src, *z_indices_dest;
    PetscMalloc1(numZ_, &z_indices_src);
    PetscMalloc1(numZ_, &z_indices_dest);
    std::iota(z_indices_src, z_indices_src + numZ_, 0);
    PetscReal *reward_vals;
    PetscMalloc1(numStates_, &reward_vals);
    const PetscInt *feasibleActions;
    ISGetIndices(A_, &feasibleActions);
    const PetscReal *availableResources, *k_vals;
    VecGetArrayRead(B_, &availableResources);
    VecGetArrayRead(k_, &k_vals);

    for(PetscInt i = 0; i < numK_; ++i) {
        for(PetscInt j = 0; j < numZ_; ++j) {
            PetscInt s = ij2s(i, j);
            if(actionInd <= feasibleActions[s]) {
                MatGetValues(P_z_, 1, &j, numZ_, z_indices_src, z_vals);
                std::iota(z_indices_dest, z_indices_dest + numZ_, actionInd * numZ_);
                MatSetValues(transitionProbabilities, 1, &s, numZ_, z_indices_dest, z_vals, INSERT_VALUES);
                reward_vals[s] = std::pow(availableResources[s] - k_vals[actionInd], riskAversionParameter_) / riskAversionParameter_;
            }
            else {
                reward_vals[s] = -1e10;
            }
        }
    }

    VecRestoreArrayRead(k_, &k_vals);
    VecRestoreArrayRead(B_, &availableResources);
    ISRestoreIndices(A_, &feasibleActions);

    PetscInt *stageCostIndices;
    PetscMalloc1(numStates_, &stageCostIndices);
    std::iota(stageCostIndices, stageCostIndices + numStates_, 0);
    VecSetValues(stageCosts, numStates_, stageCostIndices, reward_vals, INSERT_VALUES);
    PetscFree(stageCostIndices);

    PetscFree(z_indices_src);
    PetscFree(z_indices_dest);
    PetscFree(z_vals);
    PetscFree(reward_vals);

    MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(stageCosts);
    MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(stageCosts);

    return 0;
}

PetscErrorCode GrowthModel::extractGreedyPolicy(Vec &V, PetscInt *policy, PetscReal &residualNorm) {

    PetscErrorCode ierr;
    PetscInt localNumStates_ = numStates_;
    const PetscReal *costValues; // stores cost (= g + gamma PV) values for each state
    PetscReal *maxRewardValues; // stores minimum cost values for each state
    PetscMalloc1(localNumStates_, &maxRewardValues);
    std::fill(maxRewardValues, maxRewardValues + localNumStates_, std::numeric_limits<PetscReal>::lowest());


    Mat P;
    Vec g;

    for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
        //LOG("action " + std::to_string(actionInd) + " of " + std::to_string(numActions_) + "...");
        constructFromPolicy(actionInd, P, g); // creates P and g => need to destroy P and g by ourselves
        //LOG("Finished construction of P and g. Calculating g + gamma PV...");
        ierr = MatScale(P, discountFactor_); CHKERRQ(ierr);
        ierr = MatMultAdd(P, V, g, g); CHKERRQ(ierr);

        ierr = VecGetArrayRead(g, &costValues); CHKERRQ(ierr);

        //LOG("Performing minimization for all local states...");
        for (int localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
            if (costValues[localStateInd] > maxRewardValues[localStateInd]) {
                maxRewardValues[localStateInd] = costValues[localStateInd];
                policy[localStateInd] = actionInd;

            }
        }
        ierr = VecRestoreArrayRead(g, &costValues); CHKERRQ(ierr);
        //LOG("Finished minimization for all local states.");
        ierr = MatDestroy(&P); CHKERRQ(ierr);
        ierr = VecDestroy(&g); CHKERRQ(ierr);
    }

    // compute residual: create vector from maxRewardValues, subtract V, take norm
    Vec residual;
    VecDuplicate(V, &residual);
    IS V_indices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, 0, 1, &V_indices);
    const PetscInt *V_indices_arr;
    ISGetIndices(V_indices, &V_indices_arr);
    VecSetValues(residual, localNumStates_, V_indices_arr, maxRewardValues, INSERT_VALUES);
    ISRestoreIndices(V_indices, &V_indices_arr);
    ISDestroy(&V_indices);
    VecAssemblyBegin(residual);
    VecAssemblyEnd(residual);
    VecAXPY(residual, -1.0, V);
    VecNorm(residual, NORM_INFINITY, &residualNorm);
    VecDestroy(&residual);

    ierr = PetscFree(costValues); CHKERRQ(ierr);
    ierr = PetscFree(maxRewardValues); CHKERRQ(ierr);

    return 0;

}




PetscErrorCode GrowthModel::iterativePolicyEvaluation(Mat &jacobian, Vec &stageCosts, Vec &V, KSPContext &ctx) {
    PetscErrorCode ierr;
    MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY);

    // ksp solver
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, jacobian, jacobian); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    //ierr = KSPSetTolerances(ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetConvergenceTest(ksp, &GrowthModel::cvgTest, &ctx, NULL); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, stageCosts, V); CHKERRQ(ierr);
    //ierr = KSPGetIterationNumber(ksp, &iter); CHKERRQ(ierr);
    //LOG("KSP converged after " + std::to_string(iter) + " iterations");

    ierr = KSPGetIterationNumber(ksp, &ctx.kspIterations); CHKERRQ(ierr);
    //LOG("KSP iterations: " + std::to_string(ctx.kspIterations) + " (max: " + std::to_string(ctx.maxIter) + ")");
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    return ierr;
}

// defines the matrix-vector product for the jacobian shell
void GrowthModel::jacobianMultiplication(Mat mat, Vec x, Vec y) {
    JacobianContext *ctx;
    MatShellGetContext(mat, (void **) &ctx); // todo static cast
    MatMult(ctx->P_pi, x, y);
    VecScale(y, -ctx->discountFactor);
    VecAXPY(y, 1.0, x);
}

// creates MPIAIJ matrix and computes jacobian = I - gamma * P_pi
PetscErrorCode GrowthModel::createJacobian(Mat &jacobian, const Mat &transitionProbabilities, JacobianContext &ctx) {
    PetscErrorCode ierr;
    ierr = MatCreateShell(PETSC_COMM_WORLD, localNumStates_, localNumStates_, numStates_, numStates_, &ctx, &jacobian); CHKERRQ(ierr);
    ierr = MatShellSetOperation(jacobian, MATOP_MULT, (void (*)(void)) jacobianMultiplication); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode GrowthModel::inexactPolicyIteration(Vec &V0, IS &policy, Vec &optimalCost) {
    LOG("Entering inexactPolicyIteration");
    jsonWriter_->add_solver_run();

    Vec V;
    VecDuplicate(V0, &V);
    VecCopy(V0, V);

    Mat transitionProbabilities, jacobian;
    Vec stageCosts;
    PetscInt *policyValues;
    PetscMalloc1(localNumStates_, &policyValues);
    PetscReal residualNorm;
    PetscLogDouble startTime, endTime;

    PetscInt PI_iteration= 0;
    for(; PI_iteration < maxIter_PI_; ++PI_iteration) {
        PetscTime(&startTime);
        // compute jacobian wrt new policy
        extractGreedyPolicy(V, policyValues, residualNorm);
        if(residualNorm < atol_PI_) {
            PetscTime(&endTime);
            jsonWriter_->add_iteration_data(PI_iteration, 0, (endTime - startTime) * 1000, residualNorm);
            LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm));
            break;
        }
        constructFromPolicy(policyValues, transitionProbabilities, stageCosts);
        JacobianContext ctxJac = {transitionProbabilities, discountFactor_};
        createJacobian(jacobian, transitionProbabilities, ctxJac);

        // solve linear system
        KSPContext ctx = {maxIter_KSP_, residualNorm * rtol_KSP_, -1};
        iterativePolicyEvaluation(jacobian, stageCosts, V, ctx);
        MatDestroy(&transitionProbabilities);
        MatDestroy(&jacobian); // avoid memory leak
        VecDestroy(&stageCosts);

        PetscTime(&endTime);
        jsonWriter_->add_iteration_data(PI_iteration, ctx.kspIterations, (endTime - startTime) * 1000, residualNorm);
        LOG("Iteration " + std::to_string(PI_iteration) + " residual norm: " + std::to_string(residualNorm));
    }

    if(PI_iteration >= maxIter_PI_) {
        LOG("Warning: maximum number of PI iterations reached. Solution might not be optimal.");
    }

    jsonWriter_->write_to_file("gm_stats.json");

    MatDestroy(&transitionProbabilities);
    MatDestroy(&jacobian);
    VecDestroy(&stageCosts);

    // output results
    VecDuplicate(V, &optimalCost);
    VecCopy(V, optimalCost);
    ISCreateGeneral(PETSC_COMM_WORLD, localNumStates_, policyValues, PETSC_COPY_VALUES, &policy);

    VecDestroy(&V);
    PetscFree(policyValues);

    return 0;
}

PetscErrorCode GrowthModel::cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx) {
    PetscErrorCode ierr;
    PetscReal threshold = static_cast<KSPContext*>(ctx)->threshold;
    PetscReal norm;

    Vec res;
    //ierr = VecDuplicate(ksp->vec_rhs, &res); CHKERRQ(ierr);
    ierr = KSPBuildResidual(ksp, NULL, NULL, &res); CHKERRQ(ierr);
    ierr = VecNorm(res, NORM_INFINITY, &norm); CHKERRQ(ierr);
    ierr = VecDestroy(&res); CHKERRQ(ierr);

    //PetscPrintf(PETSC_COMM_WORLD, "it = %d: residual norm = %f\n", it, norm);

    if(it == 0) *reason = KSP_CONVERGED_ITERATING;
    else if(norm < threshold) *reason = KSP_CONVERGED_RTOL;
    else if(it >= static_cast<KSPContext*>(ctx)->maxIter) *reason = KSP_DIVERGED_ITS;
    else *reason = KSP_CONVERGED_ITERATING;

    return 0;
}