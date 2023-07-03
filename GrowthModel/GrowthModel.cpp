//
// Created by robin on 28.06.23.
//

#include "GrowthModel.h"


GrowthModel::GrowthModel() {
    P_z_ = nullptr;
    z_ = nullptr;
}

GrowthModel::~GrowthModel() {
    MatDestroy(&P_z_);
    VecDestroy(&z_);
    VecDestroy(&k_);
    VecDestroy(&B_);
    ISDestroy(&A_);
}

PetscErrorCode GrowthModel::generateKInterval() {
    PetscReal z_vals[2];
    PetscInt z_indices[2] = {0, numZ_ - 1};
    VecGetValues(z_, 2, z_indices, z_vals);
    //VecCreateSeq(PETSC_COMM_SELF, numK_, &k_);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, numK_, &k_);
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
    //VecCreateSeq(PETSC_COMM_SELF, numK_ * numZ_, &B_);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, numK_ * numZ_, &B_);

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


PetscErrorCode GrowthModel::constructTransitionProbabilitiesRewards() {
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetFromOptions(transitionProbabilityTensor_);
    MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_);
    PetscInt numLocalCols;
    MatGetLocalSize(transitionProbabilityTensor_, NULL, &numLocalCols);
    MatMPIAIJSetPreallocation(transitionProbabilityTensor_, std::min(numZ_, numLocalCols), NULL, numZ_, NULL); // allocate numZ_ entries both in diagonal and off-diagonal blocks
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);

    MatCreateDense(PETSC_COMM_WORLD, localNumStates_, PETSC_DECIDE, numStates_, numActions_, NULL, &stageCostMatrix_);

    PetscInt *srcIndices;
    PetscMalloc1(numZ_, &srcIndices);
    std::iota(srcIndices, srcIndices + numZ_, 0); // destIndices for extracting values from P_z_
    PetscInt *destIndices;
    PetscMalloc1(numZ_, &destIndices);
    PetscReal *zValues;
    PetscMalloc1(numZ_, &zValues);

    const PetscInt *AValues;
    ISGetIndices(A_, &AValues);
    const PetscReal *BValues, *kValues;
    VecGetArrayRead(B_, &BValues);
    VecGetArrayRead(k_, &kValues);

    // fill values
    //PetscInt k_start = P_start_ / numZ_;
    PetscInt P_pi_start = P_start_ / numActions_;
    //PetscInt i, j, a;
    for(PetscInt stateInd = P_pi_start; stateInd < P_pi_start + localNumStates_; ++stateInd) {
        //i = stateInd / numZ_; // k index
        //j = stateInd % numZ_; // z index
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            PetscInt localStateIndex = stateInd - P_pi_start;
            PetscInt srcRow = stateInd % numZ_; // z index
            if(actionInd <= AValues[localStateIndex]) { // action must be feasible
                // transition probabilities
                std::iota(destIndices, destIndices + numZ_, actionInd * numZ_);
                MatGetValues(P_z_, 1, &srcRow, numZ_, srcIndices, zValues);
                PetscInt destRow = stateInd * numActions_ + actionInd;
                MatSetValues(transitionProbabilityTensor_, 1, &destRow, numZ_, destIndices, zValues, INSERT_VALUES);
                // reward
                PetscReal reward = std::pow(BValues[localStateIndex] - kValues[actionInd], riskAversionParameter_) / riskAversionParameter_;
                MatSetValue(stageCostMatrix_, stateInd, actionInd, reward, INSERT_VALUES);
            }
            else {
                // set reward to -inf for infeasible actions
                MatSetValue(stageCostMatrix_, stateInd, actionInd, std::numeric_limits<PetscReal>::lowest(), INSERT_VALUES);
            }
        }
    }

    MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY);

    ISRestoreIndices(A_, &AValues);
    VecRestoreArrayRead(B_, &BValues);
    VecRestoreArrayRead(k_, &kValues);

    PetscFree(srcIndices);
    PetscFree(destIndices);
    PetscFree(zValues);


    return 0;
}

PetscErrorCode GrowthModel::setValuesFromOptions() {
    PetscErrorCode ierr;
    PetscBool flg;

    ierr = PetscOptionsGetInt(NULL, NULL, "-numK", &numK_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of capital stock states not specified. Use -numK <int>.");
    }
    jsonWriter_->add_data("numK", numK_);

    ierr = PetscOptionsGetInt(NULL, NULL, "-numZ", &numZ_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of productivity states not specified. Use -numZ <int>.");
    }
    jsonWriter_->add_data("numZ", numZ_);

    ierr = PetscOptionsGetReal(NULL, NULL, "-discountFactor", &discountFactor_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Discount factor not specified. Use -discountFactor <double>.");
    }
    jsonWriter_->add_data("discountFactor", discountFactor_);

    ierr = PetscOptionsGetReal(NULL, NULL, "-riskAversion", &riskAversionParameter_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Risk aversion parameter not specified. Use -riskAversion <double>.");
    }
    jsonWriter_->add_data("riskAversionParameter", riskAversionParameter_);

    ierr = PetscOptionsGetInt(NULL, NULL, "-maxIter_PI", &maxIter_PI_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Maximum number of policy iterations not specified. Use -maxIter_PI <int>.");
    }
    jsonWriter_->add_data("maxIter_PI", maxIter_PI_);

    ierr = PetscOptionsGetInt(NULL, NULL, "-maxIter_KSP", &maxIter_KSP_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Maximum number of KSP iterations not specified. Use -maxIter_KSP <int>.");
    }
    jsonWriter_->add_data("maxIter_KSP", maxIter_KSP_);

    ierr = PetscOptionsGetInt(NULL, NULL, "-numPIRuns", &numPIRuns_, &flg); CHKERRQ(ierr);
    if(!flg) {
        //SETERRQ(PETSC_COMM_WORLD, 1, "Maximum number of KSP iterations not specified. Use -maxIter_KSP <int>.");
        LOG("Number of PI runs for benchmarking not specified. Use -numPIRuns <int>. Default: 1");
        numPIRuns_ = 1;
    }
    jsonWriter_->add_data("numPIRuns", numPIRuns_);

    ierr = PetscOptionsGetReal(NULL, NULL, "-rtol_KSP", &rtol_KSP_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Relative tolerance for KSP not specified. Use -rtol_KSP <double>.");
    }
    jsonWriter_->add_data("rtol_KSP", rtol_KSP_);

    ierr = PetscOptionsGetReal(NULL, NULL, "-atol_PI", &atol_PI_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Absolute tolerance for policy iteration not specified. Use -atol_PI <double>.");
    }
    jsonWriter_->add_data("atol_PI", atol_PI_);

    ierr = PetscOptionsGetString(NULL, NULL, "-file_policy", file_policy_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        LOG("Filename for policy not specified. Optimal policy will not be written to file.");
        file_policy_[0] = '\0';
    }

    ierr = PetscOptionsGetString(NULL, NULL, "-file_cost", file_cost_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        LOG("Filename for cost not specified. Optimal cost will not be written to file.");
        file_cost_[0] = '\0';
    }

    ierr = PetscOptionsGetString(NULL, NULL, "-file_stats", file_stats_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Filename for statistics not specified. Use -file_stats <string>. (max length: 4096 chars");
    }

    PetscChar inputMode[20];
    ierr = PetscOptionsGetString(NULL, NULL, "-mode", inputMode, 20, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Input mode not specified. Use -mode MINCOST or MAXREWARD.");
    }
    if (strcmp(inputMode, "MINCOST") == 0) {
        mode_ = MINCOST;
        jsonWriter_->add_data("mode", "MINCOST");
    } else if (strcmp(inputMode, "MAXREWARD") == 0) {
        mode_ = MAXREWARD;
        jsonWriter_->add_data("mode", "MAXREWARD");
    } else {
        SETERRQ(PETSC_COMM_WORLD, 1, "Input mode not recognized. Use -mode MINCOST or MAXREWARD.");
    }


    // set derived parameters
    numStates_ = numK_ * numZ_;
    numActions_ = numK_;
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    LOG("owns " + std::to_string(localNumStates_) + " states.");

    return 0;
}


#if 0
// user must destroy P and g by himself. Function will create them. [used in inexactPolicyIteration]
PetscErrorCode GrowthModel::constructFromPolicy(PetscInt *policy, Mat &transitionProbabilities, Vec &stageCosts) {
    // compute where local ownership of new P_pi matrix starts
    PetscInt P_pi_start; // start of ownership of new matrix (to be created)
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_pi_start, NULL);
    P_pi_start /= numActions_;

    // allocate memory for values
    PetscInt *P_rowIndexValues;
    PetscMalloc1(localNumStates_, &P_rowIndexValues);
    PetscReal *g_pi_values;
    PetscMalloc1(localNumStates_, &g_pi_values);
    const PetscReal *kValues;
    VecGetArrayRead(k_, &kValues);


    // compute global row indices for P and get values for g_pi
    PetscInt g_srcRow, actionInd;
    for(PetscInt localStateInd = 0; localStateInd < localNumStates_; ++localStateInd) {
        actionInd = policy[localStateInd];
        // compute values for row index set
        P_rowIndexValues[localStateInd] = P_start_ + localStateInd * numActions_ + actionInd;
        // get values for stageCosts
        g_srcRow  = g_start_ + localStateInd;
        //MatGetValue(stageCostMatrix_, g_srcRow, actionInd, &g_pi_values[localStateInd]);
    }

    // generate index sets
    IS P_rowIndices;
    ISCreateGeneral(PETSC_COMM_WORLD, localNumStates_, P_rowIndexValues, PETSC_COPY_VALUES, &P_rowIndices);
    IS g_pi_rowIndices;
    ISCreateStride(PETSC_COMM_WORLD, localNumStates_, g_start_, 1, &g_pi_rowIndices);

    //LOG("Creating transitionProbabilities submatrix");
    MatCreateSubMatrix(transitionProbabilityTensor_, P_rowIndices, NULL, MAT_INITIAL_MATRIX, &transitionProbabilities);

    //LOG("Creating stageCosts vector");
    VecCreateMPI(PETSC_COMM_WORLD, localNumStates_, numStates_, &stageCosts);
    const PetscInt *g_pi_rowIndexValues; // global indices
    ISGetIndices(g_pi_rowIndices, &g_pi_rowIndexValues);
    VecSetValues(stageCosts, localNumStates_, g_pi_rowIndexValues, g_pi_values, INSERT_VALUES);
    ISRestoreIndices(g_pi_rowIndices, &g_pi_rowIndexValues);

    //LOG("Assembling transitionProbabilities and stageCosts");
    MatAssemblyBegin(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(stageCosts);
    MatAssemblyEnd(transitionProbabilities, MAT_FINAL_ASSEMBLY);
    VecAssemblyEnd(stageCosts);

    ISDestroy(&P_rowIndices);
    ISDestroy(&g_pi_rowIndices);
    PetscFree(P_rowIndexValues);
    PetscFree(g_pi_values);
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

#endif


