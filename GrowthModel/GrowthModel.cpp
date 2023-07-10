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
    //VecDestroy(&k_);
    //VecDestroy(&B_);
    //ISDestroy(&A_);
}


PetscErrorCode GrowthModel::generateKInterval() {
    VecCreate(PETSC_COMM_SELF, &k_);
    VecSetType(k_, VECSEQ);
    VecSetSizes(k_, PETSC_DECIDE, numK_);
    VecSetFromOptions(k_);

    PetscReal *k_vals;
    PetscMalloc1(numK_, &k_vals);
    PetscInt *k_indices;
    PetscMalloc1(numK_, &k_indices);

    PetscReal z_vals[2];
    PetscInt z_indices[2] = {0, numZ_ - 1}; // get first and last element of z
    VecGetValues(z_, 2, z_indices, z_vals);

    PetscReal k_star_z1 = std::pow(discountFactor_ * rho_ * z_vals[0] / (1 - discountFactor_), 1 / (1 - rho_));
    PetscReal k_star_z2 = std::pow(discountFactor_ * rho_ * z_vals[1] / (1 - discountFactor_), 1 / (1 - rho_));
    PetscReal k_min = k_star_z1 - 0.1 * (k_star_z2 - k_star_z1);
    PetscReal k_max = k_star_z2 + 0.1 * (k_star_z2 - k_star_z1);
    PetscReal k_incr = (k_max - k_min) / (numK_ - 1);
    PetscReal val = k_min - k_incr;
    std::generate(k_vals, k_vals + numK_, [&val, k_incr](){ return val += k_incr; });
    std::iota(k_indices, k_indices + numK_, 0);
    VecSetValues(k_, numK_, k_indices, k_vals, INSERT_VALUES);
    VecAssemblyBegin(k_);
    VecAssemblyEnd(k_);

    PetscFree(k_vals);
    PetscFree(k_indices);
    return 0;
}

PetscErrorCode GrowthModel::constructTransitionProbabilitiesRewards() {
    // create transition probabilities tensor
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetType(transitionProbabilityTensor_, MATMPIAIJ);
    MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, PETSC_DECIDE, numStates_*numActions_, numStates_);
    PetscInt numLocalCols;
    MatGetLocalSize(transitionProbabilityTensor_, NULL, &numLocalCols);
    MatMPIAIJSetPreallocation(transitionProbabilityTensor_, std::min(numZ_, numLocalCols), NULL, numZ_, NULL); // allocate numZ_ entries both in diagonal and off-diagonal blocks
    MatSetFromOptions(transitionProbabilityTensor_);
    MatSetUp(transitionProbabilityTensor_);
    MatSetOption(transitionProbabilityTensor_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE); // todo: true -> error if new nonzero is inserted; false -> silently overwrite
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);

    // create stage costs (reward) matrix
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetType(stageCostMatrix_, MATDENSE);
    MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_);
    MatSetUp(stageCostMatrix_);
    MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);

    LOG("P start: " + std::to_string(P_start_) + ", P end: " + std::to_string(P_end_));
    LOG("g start: " + std::to_string(g_start_) + ", g end: " + std::to_string(g_end_));

    const PetscReal *z_vals, *k_vals;
    VecGetArrayRead(z_, &z_vals);
    VecGetArrayRead(k_, &k_vals);
    PetscInt *srcIndices, *destIndices;
    PetscMalloc1(numZ_, &srcIndices);
    PetscMalloc1(numZ_, &destIndices);
    std::iota(srcIndices, srcIndices + numZ_, 0);
    PetscReal *Pz_vals;
    PetscMalloc1(numZ_, &Pz_vals);


    for(PetscInt stateInd = g_start_; stateInd < g_end_; ++stateInd) {
        auto [i, j] = s2ij(stateInd);
        PetscReal availableResources = z_vals[j] * std::pow(k_vals[i], rho_) + k_vals[i];
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            if (availableResources > k_vals[actionInd]) { // feasible action
                // reward
                PetscReal reward = std::pow(availableResources - k_vals[actionInd], riskAversionParameter_) / riskAversionParameter_;
                MatSetValue(stageCostMatrix_, stateInd, actionInd, reward, INSERT_VALUES);
                // transition probabilities
                PetscInt destRow = stateInd * numActions_ + actionInd;
                std::iota(destIndices, destIndices + numZ_, actionInd * numZ_);
                MatGetValues(P_z_, 1, &j, numZ_, srcIndices, Pz_vals);
                MatSetValues(transitionProbabilityTensor_, 1, &destRow, numZ_, destIndices, Pz_vals, INSERT_VALUES);
            }
            else {
                // reward is -inf, probability is 0
                MatSetValue(stageCostMatrix_, stateInd, actionInd, std::numeric_limits<PetscReal>::lowest(), INSERT_VALUES);
            }
        }
    }

    MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);

    VecRestoreArrayRead(z_, &z_vals);
    VecRestoreArrayRead(k_, &k_vals);
    PetscFree(srcIndices);
    PetscFree(destIndices);
    PetscFree(Pz_vals);

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
    localNumK_ = (rank_ < numK_ % size_) ? numK_ / size_ + 1 : numK_ / size_; // first numK_ % size_ ranks get one more state
    //LOG("owns " + std::to_string(localNumK_) + " capital values.");
    localNumStates_ = localNumK_ * numZ_;
    //localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    LOG("owns " + std::to_string(localNumK_) + "/" + std::to_string(numK_) + " capital values.");
    LOG("owns " + std::to_string(localNumStates_) + "/" + std::to_string(numStates_) + " states.");

    return 0;
}
