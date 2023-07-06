//
// Created by robin on 06.07.23.
//

#include "InfectiousDiseaseModel.h"
#include "../utils/Logger.h"
#include <boost/math/distributions/binomial.hpp>

InfectiousDiseaseModel::InfectiousDiseaseModel() {}

InfectiousDiseaseModel::~InfectiousDiseaseModel() {}

// 1d action index to 2d action index
std::pair<PetscInt, PetscInt> InfectiousDiseaseModel::a2ij(PetscInt a) const {
    return std::make_pair(a % numA1_, a / numA1_);
}


PetscReal InfectiousDiseaseModel::ch(PetscInt state) const {
    return std::pow(static_cast<double>(populationSize_ - state), 1.1); // #infections^1.1
}


PetscReal InfectiousDiseaseModel::g(PetscInt state, PetscInt action) const {
    auto [a1, a2] = a2ij(action);
    PetscReal cf = cf_a1_[a1] + cf_a2_[a2];
    PetscReal cq = cq_a1_[a1] * cq_a2_[a2];
    return weights_[0] * cf - weights_[1] * cq + weights_[2] * ch(state);
}

PetscReal InfectiousDiseaseModel::q(PetscInt state, PetscInt action) const {
    auto [a1, a2] = a2ij(action);
    PetscReal beta = 1.0 - 1.0 * state / populationSize_;
    return 1 - std::exp(-beta * r_[a1] * lambda_[a2]);
}


PetscErrorCode InfectiousDiseaseModel::generateStageCosts() {

    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetType(stageCostMatrix_, MATDENSE);
    MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_);
    //MatMPIAIJSetPreallocation(stageCostMatrix_, PETSC_DECIDE, NULL, PETSC_DECIDE, NULL);
    MatSetUp(stageCostMatrix_);

    PetscInt start, end;
    MatGetOwnershipRange(stageCostMatrix_, &start, &end);
    for(PetscInt state = start; state < end; ++state) {
        for(PetscInt action = 0; action < numActions_; ++action) {
            MatSetValue(stageCostMatrix_, state, action, g(state, action), INSERT_VALUES);
        }
    }
    MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY);

    return 0;
}

PetscErrorCode InfectiousDiseaseModel::generateTransitionProbabilities() {
    PetscErrorCode ierr;
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetType(transitionProbabilityTensor_, MATMPIAIJ);
    MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, PETSC_DECIDE, numStates_*numActions_, numStates_);
    MatSetOption(transitionProbabilityTensor_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
    MatSetUp(transitionProbabilityTensor_); // todo preallocation

    PetscInt start, end;
    MatGetOwnershipRange(transitionProbabilityTensor_, &start, &end);
    start /= numActions_;
    end /= numActions_;
    PetscInt nextState;
    for(PetscInt stateInd = 0; stateInd < end; ++stateInd) {
        //PetscInt stateInd = startState + stateInd;
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            if(stateInd == populationSize_) {
                ierr = MatSetValue(transitionProbabilityTensor_, stateInd*numActions_ + actionInd, populationSize_, 1.0, INSERT_VALUES); CHKERRQ(ierr); // absorbing state
                continue;
            }
            boost::math::binomial_distribution<PetscReal> binom(stateInd, q(stateInd, actionInd));
            PetscPrintf(PETSC_COMM_WORLD, "  q(%d, %d) = %f\n", stateInd, actionInd, q(stateInd, actionInd));
            for(PetscInt i = 0; i <= stateInd; ++i) {
                nextState = populationSize_ - i;
                PetscReal prob = boost::math::pdf(binom, i);
                ierr = MatSetValue(transitionProbabilityTensor_, stateInd*numActions_ + actionInd, nextState, prob, INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }
    MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);


    return 0;
}



PetscErrorCode InfectiousDiseaseModel::setValuesFromOptions() {
    PetscErrorCode ierr;
    PetscBool flg;

    // population size
    ierr = PetscOptionsGetInt(NULL, NULL, "-populationSize", &populationSize_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "population size not specified. Use -populationSize <int>.");
    }
    jsonWriter_->add_data("populationSize", populationSize_);

    // discount Factor
    ierr = PetscOptionsGetReal(NULL, NULL, "-discountFactor", &discountFactor_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "discount factor not specified. Use -discountFactor <double>.");
    }
    jsonWriter_->add_data("discountFactor", discountFactor_);

    // hygiene measures (HM), 5 values
    PetscInt count = numA1_;
    ierr = PetscOptionsGetRealArray(NULL, NULL, "-HM", r_, &count, &flg); CHKERRQ(ierr);
    if(count != numA1_ || !flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "hygiene measures not specified. Use -HM <double> <double> <double> <double> <double>.");
    }
    jsonWriter_->add_data("HM", std::vector<PetscReal>(r_, r_ + numA1_));

    // social distancing (SD), 4 values
    count = numA2_;
    ierr = PetscOptionsGetIntArray(NULL, NULL, "-SD", lambda_, &count, &flg); CHKERRQ(ierr);
    if(count != numA2_ || !flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "social distancing not specified. Use -SD <double> <double> <double> <double>.");
    }
    jsonWriter_->add_data("SD", std::vector<PetscReal>(lambda_, lambda_ + numA2_));

    // financial cost of hygiene measures, 5 values
    count = numA1_;
    ierr = PetscOptionsGetRealArray(NULL, NULL, "-HM-cf", cf_a1_, &count, &flg); CHKERRQ(ierr);
    if(count != numA1_ || !flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "financial cost of hygiene measures not specified. Use -HM-cf <double> <double> <double> <double> <double>.");
    }
    jsonWriter_->add_data("HM-cf", std::vector<PetscReal>(cf_a1_, cf_a1_ + numA1_));

    // financial cost of social distancing, 4 values
    count = numA2_;
    ierr = PetscOptionsGetRealArray(NULL, NULL, "-SD-cf", cf_a2_, &count, &flg); CHKERRQ(ierr);
    if(count != numA2_ || !flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "financial cost of social distancing not specified. Use -SD-cf <double> <double> <double> <double>.");
    }
    jsonWriter_->add_data("SD-cf", std::vector<PetscReal>(cf_a2_, cf_a2_ + numA2_));

    // quality of life of hygiene measures, 5 values
    count = numA1_;
    ierr = PetscOptionsGetRealArray(NULL, NULL, "-HM-cq", cq_a1_, &count, &flg); CHKERRQ(ierr);
    if(count != numA1_ || !flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "quality of life of hygiene measures not specified. Use -HM-cq <double> <double> <double> <double> <double>.");
    }
    jsonWriter_->add_data("HM-cq", std::vector<PetscReal>(cq_a1_, cq_a1_ + numA1_));

    // quality of life of social distancing, 4 values
    count = numA2_;
    ierr = PetscOptionsGetRealArray(NULL, NULL, "-SD-cq", cq_a2_, &count, &flg); CHKERRQ(ierr);
    if(count != numA2_ || !flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "quality of life of social distancing not specified. Use -SD-cq <double> <double> <double> <double>.");
    }
    jsonWriter_->add_data("SD-cq", std::vector<PetscReal>(cq_a2_, cq_a2_ + numA2_));

    // weights, 3 values
    count = 3;
    ierr = PetscOptionsGetRealArray(NULL, NULL, "-weights", weights_, &count, &flg); CHKERRQ(ierr);
    if(count != 3 || !flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "weights not specified. Use -weights <financial cost> <quality of life cost> <health cost>. (doubles)");
    }

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


    numStates_ = populationSize_ + 1;
    numActions_ = numA1_ * numA2_;
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    LOG("owns " + std::to_string(localNumStates_) + " states.");

    return 0;
}