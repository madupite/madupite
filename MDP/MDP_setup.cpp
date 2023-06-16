//
// Created by robin on 05.06.23.
//

#include "MDP.h"
#include "../utils/Logger.h"
#include <mpi.h>

MDP::MDP() {
    // MPI parallelization initialization
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank_);
    MPI_Comm_size(PETSC_COMM_WORLD, &size_);

    jsonWriter_ = new JsonWriter(rank_);
    transitionProbabilityTensor_ = nullptr;
    stageCostMatrix_ = nullptr;

    setValuesFromOptions();

    jsonWriter_->add_data("numRanks", size_);

    // distribute states to ranks
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file
    LOG("owns " + std::to_string(localNumStates_) + " states.");

    if(file_P_[0] != '\0' && file_g_[0] != '\0') {
        loadFromBinaryFile(file_P_, file_g_);
    }
}

MDP::~MDP() {
    MatDestroy(&transitionProbabilityTensor_);
    MatDestroy(&stageCostMatrix_);
    //delete jsonWriter_; // todo fix this (double free or corruption error)
}

PetscErrorCode MDP::setValuesFromOptions() {
    PetscErrorCode ierr;
    PetscBool flg;

    ierr = PetscOptionsGetInt(NULL, NULL, "-states", &numStates_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of states not specified. Use -states <int>.");
    }
    jsonWriter_->add_data("numStates", numStates_);
    ierr = PetscOptionsGetInt(NULL, NULL, "-actions", &numActions_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of actions not specified. Use -actions <int>.");
    }
    jsonWriter_->add_data("numActions", numActions_);
    ierr = PetscOptionsGetReal(NULL, NULL, "-discountFactor", &discountFactor_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Discount factor not specified. Use -discountFactor <double>.");
    }
    jsonWriter_->add_data("discountFactor", discountFactor_);
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
    ierr = PetscOptionsGetString(NULL, NULL, "-file_P", file_P_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Filename for transition probability tensor not specified. Use -file_P <string>. (max length: 4096 chars");
    }
    ierr = PetscOptionsGetString(NULL, NULL, "-file_g", file_g_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Filename for stage cost matrix not specified. Use -file_g <string>. (max length: 4096 chars");
    }
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
    return 0;
}

PetscErrorCode MDP::loadFromBinaryFile(std::string filename_P, std::string filename_g) {
    LOG("Loading MDP from binary file: " + filename_P + ", " + filename_g);
    PetscErrorCode ierr = 0;
    PetscViewer viewer;

    // load transition probability tensor
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetFromOptions(transitionProbabilityTensor_);
    ierr = MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, PETSC_DECIDE, PETSC_DECIDE, numStates_); CHKERRQ(ierr);
    MatSetUp(transitionProbabilityTensor_);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(viewer, FILE_MODE_READ);
    PetscViewerFileSetName(viewer, filename_P.c_str());
    MatLoad(transitionProbabilityTensor_, viewer);
    PetscViewerDestroy(&viewer);
    // PetscViewerBinaryOpen?

    // load stage cost matrix
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetFromOptions(stageCostMatrix_);
    ierr = MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, PETSC_DECIDE, numActions_); CHKERRQ(ierr);
    MatSetUp(stageCostMatrix_);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
    PetscViewerSetType(viewer, PETSCVIEWERBINARY);
    PetscViewerFileSetMode(viewer, FILE_MODE_READ);
    PetscViewerFileSetName(viewer, filename_g.c_str());
    MatLoad(stageCostMatrix_, viewer);
    PetscViewerDestroy(&viewer);

    // Information about distribution on processes
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
    MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
    LOG("owns rows " + std::to_string(P_start_) + " to " + std::to_string(P_end_) + " of P.");
    LOG("owns rows " + std::to_string(g_start_) + " to " + std::to_string(g_end_) + " of g.");

    return ierr;
}

// gathers values on rank 0 and writes to file
PetscErrorCode MDP::writeResultCost(const Vec &optimalCost) {

    /*if(rank_ == 0) {
        PetscReal *values = new PetscReal[numStates_];
    }
    else {
        PetscReal *values = new PetscReal[localNumStates_];
    }*/

    VecScatter ctx;
    Vec MPIVec;
    if(rank_ == 0) {
        VecCreateSeq(PETSC_COMM_SELF, numStates_, &MPIVec);
    }
    else {
        VecCreateSeq(PETSC_COMM_SELF, localNumStates_, &MPIVec);
    }

    VecScatterCreateToAll(optimalCost, &ctx, &MPIVec);
    VecScatterBegin(ctx, optimalCost, MPIVec, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, optimalCost, MPIVec, INSERT_VALUES, SCATTER_FORWARD);

    const PetscReal *values;
    VecGetArrayRead(MPIVec, &values);

    // Rank 0 writes to file
    if(rank_ == 0) {
        std::ofstream out(file_cost_);
        for(PetscInt i = 0; i < numStates_; ++i) {
            out << values[i] << "\n";
        }
        out.close();
    }

    VecRestoreArrayRead(MPIVec, &values);

    VecScatterDestroy(&ctx);
    VecDestroy(&MPIVec);

    return 0;
}

// gathers values on rank 0 and writes to file
PetscErrorCode MDP::writeResultPolicy(const IS &optimalPolicy) {
    const PetscInt *indices;
    ISGetIndices(optimalPolicy, &indices);
    PetscInt localSize = localNumStates_;
    PetscInt *allIndices = NULL;
    PetscInt *recvcounts = NULL;
    PetscInt *displs = NULL;
    if(rank_ == 0) {
        allIndices = new PetscInt[numStates_];
        recvcounts = new PetscInt[size_];
        displs = new PetscInt[size_];
    }
    MPI_Gather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    if(rank_ == 0) {
        displs[0] = 0;
        for(PetscInt i = 1; i < size_; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    MPI_Gatherv(indices, localSize, MPI_INT, allIndices, recvcounts, displs, MPI_INT, 0, PETSC_COMM_WORLD);

    // Rank 0 writes to file
    if(rank_ == 0) {
        std::ofstream out(file_policy_);
        for(PetscInt i = 0; i < numStates_; ++i) {
            out << allIndices[i] << "\n";
        }
        out.close();

        delete[] allIndices;
        delete[] recvcounts;
        delete[] displs;
    }
}
