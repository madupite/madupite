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
    jsonWriter_->add_data("numRanks", size_);
    transitionProbabilityTensor_ = nullptr;
    stageCostMatrix_ = nullptr;
    costMatrix_ = nullptr;
    costVector_ = nullptr;

    Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file
}

MDP::~MDP() {
    MatDestroy(&transitionProbabilityTensor_);
    MatDestroy(&stageCostMatrix_);
    MatDestroy(&costMatrix_);
    VecDestroy(&costVector_);
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
    ierr = PetscOptionsGetString(NULL, NULL, "-file_P", file_P_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        //SETERRQ(PETSC_COMM_WORLD, 1, "Filename for transition probability tensor not specified. Use -file_P <string>. (max length: 4096 chars");
        LOG("Warning: Filename for transition probability tensor not specified. Use -file_P <string>. (max length: 4096 chars");
        file_P_[0] = '\0';
    }
    ierr = PetscOptionsGetString(NULL, NULL, "-file_g", file_g_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        //SETERRQ(PETSC_COMM_WORLD, 1, "Filename for stage cost matrix not specified. Use -file_g <string>. (max length: 4096 chars");
        LOG("Warning: Filename for stage cost matrix not specified. Use -file_g <string>. (max length: 4096 chars");
        file_g_[0] = '\0';
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

    // set local number of states (for this rank)
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    LOG("owns " + std::to_string(localNumStates_) + " states.");

    return 0;
}

PetscErrorCode MDP::setOption(const char *option, const char *value) {
    PetscErrorCode ierr;
    ierr = PetscOptionsSetValue(NULL, option, value); CHKERRQ(ierr);
    ierr = setValuesFromOptions(); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode MDP::loadFromBinaryFile(std::string filename_P, std::string filename_g) {
    LOG("Loading MDP from binary file: " + filename_P + ", " + filename_g);
    PetscErrorCode ierr = 0;
    PetscViewer viewer;

    // load transition probability tensor
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetFromOptions(transitionProbabilityTensor_);
    ierr = MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, localNumStates_, numStates_*numActions_, numStates_); CHKERRQ(ierr);
    MatSetUp(transitionProbabilityTensor_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename_P.c_str(), FILE_MODE_READ, &viewer);
    MatLoad(transitionProbabilityTensor_, viewer);
    PetscViewerDestroy(&viewer);

    // load stage cost matrix
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetFromOptions(stageCostMatrix_);
    ierr = MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_); CHKERRQ(ierr);
    MatSetUp(stageCostMatrix_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename_g.c_str(), FILE_MODE_READ, &viewer);
    MatLoad(stageCostMatrix_, viewer);
    PetscViewerDestroy(&viewer);

    // convert stage cost matrix to dense matrix
    MatConvert(stageCostMatrix_, MATDENSE, MAT_INPLACE_MATRIX, &stageCostMatrix_);

    // Information about distribution on processes
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
    MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
    LOG("owns rows " + std::to_string(P_start_) + " to " + std::to_string(P_end_) + " of P.");
    LOG("owns rows " + std::to_string(g_start_) + " to " + std::to_string(g_end_) + " of g.");

    return ierr;
}

PetscErrorCode MDP::writeVec(const Vec &vec, const char *filename) {
    PetscErrorCode ierr;
    VecScatter ctx;
    Vec MPIVec;
    PetscInt size;

    ierr = VecGetSize(vec, &size); CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF, size, &MPIVec); CHKERRQ(ierr);

    ierr = VecScatterCreateToAll(vec, &ctx, &MPIVec); CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx, vec, MPIVec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx, vec, MPIVec, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    const PetscScalar *values;
    ierr = VecGetArrayRead(MPIVec, &values); CHKERRQ(ierr);

    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    // Rank 0 writes to file
    if(rank == 0) {
        std::ofstream out(filename);
        for(PetscInt i = 0; i < size; ++i) {
            out << values[i] << "\n";
        }
        out.close();
    }

    ierr = VecRestoreArrayRead(MPIVec, &values); CHKERRQ(ierr);

    ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);
    ierr = VecDestroy(&MPIVec); CHKERRQ(ierr);

    return 0;
}

PetscErrorCode MDP::writeIS(const IS &is, const char *filename) {
    PetscErrorCode ierr;
    const PetscInt *indices;
    PetscInt localSize;
    PetscInt size;
    PetscInt *allIndices = NULL;
    PetscInt *recvcounts = NULL;
    PetscInt *displs = NULL;

    ierr = ISGetLocalSize(is, &localSize); CHKERRQ(ierr);
    ierr = ISGetSize(is, &size); CHKERRQ(ierr);

    ierr = ISGetIndices(is, &indices); CHKERRQ(ierr);

    PetscMPIInt rank;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    if(rank == 0) {
        ierr = PetscMalloc1(size, &allIndices); CHKERRQ(ierr);
        ierr = PetscMalloc1(size, &recvcounts); CHKERRQ(ierr);
        ierr = PetscMalloc1(size, &displs); CHKERRQ(ierr);

        ierr = MPI_Gather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

        displs[0] = 0;
        for(PetscInt i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }

        ierr = MPI_Gatherv(indices, localSize, MPI_INT, allIndices, recvcounts, displs, MPI_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

        // Rank 0 writes to file
        std::ofstream out(filename);
        for(PetscInt i = 0; i < size; ++i) {
            out << allIndices[i] << "\n";
        }
        out.close();

        ierr = PetscFree(allIndices); CHKERRQ(ierr);
        ierr = PetscFree(recvcounts); CHKERRQ(ierr);
        ierr = PetscFree(displs); CHKERRQ(ierr);
    }
    else {
        ierr = MPI_Gather(&localSize, 1, MPI_INT, NULL, 0, MPI_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = MPI_Gatherv(indices, localSize, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
    }

    ierr = ISRestoreIndices(is, &indices); CHKERRQ(ierr);

    return 0;
}
