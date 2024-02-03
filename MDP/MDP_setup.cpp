//
// Created by robin on 05.06.23.
//

#include "MDP.h"
// #include "../utils/Logger.h"
#include <mpi.h>
#include <string>
#include<iostream>

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

    // Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    // Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file

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

    ierr = PetscOptionsGetInt(NULL, NULL, "-numStates", &numStates_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of states not specified. Use -numStates <int>.");
    }
    else { // set local num states here if numStates_ is set (e.g. not the case for loading from file)
        localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    }
    jsonWriter_->add_data("numStates", numStates_);
    ierr = PetscOptionsGetInt(NULL, NULL, "-numActions", &numActions_, &flg); CHKERRQ(ierr);
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of actions not specified. Use -numActions <int>.");
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
        // LOG("Number of PI runs for benchmarking not specified. Use -numPIRuns <int>. Default: 1");
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
        // LOG("Warning: Filename for transition probability tensor not specified. Use -file_P <string>. (max length: 4096 chars");
        file_P_[0] = '\0';
    }
    ierr = PetscOptionsGetString(NULL, NULL, "-file_g", file_g_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        //SETERRQ(PETSC_COMM_WORLD, 1, "Filename for stage cost matrix not specified. Use -file_g <string>. (max length: 4096 chars");
        // LOG("Warning: Filename for stage cost matrix not specified. Use -file_g <string>. (max length: 4096 chars");
        file_g_[0] = '\0';
    }
    ierr = PetscOptionsGetString(NULL, NULL, "-file_policy", file_policy_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        // LOG("Filename for policy not specified. Optimal policy will not be written to file.");
        file_policy_[0] = '\0';
    }
    ierr = PetscOptionsGetString(NULL, NULL, "-file_cost", file_cost_, PETSC_MAX_PATH_LEN, &flg); CHKERRQ(ierr);
    if(!flg) {
        // LOG("Filename for cost not specified. Optimal cost will not be written to file.");
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
    // localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    // LOG("owns " + std::to_string(localNumStates_) + " states.");

    return 0;
}

PetscErrorCode MDP::setOption(const char *option, const char *value) {
    // todo: should only be possible for:
    // -discountFactor, -maxIter_PI, -maxIter_KSP, -numPIRuns, -rtol_KSP, -atol_PI, -file_policy, -file_cost, -file_stats, -mode
    PetscErrorCode ierr;
    ierr = PetscOptionsSetValue(NULL, option, value); CHKERRQ(ierr);
    // ierr = setValuesFromOptions(); CHKERRQ(ierr); // todo need to update member variables?
    return 0;
}


PetscErrorCode MDP::loadFromBinaryFile() {
    // LOG("Loading MDP from binary file: " + std::string(file_P_) + ", " + std::string(file_g_));
    std::cout << "Loading MDP from binary file: " << file_P_ << ", " << file_g_ << std::endl;
    PetscErrorCode ierr = 0;
    PetscViewer viewer;

    // Read number of states and actions from file
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_g_, FILE_MODE_READ, &viewer);
    PetscInt sizes[4]; // ClassID, Rows, Cols, NNZ
    PetscViewerBinaryRead(viewer, sizes, 4, PETSC_NULLPTR, PETSC_INT);
    numStates_ = sizes[1];
    numActions_ = sizes[2];
    PetscViewerDestroy(&viewer);
    
    // assert P and g are compatible (P: nm x n, g: n x m)
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_P_, FILE_MODE_READ, &viewer);
    PetscViewerBinaryRead(viewer, sizes, 4, PETSC_NULLPTR, PETSC_INT);
    if (sizes[1] != numStates_ * numActions_ || sizes[2] != numStates_) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Sizes of cost matrix and transition probability tensor not compatible.\nIt should hold that P: nm x n, g: n x m,\nwhere n is the number of states and m is the number of actions.\n");
    }
    PetscViewerDestroy(&viewer);

    // PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d\n", sizes[0], sizes[1], sizes[2], sizes[3]); // ClassID, Rows, Cols, NNZ

    // set local number of states (for this rank) 
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    // LOG("owns " + std::to_string(localNumStates_) + " states.");
    jsonWriter_->add_data("numStates", numStates_);
    jsonWriter_->add_data("numActions", numActions_);

    // load transition probability tensor
    MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_);
    MatSetFromOptions(transitionProbabilityTensor_);
    ierr = MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, localNumStates_, numStates_*numActions_, numStates_); CHKERRQ(ierr);
    MatSetUp(transitionProbabilityTensor_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_P_, FILE_MODE_READ, &viewer);
    MatLoad(transitionProbabilityTensor_, viewer);
    PetscViewerDestroy(&viewer);

    // load stage cost matrix
    MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_);
    MatSetFromOptions(stageCostMatrix_);
    ierr = MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_); CHKERRQ(ierr);
    MatSetUp(stageCostMatrix_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_g_, FILE_MODE_READ, &viewer);
    MatLoad(stageCostMatrix_, viewer);
    PetscViewerDestroy(&viewer);

    // convert stage cost matrix to dense matrix
    MatConvert(stageCostMatrix_, MATDENSE, MAT_INPLACE_MATRIX, &stageCostMatrix_);

    // Information about distribution on processes
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
    MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
    // LOG("owns rows " + std::to_string(P_start_) + " to " + std::to_string(P_end_) + " of P.");
    // LOG("owns rows " + std::to_string(g_start_) + " to " + std::to_string(g_end_) + " of g.");

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


PetscErrorCode MDP::generateCostMatrix(double (*g)(PetscInt, PetscInt)) {
    PetscErrorCode ierr;
    
    // assert numStates_ and numActions_ are set
    if (numStates_ == 0 || numActions_ == 0) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of states and actions not set.");
    }

    // create stage cost matrix
    // ierr = MatCreateDense(PETSC_COMM_WORLD, localNumStates_, PETSC_DECIDE, numStates_, numActions_, PETSC_NULLPTR, &stageCostMatrix_); CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_); CHKERRQ(ierr);
    ierr = MatSetType(stageCostMatrix_, MATDENSE); CHKERRQ(ierr);
    ierr = MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_); CHKERRQ(ierr);
    MatSetFromOptions(stageCostMatrix_);
    MatSetUp(stageCostMatrix_);

    // fill stage cost matrix
    MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_);
    for (PetscInt i = g_start_; i < g_end_; ++i) {
        for (PetscInt j = 0; j < numActions_; ++j) {
            MatSetValue(stageCostMatrix_, i, j, g(i, j), INSERT_VALUES);
        }
    }

    MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY);
    
    return 0;
}


PetscErrorCode MDP::generateTransitionProbabilityTensor(double (*P)(PetscInt, PetscInt, PetscInt), PetscInt d_nz, const PetscInt *d_nnz, PetscInt o_nz, const PetscInt *o_nnz) {
    PetscErrorCode ierr;

    // assert numStates_ and numActions_ are set
    if (numStates_ == 0 || numActions_ == 0) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Number of states and actions not set.");
    }

    // create transition probability tensor
    ierr = MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_); CHKERRQ(ierr);
    ierr = MatSetType(transitionProbabilityTensor_, MATMPIAIJ); CHKERRQ(ierr);
    ierr = MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, localNumStates_, numStates_*numActions_, numStates_); CHKERRQ(ierr);
    ierr = MatSetFromOptions(transitionProbabilityTensor_); CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(transitionProbabilityTensor_, d_nz, d_nnz, o_nz, o_nnz); CHKERRQ(ierr);
    ierr = MatSetUp(transitionProbabilityTensor_); CHKERRQ(ierr);

    // fill transition probability tensor
    MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_);
    for (PetscInt stateInd = P_start_ / numActions_; stateInd < P_end_ / numActions_; ++stateInd) {
        for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            PetscInt row = stateInd * numActions_ + actionInd;
            for (PetscInt nextStateInd = 0; nextStateInd < numStates_; ++nextStateInd) {
                PetscReal prob = P(stateInd, actionInd, nextStateInd);
                if (prob != 0.0) {
                    ierr = MatSetValue(transitionProbabilityTensor_, row, nextStateInd, prob, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
    }

    MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY);

    return 0;
}

