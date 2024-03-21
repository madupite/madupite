//
// Created by robin on 05.06.23.
//

#include "MDP.h"
// #include <mpi.h>
#include <string>
#include<iostream> // todo: replace with logging

MDP::MDP() {
    // MPI parallelization initialization
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank_);
    MPI_Comm_size(PETSC_COMM_WORLD, &size_);

    jsonWriter_ = new JsonWriter(rank_);

    transitionProbabilityTensor_ = nullptr;
    stageCostMatrix_ = nullptr;
    costMatrix_ = nullptr;
    costVector_ = nullptr;

    numStates_ = -1; // todo: change to unsigned int for better scalability? then remove -1
    numActions_ = -1;

    // Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    // Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file
}

MDP::~MDP() {
    PetscCallNoThrow(MatDestroy(&transitionProbabilityTensor_));
    PetscCallNoThrow(MatDestroy(&stageCostMatrix_));
    PetscCallNoThrow(MatDestroy(&costMatrix_));
    PetscCallNoThrow(VecDestroy(&costVector_));
    //delete jsonWriter_; // todo fix this (double free or corruption error)
}


void MDP::splitOwnership() {
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
}

PetscErrorCode MDP::setValuesFromOptions() {
    PetscBool flg;

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_states", &numStates_, &flg));
    if(flg) { // set local num states here if numStates_ is set (e.g. not the case for loading from file)
        splitOwnership();
    }
    // jsonWriter_->add_data("numStates", numStates_);
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_actions", &numActions_, &flg));

    PetscCall(PetscOptionsGetReal(NULL, NULL, "-discount_factor", &discountFactor_, &flg));
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Discount factor not specified. Use -discountFactor <double>.");
    }
    // jsonWriter_->add_data("discountFactor", discountFactor_);
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_iter_pi", &maxIter_PI_, &flg));
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Maximum number of policy iterations not specified. Use -maxIter_PI <int>.");
    }
    // jsonWriter_->add_data("maxIter_PI", maxIter_PI_);
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_iter_ksp", &maxIter_KSP_, &flg));
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Maximum number of KSP iterations not specified. Use -maxIter_KSP <int>.");
    }
    // jsonWriter_->add_data("maxIter_KSP", maxIter_KSP_);
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_pi_runs", &numPIRuns_, &flg));
    if(!flg) {
        //SETERRQ(PETSC_COMM_WORLD, 1, "Maximum number of KSP iterations not specified. Use -maxIter_KSP <int>.");
        // LOG("Number of PI runs for benchmarking not specified. Use -numPIRuns <int>. Default: 1");
        numPIRuns_ = 1;
    }
    // jsonWriter_->add_data("numPIRuns", numPIRuns_);
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol_ksp", &rtol_KSP_, &flg));
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Relative tolerance for KSP not specified. Use -rtol_KSP <double>.");
    }
    // jsonWriter_->add_data("rtol_KSP", rtol_KSP_);
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-atol_pi", &atol_PI_, &flg));
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Absolute tolerance for policy iteration not specified. Use -atol_PI <double>.");
    }
    // jsonWriter_->add_data("atol_PI", atol_PI_);
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_probabilities", file_P_, PETSC_MAX_PATH_LEN, &flg));
    if(!flg) {
        //SETERRQ(PETSC_COMM_WORLD, 1, "Filename for transition probability tensor not specified. Use -file_P <string>. (max length: 4096 chars");
        // LOG("Warning: Filename for transition probability tensor not specified. Use -file_P <string>. (max length: 4096 chars");
        file_P_[0] = '\0';
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_costs", file_g_, PETSC_MAX_PATH_LEN, &flg));
    if(!flg) {
        //SETERRQ(PETSC_COMM_WORLD, 1, "Filename for stage cost matrix not specified. Use -file_g <string>. (max length: 4096 chars");
        // LOG("Warning: Filename for stage cost matrix not specified. Use -file_g <string>. (max length: 4096 chars");
        file_g_[0] = '\0';
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_policy", file_policy_, PETSC_MAX_PATH_LEN, &flg));
    if(!flg) {
        // LOG("Filename for policy not specified. Optimal policy will not be written to file.");
        file_policy_[0] = '\0';
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_cost", file_cost_, PETSC_MAX_PATH_LEN, &flg));
    if(!flg) {
        // LOG("Filename for cost not specified. Optimal cost will not be written to file.");
        file_cost_[0] = '\0';
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_stats", file_stats_, PETSC_MAX_PATH_LEN, &flg));
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Filename for statistics not specified. Use -file_stats <string>. (max length: 4096 chars");
    }
    PetscChar inputMode[20];
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mode", inputMode, 20, &flg));
    if(!flg) {
        SETERRQ(PETSC_COMM_WORLD, 1, "Input mode not specified. Use -mode MINCOST or MAXREWARD.");
    }
    if (strcmp(inputMode, "MINCOST") == 0) {
        mode_ = MINCOST;
        // jsonWriter_->add_data("mode", "MINCOST");
    } else if (strcmp(inputMode, "MAXREWARD") == 0) {
        mode_ = MAXREWARD;
        // jsonWriter_->add_data("mode", "MAXREWARD");
    } else {
        SETERRQ(PETSC_COMM_WORLD, 1, "Input mode not recognized. Use -mode MINCOST or MAXREWARD.");
    }

    // set local number of states (for this rank)
    // localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
    // LOG("owns " + std::to_string(localNumStates_) + " states.");

    return 0;
}

void MDP::setOption(const char *option, const char *value, bool setValues) {
    // todo: should only be possible for:
    // -discountFactor, -maxIter_PI, -maxIter_KSP, -numPIRuns, -rtol_KSP, -atol_PI, -file_policy, -file_cost, -file_stats, -mode
    PetscCallThrow(PetscOptionsSetValue(NULL, option, value));
    if (setValues) {
        PetscCallThrow(setValuesFromOptions());
    }
}


void MDP::loadFromBinaryFile() {
    // LOG("Loading MDP from binary file: " + std::string(file_P_) + ", " + std::string(file_g_));
    std::cout << "Loading MDP from binary file: " << file_P_ << ", " << file_g_ << std::endl;
    PetscViewer viewer;

    // Read number of states and actions from file
    PetscCallThrow(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_g_, FILE_MODE_READ, &viewer));
    PetscInt sizes[4]; // ClassID, Rows, Cols, NNZ
    PetscCallThrow(PetscViewerBinaryRead(viewer, sizes, 4, PETSC_NULLPTR, PETSC_INT));
    numStates_ = sizes[1];
    numActions_ = sizes[2];
    PetscCallThrow(PetscViewerDestroy(&viewer));
    
    // assert P and g are compatible (P: nm x n, g: n x m)
    PetscCallThrow(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_P_, FILE_MODE_READ, &viewer));
    PetscCallThrow(PetscViewerBinaryRead(viewer, sizes, 4, PETSC_NULLPTR, PETSC_INT));
    if (sizes[1] != numStates_ * numActions_ || sizes[2] != numStates_) {
        PetscThrow(PETSC_COMM_WORLD, 1, "Sizes of cost matrix and transition probability tensor not compatible.\nIt should hold that P: nm x n, g: n x m,\nwhere n is the number of states and m is the number of actions.\n");
    }
    PetscCallThrow(PetscViewerDestroy(&viewer));

    // PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d\n", sizes[0], sizes[1], sizes[2], sizes[3]); // ClassID, Rows, Cols, NNZ

    // set local number of states (for this rank) 
    splitOwnership();
    // LOG("owns " + std::to_string(localNumStates_) + " states.");
    // jsonWriter_->add_data("numStates", numStates_);
    // jsonWriter_->add_data("numActions", numActions_);

    // load transition probability tensor
    PetscCallThrow(MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, localNumStates_, numStates_*numActions_, numStates_));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscCallThrow(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_P_, FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(transitionProbabilityTensor_, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));

    // load stage cost matrix
    PetscCallThrow(MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_));
    PetscCallThrow(MatSetFromOptions(stageCostMatrix_));
    PetscCallThrow(MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetUp(stageCostMatrix_));
    PetscCallThrow(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_g_, FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(stageCostMatrix_, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));

    // convert stage cost matrix to dense matrix
    PetscCallThrow(MatConvert(stageCostMatrix_, MATDENSE, MAT_INPLACE_MATRIX, &stageCostMatrix_));

    // Information about distribution on processes
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_));
    PetscCallThrow(MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_));
    // LOG("owns rows " + std::to_string(P_start_) + " to " + std::to_string(P_end_) + " of P.");
    // LOG("owns rows " + std::to_string(g_start_) + " to " + std::to_string(g_end_) + " of g.");
}

void MDP::writeVec(const Vec &vec, const char *filename) {
    VecScatter ctx;
    Vec MPIVec;
    PetscInt size;

    PetscCallThrow(VecGetSize(vec, &size));

    PetscCallThrow(VecCreateSeq(PETSC_COMM_SELF, size, &MPIVec));

    PetscCallThrow(VecScatterCreateToAll(vec, &ctx, &MPIVec));
    PetscCallThrow(VecScatterBegin(ctx, vec, MPIVec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCallThrow(VecScatterEnd(ctx, vec, MPIVec, INSERT_VALUES, SCATTER_FORWARD));

    const PetscScalar *values;
    PetscCallThrow(VecGetArrayRead(MPIVec, &values));

    PetscMPIInt rank;
    PetscCallThrow(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    // Rank 0 writes to file
    if(rank == 0) {
        std::ofstream out(filename);
        for(PetscInt i = 0; i < size; ++i) {
            out << values[i] << "\n";
        }
        out.close();
    }

    PetscCallThrow(VecRestoreArrayRead(MPIVec, &values));

    PetscCallThrow(VecScatterDestroy(&ctx));
    PetscCallThrow(VecDestroy(&MPIVec));
}

void MDP::writeIS(const IS &is, const char *filename) {
    const PetscInt *indices;
    PetscInt localSize;
    PetscInt size;
    PetscInt *allIndices = NULL;
    PetscInt *recvcounts = NULL;
    PetscInt *displs = NULL;

    PetscCallThrow(ISGetLocalSize(is, &localSize));
    PetscCallThrow(ISGetSize(is, &size));

    PetscCallThrow(ISGetIndices(is, &indices));

    PetscMPIInt rank;
    PetscCallThrow(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    if(rank == 0) {
        PetscCallThrow(PetscMalloc1(size, &allIndices));
        PetscCallThrow(PetscMalloc1(size, &recvcounts));
        PetscCallThrow(PetscMalloc1(size, &displs));

        PetscCallThrow(MPI_Gather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, PETSC_COMM_WORLD));

        displs[0] = 0;
        for(PetscInt i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }

        PetscCallThrow(MPI_Gatherv(indices, localSize, MPI_INT, allIndices, recvcounts, displs, MPI_INT, 0, PETSC_COMM_WORLD));

        // Rank 0 writes to file
        std::ofstream out(filename);
        for(PetscInt i = 0; i < size; ++i) {
            out << allIndices[i] << "\n";
        }
        out.close();

        PetscCallThrow(PetscFree(allIndices));
        PetscCallThrow(PetscFree(recvcounts));
        PetscCallThrow(PetscFree(displs));
    }
    else {
        PetscCallThrow(MPI_Gather(&localSize, 1, MPI_INT, NULL, 0, MPI_INT, 0, PETSC_COMM_WORLD));
        PetscCallThrow(MPI_Gatherv(indices, localSize, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, PETSC_COMM_WORLD));
    }

    PetscCallThrow(ISRestoreIndices(is, &indices));
}


void MDP::generateCostMatrix(double (*g)(PetscInt, PetscInt)) {
    // assert numStates_ and numActions_ are set
    if (numStates_ == 0 || numActions_ == 0) {
        PetscThrow(PETSC_COMM_WORLD, 1, "Number of states and actions not set.");
    }

    // create stage cost matrix
    // ierr = MatCreateDense(PETSC_COMM_WORLD, localNumStates_, PETSC_DECIDE, numStates_, numActions_, PETSC_NULLPTR, &stageCostMatrix_); CHKERRQ(ierr);
    PetscCallThrow(MatCreate(PETSC_COMM_WORLD, &stageCostMatrix_));
    PetscCallThrow(MatSetType(stageCostMatrix_, MATDENSE));
    PetscCallThrow(MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetFromOptions(stageCostMatrix_));
    PetscCallThrow(MatSetUp(stageCostMatrix_));

    // fill stage cost matrix
    PetscCallThrow(MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_));
    for (PetscInt i = g_start_; i < g_end_; ++i) {
        for (PetscInt j = 0; j < numActions_; ++j) {
            PetscCallThrow(MatSetValue(stageCostMatrix_, i, j, g(i, j), INSERT_VALUES));
        }
    }

    PetscCallThrow(MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
}


void MDP::generateTransitionProbabilityTensor(double (*P)(PetscInt, PetscInt, PetscInt), PetscInt d_nz, const PetscInt *d_nnz, PetscInt o_nz, const PetscInt *o_nnz) {
    // assert numStates_ and numActions_ are set
    if (numStates_ == -1 || numActions_ == -1) {
        PetscThrow(PETSC_COMM_WORLD, 1, "Number of states and actions not set.");
    }

    // create transition probability tensor
    PetscCallThrow(MatCreate(PETSC_COMM_WORLD, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetType(transitionProbabilityTensor_, MATMPIAIJ));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_*numActions_, localNumStates_, numStates_*numActions_, numStates_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatMPIAIJSetPreallocation(transitionProbabilityTensor_, d_nz, d_nnz, o_nz, o_nnz));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));

    // fill transition probability tensor
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &P_start_, &P_end_));
    for (PetscInt stateInd = P_start_ / numActions_; stateInd < P_end_ / numActions_; ++stateInd) {
        for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            PetscInt row = stateInd * numActions_ + actionInd;
            for (PetscInt nextStateInd = 0; nextStateInd < numStates_; ++nextStateInd) {
                PetscReal prob = P(stateInd, actionInd, nextStateInd);
                if (prob != 0.0) {
                    PetscCallThrow(MatSetValue(transitionProbabilityTensor_, row, nextStateInd, prob, INSERT_VALUES));
                }
            }
        }
    }

    PetscCallThrow(MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
}

void MDP::writeJSONmetadata() {
    jsonWriter_->add_data("numStates", numStates_);
    jsonWriter_->add_data("numActions", numActions_);
    jsonWriter_->add_data("discountFactor", discountFactor_);
}

