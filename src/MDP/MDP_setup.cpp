//
// Created by robin on 05.06.23.
//

#include "MDP.h"
// #include <mpi.h>
#include <iostream> // todo: replace with logging
#include <string>

MDP::MDP(MPI_Comm comm)
    : comm_(comm), p_file_name_(PETSC_MAX_PATH_LEN, '_'), g_file_name_(PETSC_MAX_PATH_LEN, '_')
{
    // MPI parallelization initialization
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    jsonWriter_ = new JsonWriter(rank_);

    transitionProbabilityTensor_ = nullptr;
    stageCostMatrix_             = nullptr;
    costMatrix_                  = nullptr;
    costVector_                  = nullptr;

    numStates_  = -1; // todo: change to unsigned int for better scalability? then remove -1
    numActions_ = -1;
    p_src_ = -1;
    g_src_ = -1;
    p_prealloc_ = PETSC_FALSE;

    // Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    // Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file
}

MDP::~MDP()
{
    PetscCallNoThrow(MatDestroy(&transitionProbabilityTensor_));
    PetscCallNoThrow(MatDestroy(&stageCostMatrix_));
    PetscCallNoThrow(MatDestroy(&costMatrix_));
    PetscCallNoThrow(VecDestroy(&costVector_));
    // delete jsonWriter_; // todo fix this (double free or corruption error)
}

// Pre: numStates_, rank_ and size_ are set
// Post: localNumStates_ is set
void MDP::splitOwnership()
{
    localNumStates_ = (rank_ < numStates_ % size_) ? numStates_ / size_ + 1 : numStates_ / size_; // first numStates_ % size_ ranks get one more state
}

PetscErrorCode MDP::setValuesFromOptions()
{
    PetscBool flg;

    PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_states", &numStates_, &flg));
    if (flg) { // set local num states here if numStates_ is set (e.g. not the case for loading from file)
        splitOwnership();
    }
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_actions", &numActions_, &flg));

    PetscCall(PetscOptionsGetReal(NULL, NULL, "-discount_factor", &discountFactor_, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Discount factor not specified. Use -discountFactor <double>.");
    }
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_iter_pi", &maxIter_PI_, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Maximum number of policy iterations not specified. Use -maxIter_PI <int>.");
    }
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-max_iter_ksp", &maxIter_KSP_, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Maximum number of KSP iterations not specified. Use -maxIter_KSP <int>.");
    }
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol_ksp", &rtol_KSP_, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Relative tolerance for KSP not specified. Use -rtol_KSP <double>.");
    }
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-atol_pi", &atol_PI_, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Absolute tolerance for policy iteration not specified. Use -atol_PI <double>.");
    }
    // jsonWriter_->add_data("atol_PI", atol_PI_);
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_policy", file_policy_, PETSC_MAX_PATH_LEN, &flg));
    if (!flg) {
        // LOG("Filename for policy not specified. Optimal policy will not be written to file.");
        file_policy_[0] = '\0';
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_cost", file_cost_, PETSC_MAX_PATH_LEN, &flg));
    if (!flg) {
        // LOG("Filename for cost not specified. Optimal cost will not be written to file.");
        file_cost_[0] = '\0';
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_stats", file_stats_, PETSC_MAX_PATH_LEN, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Filename for statistics not specified. Use -file_stats <string>. (max length: 4096 chars");
    }
    PetscChar inputMode[20];
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mode", inputMode, 20, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Input mode not specified. Use -mode MINCOST or MAXREWARD.");
    }
    if (strcmp(inputMode, "MINCOST") == 0) {
        mode_ = MINCOST;
        // jsonWriter_->add_data("mode", "MINCOST");
    } else if (strcmp(inputMode, "MAXREWARD") == 0) {
        mode_ = MAXREWARD;
        // jsonWriter_->add_data("mode", "MAXREWARD");
    } else {
        SETERRQ(comm_, 1, "Input mode not recognized. Use -mode MINCOST or MAXREWARD.");
    }
    // -source_p, -source_g: either "FILE" or "FUNCTION", set p_src_ and g_src_ accordingly
    PetscCall(PetscOptionsGetString(NULL, NULL, "-source_p", inputMode, 20, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Source of transition probability tensor not specified. Use -source_p FILE or FUNCTION.");
    }
    if (strcmp(inputMode, "FILE") == 0) {
        p_src_ = 0;
    } else if (strcmp(inputMode, "FUNCTION") == 0) {
        p_src_ = 1;
    } else {
        SETERRQ(comm_, 1, "Source of transition probability tensor not recognized. Use -source_p FILE or FUNCTION.");
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-source_g", inputMode, 20, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Source of stage cost matrix not specified. Use -source_g FILE or FUNCTION.");
    }
    if (strcmp(inputMode, "FILE") == 0) {
        g_src_ = FILE;
    } else if (strcmp(inputMode, "FUNCTION") == 0) {
        g_src_ = FUNCTION;
    } else {
        SETERRQ(comm_, 1, "Source of stage cost matrix not recognized. Use -source_g FILE or FUNCTION.");
    }

    return 0;
}

void MDP::setOption(const char* option, const char* value, bool setValues)
{
    // todo: should only be possible for:
    // -discountFactor, -maxIter_PI, -maxIter_KSP, -numPIRuns, -rtol_KSP, -atol_PI, -file_policy, -file_cost, -file_stats, -mode
    PetscCallThrow(PetscOptionsSetValue(NULL, option, value));
    if (setValues) {
        PetscCallThrow(setValuesFromOptions());
    }
}


void MDP::assembleMatrix(int matrix)
{                      // assembles the matrix; blocking call
    if (matrix == 0) { // transitionProbabilityTensor
        PetscCallThrow(MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
        PetscCallThrow(MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
    } else if (matrix == 1) { // stageCostMatrix
        PetscCallThrow(MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
        PetscCallThrow(MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
    } else {
        std::cerr << "Invalid matrix type" << std::endl;
        exit(1);
    }
}


void MDP::setSourceStageCostMatrix(const char* filename) {
    if (g_src_ != FILE) {
        PetscThrow(comm_, 1, "Source of stage cost matrix not recognized. Use -source_g FILE.");
    }
    strncpy(g_file_name_, filename, PETSC_MAX_PATH_LEN);
    std::cout << "Loading stage cost matrix from binary file: " << filename << std::endl;
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, g_file_name_, FILE_MODE_READ, &viewer));
    PetscCallThrow(PetscViewerBinaryRead(viewer, &g_file_meta_, 4, PETSC_NULLPTR, PETSC_INT));
    PetscCallThrow(PetscViewerDestroy(&viewer));
}

void MDP::setSourceStageCostMatrix(const Costfunc g) {
    if (g_src_ != FUNCTION) {
        PetscThrow(comm_, 1, "Source of stage cost matrix not recognized. Use -source_g FUNCTION.");
    }
    g_func_ = g;
}


void MDP::setSourceTransitionProbabilityTensor(const char* filename) {
    if (p_src_ != FILE) {
        PetscThrow(comm_, 1, "Source of transition probability tensor not recognized. Use -source_p FILE.");
    }
    std::cout << "Loading transition probability tensor from binary file: " << filename << std::endl;
    strncpy(p_file_name_, filename, PETSC_MAX_PATH_LEN);
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, p_file_name_, FILE_MODE_READ, &viewer));
    PetscCallThrow(PetscViewerBinaryRead(viewer, &p_file_meta_, 4, PETSC_NULLPTR, PETSC_INT));
    PetscCallThrow(PetscViewerDestroy(&viewer));
}

void MDP::setSourceTransitionProbabilityTensor(const Probfunc P) {
    if (p_src_ != FUNCTION) {
        PetscThrow(comm_, 1, "Source of transition probability tensor not recognized. Use -source_p FUNCTION.");
    }
    p_func_ = P;
}

void MDP::setSourceTransitionProbabilityTensor(const Probfunc P, PetscInt d_nz, const std::vector<int> &d_nnz,
                                               PetscInt o_nz, const std::vector<int> &o_nnz) {
    if (p_src_ != FUNCTION) {
        PetscThrow(comm_, 1, "Source of transition probability tensor not recognized. Use -source_p FUNCTION.");
    }
    p_func_ = P;
    p_prealloc_ = PETSC_TRUE;
    p_nnz_ = std::make_tuple(d_nz, d_nnz, o_nz, o_nnz);
}

void MDP::loadTransitionProbabilityTensor() {
    if (transitionProbabilityTensor_ != nullptr) {
        PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
    }
    PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, localNumStates_, numStates_ * numActions_, numStates_));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, p_file_name_, FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(transitionProbabilityTensor_, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));
}

void MDP::loadStageCostMatrix() {
    if (stageCostMatrix_ != nullptr) {
        PetscCallThrow(MatDestroy(&stageCostMatrix_));
    }
    PetscCallThrow(MatCreate(comm_, &stageCostMatrix_));
    PetscCallThrow(MatSetFromOptions(stageCostMatrix_));
    PetscCallThrow(MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetUp(stageCostMatrix_));
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, g_file_name_, FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(stageCostMatrix_, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
    PetscCallThrow(MatConvert(stageCostMatrix_, MATDENSE, MAT_INPLACE_MATRIX, &stageCostMatrix_)); // convert to dense matrix!
    PetscCallThrow(MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_));
}

// Pre: localNumStates_, numStates_, numActions_ are set
void MDP::createStageCostMatrix()
{
    if (stageCostMatrix_ != nullptr) {
        PetscCallThrow(MatDestroy(&stageCostMatrix_));
    }
    PetscCallThrow(MatCreate(comm_, &stageCostMatrix_));
    PetscCallThrow(MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetType(stageCostMatrix_, MATDENSE));
    // MatSetFromOptions(stageCostMatrix_);
    PetscCallThrow(MatSetUp(stageCostMatrix_));
    PetscCallThrow(MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_));

    // fill
    double value;
    for(PetscInt stateInd = g_start_; stateInd < g_end_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            value = g_func_(stateInd, actionInd);
            PetscCallThrow(MatSetValue(stageCostMatrix_, stateInd, actionInd, value, INSERT_VALUES));
        }
    }
    assembleMatrix(1);
}

void MDP::createTransitionProbabilityTensorPrealloc()
{
    if (transitionProbabilityTensor_ != nullptr) {
        PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
    }
    auto [d_nz, d_nnz, o_nz, o_nnz] = p_nnz_;
    const PetscInt *d_nnz_ptr = d_nnz.data(), *o_nnz_ptr = o_nnz.data();
    if (d_nnz.empty())
        d_nnz_ptr = nullptr;
    if (o_nnz.empty())
        o_nnz_ptr = nullptr;
    PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatMPIAIJSetPreallocation(transitionProbabilityTensor_, d_nz, d_nnz_ptr, o_nz, o_nnz_ptr));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));

    // fill
    for(PetscInt stateInd = p_start_ / numActions_; stateInd < p_end_ / numActions_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            auto [values, indices] = p_func_(stateInd, actionInd);
            PetscInt rowInd = stateInd * numActions_ + actionInd;
            PetscCallThrow(MatSetValues(transitionProbabilityTensor_, 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    assembleMatrix(0);
}

void MDP::createTransitionProbabilityTensor()
{
    if (transitionProbabilityTensor_ != nullptr) {
        PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
    }
    PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));

    // fill
    for(PetscInt stateInd = p_start_ / numActions_; stateInd < p_end_ / numActions_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            auto [values, indices] = p_func_(stateInd, actionInd);
            PetscInt rowInd = stateInd * numActions_ + actionInd;
            PetscCallThrow(MatSetValues(transitionProbabilityTensor_, 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    assembleMatrix(0);
}


void MDP::setUp() {
    if (p_src_ == FILE && g_src_ == FILE) {
        // P: nm x n, g: n x m
        // meta = [ClassID, Rows, Cols, NNZ]
        if (p_file_meta_[1] != g_file_meta_[1]*g_file_meta_[2] || p_file_meta_[2] != g_file_meta_[1]) {
            PetscThrow(comm_, 1, "Sizes of cost matrix and transition probability tensor not compatible.\nIt should hold that P: nm x n, g: n x m,\nwhere n is the number of states and m is the number of actions.\n");
        }
        numStates_ = p_file_meta_[2];
        numActions_ = g_file_meta_[2];
        splitOwnership();
        loadTransitionProbabilityTensor();
        loadStageCostMatrix();
    }
    else if (p_src_ == FUNCTION && g_src_ == FUNCTION) {
        splitOwnership();
        if(p_prealloc_) createTransitionProbabilityTensorPrealloc();
        else createTransitionProbabilityTensor();
        createStageCostMatrix();
    }
    else if (p_src_ == FILE && g_src_ == FUNCTION) {
        if(p_file_meta_[1] != numStates_*numActions_ || p_file_meta_[2] != numStates_) {
            PetscThrow(comm_, 1, "Size of transition probability tensor in the file doesn't match the specified options -num_states and -num_actions.");
        }
        splitOwnership();
        loadTransitionProbabilityTensor();
        createStageCostMatrix();
    }
    else if (p_src_ == FUNCTION && g_src_ == FILE) {
        if(g_file_meta_[1] != numStates_ || g_file_meta_[2] != numActions_) {
            PetscThrow(comm_, 1, "Size of stage cost matrix in the file doesn't match the specified options -num_states and -num_actions.");
        }
        splitOwnership();
        if(p_prealloc_) createTransitionProbabilityTensorPrealloc();
        else createTransitionProbabilityTensor();
        loadStageCostMatrix();
    }

}


void MDP::writeVec(const Vec& vec, const char* filename)
{
    VecScatter ctx;
    Vec        MPIVec;
    PetscInt   size;

    PetscCallThrow(VecGetSize(vec, &size));

    PetscCallThrow(VecCreateSeq(PETSC_COMM_SELF, size, &MPIVec));

    PetscCallThrow(VecScatterCreateToAll(vec, &ctx, &MPIVec));
    PetscCallThrow(VecScatterBegin(ctx, vec, MPIVec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCallThrow(VecScatterEnd(ctx, vec, MPIVec, INSERT_VALUES, SCATTER_FORWARD));

    const PetscScalar* values;
    PetscCallThrow(VecGetArrayRead(MPIVec, &values));

    PetscMPIInt rank;
    PetscCallThrow(MPI_Comm_rank(comm_, &rank));

    // Rank 0 writes to file
    if (rank == 0) {
        std::ofstream out(filename);
        for (PetscInt i = 0; i < size; ++i) {
            out << values[i] << "\n";
        }
        out.close();
    }

    PetscCallThrow(VecRestoreArrayRead(MPIVec, &values));

    PetscCallThrow(VecScatterDestroy(&ctx));
    PetscCallThrow(VecDestroy(&MPIVec));
}

void MDP::writeIS(const IS& is, const char* filename)
{
    const PetscInt* indices;
    PetscInt        localSize;
    PetscInt        size;
    PetscInt*       allIndices = NULL;
    PetscInt*       recvcounts = NULL;
    PetscInt*       displs     = NULL;

    PetscCallThrow(ISGetLocalSize(is, &localSize));
    PetscCallThrow(ISGetSize(is, &size));

    PetscCallThrow(ISGetIndices(is, &indices));

    PetscMPIInt rank;
    PetscCallThrow(MPI_Comm_rank(comm_, &rank));

    if (rank == 0) {
        PetscCallThrow(PetscMalloc1(size, &allIndices));
        PetscCallThrow(PetscMalloc1(size, &recvcounts));
        PetscCallThrow(PetscMalloc1(size, &displs));

        PetscCallThrow(MPI_Gather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm_));

        displs[0] = 0;
        for (PetscInt i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        PetscCallThrow(MPI_Gatherv(indices, localSize, MPI_INT, allIndices, recvcounts, displs, MPI_INT, 0, comm_));

        // Rank 0 writes to file
        std::ofstream out(filename);
        for (PetscInt i = 0; i < size; ++i) {
            out << allIndices[i] << "\n";
        }
        out.close();

        PetscCallThrow(PetscFree(allIndices));
        PetscCallThrow(PetscFree(recvcounts));
        PetscCallThrow(PetscFree(displs));
    } else {
        PetscCallThrow(MPI_Gather(&localSize, 1, MPI_INT, NULL, 0, MPI_INT, 0, comm_));
        PetscCallThrow(MPI_Gatherv(indices, localSize, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, comm_));
    }

    PetscCallThrow(ISRestoreIndices(is, &indices));
}

void MDP::writeJSONmetadata()
{
    // writes model specifics to file per launch of MDP::ineaxctPolicyIteration
    // ksp_type is written in MDP::iterativePolicyEvaluation (since it's only known there)
    jsonWriter_->add_data("num_states", numStates_);
    jsonWriter_->add_data("num_actions", numActions_);
    jsonWriter_->add_data("discount_factor", discountFactor_);
    jsonWriter_->add_data("max_iter_pi", maxIter_PI_);
    jsonWriter_->add_data("max_iter_ksp", maxIter_KSP_);
    jsonWriter_->add_data("rtol_ksp", rtol_KSP_);
    jsonWriter_->add_data("atol_pi", atol_PI_);
    jsonWriter_->add_data("num_ranks", size_);

    if (file_policy_[0] != '\0') {
        jsonWriter_->add_data("file_policy", file_policy_);
    }
    if (file_cost_[0] != '\0') {
        jsonWriter_->add_data("file_cost", file_cost_);
    }
    if (file_stats_[0] != '\0') {
        jsonWriter_->add_data("file_stats", file_stats_);
    }

    // TODO: if file loaded by user, also write filenames
}
