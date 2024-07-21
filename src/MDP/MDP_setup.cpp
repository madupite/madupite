//
// Created by robin on 05.06.23.
//

#include "MDP.h"
// #include <mpi.h>
#include <iostream> // todo: replace with logging
#include <memory>
#include <string>

std::shared_ptr<Madupite> Madupite::instance;
std::mutex                Madupite::mtx;

std::shared_ptr<Madupite> Madupite::initialize(int* argc, char*** argv)
{
    // Inner class that allows private constructor access via inheritance.
    struct EnableMakeShared : Madupite {
        EnableMakeShared()
            : Madupite()
        {
        }
    };

    int    zero      = 0;
    char** nullchars = nullptr;
    if ((argc && *argc) != !!argv) {
        throw MadupiteException("Both argc and argv must be specified or neither");
    }
    if (!argc || !*argc) {
        argc = &zero;
        argv = &nullchars;
    }

    std::lock_guard<std::mutex> lock(mtx);
    if (!instance) {
        instance = std::make_shared<EnableMakeShared>();
        PetscCallThrow(PetscInitialize(argc, argv, PETSC_NULLPTR, PETSC_NULLPTR));
    }
    return instance;
}

MDP::MDP(std::shared_ptr<Madupite> madupite, MPI_Comm comm)
    : madupite_(madupite)
    , comm_(comm)
    , jsonWriter_(std::make_unique<JsonWriter>(rank_))
{
    // MPI parallelization initialization
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    // Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    // Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file
}

MDP::~MDP()
{
    PetscCallNoThrow(MatDestroy(&transitionProbabilityTensor_));
    PetscCallNoThrow(MatDestroy(&stageCostMatrix_));
    PetscCallNoThrow(MatDestroy(&costMatrix_));
    PetscCallNoThrow(VecDestroy(&costVector_));
}

// Pre: numStates_ is set
// Post: localNumStates_ is set
void MDP::splitOwnership()
{
    localNumStates_ = PETSC_DECIDE;
    PetscCallThrow(PetscSplitOwnership(comm_, &localNumStates_, &numStates_));
    // std::cout << "Rank " << rank_ << " owns " << localNumStates_ << " states." << std::endl;
}

PetscErrorCode MDP::setValuesFromOptions()
{
    PetscBool flg;
    PetscChar buf[PETSC_MAX_PATH_LEN];

    setupCalled = false;

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
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &alpha_, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Relative tolerance for KSP not specified. Use -alpha <double>.");
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

    PetscCall(PetscOptionsGetString(NULL, NULL, "-mode", buf, PETSC_MAX_PATH_LEN, &flg));
    if (!flg) {
        SETERRQ(comm_, 1, "Input mode not specified. Use -mode MINCOST or MAXREWARD.");
    }
    if (strcmp(buf, "MINCOST") == 0) {
        mode_ = MINCOST;
        // jsonWriter_->add_data("mode", "MINCOST");
    } else if (strcmp(buf, "MAXREWARD") == 0) {
        mode_ = MAXREWARD;
        // jsonWriter_->add_data("mode", "MAXREWARD");
    } else {
        SETERRQ(comm_, 1, "Input mode not recognized. Use -mode MINCOST or MAXREWARD.");
    }

    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_p", buf, PETSC_MAX_PATH_LEN, &flg));
    if (flg) {
        setSourceTransitionProbabilityTensor(buf);
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file_g", buf, PETSC_MAX_PATH_LEN, &flg));
    if (flg) {
        setSourceStageCostMatrix(buf);
    }

    return 0;
}

void MDP::setOption(const char* option, const char* value)
{
    setupCalled = false;
    // todo: should only be possible for:
    // -discountFactor, -maxIter_PI, -maxIter_KSP, -numPIRuns, -alpha, -atol_PI, -file_policy, -file_cost, -file_stats, -mode
    PetscCallThrow(PetscOptionsSetValue(NULL, option, value));
}

void MDP::clearOptions() { PetscCallThrow(PetscOptionsClear(NULL)); }

void MDP::setSourceStageCostMatrix(const std::string filename)
{
    setupCalled  = false;
    g_file_name_ = filename;
    g_src_       = FILE;
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, g_file_name_.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(PetscViewerBinaryRead(viewer, &g_file_meta_, 4, PETSC_NULLPTR, PETSC_INT));
    PetscCallThrow(PetscViewerDestroy(&viewer));
}

void MDP::setSourceStageCostMatrix(const Costfunc& g)
{
    setupCalled = false;
    g_src_      = FUNCTION;
    g_func_     = g;
}

void MDP::setSourceTransitionProbabilityTensor(const std::string filename)
{
    setupCalled  = false;
    p_file_name_ = filename;
    p_src_       = FILE;
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, p_file_name_.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(PetscViewerBinaryRead(viewer, &p_file_meta_, 4, PETSC_NULLPTR, PETSC_INT));
    PetscCallThrow(PetscViewerDestroy(&viewer));
}

void MDP::setSourceTransitionProbabilityTensor(const Probfunc& P)
{
    setupCalled = false;
    p_src_      = FUNCTION;
    p_func_     = P;
}

void MDP::setSourceTransitionProbabilityTensor(
    const Probfunc& P, PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz)
{
    setupCalled = false;
    p_src_      = FUNCTION;
    p_func_     = P;
    p_prealloc_ = true;
    p_nnz_      = std::make_tuple(d_nz, d_nnz, o_nz, o_nnz);
}

void MDP::loadTransitionProbabilityTensor()
{
    std::cout << "Loading transition probability tensor from binary file: " << p_file_name_ << std::endl;
    if (transitionProbabilityTensor_ != nullptr) {
        PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
    }
    PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, localNumStates_, numStates_ * numActions_, numStates_));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, p_file_name_.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(transitionProbabilityTensor_, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));
}

void MDP::loadStageCostMatrix()
{
    std::cout << "Loading stage cost matrix from binary file: " << g_file_name_ << std::endl;
    if (stageCostMatrix_ != nullptr) {
        PetscCallThrow(MatDestroy(&stageCostMatrix_));
    }
    PetscCallThrow(MatCreate(comm_, &stageCostMatrix_));
    PetscCallThrow(MatSetFromOptions(stageCostMatrix_));
    PetscCallThrow(MatSetSizes(stageCostMatrix_, localNumStates_, PETSC_DECIDE, numStates_, numActions_));
    PetscCallThrow(MatSetUp(stageCostMatrix_));
    PetscViewer viewer;
    PetscCallThrow(PetscViewerBinaryOpen(comm_, g_file_name_.c_str(), FILE_MODE_READ, &viewer));
    PetscCallThrow(MatLoad(stageCostMatrix_, viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
    PetscCallThrow(MatConvert(stageCostMatrix_, MATDENSE, MAT_INPLACE_MATRIX, &stageCostMatrix_)); // convert to dense matrix!
    PetscCallThrow(MatGetOwnershipRange(stageCostMatrix_, &g_start_, &g_end_));
}

// Pre: localNumStates_, numStates_, numActions_ are set
void MDP::createStageCostMatrix()
{
    PetscLogDouble t0, t1;
    PetscTime(&t0);
    // std::cout << "Creating stage cost matrix from function" << std::endl;
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
    for (PetscInt stateInd = g_start_; stateInd < g_end_; ++stateInd) {
        for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            value = g_func_(stateInd, actionInd);
            PetscCallThrow(MatSetValue(stageCostMatrix_, stateInd, actionInd, value, INSERT_VALUES));
        }
    }
    PetscCallThrow(MatAssemblyBegin(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(stageCostMatrix_, MAT_FINAL_ASSEMBLY));
    PetscTime(&t1);
    PetscPrintf(comm_, "Time to create stage cost matrix: %f\n", t1 - t0);
}

void MDP::createTransitionProbabilityTensorPrealloc()
{
    PetscLogDouble t0, t1;
    PetscTime(&t0);
    // std::cout << "Creating transition probability tensor from function with preallocation" << std::endl;
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
    for (PetscInt stateInd = p_start_ / numActions_; stateInd < p_end_ / numActions_; ++stateInd) {
        for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            auto [values, indices] = p_func_(stateInd, actionInd);
            PetscInt rowInd        = stateInd * numActions_ + actionInd;
            PetscCallThrow(MatSetValues(transitionProbabilityTensor_, 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    PetscCallThrow(MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
    PetscTime(&t1);
    PetscPrintf(comm_, "Time to create transition probability tensor: %f\n", t1 - t0);
}

void MDP::createTransitionProbabilityTensor()
{
    PetscLogDouble t0, t1;
    PetscTime(&t0);
    // std::cout << "Creating transition probability tensor from function without preallocation" << std::endl;
    if (transitionProbabilityTensor_ != nullptr) {
        PetscCallThrow(MatDestroy(&transitionProbabilityTensor_));
    }
    PetscCallThrow(MatCreate(comm_, &transitionProbabilityTensor_));
    PetscCallThrow(MatSetSizes(transitionProbabilityTensor_, localNumStates_ * numActions_, PETSC_DECIDE, numStates_ * numActions_, numStates_));
    PetscCallThrow(MatSetFromOptions(transitionProbabilityTensor_));
    PetscCallThrow(MatSetUp(transitionProbabilityTensor_));
    PetscCallThrow(MatGetOwnershipRange(transitionProbabilityTensor_, &p_start_, &p_end_));

    // fill
    for (PetscInt stateInd = p_start_ / numActions_; stateInd < p_end_ / numActions_; ++stateInd) {
        for (PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            auto [values, indices] = p_func_(stateInd, actionInd);
            PetscInt rowInd        = stateInd * numActions_ + actionInd;
            PetscCallThrow(MatSetValues(transitionProbabilityTensor_, 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    PetscCallThrow(MatAssemblyBegin(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
    PetscCallThrow(MatAssemblyEnd(transitionProbabilityTensor_, MAT_FINAL_ASSEMBLY));
    PetscTime(&t1);
    PetscPrintf(comm_, "Time to create transition probability tensor: %f\n", t1 - t0);
}

void MDP::setUp()
{
    if (setupCalled)
        return;

    setValuesFromOptions();

    if (p_src_ == FILE && g_src_ == FILE) {
        // P: nm x n, g: n x m
        // meta = [ClassID, Rows, Cols, NNZ]
        if (p_file_meta_[1] != g_file_meta_[1] * g_file_meta_[2] || p_file_meta_[2] != g_file_meta_[1]) {
            PetscThrow(comm_, 1,
                "Sizes of cost matrix and transition probability tensor not compatible.\n"
                "It should hold that P: nm x n, g: n x m,\n"
                "where n is the number of states and m is the number of actions.\n");
        }
        numStates_  = p_file_meta_[2];
        numActions_ = g_file_meta_[2];
        splitOwnership();
        loadTransitionProbabilityTensor();
        loadStageCostMatrix();
    } else if (p_src_ == FUNCTION && g_src_ == FUNCTION) {
        splitOwnership();
        if (p_prealloc_)
            createTransitionProbabilityTensorPrealloc();
        else
            createTransitionProbabilityTensor();
        createStageCostMatrix();
    } else if (p_src_ == FILE && g_src_ == FUNCTION) {
        if (p_file_meta_[1] != numStates_ * numActions_ || p_file_meta_[2] != numStates_) {
            PetscThrow(
                comm_, 1, "Size of transition probability tensor in the file doesn't match the specified options -num_states and -num_actions.");
        }
        splitOwnership();
        loadTransitionProbabilityTensor();
        createStageCostMatrix();
    } else if (p_src_ == FUNCTION && g_src_ == FILE) {
        if (g_file_meta_[1] != numStates_ || g_file_meta_[2] != numActions_) {
            PetscThrow(comm_, 1, "Size of stage cost matrix in the file doesn't match the specified options -num_states and -num_actions.");
        }
        splitOwnership();
        if (p_prealloc_)
            createTransitionProbabilityTensorPrealloc();
        else
            createTransitionProbabilityTensor();
        loadStageCostMatrix();
    }

    setupCalled = true;
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
    jsonWriter_->add_data("alpha", alpha_);
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
    if (p_src_ == FILE) {
        jsonWriter_->add_data("file_transition_probabilities", p_file_name_);
    }
    if (g_src_ == FILE) {
        jsonWriter_->add_data("file_stage_costs", g_file_name_);
    }
}
