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
{
    // MPI parallelization initialization
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    jsonWriter_ = std::make_unique<JsonWriter>(rank_);

    // Logger::setPrefix("[R" + std::to_string(rank_) + "] ");
    // Logger::setFilename("log_R" + std::to_string(rank_) + ".txt"); // remove if all ranks should output to the same file
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

void MDP::setStageCostMatrix(Matrix* g)
{
    setupCalled      = false;
    stageCostMatrix_ = g;
}

void MDP::setTransitionProbabilityTensor(Matrix* P)
{
    setupCalled                  = false;
    transitionProbabilityTensor_ = P;
}

void MDP::setUp()
{
    if (setupCalled)
        return;

    setValuesFromOptions();

    setupCalled = true;
}

void MDP::writeVec(const Vec& vec, const char* filename)
{
    VecScatter ctx;
    Vec        MPIVec;
    PetscInt   size;

    PetscCallThrow(VecGetSize(vec, &size));

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
