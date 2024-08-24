#include <iostream> // todo: replace with logging
#include <memory>
#include <string>

#include "mdp.h"
#include "utils.h"

std::shared_ptr<Madupite> Madupite::instance;
std::mutex                Madupite::mtx;
MPI_Comm                  Madupite::comm_;

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
    comm_ = PETSC_COMM_WORLD;
    return instance;
}

MDP::MDP(std::shared_ptr<Madupite> madupite)
    : madupite_(madupite)
    , comm_(Madupite::getCommWorld())
    {}
    

template <typename comm_t>
MDP::MDP(std::shared_ptr<Madupite> madupite, comm_t comm_arg)
    : madupite_(madupite)
    , comm_(convertComm(comm_arg))
{
    json_writer_ = std::make_unique<JsonWriter>(Madupite::getCommWorld());
}
template MDP::MDP(std::shared_ptr<Madupite> madupite, int comm);

#define GET_OPTION(type, name, default_value, parse_fn, print_fn)                                                                                    \
    do {                                                                                                                                             \
        PetscBool flg;                                                                                                                               \
        PetscCall(parse_fn(NULL, NULL, "-" #name, &name##_, &flg));                                                                                  \
        if (!flg) {                                                                                                                                  \
            print_fn(Madupite::getCommWorld(), #name " not specified. Using default value: " #default_value ".\n");                                  \
            name##_ = default_value;                                                                                                                 \
        }                                                                                                                                            \
    } while (0)

#define GET_STRING_OPTION(name, buffer)                                                                                                              \
    do {                                                                                                                                             \
        PetscBool flg;                                                                                                                               \
        PetscCall(PetscOptionsGetString(NULL, NULL, "-" #name, buffer, PETSC_MAX_PATH_LEN, &flg));                                                   \
        if (!flg) {                                                                                                                                  \
            buffer[0] = '\0';                                                                                                                        \
        }                                                                                                                                            \
    } while (0)

PetscErrorCode MDP::setValuesFromOptions()
{
    PetscBool flg;
    PetscChar buf[PETSC_MAX_PATH_LEN];

    setup_called = false;

    // Mandatory options
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-discount_factor", &discount_factor_, &flg));
    if (!flg) {
        SETERRQ(Madupite::getCommWorld(), 1, "Discount factor not specified. Use -discount_factor <double>.\n");
    }
    PetscCall(PetscOptionsGetString(NULL, NULL, "-mode", buf, PETSC_MAX_PATH_LEN, &flg));
    if (!flg) {
        SETERRQ(Madupite::getCommWorld(), 1, "Input mode not specified. Use -mode MINCOST or MAXREWARD.\n");
    } else if (strcmp(buf, "MINCOST") == 0) {
        mode_ = MINCOST;
    } else if (strcmp(buf, "MAXREWARD") == 0) {
        mode_ = MAXREWARD;
    } else {
        SETERRQ(Madupite::getCommWorld(), 1, "Input mode not recognized. Use -mode MINCOST or MAXREWARD.\n");
    }
    // Optional options with defaults
    GET_OPTION(int, max_iter_pi, 1000, PetscOptionsGetInt, PetscPrintf);
    GET_OPTION(int, max_iter_ksp, 1000, PetscOptionsGetInt, PetscPrintf);
    GET_OPTION(double, alpha, 1e-4, PetscOptionsGetReal, PetscPrintf);
    GET_OPTION(double, atol_pi, 1e-8, PetscOptionsGetReal, PetscPrintf);

    GET_STRING_OPTION(file_policy, file_policy_);
    GET_STRING_OPTION(file_cost, file_cost_);
    GET_STRING_OPTION(file_stats, file_stats_);
    GET_STRING_OPTION(export_optimal_transition_probabilities, file_optimal_transition_probabilities_);
    GET_STRING_OPTION(export_optimal_stage_costs, file_optimal_stage_costs_);
    return 0;
}

PetscErrorCode MDP::setOption(const char* option, const char* value)
{
    setup_called = false;
    if (strcmp(option, "-discount_factor") == 0 || strcmp(option, "-max_iter_pi") == 0 || strcmp(option, "-max_iter_ksp") == 0
        || strcmp(option, "-num_pi_runs") == 0 || strcmp(option, "-alpha") == 0 || strcmp(option, "-atol_pi") == 0
        || strcmp(option, "-file_policy") == 0 || strcmp(option, "-file_cost") == 0 || strcmp(option, "-file_stats") == 0
        || strcmp(option, "-mode") == 0 || strcmp(option, "-ksp_type") == 0 || strcmp(option, "-export_optimal_transition_probabilities") == 0
        || strcmp(option, "-export_optimal_stage_costs") == 0) {
        PetscCallThrow(PetscOptionsSetValue(NULL, option, value));
    } else {
        SETERRQ(Madupite::getCommWorld(), 1, "%s", ("Invalid option: " + std::string(option)).c_str());
    }
    return 0;
}

void MDP::clearOptions() { PetscCallThrow(PetscOptionsClear(NULL)); }

void MDP::setStageCostMatrix(const Matrix& g)
{
    setup_called       = false;
    stage_cost_matrix_ = g;
}

void MDP::setTransitionProbabilityTensor(const Matrix& P)
{
    setup_called                   = false;
    transition_probability_tensor_ = P;
}

void MDP::setUp()
{
    if (setup_called)
        return;

    setValuesFromOptions();

    // check matrix sizes agree
    if (transition_probability_tensor_.colLayout().localSize() != stage_cost_matrix_.rowLayout().localSize()) {
        PetscThrow(Madupite::getCommWorld(), 1, "Error: number of states do not agree (P != g): : %" PetscInt_FMT " != %" PetscInt_FMT,
            transition_probability_tensor_.colLayout().localSize(), stage_cost_matrix_.rowLayout().localSize());
    }
    if (transition_probability_tensor_.rowLayout().size() / transition_probability_tensor_.colLayout().size()
        != stage_cost_matrix_.colLayout().size()) {
        PetscThrow(Madupite::getCommWorld(), 1, "Error: number of actions do not agree (P != g): %" PetscInt_FMT " != %" PetscInt_FMT,
            transition_probability_tensor_.rowLayout().size() / transition_probability_tensor_.colLayout().size(),
            stage_cost_matrix_.colLayout().size());
    }
    num_states_       = stage_cost_matrix_.rowLayout().size();
    local_num_states_ = stage_cost_matrix_.rowLayout().localSize();
    num_actions_      = stage_cost_matrix_.colLayout().size();
    g_start_          = stage_cost_matrix_.rowLayout().start();
    g_end_            = stage_cost_matrix_.rowLayout().end();

    setup_called = true;
}

// write MPIAIJ matrix as ASCII in COO format to file
// TODO: same function as Matrix::writeToFile. This here should be removed, once Matrix is used in MDP
void MDP::writeMat(const Mat& mat, const char* filename)
{
    PetscInt    m, n, rstart, rend;
    PetscViewer viewer;
    PetscMPIInt rank, size;
    MatInfo     info;
    PetscInt    nz_global;

    PetscCallThrow(MatGetSize(mat, &m, &n));
    PetscCallThrow(MatGetOwnershipRange(mat, &rstart, &rend));
    PetscCallThrow(MPI_Comm_rank(PetscObjectComm((PetscObject)mat), &rank));
    PetscCallThrow(MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size));

    // Get matrix info
    PetscCallThrow(MatGetInfo(mat, MAT_GLOBAL_SUM, &info));
    nz_global = (PetscInt)info.nz_used;

    PetscCallThrow(PetscViewerCreate(PetscObjectComm((PetscObject)mat), &viewer));
    PetscCallThrow(PetscViewerSetType(viewer, PETSCVIEWERASCII));
    PetscCallThrow(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
    PetscCallThrow(PetscViewerFileSetName(viewer, filename));

    // Write the first line with matrix dimensions and global non-zeros (global_rows, global_cols, global_nz)
    if (rank == 0) {
        PetscCallThrow(PetscViewerASCIIPrintf(viewer, "%d,%d,%d\n", m, n, nz_global));
    }

    PetscCallThrow(PetscViewerASCIIPushSynchronized(viewer));

    for (PetscInt row = rstart; row < rend; row++) {
        PetscInt           ncols;
        const PetscInt*    cols;
        const PetscScalar* vals;
        PetscCallThrow(MatGetRow(mat, row, &ncols, &cols, &vals));
        for (PetscInt j = 0; j < ncols; j++) {
            // rowidx, colidx, value
            PetscCallThrow(PetscViewerASCIISynchronizedPrintf(viewer, "%d,%d,%.15e\n", row, cols[j], (double)PetscRealPart(vals[j])));
        }
        PetscCallThrow(MatRestoreRow(mat, row, &ncols, &cols, &vals));
    }

    PetscCallThrow(PetscViewerFlush(viewer));
    PetscCallThrow(PetscViewerASCIIPopSynchronized(viewer));
    PetscCallThrow(PetscViewerDestroy(&viewer));
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
    PetscCallThrow(MPI_Comm_rank(Madupite::getCommWorld(), &rank));

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
    PetscCallThrow(MPI_Comm_rank(Madupite::getCommWorld(), &rank));

    if (rank == 0) {
        PetscCallThrow(PetscMalloc1(size, &allIndices));
        PetscCallThrow(PetscMalloc1(size, &recvcounts));
        PetscCallThrow(PetscMalloc1(size, &displs));

        PetscCallThrow(MPI_Gather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, Madupite::getCommWorld()));

        displs[0] = 0;
        for (PetscInt i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        PetscCallThrow(MPI_Gatherv(indices, localSize, MPI_INT, allIndices, recvcounts, displs, MPI_INT, 0, Madupite::getCommWorld()));

        std::ofstream out(filename);
        for (PetscInt i = 0; i < size; ++i) {
            out << allIndices[i] << "\n";
        }
        out.close();

        PetscCallThrow(PetscFree(allIndices));
        PetscCallThrow(PetscFree(recvcounts));
        PetscCallThrow(PetscFree(displs));
    } else {
        PetscCallThrow(MPI_Gather(&localSize, 1, MPI_INT, NULL, 0, MPI_INT, 0, Madupite::getCommWorld()));
        PetscCallThrow(MPI_Gatherv(indices, localSize, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, Madupite::getCommWorld()));
    }

    PetscCallThrow(ISRestoreIndices(is, &indices));
}

void MDP::writeJSONmetadata()
{
    PetscMPIInt size;
    MPI_Comm_size(Madupite::getCommWorld(), &size);

    // writes model specifics to file per launch of MDP::solve
    // ksp_type is written in MDP::solve (since it's only known there)
    json_writer_->add_data("num_states", num_states_);
    json_writer_->add_data("num_actions", num_actions_);
    json_writer_->add_data("discount_factor", discount_factor_);
    json_writer_->add_data("max_iter_pi", max_iter_pi_);
    json_writer_->add_data("max_iter_ksp", max_iter_ksp_);
    json_writer_->add_data("alpha", alpha_);
    json_writer_->add_data("atol_pi", atol_pi_);
    json_writer_->add_data("num_ranks", size);

    if (file_policy_[0] != '\0') {
        json_writer_->add_data("file_policy", file_policy_);
    }
    if (file_cost_[0] != '\0') {
        json_writer_->add_data("file_cost", file_cost_);
    }
    if (file_stats_[0] != '\0') {
        json_writer_->add_data("file_stats", file_stats_);
    }
}
