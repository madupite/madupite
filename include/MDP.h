//
// Created by robin on 27.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <mpi.h> //TODO this should not be needed if petscerrror.h includes mpi.h
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "JsonWriter.h"

using Costfunc = std::function<double(int, int)>;
using Probfunc = std::function<std::pair<std::vector<double>, std::vector<int>>(int, int)>;

class PetscException : public std::exception {
    int         ierr;
    std::string message;

public:
    PetscException(int ierr, const std::string& message)
        : ierr(ierr)
        , message(message)
    {
    }

    const char* what() const noexcept override { return message.c_str(); }

    int code() const noexcept { return ierr; }
};

class MadupiteException : public std::exception {
    std::string message;

public:
    MadupiteException(const std::string& message)
        : message(message)
    {
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#define PetscCallNoThrow(...)                                                                                                                        \
    do {                                                                                                                                             \
        PetscStackUpdateLine;                                                                                                                        \
        PetscErrorCode ierr = __VA_ARGS__;                                                                                                           \
        if (PetscUnlikely(ierr != PETSC_SUCCESS)) {                                                                                                  \
            PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_IN_CXX, PETSC_NULLPTR);                           \
        }                                                                                                                                            \
    } while (0)

#define PetscCallThrow(...)                                                                                                                          \
    do {                                                                                                                                             \
        PetscStackUpdateLine;                                                                                                                        \
        PetscErrorCode ierr = __VA_ARGS__;                                                                                                           \
        if (PetscUnlikely(ierr != PETSC_SUCCESS)) {                                                                                                  \
            char* msg;                                                                                                                               \
            PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_IN_CXX, PETSC_NULLPTR);                           \
            PetscErrorMessage(ierr, PETSC_NULLPTR, &msg);                                                                                            \
            throw PetscException(ierr, std::string(msg));                                                                                            \
        }                                                                                                                                            \
    } while (0)

#define PetscThrow(comm, ierr, ...)                                                                                                                  \
    do {                                                                                                                                             \
        char* msg;                                                                                                                                   \
        PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_INITIAL, __VA_ARGS__);                                           \
        PetscErrorMessage(ierr, PETSC_NULLPTR, &msg);                                                                                                \
        throw PetscException(ierr, std::string(msg));                                                                                                \
    } while (0)

struct KSPContext {
    PetscInt  maxIter;       // input
    PetscReal threshold;     // input
    PetscInt  kspIterations; // output
};

struct JacobianContext {
    Mat       P_pi;
    PetscReal discountFactor;
};

class Madupite {
    static std::shared_ptr<Madupite> instance;
    static std::mutex                mtx;

    Madupite()                                 = default;
    Madupite(const Madupite&)                  = delete;
    const Madupite& operator=(const Madupite&) = delete;

public:
    static std::shared_ptr<Madupite> initialize(int* argc = nullptr, char*** argv = nullptr);

    static std::shared_ptr<Madupite> get()
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!instance)
            throw MadupiteException("Madupite not initialized");
        return instance;
    }

    ~Madupite()
    {
        std::lock_guard<std::mutex> lock(mtx);
        // Finalize MPI and PETSc
        PetscFinalize();
    }
};

class MDP {
public:
    MDP(std::shared_ptr<Madupite> madupite, MPI_Comm comm = PETSC_COMM_WORLD);
    ~MDP();
    void           setOption(const char* option, const char* value, bool setValues = false);
    PetscErrorCode setValuesFromOptions();
    void           setSourceTransitionProbabilityTensor(const char* filename);
    void           setSourceTransitionProbabilityTensor(const Probfunc P); // no preallocation
    void           setSourceTransitionProbabilityTensor(
                  const Probfunc P, PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz); // full preallocation freedom
    void setSourceStageCostMatrix(const char* filename);
    void setSourceStageCostMatrix(const Costfunc g);
    void setUp(); // call after setting sources
    void inexactPolicyIteration();

    static constexpr std::vector<PetscInt> emptyVec = {}; // can be used by the user for d_nnz and o_nnz

private:
    // MDP Setup
    void splitOwnership();
    void loadTransitionProbabilityTensor();
    void loadStageCostMatrix();
    void createStageCostMatrix(); // no preallocation needed since it's a dense matrix
    void createTransitionProbabilityTensorPrealloc();
    void createTransitionProbabilityTensor();
    void assembleMatrix(int matrix);

    // functions not needed right now but maybe for cython wrapper
    std::pair<int, int> request_states(int nstates, int mactions, int matrix,
        int prealloc); // matrix = 0: transitionProbabilityTensor_, matrix = 1: stageCostMatrix_; maybe needed for cython wrapper
    void                fillRow(std::vector<int>& idxs, std::vector<double>& vals, int i, int matrix); // maybe needed for cython wrapper
    std::pair<int, int> getStateOwnershipRange();                                                      // maybe needed for cython wrapper
    std::pair<int, int> getMDPSize();                                                                  // maybe needed for cython wrapper

    // MDP Algorithm
    void extractGreedyPolicy(const Vec& V, PetscInt* policy, PetscReal& residualNorm);
    void constructFromPolicy(const PetscInt* policy, Mat& transitionProbabilities, Vec& stageCosts);
    void iterativePolicyEvaluation(const Mat& jacobian, const Vec& stageCosts, Vec& V, KSPContext& ctx);
    void createJacobian(Mat& jacobian, const Mat& transitionProbabilities, JacobianContext& ctx);

    // maybe private, depends on usage of output / storing results
    void writeVec(const Vec& vec, const PetscChar* filename);
    void writeIS(const IS& is, const PetscChar* filename);

    // probably private
    static void cvgTest(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason* reason,
        void* ctx); // Test if residual norm is smaller than alpha * r0_norm; todo: keep this or move to documentation to show user how to implement
                    // own cvg test; not used in madupite for performance reasons
    static void jacobianMultiplication(Mat mat, Vec x, Vec y); // defines matrix vector product for jacobian shell
    static void jacobianMultiplicationTranspose(
        Mat mat, Vec x, Vec y); // defines tranposed matrix vector product for jacobian shell (needed for some KSP methods)
    void writeJSONmetadata();

    // Madupite, MPI, JSON output
    const std::shared_ptr<Madupite> madupite_;
    const MPI_Comm                  comm_;       // MPI communicator
    std::unique_ptr<JsonWriter>     jsonWriter_; // used to write statistics (residual norm, times etc.) to file

    // user specified options
    enum mode { MINCOST, MAXREWARD };
    enum source { FILE, FUNCTION };
    mode      mode_;
    PetscInt  numStates_;  // global; read from file or via setOption
    PetscInt  numActions_; // global; read from file or via setOption
    PetscReal discountFactor_;
    PetscInt  maxIter_PI_;
    PetscInt  maxIter_KSP_;
    PetscReal rtol_KSP_;
    PetscReal atol_PI_;
    PetscChar file_policy_[PETSC_MAX_PATH_LEN]; // output
    PetscChar file_cost_[PETSC_MAX_PATH_LEN];   // output
    PetscChar file_stats_[PETSC_MAX_PATH_LEN];  // output
    PetscInt  p_src_;                           // 0: from file, 1: from function, -1: not set
    PetscInt  g_src_;                           // 0: from file, 1: from function, -1: not set
    PetscChar p_file_name_[PETSC_MAX_PATH_LEN];
    PetscChar g_file_name_[PETSC_MAX_PATH_LEN];
    Probfunc  p_func_;
    Costfunc  g_func_;
    PetscBool p_prealloc_;

    std::tuple<PetscInt, std::vector<int>, PetscInt, std::vector<int>> p_nnz_; // preallocation for P (if passed by user) [d_nz, d_nnz, o_nz, o_nnz]

    // derived parameters
    PetscInt                localNumStates_;  // number of states owned by this rank
    PetscInt                rank_;            // rank of this process
    PetscInt                size_;            // number of processes
    PetscInt                p_start_, p_end_; // local row range of transitionProbabilityTensor_
    PetscInt                g_start_, g_end_; // local row range of stageCostMatrix_
    std::array<PetscInt, 4> p_file_meta_;     // metadata when P is loaded from file (ClassID, rows, cols, nnz)
    std::array<PetscInt, 4> g_file_meta_;     // metadata when g is loaded from file (ClassID, rows, cols, nnz)

    // MDP data
    Mat transitionProbabilityTensor_; // transition probability tensor (nm x n; MPIAIJ)
    Mat stageCostMatrix_;             // stage cost matrix (also rewards possible) (n x m; DENSE)
    Mat costMatrix_;                  // cost matrix used in extractGreedyPolicy, as member to avoid reallocation (n x m; DENSE)
    Vec costVector_;                  // cost vector used in extractGreedyPolicy, as member to avoid reallocation (n; DENSE)
};

#endif // DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
