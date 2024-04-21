//
// Created by robin on 27.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <mpi.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsystypes.h>
#include <petscvec.h>

#include <exception>
#include <string>
#include <vector>

#include "JsonWriter.h"

using Costfunc = std::function<double(int, int)>;
using Probfunc = std::function<std::pair<std::vector<double>, std::vector<int>>(int, int)>;

class PetscException : public std::exception {
private:
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

class MDP {
public:
    MDP(MPI_Comm comm = PETSC_COMM_WORLD);
    ~MDP();

    // MDP Setup
    void           splitOwnership();
    PetscErrorCode setValuesFromOptions();
    void           setOption(const char* option, const char* value, bool setValues = false);
    void           loadTransitionProbabilityTensor();
    void           loadStageCostMatrix();
    void           createStageCostMatrix(); // no preallocation needed since it's a dense matrix
    void           createTransitionProbabilityTensorPrealloc();
    void           createTransitionProbabilityTensor();
    void           assembleMatrix(int matrix);

    // functions not needed right now but maybe for cython wrapper
    std::pair<int, int> request_states(int nstates, int mactions, int matrix,
        int prealloc); // matrix = 0: transitionProbabilityTensor_, matrix = 1: stageCostMatrix_; maybe needed for cython wrapper
    void                fillRow(std::vector<int>& idxs, std::vector<double>& vals, int i, int matrix); // maybe needed for cython wrapper
    std::pair<int, int> getStateOwnershipRange();                                                      // maybe needed for cython wrapper
    std::pair<int, int> getMDPSize();                                                                  // maybe needed for cython wrapper

    void setSourceTransitionProbabilityTensor(const char* filename);
    void setSourceTransitionProbabilityTensor(const Probfunc P); // no preallocation
    void setSourceTransitionProbabilityTensor(
        const Probfunc P, PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz); // full preallocation freedom
    void setSourceStageCostMatrix(const char* filename);
    void setSourceStageCostMatrix(const Costfunc g);

    void setUp(); // call after setting sources

    // MDP Algorithm
    void extractGreedyPolicy(const Vec& V, PetscInt* policy, PetscReal& residualNorm);
    void constructFromPolicy(const PetscInt* policy, Mat& transitionProbabilities, Vec& stageCosts);
    void iterativePolicyEvaluation(const Mat& jacobian, const Vec& stageCosts, Vec& V, KSPContext& ctx);
    void createJacobian(Mat& jacobian, const Mat& transitionProbabilities, JacobianContext& ctx);
    void inexactPolicyIteration();

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

    // user specified options
    enum mode { MINCOST, MAXREWARD };
    enum source { FILE, FUNCTION };
    mode      mode_;
    PetscInt  numStates_;  // global; read from file or via setOption
    PetscInt  numActions_; // global; read from file or via setOption
    PetscReal discountFactor_;
    PetscInt  maxIter_PI_;
    PetscInt  maxIter_KSP_;
    // PetscInt  numPIRuns_; // used for MDP::benchmarkIPI()
    PetscReal rtol_KSP_;
    PetscReal atol_PI_;
    // PetscChar file_P_[PETSC_MAX_PATH_LEN];      // input
    // PetscChar file_g_[PETSC_MAX_PATH_LEN];      // input
    PetscChar file_policy_[PETSC_MAX_PATH_LEN]; // output
    PetscChar file_cost_[PETSC_MAX_PATH_LEN];   // output
    PetscChar file_stats_[PETSC_MAX_PATH_LEN];  // output
    PetscInt  p_src_;                           // 0: from file, 1: from function, -1: not set
    PetscInt  g_src_;                           // 0: from file, 1: from function, -1: not set

    static constexpr std::vector<PetscInt> emptyVec = {}; // can be used by the user for d_nnz and o_nnz

    const MPI_Comm comm_; // MPI communicator

    // derived parameters
    PetscInt localNumStates_;  // number of states owned by this rank
    PetscInt rank_;            // rank of this process
    PetscInt size_;            // number of processes
    PetscInt p_start_, p_end_; // local row range of transitionProbabilityTensor_
    PetscInt g_start_, g_end_; // local row range of stageCostMatrix_

    Mat transitionProbabilityTensor_; // transition probability tensor
    Mat stageCostMatrix_;             // stage cost matrix (also rewards possible)
    Mat costMatrix_;                  // cost matrix used in extractGreedyPolicy
    Vec costVector_;                  // cost vector used in extractGreedyPolicy

    PetscChar                                                          p_file_name_[PETSC_MAX_PATH_LEN];
    PetscChar                                                          g_file_name_[PETSC_MAX_PATH_LEN];
    Probfunc                                                           p_func_;
    Costfunc                                                           g_func_;
    std::array<PetscInt, 4>                                            p_file_meta_; // metadata when P is loaded from file (ClassID, rows, cols, nnz)
    std::array<PetscInt, 4>                                            g_file_meta_; // metadata when g is loaded from file (ClassID, rows, cols, nnz)
    PetscBool                                                          p_prealloc_;
    std::tuple<PetscInt, std::vector<int>, PetscInt, std::vector<int>> p_nnz_; // preallocation for P (if passed by user) [d_nz, d_nnz, o_nz, o_nnz]

    JsonWriter* jsonWriter_; // used to write statistics (residual norm, times etc.) to file
};

#endif // DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
