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
#include <memory>
#include <string>
#include <vector>

#include "JsonWriter.h"

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
    const MPI_Comm              comm_;       // MPI communicator
    std::unique_ptr<JsonWriter> jsonWriter_; // used to write statistics (residual norm, times etc.) to file
public:
    MDP(MPI_Comm comm = PETSC_COMM_WORLD);
    ~MDP();

    // MDP Setup
    void           splitOwnership();
    PetscErrorCode setValuesFromOptions();
    void           setOption(const char* option, const char* value, bool setValues = false);
    void           loadFromBinaryFile(); // TODO split into P and g

    // functions needed for parallel matrix generation
    std::pair<int, int> request_states(
        int nstates, int mactions, int matrix, int prealloc); // matrix = 0: transitionProbabilityTensor_, matrix = 1: stageCostMatrix_
    void fillRow(std::vector<int>& idxs, std::vector<double>& vals, int i, int matrix);
    void assembleMatrix(int matrix);

    std::pair<int, int> getStateOwnershipRange();
    std::pair<int, int> getMDPSize();
    void                createCostMatrix(); // no preallocation needed since it's a dense matrix
    void                createTransitionProbabilityTensor(
                       PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz); // full preallocation freedom
    void createTransitionProbabilityTensor();                                                                       // no preallocation

    template <typename Func> void generateStageCostMatrix(Func g);
    template <typename Func> void generateTransitionProbabilityTensor(Func P);
    template <typename Func>
    void generateTransitionProbabilityTensor(Func P, PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz);

    // MDP Algorithm
    void extractGreedyPolicy(const Vec& V, PetscInt* policy, PetscReal& residualNorm);
    void constructFromPolicy(const PetscInt* policy, Mat& transitionProbabilities, Vec& stageCosts);
    void iterativePolicyEvaluation(const Mat& jacobian, const Vec& stageCosts, Vec& V, KSPContext& ctx);
    void createJacobian(Mat& jacobian, const Mat& transitionProbabilities, JacobianContext& ctx);
    void inexactPolicyIteration();
    // virtual PetscErrorCode benchmarkIPI(const Vec &V0, IS &policy, Vec &optimalCost);

    // maybe private, depends on usage of output / storing results
    void writeVec(const Vec& vec, const PetscChar* filename);
    void writeIS(const IS& is, const PetscChar* filename);

    // probably private
    static void cvgTest(
        KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason* reason, void* ctx); // Test if residual norm is smaller than alpha * r0_norm
    static void jacobianMultiplication(Mat mat, Vec x, Vec y);                         // defines matrix vector product for jacobian shell
    static void jacobianMultiplicationTranspose(
        Mat mat, Vec x, Vec y); // defines tranposed matrix vector product for jacobian shell (needed for some KSP methods)
    void writeJSONmetadata();

    // user specified options
    enum mode { MINCOST, MAXREWARD };
    mode      mode_;
    PetscInt  numStates_;  // global; read from file or via setOption
    PetscInt  numActions_; // global; read from file or via setOption
    PetscReal discountFactor_;
    PetscInt  maxIter_PI_;
    PetscInt  maxIter_KSP_;
    PetscInt  numPIRuns_; // used for MDP::benchmarkIPI()
    PetscReal rtol_KSP_;
    PetscReal atol_PI_;
    PetscChar file_P_[PETSC_MAX_PATH_LEN];      // input
    PetscChar file_g_[PETSC_MAX_PATH_LEN];      // input
    PetscChar file_policy_[PETSC_MAX_PATH_LEN]; // output
    PetscChar file_cost_[PETSC_MAX_PATH_LEN];   // output
    PetscChar file_stats_[PETSC_MAX_PATH_LEN];  // output

    static constexpr std::vector<PetscInt> emptyVec = {}; // can be used by the user for d_nnz and o_nnz

    // derived parameters
    PetscInt localNumStates_;  // number of states owned by this rank
    PetscInt rank_;            // rank of this process
    PetscInt size_;            // number of processes
    PetscInt P_start_, P_end_; // local row range of transitionProbabilityTensor_
    PetscInt g_start_, g_end_; // local row range of stageCostMatrix_

    Mat transitionProbabilityTensor_; // transition probability tensor
    Mat stageCostMatrix_;             // stage cost matrix (also rewards possible)
    Mat costMatrix_;                  // cost matrix used in extractGreedyPolicy
    Vec costVector_;                  // cost vector used in extractGreedyPolicy
};

#include "MDP/MDP_setup.tpp"

#endif // DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
