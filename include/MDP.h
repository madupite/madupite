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

/**
 * @brief Context struct to provide and obtain information for the Krylov subspace method.
 *
 */
struct KSPContext {
    PetscInt  maxIter;       /// input to KSP
    PetscReal threshold;     /// input to KSP
    PetscInt  kspIterations; /// output from KSP
};

/**
 * @brief Context struct to provide information for the Jacobian matrix shell.
 *
 */
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

/**
 * @brief Data structure to represent a Markov Decision Process (MDP)
 */
class MDP {
public:
    /**
     * @brief Constructor
     *
     * @param madupite madupite instance that is used to initialize/finalize MPI and PETSc
     * @param comm MPI communicator that is used for the MDP (default: PETSC_COMM_WORLD)
     */
    MDP(std::shared_ptr<Madupite> madupite, MPI_Comm comm = PETSC_COMM_WORLD);
    /**
     * @brief Destroy the MDP object
     *
     */
    ~MDP();
    /**
     * @brief Add options to the PETSc options database
     *
     * @param option name of the option, should have a dash "-" prepended
     * @param value value of the option, can be NULL
     * @param setValues to be removed (?, TODO)
     */
    void setOption(const char* option, const char* value = NULL, bool setValues = false);
    /**
     * @brief erase all options from the PETSc options database
     *
     */
    void           clearOptions();
    PetscErrorCode setValuesFromOptions();
    void           setSourceTransitionProbabilityTensor(const char* filename);
    void           setSourceTransitionProbabilityTensor(const Probfunc& P); // no preallocation
    void           setSourceTransitionProbabilityTensor(
                  const Probfunc& P, PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz);
    void setSourceStageCostMatrix(const char* filename);
    void setSourceStageCostMatrix(const Costfunc& g);
    void setUp(); // call after setting sources
    /**
     * @brief Starts the inexact policy iteration algorithm using the specified sources and options.
     *
     */
    void solve();

private:
    // MDP Setup
    /**
     * @brief calculates the local number of states that a rank on the defined communicator owns. Internally,
     * [PetscSplitOwnership](https://petsc.org/release/manualpages/Sys/PetscSplitOwnership/) is used.
     *
     */
    void splitOwnership();
    /**
     * @brief Loads the transition probability tensor from a file.
     *
     */
    void loadTransitionProbabilityTensor();
    /**
     * @brief Loads the stage cost matrix from a file.
     *
     */
    void loadStageCostMatrix();
    /**
     * @brief Create a Stage Cost Matrix object based on the user specified options regarding size and filled using the user specified function g of
     * type Costfunc. The user does not need to provide information on the non-zeroes / sparsity pattern of the matrix because it is a dense matrix.
     *
     */
    void createStageCostMatrix();
    /**
     * @brief Create a Transition Probability Tensor Prealloc object based on the user specified options regarding size and filled using the user
     * specified function P of type Probfunc. The function also takes the sparsity pattern of the function provided by the user into account.
     *
     */
    void createTransitionProbabilityTensorPrealloc();
    /**
     * @brief Create a Transition Probability Tensor object based on the user specified options regarding size and filled using the user specified
     * function P of type Probfunc. This function is called when the user does not provide any information on the non-zeroes / sparsity pattern of the
     * matrix. Efficient preallocation is not possible in this case.
     *
     */
    void createTransitionProbabilityTensor();

    // functions not needed right now but maybe for cython wrapper
    std::pair<int, int> request_states(int nstates, int mactions, int matrix,
        int prealloc); // matrix = 0: transitionProbabilityTensor_, matrix = 1: stageCostMatrix_; maybe needed for cython wrapper
    void                fillRow(std::vector<int>& idxs, std::vector<double>& vals, int i, int matrix); // maybe needed for cython wrapper
    std::pair<int, int> getStateOwnershipRange();                                                      // maybe needed for cython wrapper
    std::pair<int, int> getMDPSize();                                                                  // maybe needed for cython wrapper

    // MDP Algorithm
    /**
     * @brief Policy improvement step: Computes the greedy policy based on the current value function $V$. $\pi = \arg\min_{\pi' \in \Pi}
     * \left\{g^{\pi'} + \gamma P^{\pi'}V\right\}$
     * TODO: does the function allocate memory?
     *
     * @param V (input) current value function as a vector of size $n$
     * @param policy (output) greedy policy as an int array of size $n$
     * @param residualNorm (output) infinity norm of the Bellman residual
     */
    void extractGreedyPolicy(const Vec& V, PetscInt* policy, PetscReal& residualNorm);
    /**
     * @brief Constructs the transition probability matrix $P^\pi$ $(n \times n)$ and stage cost vector $g^\pi$ $(n)$ based on the policy $\pi$.
     * TODO: does the function allocate memory?
     *
     * @param policy (input) policy as an int array of size $n$
     * @param transitionProbabilities (output) transition probability matrix $P^\pi$ $(n \times n)$
     * @param stageCosts (output) stage cost vector $g^\pi$ $(n)$
     */
    void constructFromPolicy(const PetscInt* policy, Mat& transitionProbabilities, Vec& stageCosts);
    /**
     * @brief Policy evaluation step: Computes the value function $V$ by solving the linear system $(I - \gamma P^\pi)V = g^\pi$. Internally, Krylov
     * subspace methods are used for efficiency.
     *
     * @param jacobian the jacobian matrix $(I - \gamma P^\pi)$ of size $n \times n$
     * @param stageCosts the stage cost vector $g^\pi$ of size $n$ (right-hand side of the linear system)
     * @param V the value function $V$ of size $n$ (solution of the linear system). Argument also serves as input for the initial guess of the
     * solution.
     * @param ctx
     */
    void iterativePolicyEvaluation(const Mat& jacobian, const Vec& stageCosts, Vec& V, KSPContext& ctx);
    /**
     * @brief Creates a matrix shell for the jacobian matrix $(I - \gamma P^\pi)$.
     *
     * @param jacobian (output) the initialized jacobian matrix shell
     * @param ctx context struct to provide information for the jacobian matrix shell ($P^\pi$, discount factor)
     */
    void createJacobian(Mat& jacobian, JacobianContext& ctx);

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
    const std::shared_ptr<Madupite>        madupite_;
    const MPI_Comm                         comm_;         // MPI communicator
    std::unique_ptr<JsonWriter>            jsonWriter_;   // used to write statistics (residual norm, times etc.) to file
    static constexpr std::vector<PetscInt> emptyVec = {}; // used internally if d_nnz or o_nnz are not set (preallocation)

    // user specified options
    enum mode { MINCOST, MAXREWARD };
    enum source { FILE, FUNCTION };
    mode      mode_;
    PetscInt  numStates_;  // global; read from file or via setOption
    PetscInt  numActions_; // global; read from file or via setOption
    PetscReal discountFactor_;
    PetscInt  maxIter_PI_;
    PetscInt  maxIter_KSP_;
    PetscReal alpha_;
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
