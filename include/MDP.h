//
// Created by robin on 27.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H

#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

#include <memory>
#include <vector>

#include "JsonWriter.h"
#include "madupite_errors.h"
#include "madupite_matrix.h"

class Madupite {
    static std::shared_ptr<Madupite> instance;
    static std::mutex                mtx;

    Madupite() = default;

    // Forbid copy and move for now
    Madupite(const Madupite&)                  = delete;
    Madupite(Madupite&&)                       = delete;
    const Madupite& operator=(const Madupite&) = delete;
    const Madupite& operator=(Madupite&&)      = delete;

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
    void setOption(const char* option, const char* value = NULL);
    void clearOptions();
    void setSourceTransitionProbabilityTensor(const std::string filename);
    void setSourceTransitionProbabilityTensor(const Probfunc& P); // no preallocation
    void setSourceTransitionProbabilityTensor(
        const Probfunc& P, PetscInt d_nz, const std::vector<int>& d_nnz, PetscInt o_nz, const std::vector<int>& o_nnz);
    void setSourceStageCostMatrix(const std::string filename);
    void setSourceStageCostMatrix(const Costfunc& g);
    void setUp(); // call after setting sources
    void solve();

    PetscErrorCode setValuesFromOptions();

private:
    // MDP Setup
    void splitOwnership();
    void loadTransitionProbabilityTensor();
    void loadStageCostMatrix();
    void createStageCostMatrix(); // no preallocation needed since it's a dense matrix
    void createTransitionProbabilityTensorPrealloc();
    void createTransitionProbabilityTensor();

    // MDP Algorithm
    void      reshapeCostVectorToCostMatrix(const Vec costVector, Mat costMatrix);
    PetscReal getGreedyPolicyAndResidualNorm(Mat costMatrix, const Vec V, const std::unique_ptr<PetscInt[]>& policy);
    Mat       getTransitionProbabilities(const std::unique_ptr<PetscInt[]>& policy);
    Vec       getStageCosts(const std::unique_ptr<PetscInt[]>& policy);
    PetscInt  iterativePolicyEvaluation(const Mat jacobian, const Vec stageCosts, PetscInt maxIter, PetscReal threshold, Vec V);
    Mat       createJacobian(const Mat transitionProbabilities, PetscReal discountFactor);

    // maybe private, depends on usage of output / storing results
    void writeVec(const Vec& vec, const PetscChar* filename);
    void writeIS(const IS& is, const PetscChar* filename);
    void writeJSONmetadata();

    // Madupite, MPI, JSON output
    const std::shared_ptr<Madupite> madupite_;
    const MPI_Comm                  comm_;       // MPI communicator
    std::unique_ptr<JsonWriter>     jsonWriter_; // used to write statistics (residual norm, times etc.) to file

    // user specified options
    enum mode { MINCOST, MAXREWARD };
    enum source { FILE, FUNCTION };
    mode mode_;
    // TODO: change to unsigned int for better scalability? then remove -1
    PetscInt  numStates_  = -1; // global; read from file or via setOption
    PetscInt  numActions_ = -1; // global; read from file or via setOption
    PetscReal discountFactor_;
    PetscInt  maxIter_PI_;
    PetscInt  maxIter_KSP_;
    PetscReal alpha_;
    PetscReal atol_PI_;
    PetscChar file_policy_[PETSC_MAX_PATH_LEN]; // output
    PetscChar file_cost_[PETSC_MAX_PATH_LEN];   // output
    PetscChar file_stats_[PETSC_MAX_PATH_LEN];  // output

    PetscInt    p_src_ = -1; // 0: from file, 1: from function, -1: not set
    PetscInt    g_src_ = -1; // 0: from file, 1: from function, -1: not set
    std::string p_file_name_;
    std::string g_file_name_;
    Probfunc    p_func_;
    Costfunc    g_func_;
    bool        p_prealloc_ = false;

    // preallocation for P (if passed by user) [d_nz, d_nnz, o_nz, o_nnz]
    std::tuple<PetscInt, std::vector<int>, PetscInt, std::vector<int>> p_nnz_;

    // derived parameters
    PetscInt localNumStates_;  // number of states owned by this rank
    PetscInt rank_;            // rank of this process
    PetscInt size_;            // number of processes
    PetscInt p_start_, p_end_; // local row range of transitionProbabilityTensor_
    PetscInt g_start_, g_end_; // local row range of stageCostMatrix_

    std::array<PetscInt, 4> p_file_meta_; // metadata when P is loaded from file (ClassID, rows, cols, nnz)
    std::array<PetscInt, 4> g_file_meta_; // metadata when g is loaded from file (ClassID, rows, cols, nnz)

    // MDP data
    Mat transitionProbabilityTensor_ = nullptr; // transition probability tensor (nm x n; MPIAIJ)
    Mat stageCostMatrix_             = nullptr; // stage cost matrix (also rewards possible) (n x m; DENSE)

    bool setupCalled = false;
};

#endif // DISTRIBUTED_INEXACT_POLICY_ITERATION_MDP_H
