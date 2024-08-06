#pragma once

#include <petscksp.h>

#include <memory>

#include "JsonWriter.h"
#include "madupite_errors.h"
#include "madupite_matrix.h"

// convenience import so that user code doesn't need to include MDP_matrix.h
#include "MDP_matrix.h"

class Madupite {
    static std::shared_ptr<Madupite> instance;
    static std::mutex                mtx;
    static MPI_Comm                  comm_;

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
    static MPI_Comm getCommWorld() { return comm_; }

    ~Madupite()
    {
        std::lock_guard<std::mutex> lock(mtx);
        PetscFinalize();
    }
};

class MDP {
public:
    MDP(std::shared_ptr<Madupite> madupite, MPI_Comm comm = Madupite::getCommWorld());
    PetscErrorCode setOption(const char* option, const char* value = NULL);
    void           clearOptions();
    void           setStageCostMatrix(const Matrix& g);
    void           setTransitionProbabilityTensor(const Matrix& P);
    void           setUp();
    void           solve();

    PetscErrorCode setValuesFromOptions();

private:
    void      reshapeCostVectorToCostMatrix(const Vec costVector, Mat costMatrix);
    PetscReal getGreedyPolicyAndResidualNorm(Mat costMatrix, const Vec V, const std::unique_ptr<PetscInt[]>& policy);
    Mat       getTransitionProbabilities(const std::unique_ptr<PetscInt[]>& policy);
    Vec       getStageCosts(const std::unique_ptr<PetscInt[]>& policy);
    PetscInt  iterativePolicyEvaluation(const Mat jacobian, const Vec stageCosts, PetscInt maxIter, PetscReal threshold, Vec V);
    Mat       createJacobian(const Mat transitionProbabilities, PetscReal discountFactor);

    // TODO: remove
    void writeMat(const Mat& mat, const PetscChar* filename);
    void writeVec(const Vec& vec, const PetscChar* filename);
    void writeIS(const IS& is, const PetscChar* filename);
    void writeJSONmetadata();

    const std::shared_ptr<Madupite> madupite_;
    const MPI_Comm                  comm_;
    std::unique_ptr<JsonWriter>     json_writer_;

    enum mode { MINCOST, MAXREWARD };
    mode mode_;
    // TODO: change to unsigned int for better scalability? then remove -1
    PetscInt  num_states_  = -1;
    PetscInt  num_actions_ = -1;
    PetscReal discount_factor_;
    PetscInt  max_iter_pi_;
    PetscInt  max_iter_ksp_;
    PetscReal alpha_;
    PetscReal atol_pi_;
    PetscChar file_policy_[PETSC_MAX_PATH_LEN];
    PetscChar file_cost_[PETSC_MAX_PATH_LEN];
    PetscChar file_stats_[PETSC_MAX_PATH_LEN];
    PetscChar file_optimal_transition_probabilities_[PETSC_MAX_PATH_LEN];
    PetscChar file_optimal_stage_costs_[PETSC_MAX_PATH_LEN];

    PetscInt local_num_states_;

    // TODO: remove - don't need to be stashed in MDP object; replace with getters
    PetscInt g_start_, g_end_;

    Matrix transition_probability_tensor_;
    Matrix stage_cost_matrix_;

    bool setup_called = false;
};
