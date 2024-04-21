#include "MDP.h"

template <typename Func>
void MDP::generateTransitionProbabilityTensor(Func P) { // no preallocation (not recommended)
    createTransitionProbabilityTensor();


}

template <typename Func>
void MDP::generateTransitionProbabilityTensor(Func P, PetscInt d_nz, const std::vector<int> &d_nnz, PetscInt o_nz, const std::vector<int> &o_nnz) {
    createTransitionProbabilityTensor(d_nz, d_nnz, o_nz, o_nnz);


}

template <typename Func>
void MDP::generateStageCostMatrix(Func g) {
    createCostMatrix(); // no preallocation needed since it's a dense matrix


}
