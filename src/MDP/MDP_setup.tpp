template <typename Func>
void MDP::generateTransitionProbabilityTensor(Func P, PetscInt d_nz, const std::vector<int> &d_nnz, PetscInt o_nz, const std::vector<int> &o_nnz) {
    // P: (s, a) -> [probabilities, indices]

    // create transition probability tensor
    createTransitionProbabilityTensor(); // no preallocation (not recommended)
    // Todo: use preallocation if user provides it
    //createTransitionProbabilityTensor(d_nz, d_nnz, o_nz, o_nnz);

    // fill 
    for(PetscInt stateInd = P_start_ / numActions_; stateInd < P_end_ / numActions_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            auto [values, indices] = P(stateInd, actionInd);
            PetscInt rowInd = stateInd * numActions_ + actionInd;
            PetscCallThrow(MatSetValues(transitionProbabilityTensor_, 1, &rowInd, indices.size(), indices.data(), values.data(), INSERT_VALUES));
        }
    }
    assembleMatrix(0);
}

template <typename Func>
void MDP::generateStageCostMatrix(Func g) {
    // create cost matrix
    createCostMatrix(); // no preallocation needed since it's a dense matrix

    // fill
    double value;
    for(PetscInt stateInd = g_start_; stateInd < g_end_; ++stateInd) {
        for(PetscInt actionInd = 0; actionInd < numActions_; ++actionInd) {
            value = g(stateInd, actionInd);
            PetscCallThrow(MatSetValue(stageCostMatrix_, stateInd, actionInd, value, INSERT_VALUES));
        }
    }
    assembleMatrix(1);
}