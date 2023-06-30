//
// Created by robin on 28.06.23.
//

#include "GrowthModel.h"

GrowthModel::GrowthModel() {
    numK_ = 20;
    numZ_ = 2;
    discountFactor_ = 0.98;
    rho_ = 0.33;
    riskAversionParameter_ = 0.5;
    P_z_ = nullptr;
    z_ = nullptr;
    numStates_ = numK_ * numZ_;
    numActions_ = numK_;

}

PetscErrorCode GrowthModel::generateKInterval() {
    PetscReal z_vals[2];
    PetscInt z_indices[2] = {0, numZ_ - 1};
    VecGetValues(z_, 2, z_indices, z_vals);
    VecCreateSeq(PETSC_COMM_SELF, numK_, &k_);
    PetscReal *k_vals;
    PetscMalloc1(numK_, &k_vals);

    PetscReal k_star_z1 = std::pow(discountFactor_ * rho_ * z_vals[0] / (1 - discountFactor_), 1 / (1 - rho_));
    PetscReal k_star_z2 = std::pow(discountFactor_ * rho_ * z_vals[1] / (1 - discountFactor_), 1 / (1 - rho_));
    PetscReal k_min = k_star_z1 - 0.1 * (k_star_z2 - k_star_z1);
    PetscReal k_max = k_star_z2 + 0.1 * (k_star_z2 - k_star_z1);

    PetscMalloc1(numK_, &k_vals);
    PetscReal k_incr = (k_max - k_min) / (numK_ - 1);
    for(PetscInt i = 0; i < numK_; ++i) {
        k_vals[i] = k_min + i * k_incr;
    }

    IS indices;
    ISCreateStride(PETSC_COMM_SELF, numK_, 0, 1, &indices);
    const PetscInt *indices_arr;
    ISGetIndices(indices, &indices_arr);
    VecSetValues(k_, numK_, indices_arr, k_vals, INSERT_VALUES);
    ISRestoreIndices(indices, &indices_arr);
    ISDestroy(&indices);

    VecAssemblyBegin(k_);
    VecAssemblyEnd(k_);

    PetscFree(k_vals);

    return 0;
}

PetscErrorCode GrowthModel::calculateAvailableResources() {
    PetscReal *B_vals;
    PetscMalloc1(numK_ * numZ_, &B_vals);
    VecCreateSeq(PETSC_COMM_SELF, numK_ * numZ_, &B_);

    const PetscReal *k_vals, *z_vals;
    VecGetArrayRead(k_, &k_vals);
    VecGetArrayRead(z_, &z_vals);
    for(PetscInt i = 0; i < numK_; ++i) {
        for(PetscInt j = 0; j < numZ_; ++j) {
            // B[i,j] = z[j] * (k[i]^rho) + k[i]
            B_vals[ij2s(i, j)] = z_vals[j] * std::pow(k_vals[i], rho_) + k_vals[i];
        }
    }
    VecRestoreArrayRead(k_, &k_vals);
    VecRestoreArrayRead(z_, &z_vals);

    IS indices;
    ISCreateStride(PETSC_COMM_SELF, numK_ * numZ_, 0, 1, &indices);
    const PetscInt *indices_arr;
    ISGetIndices(indices, &indices_arr);
    VecSetValues(B_, numK_ * numZ_, indices_arr, B_vals, INSERT_VALUES);
    ISRestoreIndices(indices, &indices_arr);
    ISDestroy(&indices);

    VecAssemblyBegin(B_);
    VecAssemblyEnd(B_);

    PetscFree(B_vals);

    return 0;
}



PetscErrorCode GrowthModel::calculateFeasibleActions() {
    // find max_a {a | 0 <= a < nk, B[i,j] - k[a] >= 0}
    PetscInt *A_vals;
    PetscMalloc1(numK_ * numZ_, &A_vals);
    const PetscReal *k_vals, *B_vals;
    VecGetArrayRead(k_, &k_vals);
    VecGetArrayRead(B_, &B_vals);

    for(PetscInt i = 0; i < numK_; ++i) {
        for(PetscInt j = 0; j < numZ_; ++j) {
            PetscInt a = 0;
            while(a < numK_ && B_vals[ij2s(i, j)] - k_vals[a] > 0) {
                ++a;
            }
            A_vals[ij2s(i, j)] = std::max(a - 1, 0);
        }
    }

    VecRestoreArrayRead(k_, &k_vals);
    VecRestoreArrayRead(B_, &B_vals);
    ISCreateGeneral(PETSC_COMM_SELF, numK_ * numZ_, A_vals, PETSC_COPY_VALUES, &A_);
    PetscFree(A_vals);

    return 0;
}