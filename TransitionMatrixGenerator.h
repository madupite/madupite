//
// Created by robin on 09.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_TRANSITIONMATRIXGENERATOR_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_TRANSITIONMATRIXGENERATOR_H

#include <random>
#include <petscmat.h>
#include <algorithm>
#include <cassert>


using idx_t = PetscInt;

double checkRowStochasticity(const Mat& A, idx_t numStates, idx_t numActions) {
    // check if matrix is row stochastic, returns inf norm of difference between 1 and sum of row
    // calculate row-wise sum of matrix: rowSum = A * ones
    // rowSum should be equal to actions (since every action should sum to 1)
    // calculate inf norm of difference between 1 and rowSum

    Vec ones, rowSum, solution;
    VecCreate(PETSC_COMM_WORLD, &ones);
    VecSetSizes(ones, PETSC_DECIDE, numStates*numActions);
    VecSetFromOptions(ones);
    VecSet(ones, 1.0);

    VecCreate(PETSC_COMM_WORLD, &rowSum);
    VecSetSizes(rowSum, PETSC_DECIDE, numStates);
    VecSetFromOptions(rowSum);

    VecCreate(PETSC_COMM_WORLD, &solution);
    VecSetSizes(solution, PETSC_DECIDE, numStates);
    VecSetFromOptions(solution);
    VecSet(solution, static_cast<PetscScalar>(numActions));

    MatMult(A, ones, rowSum);
    VecAXPY(solution, -1, rowSum);

    double norm;
    VecNorm(solution, NORM_INFINITY, &norm);

    VecDestroy(&ones);
    VecDestroy(&rowSum);
    VecDestroy(&solution);

    return norm;
}

void generateTransitionMatrix(Mat& A, idx_t numStates, idx_t numActions, double sparsityFactor, int seed) {
    // matrix A is numStates x numStates*numActions, in CRS format TODO: CRS
    // fill matrix with sparsityFactor*numStates*numActions non-zero entries per row
    // every block j*numStates:(j+1)*numStates is a transition matrix for action j => block must sum to 1

    double perturbFactor = 0.05;
    double stdDev = 0.01;


    // random engine setup
    std::mt19937_64 gen(seed);
    std::mt19937_64 gen2(seed + 42);
    std::mt19937_64 gen3(seed + 314);
    std::uniform_real_distribution<double> uniform_dis(0.0, 1.0);   // to generate probabilities
    std::normal_distribution<double> normal_dis(0.0, stdDev);      // to perturb nnz per action
    std::uniform_int_distribution<idx_t> ind_dis(0, numStates-1);   // to generate random indices
    //const idx_t nnzPerAction = (idx_t)(sparsityFactor * numStates);

    // create matmpiaij matrix
    //MatCreate(PETSC_COMM_WORLD, &A);
    //MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, numStates, numStates*numActions);
    //MatSetType(A, MATMPIAIJ);
    //MatMPIAIJSetPreallocation(A, nnzPerAction, PETSC_NULL, nnzPerAction, PETSC_NULL);

    // dense
    MatCreateSeqDense(PETSC_COMM_WORLD, numStates, numStates*numActions, PETSC_NULL, &A);

    // fill matrix
    for(idx_t rowInd = 0; rowInd < numStates; ++rowInd) {
        // for actions
            // create vector of size nnzPerAction with random values, normalize vector
            // generate random indices for these values in the range [i, (i+1)*numStates)
            // insert values into matrix

        for(idx_t actionInd = 0; actionInd < numActions; ++actionInd) {
            idx_t nnzPerAction = static_cast<idx_t>((sparsityFactor + normal_dis(gen2)) * numStates);
            //printf("row: %d, action: %d, nnz: %d\n", rowInd, actionInd, nnzPerAction);
            if(nnzPerAction > numStates) nnzPerAction = numStates;
            else if(nnzPerAction < 1) nnzPerAction = 1;

            std::vector<double> values(nnzPerAction);
            double sum = 0;

            // generate random values
            // TODO: perturb values for less uniformity
            for(idx_t i = 0; i < nnzPerAction; ++i) {
                double val = uniform_dis(gen);
                if(val < perturbFactor) {
                    val *= 5;
                }
                values[i] = val;
                sum += val;
            }

            // normalize values s.t. they are row-stochastic
            for(idx_t i = 0; i < nnzPerAction; ++i) {
                values[i] /= sum;
            }

            // generate random indices
            std::vector<idx_t> indices(nnzPerAction, std::numeric_limits<idx_t>::max());
            idx_t ind = 0;
            for(auto &i : indices) {
                do {
                    ind = actionInd * numStates + ind_dis(gen3); // global index
                } while (std::find(indices.begin(), indices.end(), ind) != indices.end()); // assert index is unique
                i = ind;
            }

            // insert values into matrix
            idx_t row[1] = {rowInd};
            MatSetValues(A, 1, row, nnzPerAction, indices.data(), values.data(), INSERT_VALUES);
            MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        }
    }
}

#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_TRANSITIONMATRIXGENERATOR_H
