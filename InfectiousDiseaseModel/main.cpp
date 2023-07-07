//
// Created by robin on 05.07.23.
//

#include <iostream>
#include <boost/math/distributions/binomial.hpp>
#include "InfectiousDiseaseModel.h"
#include "../utils/Timer.h"

int main(int argc, char **argv) {
#if 0
    int n = 10; // Number of trials
    double p = 0.5; // Probability of success

    // Create a binomial distribution with n trials and probability of success p
    boost::math::binomial_distribution<double> binom(n, p);

    std::cout << "Binomial probabilities for " << n << " trials with probability of success " << p << ":\n";
    for (int k = 0; k <= n; ++k) {
        double prob = boost::math::pdf(binom, k);
        std::cout << "P(X = " << k << ") = " << prob << "\n";
    }
#endif


    PetscInitialize(&argc, &argv, nullptr, nullptr);

    Timer t;

    InfectiousDiseaseModel idm;
    idm.setValuesFromOptions();

    t.start();
    idm.generateStageCosts();
    t.stop("Generating stage costs took: ");
    //MatView(idm.stageCostMatrix_, PETSC_VIEWER_STDOUT_WORLD);
    t.start();
    idm.generateTransitionProbabilities();
    t.stop("Generating transition probabilities took: ");
#if 0
    // save to csv file
    std::ofstream file("transition_probabilities.csv");
    PetscReal value;
    for(int i = 0; i < idm.numStates_; ++i) {
        const PetscInt row = i * idm.numActions_;
        for(int j = 0; j < idm.numStates_; ++j) {
            MatGetValues(idm.transitionProbabilityTensor_, 1, &row, 1, &j, &value);
            file << value << ",";
        }
        file << std::endl;
    }
    file.close();
#endif

    Vec V0;
    VecCreateMPI(PETSC_COMM_WORLD, idm.localNumStates_, idm.numStates_, &V0);
    VecSet(V0, 1.0);

    Vec optimalCost;
    IS optimalPolicy;
    t.start();
    idm.inexactPolicyIteration(V0, optimalPolicy, optimalCost);
    t.stop("Inexact policy iteration took: ");
    PetscPrintf(PETSC_COMM_WORLD, "Optimal cost:\n");
    VecView(optimalCost, PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Optimal policy:\n");
    ISView(optimalPolicy, PETSC_VIEWER_STDOUT_WORLD);

    idm.writeVec(optimalCost, idm.file_cost_);
    idm.writeIS(optimalPolicy, idm.file_policy_);

    idm.~InfectiousDiseaseModel();
    PetscFinalize();

    return 0;
}