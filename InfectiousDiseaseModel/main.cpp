//
// Created by robin on 05.07.23.
//

// Run: cd build; ./infectious_disease_model -options_file ../InfectiousDiseaseModel/petsc_options.txt

#include <boost/math/distributions/binomial.hpp>
#include "InfectiousDiseaseModel.h"
#include "../utils/Timer.h"

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    Timer t;

    // model setup
    InfectiousDiseaseModel idm;
    idm.setValuesFromOptions();
    t.start();
    idm.generateStageCosts();
    t.stop("Generating stage costs took: ");
    t.start();
    idm.generateTransitionProbabilities();
    t.stop("Generating transition probabilities took: ");

    // run iPI
    t.start();
    idm.inexactPolicyIteration();
    // idm.benchmarkIPI(V0, optimalPolicy, optimalCost);
    t.stop("Inexact policy iteration took: ");

    // change discountFactor to 0.1
    PetscOptionsSetValue(PETSC_NULLPTR, "-discountFactor", "0.1");
    idm.setValuesFromOptions();
    t.start();
    idm.inexactPolicyIteration();
    t.stop("Inexact policy iteration took: ");

    // change discount factor using member function
    idm.setOption("-discountFactor", "0.9999");
    t.start();
    idm.inexactPolicyIteration();
    t.stop("Inexact policy iteration took: ");

    idm.~InfectiousDiseaseModel();
    PetscFinalize();
    return 0;
}