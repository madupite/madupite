#include "MDP.h"

int main() {

    int states = 3, actions = 2;

    MDP mdp(states, actions);
    mdp.generateTransitionMatrix();
    mdp.generateStageCosts();
    mdp.outputInfo();


    return 0;
}
