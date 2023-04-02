//
// Created by robin on 02.04.23.
//

#include <iostream>
#include <string>
#include <petsc.h>



int main(int argc, char *argv[]) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "Hello World from using PETSc!\n";

    PetscFinalize();
    return 0;
}