//
// Created by robin on 09.04.23.
//

#ifndef DISTRIBUTED_INEXACT_POLICY_ITERATION_FILEWRITER_H
#define DISTRIBUTED_INEXACT_POLICY_ITERATION_FILEWRITER_H

#include <petscmat.h>

void matrixToBin(const Mat& A, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_WRITE, &viewer);
    MatView(A, viewer);
    PetscViewerDestroy(&viewer);
}

void vectorToBin(const Vec& v, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_WRITE, &viewer);
    VecView(v, viewer);
    PetscViewerDestroy(&viewer);
}

void matrixToAscii(const Mat& A, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer);
    MatView(A, viewer);
    PetscViewerDestroy(&viewer);
}

void vectorToAscii(const Vec& v, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer);
    VecView(v, viewer);
    PetscViewerDestroy(&viewer);
}

void matrixFromBin(Mat& A, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ, &viewer);
    MatLoad(A, viewer);
    PetscViewerDestroy(&viewer);
}

void vectorFromBin(Vec& v, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename.c_str(), FILE_MODE_READ, &viewer);
    VecLoad(v, viewer);
    PetscViewerDestroy(&viewer);
}

void matrixFromAscii(Mat& A, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer);
    MatLoad(A, viewer);
    PetscViewerDestroy(&viewer);
}

void vectorFromAscii(Vec& v, const std::string& filename) {
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer);
    VecLoad(v, viewer);
    PetscViewerDestroy(&viewer);
}


#endif //DISTRIBUTED_INEXACT_POLICY_ITERATION_FILEWRITER_H
