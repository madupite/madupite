#include <petsc.h>

int rankPETSCWORLD() {
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  return rank;
}
