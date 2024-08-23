#pragma once
#include <mpi.h>
#include <petscsys.h>
#include <type_traits>
#include <utility>

// Utility function to handle comm conversion
template <typename comm_T> MPI_Comm convertComm(comm_T comm_arg)
{
    MPI_Comm comm;
    if constexpr (std::is_integral<comm_T>::value) {
        if (comm_arg == 0) {
            comm = PETSC_COMM_WORLD;
        } else {
            comm = MPI_Comm_f2c(comm_arg);
        }
    } else {
        comm = comm_arg;
    }
    return comm;
}

inline std::pair<int, int> mpi_rank_size(int comm_arg)
{
    MPI_Comm comm = convertComm(comm_arg);
    int      rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    return std::make_pair(rank, size);
}
