#pragma once
#include <filesystem>
#include <mpi.h>
#include <petscsys.h>
#include <sstream>
#include <string>
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

// returns filename_{i}.ext if filename.ext already exists such that no files are overwritten
inline std::string get_safe_filename(const std::string& filename, bool overwrite = false)
{
    if (filename.empty() || overwrite) {
        return filename;
    }

    std::filesystem::path filepath(filename);
    std::string           base_filename = filepath.stem().string();
    std::string           extension     = filepath.extension().string();
    std::string           directory     = filepath.parent_path().string();

    std::string new_filename = filename;
    int         counter      = 1;

    while (std::filesystem::exists(new_filename)) {
        std::stringstream ss;
        ss << directory;
        if (!directory.empty() && directory.back() != std::filesystem::path::preferred_separator) {
            ss << std::filesystem::path::preferred_separator;
        }
        ss << base_filename << "_" << counter << extension;
        new_filename = ss.str();
        counter++;
    }

    return new_filename;
}
