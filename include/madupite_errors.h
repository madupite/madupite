#pragma once

#include <exception>

#include <mpi.h> //TODO this should not be needed if petscerrror.h includes mpi.h
#include <petscerror.h>

class PetscException : public std::exception {
    int         ierr;
    std::string message;

public:
    PetscException(int ierr, const std::string& message)
        : ierr(ierr)
        , message(message)
    {
    }

    const char* what() const noexcept override { return message.c_str(); }

    int code() const noexcept { return ierr; }
};

class MadupiteException : public std::exception {
    std::string message;

public:
    MadupiteException(const std::string& message)
        : message(message)
    {
    }

    const char* what() const noexcept override { return message.c_str(); }
};

#define PetscCallNoThrow(...)                                                                                                                        \
    do {                                                                                                                                             \
        PetscStackUpdateLine;                                                                                                                        \
        PetscErrorCode ierr = __VA_ARGS__;                                                                                                           \
        if (PetscUnlikely(ierr != PETSC_SUCCESS)) {                                                                                                  \
            PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_IN_CXX, PETSC_NULLPTR);                           \
        }                                                                                                                                            \
    } while (0)

#define PetscCallThrow(...)                                                                                                                          \
    do {                                                                                                                                             \
        PetscStackUpdateLine;                                                                                                                        \
        PetscErrorCode ierr = __VA_ARGS__;                                                                                                           \
        if (PetscUnlikely(ierr != PETSC_SUCCESS)) {                                                                                                  \
            char* msg;                                                                                                                               \
            PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_IN_CXX, PETSC_NULLPTR);                           \
            PetscErrorMessage(ierr, PETSC_NULLPTR, &msg);                                                                                            \
            throw PetscException(ierr, std::string(msg));                                                                                            \
        }                                                                                                                                            \
    } while (0)

#define PetscThrow(comm, ierr, ...)                                                                                                                  \
    do {                                                                                                                                             \
        char* msg;                                                                                                                                   \
        PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_INITIAL, __VA_ARGS__);                                           \
        PetscErrorMessage(ierr, PETSC_NULLPTR, &msg);                                                                                                \
        throw PetscException(ierr, std::string(msg));                                                                                                \
    } while (0)
