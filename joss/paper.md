---
title: 'madupite: A High-Performance Distributed Solver for Large-Scale Markov Decision Processes'
tags:
  - Python
  - C++
  - PETSc
  - Markov decision processes
  - dynamic programming
  - inexact policy iteration methods
  - high-performance computing
  - distributed computing
authors:
  - name: Matilde Gargiani
    orcid: 0000-0001-8615-6214
    equal-contrib: true
    corresponding: true 
    affiliation: 1
  - name: Philip Pawlowsky
    equal-contrib: true 
    corresponding: true
    affiliation: 1
  - name: Robin Sieber
    orcid: 0009-0002-8592-8387
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Václav Hapla
    orcid: 0000-0002-9190-2207
    corresponding: false
    affiliation: "2, 3"
  - name: John Lygeros
    orcid: 0000-0002-6159-1962
    corresponding: true
    affiliation: 1
affiliations:
 - name: Automatic Control Laboratory (IfA), ETH Zurich, 8092 Zurich, Switzerland 
   index: 1
 - name: Department of Earth and Planetary Sciences, ETH Zurich, 8092 Zurich, Switzerland 
   index: 2
 - name: Department of Applied Mathematics, FEECS at VSB-TU Ostrava, Czechia 
   index: 3
date: 9 September 2024
bibliography: paper.bib

# We have not submitted this work to any other journal.
---

![Logo of ``madupite``.](madupite_logo.png){ width=45% }

# Summary

We propose `madupite`, a distributed high-performance solver for Markov Decision Processes (MDPs).
MDPs are a powerful mathematical tool to model a variety of problems arising in different fields [@realworld_mdp; @mdp_methods_applications], from finance [@mdp_finance] to epidemiology [@mdp_epidemiology] and traffic control [@mdp_trafficcontrol]. In general terms, MDPs are used to mathematically characterize dynamical systems whose state is evolving in time as a consequence of actions that we play on the system and disturbances that are acting on it. The goal is generally to select actions in order to minimize in expectation a certain cumulative discounted metric over time, *e.g.*, deviations from a reference state and/or costs incurred by selecting certain actions in specific states of the system [@bertsekas_book_2; @bellman_book]. 

MDPs arising from real-world applications tend to be extremely high-dimensional and in some cases the number of states in the system grows exponentially with the number of certain parameters. This phenomenon is known as *curse-of-dimensionality*  and it is generally tackled in reinforcement learning with the deployment of function approximations [@sutton_RL]. The deployment of the latter leads to a smaller size optimization problem since, instead of optimizing for the number of states, there is only need to optimize for the number of parameters deployed in the function approximation, which is generally much smaller than the original state space size. This comes at the price of introducing sub-optimality with respect to the original solution. 

# Statement of need

Modern high-performance clusters and super-computers offer the possibility of simulating, storing and solving gigantic size MDPs. To exploit modern computational resources, solution methods which can efficiently distribute the computation as well as adequate high-performance software packages are needed. Even though there are a number of toolboxes to solve MDPs, such as `pymdptoolbox` [@mdptoolbox] and the recent `mdpsolver` [@mdp_solver], to the best of our knowledge there is no existing solver that combines high-performance distributed computing with the possibility of selecting a solution method that is tailored to the application at hand. `pymdptoolbox` is coded in plain Python and this results in poor scalability. In addition, it does not support parallel and distributed computing. `mdpsolver` is instead written in C++, supports parallel computing and comes with a user-friendly Python API. On the other hand, the solution methods available are limited to modified policy iteration, a dynamic programming (DP) method which has shown poor performance for a significant class of problems of practical interest [@gargiani_igmrespi_2023; @gargiani_ipi_2024]. On the other hand, `mdpsolver` makes certain implementation choices that limit its applicability; *e.g.*, matrices with values and indices being stored in nested `std::vector` independently of their sparsity degree and thus precluding the use of available optimized linear algebra routines. Finally, `mdpsolver` and `madupite` were developed during the same time-frame, which makes them concurrent contributions.

The vision behind our solver is to enable the solution of large-scale MDPs with more than a million states by exploiting modern computational resources. `madupite` is a high-performance distributed solver which is capable of efficiently distributing the memory load and computation, comes with a wide range of choices for solution methods enabling the user to select the one that is best tailored to its specific application, and, last but not least, its core is in C++ but it is equipped with a user-friendly Python API to enable its deployment to a broader range of users. 
The combination of distributed computing, user friendly Python API, and support for a wide range of methods in `madupite` will enable researchers and engineers to solve exactly gigantic scale MDPs which previously could only be tackled via function approximations.

# Problem Setting and Solution Methods

`madupite` solves infinite-horizon discounted MDPs with finite state and action spaces. This problem class can be efficiently tackled with *inexact policy iteration methods* (iPI) [@gargiani_igmrespi_2023; @gargiani_ipi_2024]. These methods are a variant of policy iteration [@bertsekas_book_2], where inexactness is introduced at the policy evaluation step for scalability reasons. iPI methods are general enough to embrace standard DP methods, such as value iteration and modified policy iteration, which are the main solution methods of `mdpsolver`. The interested readers should refer to [@gargiani_ipi_2024] for an in-depth description of the problem setting and a thorough mathematical analysis of iPI methods. The versatility of iPI methods makes them particularly suited to efficiently solve different large-scale instances, while their structure is also favorable for distributed implementations, which are needed to exploit high-performance computing clusters.

The great flexibility of `madupite` on the algorithmic side relies on the possibility of customizing the iPI method deployed for the specific problem at hand by tuning the level of inexactness and selecting among a wide range of inner solvers for the approximate policy evaluation step (step 8 of Algorithm 3 in [@gargiani_ipi_2024]). It is indeed empirically and theoretically demonstrated that, depending on the specific structure of the problem, different inner solvers may enhance the convergence performance [@gargiani_igmrespi_2023; @gargiani_ipi_2024].   

# Implementation 

The core of `madupite` is written in C++ and relies on PETSc (Portable, Extensible Toolkit for Scientific Computation) for the distributed implementation of iPI methods [@petsc-web-page; @petsc-user-ref; @petsc-efficient]. PETSc is an open-source high-performance C library and, despite being developed specifically for solving partial differential equations, it comes with a wide range of highly-optimized linear system solvers and memory-efficient sparse linear algebra data types and routines that can be used for a variety of problems, including DP. We rely on PETSc also as it natively adopts a distributed memory parallelism using the MPI standard to enable scalability beyond one CPU, which is one of the innovative key-features of our solver with respect to its competitors. MPI allows users to abstract the parallelism away from the underlying hardware and enables them to run the same code in parallel on their personal device or even on multiple nodes of a high-performance computing cluster. `madupite` itself is a fully-featured C++20 library and, by leveraging the `nanobind` binding library [@nanobind], offers an equivalent API in Python that can be installed as a package using `pip`.

Finally, ``madupite`` allows the user to create an MDP by loading offline data collected from previously run experiments as well as from online simulations, offering the possibility of carrying out both the simulations and the solution in a completely parallel/distributed fashion. More details on how to use ``madupite`` and all of its functionalities can be found in its documentation and in the ``examples`` folder, where we provide an extensive selection of code-examples on how to use this library in Python and C++.

# Acknowledgements

This work was supported by the European Research Council under the Horizon 2020 Advanced under Grant 787845 (OCAL) and by the SNSF through NCCR Automation (Grant Number 180545).

# References
