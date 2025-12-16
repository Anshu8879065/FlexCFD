# FlexCFD

**FlexCFD** is a modular C++ framework for solving partial differential equations (PDEs) using finite-difference methods, with a focus on **computational fluid dynamics (CFD)** and **shallow water equations (SWE)**. It provides structured-grid discretizations, scalable solvers, and modular tools for research, prototyping, and educational purposes.

---

## Key Features

- Separation of **physics** (PDE definitions) and **numerics** (solvers and iterative methods)
- Support for **shallow water equations (SWE)** and general PDE systems
- Scalable solvers with **PETSc** integration
- Modular design with reusable tools for different numerical schemes
- Includes **Casulli-type SWE discretization** and solver utilities
- Supports **unit testing** and example simulations

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-20-blue)](https://isocpp.org)
[![Build](https://img.shields.io/badge/build-CMake-brightgreen)](https://cmake.org)
[![PETSc](https://img.shields.io/badge/dependency-PETSc-orange)](https://petsc.org/)

## Repository Structure

```text
FlexCFD/
├── SolTools/                 # Solver tools and wrappers
│   ├── casulliwrap/          # Casulli-type SWE solver
│   │   ├── Casulliwrap.hpp
│   │   └── CG.hpp             # Conjugate Gradient solver for SPD systems
│   └── petscwrap/            # PETSc-based solver wrappers
│       ├── GetIndex.hpp
│       ├── IterCallbacks.hpp
│       ├── PetscInitException.hpp
│       ├── PetscOptions.hpp
│       └── PetscWrap.hpp
│   └── SolTools.hpp          # Base class for solver utilities
├── PdeSystem/                # PDE model definitions
│   ├── BoundaryCondition.hpp
│   ├── GridParams.hpp
│   ├── ModelParams.hpp
│   ├── PDEParams.hpp
│   ├── PDESystem.hpp
│   ├── PDEType.hpp
│   ├── SVE.hpp
│   └── SWE.hpp
├── Utils/                    # Utility headers
│   └── itertools/            # Iteration tools
│       ├── itertools.hpp
│       ├── SquareRange.hpp
│       └── TriangleRange.hpp
├── extras/                   # Example programs and scripts
│   ├── FlexCFD.cpp
│   ├── FlexCFD.hpp
│   ├── FlexCFD_types.hpp
│   ├── ConsoleApplication.cpp
│   ├── Makefile
│   └── saintvenant/          # SWE example code
├── external/                 # External dependencies
│   └── petsc/                # PETSc installation/config files
├── docs/                     # Documentation
│   ├── conf.py.in
│   ├── Doxyfile.in
│   └── pages/                # Project-specific documentation
├── cmake/                    # CMake modules and utilities
├── test/                     # Unit and integration tests
│   └── source/
│       └── flexcfd_test.cpp
├── .github/                  # CI/CD workflows and scripts
└── README.md                 # This file
