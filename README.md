# flexcfd
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-20-blue)](https://isocpp.org)
[![Build](https://img.shields.io/badge/build-CMake-brightgreen)](https://cmake.org)
[![PETSc](https://img.shields.io/badge/dependency-PETSc-orange)](https://petsc.org/)

This is the flexcfd project.

**FlexCFD** is a modular C++ framework for solving partial differential equations (PDEs) on structured grids using finite-difference methods, with a focus on **computational fluid dynamics (CFD)** and **shallow water equations (SWE)**.

It provides a clean separation between:
- **Physics** (PDE models, parameters, boundary conditions)
- **Numerics** (solvers, time integrators, discretization wrappers)

FlexCFD currently provides two solver backends:
- **PETScWrap**: PETSc TS/DMDA-based time integration
- **CasulliWrap**: Casulli-type semi-implicit SWE discretization

---

## Key Features

- Modular C++ design with header-driven PDE definitions
- Structured-grid finite-difference discretization
- PETSc integration (implicit / explicit / IMEX time stepping)
- Casulli-type SWE discretization support
- Example executables for PETScWrap and CasulliWrap
- CMake build system with presets and helper modules

---

## Repository Structure

```text
.
├── BUILDING.md                      # Build instructions and dependency notes
├── CMakeLists.txt                   # Top-level build entry (builds the main executables)
├── CMakePresets.json                # CMake presets for common configurations
├── CODE_OF_CONDUCT.md               # Community guidelines
├── CONTRIBUTING.md                  # Contribution workflow and rules
├── Dockerfile                       # Container setup (optional dev environment)
├── Example                           # Example executables and their local build glue
│   ├── Casulliwrap.cpp              # CasulliWrap example driver (Casulli backend demo)
│   ├── CmakeLists.txt               # Example-level build rules (Casulliwrap.cpp / petscmain.cpp targets)
│   └── petscmain.cpp                # PETScWrap driver executable (supports -dim 1 or -dim 2)
├── HACKING.md                       # Developer notes and internal conventions
├── Numerics                          # Numerical backends + solver utilities
│   ├── CMakeLists.txt               # Builds Numerics library (petscwrap/casulliwrap code)
│   ├── GetIndex.hpp                 # Structured-grid linear indexing utilities
│   ├── IterCallbacks.hpp            # Iteration/evaluation callback helpers
│   ├── SolTools.hpp                 # Base interface for solver tools/wrappers
│   ├── casulliwrap                  # Casulli-type SWE backend
│   │   ├── CasulliMethodProps.hpp   # Method properties/options for Casulli discretization
│   │   └── CasulliWrap.hpp          # CasulliWrap solver wrapper (semi-implicit SWE)
│   └── petscwrap                    # PETSc backend wrapper
│       ├── PetscInitException.hpp   # PETSc init error handling wrapper
│       ├── PetscOptions.hpp         # PETSc options helpers / option parsing
│       ├── PetscWrap.cpp            # PETScWrap implementation (.cpp side)
│       └── PetscWrap.hpp            # PETScWrap interface (DMDA + TS glue)
├── PDEs                              # Physics layer (PDE models + parameters)
│   ├── BoundaryCondition.hpp        # Boundary condition definitions
│   ├── CMakeLists.txt               # Builds PDE model library
│   ├── GridParams.hpp               # Grid configuration/metadata structs
│   ├── ModelParams.hpp              # Model-specific parameter structs (1D/2D model params)
│   ├── PDEParams.hpp                # PDE parameter containers/common settings
│   ├── PDESystem.hpp                # Base PDE system interface (physics API)
│   ├── PDEType.hpp                  # PDE type enum/identifiers
│   ├── SVE                           # 1D Saint-Venant equations
│   │   ├── SVE.hpp                  # SVE PDE definition (PETSc backend usage)
│   │   └── SVECas.hpp               # SVE Casulli-form (Casulli backend usage)
│   └── SWE                           # 2D Shallow Water equations
│       ├── SWE2d.hpp                # SWE2d PDE definition (PETSc backend usage)
│       └── SWE2dCas.hpp             # SWE2d Casulli-form (Casulli backend usage)
├── README.md                         # Project overview (this file)
├── Utils                             # Utility headers shared across the project
│   └── itertools                    # Iteration utilities (grid loops, ranges)
│       ├── CMakeLists.txt           # Builds itertools utilities
│       ├── IterTools.hpp            # Iteration primitives and states
│       ├── SquareRange.hpp          # 2D range helper
│       └── TriangleRange.hpp        # Triangular iteration helper
├── cmake                             # CMake helper modules (lint/docs/install/etc.)
│   ├── coverage.cmake
│   ├── dev-mode.cmake
│   ├── docs-ci.cmake
│   ├── docs.cmake
│   ├── folders.cmake
│   ├── install-config.cmake
│   ├── install-rules.cmake
│   ├── lint-targets.cmake
│   ├── lint.cmake
│   ├── prelude.cmake
│   ├── project-is-top-level.cmake
│   ├── spell-targets.cmake
│   ├── spell.cmake
│   └── variables.cmake
├── docs                              # Documentation configuration (Doxygen/Sphinx)
│   ├── Doxyfile.in                  # Doxygen template
│   ├── conf.py.in                   # Sphinx config template
│   └── pages
│       └── about.dox                # Project docs page stub
└── structure.txt                     # Repository structure notes / reference output


## Background

This project implements two solvers for PDE systems: CasulliWrap and PetscWrap. PetscWrap is a PDE-agnostic wrapper around [PETSc](https://petsc.org/release/) routines.

As it is PDE-agnostic, to implement a solver for your particular PDE, you have to extend the PDESystem class found in PDEs/PDESystem.hpp, through which you will provide implementations for functions that define how boundary conditions, initial conditions, etc. are handled. These procedures are then fed in to PETSc through PetscWrap for its use.

Examples of solvers for PDEs are present in the PDEs/SVE/SVE.hpp file, which implements the Saint-Venant equations, and PDEs/SWE/SWE2d.hpp file, which implements the 2D Shallow Water equations.

# Building and installing

See the [BUILDING](BUILDING.md) document.




