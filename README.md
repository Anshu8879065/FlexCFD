# FlexCFD
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-20-blue)](https://isocpp.org)
[![Build](https://img.shields.io/badge/build-CMake-brightgreen)](https://cmake.org)
[![PETSc](https://img.shields.io/badge/dependency-PETSc-orange)](https://petsc.org/)


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

## Repository structure

```text
cfd/
├── BUILDING.md                      # Build instructions and dependency notes
├── CMakeLists.txt                   # Top-level CMake entry (configures whole project)
├── CMakePresets.json                # Presets for common CMake configure/build setups
├── CODE_OF_CONDUCT.md               # Community guidelines
├── CONTRIBUTING.md                  # Contribution workflow and rules
├── Dockerfile                       # Optional dev container / CI-friendly build environment
├── HACKING.md                       # Developer notes and internal conventions
├── README.md                        # Project overview (this file)
├── cmake/                           # CMake helper modules (lint/docs/install/etc.)
│   ├── configure.cmake              # Project configuration helpers
│   ├── coverage.cmake               # Coverage flags/targets
│   ├── dev-mode.cmake               # Developer-mode toggles (sanitizers, warnings, etc.)
│   ├── docs-ci.cmake                # Docs CI helpers
│   ├── docs.cmake                   # Docs build helpers
│   ├── folders.cmake                # Folder/layout helper utilities
│   ├── install-config.cmake         # Install configuration helpers
│   ├── install-rules.cmake          # Install rules for targets/headers
│   ├── lint-targets.cmake           # Lint targets (clang-tidy/format hooks etc.)
│   ├── lint.cmake                   # Lint configuration
│   ├── prelude.cmake                # Common CMake prelude utilities
│   ├── project-is-top-level.cmake   # Detect top-level vs subproject
│   ├── spell-targets.cmake          # Spell-check targets (if enabled)
│   ├── spell.cmake                  # Spell-check configuration
│   └── variables.cmake              # Shared CMake variables
├── docs/                            # Documentation configuration (Doxygen/Sphinx)
│   ├── Doxyfile.in                  # Doxygen template
│   ├── conf.py.in                   # Sphinx config template
│   └── pages/
│       └── about.dox                # Project docs page stub
├── examples/                        # Example executables (small drivers)
│   ├── CMakeLists.txt               # Builds examples targets
│   ├── Casulliwrap/                 # Examples using CasulliWrap backend
│   │   ├── SVECas.cpp               # 1D SVE (Casulli-form) example driver
│   │   └── SWE2dCas.cpp             # 2D SWE (Casulli-form) example driver
│   └── petscwrap/                   # Examples using PetscWrap backend
│       ├── SVE.cpp                  # 1D SVE example using PETSc TS/DMDA
│       └── SWE2d.cpp                # 2D SWE example using PETSc TS/DMDA
└── source/                          # Library code (headers/impl)
    ├── CMakeLists.txt               # Builds the core libraries under source/
    ├── numerics/                    # Numerical backends + solver utilities
    │   ├── CMakeLists.txt           # Builds numerics library
    │   ├── GetIndex.hpp             # Structured-grid linear indexing helpers
    │   ├── IterCallbacks.hpp        # Iteration/evaluation callback helpers
    │   ├── SolTools.hpp             # Base interface for solver tools/wrappers
    │   ├── casulliwrap/             # Casulli semi-implicit SWE backend
    │   │   ├── CasulliMethodProps.hpp # Method properties/options for Casulli solver
    │   │   └── CasulliWrap.hpp        # CasulliWrap solver wrapper
    │   └── petscwrap/               # PETSc TS/DMDA backend wrapper
    │       ├── PetscInitException.hpp # PETSc init error wrapper
    │       ├── PetscOptions.hpp       # PETSc options helper structs/utilities
    │       ├── PetscUtils.cpp         # PETSc helper implementations
    │       ├── PetscUtils.hpp         # PETSc helper declarations
    │       └── PetscWrap.hpp          # PetscWrap interface (DMDA + TS glue)
    ├── pdes/                        # Physics layer (PDE models + parameters)
    │   ├── BoundaryCondition.hpp     # Boundary condition types/utilities
    │   ├── CMakeLists.txt            # Builds PDE model library
    │   ├── GridParams.hpp            # Grid configuration/metadata structs
    │   ├── ModelParams.hpp           # Model parameter structs (1D/2D params)
    │   ├── PDEParams.hpp             # PDE parameter container/common settings
    │   ├── PDESystem.hpp             # Base PDE system interface (physics API)
    │   ├── PDEType.hpp               # PDE type enum/identifiers
    │   ├── sve/                      # 1D Saint-Venant equations
    │   │   ├── SVE.hpp               # SVE definition (PETSc-oriented form)
    │   │   └── SVECas.hpp            # SVE Casulli-form (CasulliWrap-oriented form)
    │   └── swe/                      # 2D Shallow Water equations
    │       ├── SWE2d.hpp             # SWE2d definition (PETSc-oriented form)
    │       └── SWE2dCas.hpp          # SWE2d Casulli-form (CasulliWrap-oriented form)
    └── utils/                       # Small general utilities shared by numerics/pdes
        ├── CMakeLists.txt            # Builds utils (if you compile it as a target)
        ├── IterTools.hpp             # Iteration primitives / iterator state helpers
        ├── SquareRange.hpp           # 2D range helper
        └── TriangleRange.hpp         # Triangular iteration helper

## Background

This project implements two solvers for PDE systems: CasulliWrap and PetscWrap. PetscWrap is a PDE-agnostic wrapper around [PETSc](https://petsc.org/release/) routines.

As it is PDE-agnostic, to implement a solver for your particular PDE, you have to extend the PDESystem class found in PDEs/PDESystem.hpp, through which you will provide implementations for functions that define how boundary conditions, initial conditions, etc. are handled. These procedures are then fed in to PETSc through PetscWrap for its use.

Examples of solvers for PDEs are present in the PDEs/SVE/SVE.hpp file, which implements the Saint-Venant equations, and PDEs/SWE/SWE2d.hpp file, which implements the 2D Shallow Water equations.

# Building and installing

See the [BUILDING](BUILDING.md) document.

