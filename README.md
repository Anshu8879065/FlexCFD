# FlexCFD

FlexCFD is a modular C++ framework for developing finite-difference numerical solvers for partial differential equations, focusing on **computational fluid dynamics (CFD)** and **shallow water equations (SWE)**.

## Key Features

- Clear separation between **physics** and **numerics**
- Structured-grid discretizations
- Scalable solvers based on **PETSc**
- Suitable for **research, prototyping, and educational use**

## Repository Structure

```text
FlexCFD/
├── casulliwrap/   # Casulli-type SWE discretization
├── cmake/         # CMake modules
├── docs/          # Documentation
├── extras/        # Example codes and utilities
├── external/      # External dependencies (e.g., PETSc)
├── PdeSystem/     # PDE model definitions
├── SolTools/      # Solver tools and wrappers
├── test/          # Unit and integration tests
├── Utils/         # Utility headers
└── README.md
