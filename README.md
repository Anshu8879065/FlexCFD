# FlexCFD

FlexCFD is a modular C++ framework for the development of
finite-difference numerical solvers for partial differential
equations, with a focus on **computational fluid dynamics (CFD)** and
**shallow water equations (SWE)**.

The framework emphasizes:
- clear separation between **physics** and **numerics**
- structured-grid discretizations
- scalable solvers based on **PETSc**

FlexCFD is intended for **research, prototyping, and educational use**.


## Repository Structure

```text
FlexCFD/
├── petscwrap/      # PETSc-based numerical infrastructure
├── casulliwrap/   # Casulli-type SWE discretization
└── README.md

See petscwrap/README.md for PETSc infrastructure.
See casulliwrap/README.md for Casulli SWE discretization.

