# CasulliWrap

`CasulliWrap` provides a C++ implementation and wrapper for
Casulli-type numerical schemes for the shallow water equations (SWE).
The focus is on structured-grid discretizations and robust treatment
of source terms, following the philosophy of semi-implicit and
well-balanced finite difference methods.

This module is designed to be used as part of the **FlexCFD** framework,
but it can also be studied independently for research and development
purposes.

---

## Features

- Shallow Water Equations (SWE) formulation
- Casulli-type semi-implicit and Euler lagrangian discretization
- Structured grid support
- Clear separation between:
  - PDE model definitions
  - Numerical operators
  - Iteration utilities

---

## Folder Structure

```text
casulliwrap/
├── Source/
│   ├── Itertools/        # Iteration utilities (ranges, indexing)
│   ├── PdeModel/         # PDE definitions and parameters
│   └── PdeNeumerics/     # Numerical operators and wrappers
└── README.md
