#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <petscsystypes.h>
#include <petscts.h>

namespace fcfd::pdenumerics
{

struct PetscOptions
{
  //! A string with the name of a PETSc TS method (the time/ODE integrators that PETSc provides)
  TSType tsType {};

  //! The number of unknown fields in the PDE system
  std::size_t numFields = 0;

  //! The names of the unknown fields in the PDE system
  std::vector<std::string> fields;
};

struct PetscSolveOpts
{
  TSType tsType = TSARKIMEX;
  PetscReal initTime = 0.0;
  PetscReal maxTime = 10.0;
  PetscReal timeStep = 0.1;
  TSExactFinalTimeOption finalTimeOption = TS_EXACTFINALTIME_MATCHSTEP;
};

}  // namespace fcfd::pdenumerics
