#pragma once

#include <string>
#include <vector>

#include <petsc/petsc.h>

namespace fcfd::pdenumerics
{

struct PetscOptions
{
  //! A string with the name of a PETSc TS method (the time/ODE integrators that PETSc provides)
  TSType tsType {};

  //! The number of options in the options database
  int numOptions = 0;

  //! A set of options in the options database that are related and should be displayed on the same
  //! window of a GUI that allows the user to set the options interactively
  std::vector<std::string> options;
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
