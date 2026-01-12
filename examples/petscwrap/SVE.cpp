#include <cstdlib>
#include <iostream>

#include "pdes/sve/SVE.hpp"

#include "numerics/petscwrap/PetscInitException.hpp"
#include "numerics/petscwrap/PetscWrap.hpp"
#include "pdes/ModelParams.hpp"

auto main(int argc, char* argv[]) -> int
{
  using fcfd::pdemodel::Model1dParams;
  using fcfd::pdemodel::SVE;
  using fcfd::pdenumerics::PetscInitException;
  using fcfd::pdenumerics::PetscWrap;

  static constexpr char help[] = "Coupled Saint-Venant System.  Option prefix -ptn_.\n"
                                   "Incorporates form  F(t,Y,dot Y) = G(t,Y)  where F() is IFunction and G() is\n"
                                   "RHSFunction().  Implements IJacobian() and RHSJacobian().  Defaults to\n"
                                   "ARKIMEX (= adaptive Runge-Kutta implicit-explicit) TS type.\n\n";

  auto sve = std::make_unique<SVE<double, Model1dParams<double>>>();

  try {
    auto pwrap = PetscWrap<Model1dParams<double>>(argc, argv, help);
    pwrap.SetPDE(std::move(sve));

    PetscInt L = 10;
    PetscReal Nx = 200;
    PetscReal dx = L / (Nx - 1);

    // First-derivative stencil weights for [i-1, i, i+1]
    std::optional<std::vector<PetscReal>> stengrid =
      std::vector<PetscReal> {-PetscReal(0.5) / dx, PetscReal(0.0), PetscReal(0.5) / dx};
    std::vector<PetscReal> stengridd = {
      1.0 / (dx * dx),  // u_{i-1}
      -2.0 / (dx * dx),  // u_i
      1.0 / (dx * dx)  // u_{i+1}
    };

    // Not used in code path -> nullopt
    std::optional<std::vector<PetscReal>> stengridd3 = std::nullopt;

    pwrap.SetGrid(1,
      {DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED},
      {static_cast<PetscInt>(Nx), 2, 1},
      stengrid,
      stengridd,
      stengridd3);
    pwrap.SetFields({"h", "qx"});
    pwrap.InitializeTS();
    pwrap.MakeProbFuns();
    pwrap.SetSolOpts();
    pwrap.NumericalSolve();

    // ---- print solution ----
    auto interpol = [](std::vector<double> v) -> std::vector<double>
    {
      return v;  // identity: prints h and qx
    };
    pwrap.EvaluateSolution(interpol);
    // ------------------------
  }
  catch (PetscInitException const& e) {
    std::clog << "Exception: " << e.what() << '\n';
  }

  return EXIT_SUCCESS;
}
