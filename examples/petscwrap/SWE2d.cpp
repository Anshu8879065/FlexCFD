#include <cstdlib>
#include <iostream>

#include "pdes/swe/SWE2d.hpp"

#include "numerics/petscwrap/PetscInitException.hpp"
#include "numerics/petscwrap/PetscWrap.hpp"
#include "pdes/ModelParams.hpp"

auto main(int argc, char* argv[]) -> int
{
  using fcfd::pdemodel::Model2dParams;
  using fcfd::pdemodel::SWE2d;
  using fcfd::pdenumerics::PetscInitException;
  using fcfd::pdenumerics::PetscWrap;

  static constexpr char help[] = "2D Shallow Water Equations.  Option prefix -ptn_.\n"
                                   "Incorporates form  F(t,Y,dot Y) = G(t,Y)  where F() is IFunction and G() is\n"
                                   "RHSFunction().  Implements IJacobian() and RHSJacobian().  Defaults to\n"
                                   "ARKIMEX (= adaptive Runge-Kutta implicit-explicit) TS type.\n\n";

  auto swe2d = std::make_unique<SWE2d<double, Model2dParams<double>>>();

  try {
    auto pwrap = PetscWrap<Model2dParams<double>>(argc, argv, help);
    pwrap.SetPDE(std::move(swe2d));

    PetscInt L = 10;
    PetscReal Nx = 200;  // Nx == Ny
    PetscReal dx = L / (Nx - 1);  // dx==dy

    std::optional<std::vector<PetscReal>> stengrid =
      std::vector<PetscReal> {-PetscReal(0.5) / dx, PetscReal(0.0), PetscReal(0.5) / dx};

    std::optional<std::vector<PetscReal>> stengridd =
      std::vector<PetscReal> {PetscReal(1.0) / (dx * dx), -PetscReal(2.0) / (dx * dx), PetscReal(1.0) / (dx * dx)};

    std::optional<std::vector<PetscReal>> stengridd3 = std::nullopt;

    pwrap.SetGrid(2,
      {DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED},
      {static_cast<PetscInt>(Nx), static_cast<PetscInt>(Nx), 3, 1},
      stengrid,
      stengridd,
      stengridd3);
    pwrap.SetFields({"h", "qx", "qy"});
    pwrap.InitializeTS();
    pwrap.MakeProbFuns();
    pwrap.SetSolOpts();
    pwrap.NumericalSolve();

    auto interpol = [](std::vector<double> v) -> std::vector<double>
    {
      return v;  // identity: prints h, qx, qy
    };
    pwrap.EvaluateSolution(interpol);
  }
  catch (PetscInitException const& e) {
    std::clog << "Exception: " << e.what() << '\n';
  }

  return EXIT_SUCCESS;
}
