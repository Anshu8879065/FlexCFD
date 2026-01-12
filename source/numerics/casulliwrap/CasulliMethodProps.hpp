#pragma once

#include <concepts>
#include <cstddef>

namespace fcfd::pdenumerics
{

// Minimal method properties required by SolTool.
// These are NUMERICAL controls, not physics.
//
// Physics stays in pdemodel::ModelParams (g, nm, radi, gammat,...)
// Fu + assembly are inside CasulliWrap.
template<std::floating_point Real>
struct CasulliMethodProps
{
  // Grid / time
  Real dt{Real(0.0)};
  Real dx{Real(0.0)};
  Real dy{Real(0.0)}; // ignored for 1D

  // Newton controls
  int  newtonMaxIts{50};
  Real newtonTol{Real(1e-10)};

  // Linear solver controls (2D uses CG in this implementation)
  int  cgMaxIts{500};
  Real cgTol{Real(1e-10)};

  // Wet/dry
  Real dryTol{Real(1e-12)};

  // Steps
  int nSteps{1};
};

} // namespace fcfd::pdenumerics
